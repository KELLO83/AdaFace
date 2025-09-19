import argparse
import copy
import math
import os
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode as I
from tqdm import tqdm

from dataset.image_folder_dataset import CustomImageFolderDataset, cv2_loader
import net


SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp',
                        '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP')


def build_dataset_for_bn(root: str, transform, swap_color: bool, output_dir: str):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    has_class_dirs = any(entry.is_dir() for entry in os.scandir(root))
    if has_class_dirs:
        return CustomImageFolderDataset(
            root=root,
            transform=transform,
            low_res_augmentation_prob=0.0,
            crop_augmentation_prob=0.0,
            photometric_augmentation_prob=0.0,
            swap_color_channel=swap_color,
            output_dir=output_dir,
        )
    return FlatImageDataset(root=root, transform=transform, swap_color=swap_color)


class FlatImageDataset(Dataset):
    """Dataset for flat image directories (no class sub-folders)."""

    def __init__(self, root: str, transform, swap_color: bool):
        self.root = root
        self.transform = transform
        self.swap_color = swap_color
        self.paths: List[str] = []
        for pattern in SUPPORTED_EXTENSIONS:
            self.paths.extend(glob(os.path.join(root, f'*{pattern}')))
        self.paths.sort()
        if not self.paths:
            raise FileNotFoundError(f"No image files found in {root} (extensions={SUPPORTED_EXTENSIONS})")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        sample = cv2_loader(path)
        if self.swap_color:
            sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, 0


class LoRAConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError('LoRA rank must be > 0')
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        device = conv.weight.device

        # Clone original conv configuration
        self.base = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode,
        )
        self.base.load_state_dict(conv.state_dict())
        self.base.to(device)
        for param in self.base.parameters():
            param.requires_grad = False

        # LoRA parameters (low-rank update)
        in_features = (conv.in_channels // conv.groups) * conv.kernel_size[0] * conv.kernel_size[1]
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(conv.out_channels, rank, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        delta = torch.matmul(self.lora_B, self.lora_A)
        delta = delta.view_as(self.base.weight)
        result = result + F.conv2d(
            x,
            delta * self.scaling,
            bias=None,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=self.base.groups,
        )
        return result

    def merge_to_conv(self) -> nn.Conv2d:
        merged = nn.Conv2d(
            in_channels=self.base.in_channels,
            out_channels=self.base.out_channels,
            kernel_size=self.base.kernel_size,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=self.base.groups,
            bias=(self.base.bias is not None),
            padding_mode=self.base.padding_mode,
        )
        merged.load_state_dict(self.base.state_dict())
        merged.to(self.base.weight.device)
        with torch.no_grad():
            delta = torch.matmul(self.lora_B, self.lora_A)
            delta = delta.view_as(self.base.weight)
            merged.weight.copy_(self.base.weight + delta * self.scaling)
        return merged


def replace_conv_with_lora(module: nn.Module,
                           target_prefixes: Iterable[str],
                           rank: int,
                           alpha: float,
                           prefix: str = '') -> List[str]:
    replaced: List[str] = []
    for name, child in list(module.named_children()):
        child_prefix = f'{prefix}.{name}' if prefix else name
        if isinstance(child, nn.Conv2d) and _matches_prefix(child_prefix, target_prefixes):
            lora_module = LoRAConv2d(child, rank=rank, alpha=alpha)
            setattr(module, name, lora_module)
            replaced.append(child_prefix)
        else:
            replaced.extend(replace_conv_with_lora(child, target_prefixes, rank, alpha, child_prefix))
    return replaced


def _matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    prefixes = list(prefixes)
    if not prefixes:
        return True
    return any(name.startswith(prefix) for prefix in prefixes)


def collect_lora_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def merge_lora_modules(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, LoRAConv2d):
            merged = child.merge_to_conv()
            setattr(module, name, merged)
        else:
            merge_lora_modules(child)


def set_batchnorm_eval(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def disable_dropout(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.eval()


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.size(0)
        if batch_size < 2:
            raise ValueError('SupConLoss requires batch_size >= 2')

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.fill_diagonal_(0.0)

        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size, device=device))
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_counts = mask.sum(dim=1)
        valid = positive_counts > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


def run_adabn(backbone: nn.Module,
              data_loader: DataLoader,
              device: torch.device,
              num_passes: int) -> None:
    """Adaptive BatchNorm statistics re-estimation."""
    if num_passes <= 0:
        return

    set_batchnorm_eval(backbone)
    backbone.to(device)
    backbone.train()
    torch.set_grad_enabled(False)

    processed = 0
    skipped = 0
    for pass_idx in range(num_passes):
        for images, _ in tqdm(data_loader, desc=f'BN pass {pass_idx + 1}/{num_passes}'):
            if images.size(0) < 2:
                skipped += 1
                continue
            images = images.to(device, non_blocking=True)
            backbone(images)
            processed += 1
    torch.set_grad_enabled(True)
    if processed == 0:
        raise RuntimeError('AdaBN received no usable mini-batches (batch_size < 2).')
    if skipped > 0:
        print(f'[INFO] AdaBN skipped {skipped} batches with fewer than 2 samples.')


def build_backbone(arch: str, img_size: int) -> nn.Module:
    builders = {
        'ir_18': net.IR_18,
        'ir_34': net.IR_34,
        'ir_50': net.IR_50,
        'ir_101': net.IR_101,
        'ir_se_50': net.IR_SE_50,
    }
    if arch not in builders:
        raise ValueError(f'Unsupported arch: {arch}')
    return builders[arch]((img_size, img_size))


def load_checkpoint(backbone: nn.Module, ckpt: Dict) -> Tuple[bool, bool]:
    loaded = False
    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
            result = backbone.load_state_dict(state_dict, strict=False)
            loaded = True
            missing_keys.extend(result.missing_keys)
            unexpected_keys.extend(result.unexpected_keys)
        elif 'backbone' in ckpt and isinstance(ckpt['backbone'], dict):
            result = backbone.load_state_dict(ckpt['backbone'], strict=False)
            loaded = True
            missing_keys.extend(result.missing_keys)
            unexpected_keys.extend(result.unexpected_keys)
    if not loaded:
        result = backbone.load_state_dict(ckpt, strict=False)
        missing_keys.extend(result.missing_keys)
        unexpected_keys.extend(result.unexpected_keys)
    return bool(missing_keys), bool(unexpected_keys)


def make_dataloader(data_dir: str,
                    batch_size: int,
                    num_workers: int,
                    img_size: int,
                    swap_color: bool,
                    seed: int,
                    shuffle: bool = True) -> DataLoader:
    torch.manual_seed(seed)
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((img_size, img_size), interpolation=I.BILINEAR, antialias=True),
        v2.RandomHorizontalFlip(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    imgs_dir = os.path.join(data_dir, 'imgs')
    root = imgs_dir if os.path.isdir(imgs_dir) else data_dir
    dataset = CustomImageFolderDataset(
        root=root,
        transform=transform,
        low_res_augmentation_prob=0.0,
        crop_augmentation_prob=0.0,
        photometric_augmentation_prob=0.0,
        swap_color_channel=swap_color,
        output_dir=data_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def determine_target_prefixes(backbone: nn.Module, last_blocks: int) -> List[str]:
    if not hasattr(backbone, 'body') or not isinstance(backbone.body, nn.Sequential):
        return []
    total_blocks = len(backbone.body)
    start_idx = max(total_blocks - last_blocks, 0)
    return [f'body.{idx}' for idx in range(start_idx, total_blocks)]


def train_lora(backbone: nn.Module,
               data_loader: DataLoader,
               device: torch.device,
               epochs: int,
               lr: float,
               temperature: float,
               grad_clip: Optional[float],
               freeze_bn: bool) -> None:
    criterion = SupConLoss(temperature=temperature)
    params = collect_lora_parameters(backbone)
    if not params:
        raise RuntimeError('No LoRA parameters were found for optimization.')
    optimizer = torch.optim.Adam(params, lr=lr)

    disable_dropout(backbone)

    for epoch in range(epochs):
        backbone.train()
        if freeze_bn:
            set_batchnorm_eval(backbone)
        total_loss = 0.0
        steps = 0
        for images, labels in tqdm(data_loader, desc=f'LoRA epoch {epoch + 1}/{epochs}'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)
            embeddings, _ = backbone(images)
            loss = criterion(embeddings, labels)
            if loss.item() == 0.0:
                continue
            optimizer.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        if steps == 0:
            print(f'[WARN] No effective steps taken in epoch {epoch + 1}.')
        else:
            print(f'[INFO] Epoch {epoch + 1} loss: {total_loss / steps:.4f}')


def main():
    parser = argparse.ArgumentParser(description='AdaBN + ConvLoRA self-training for AdaFace backbones')
    parser.add_argument('--data_dir', type=str, default = 'adabn_data/label', help='Root directory with identity sub-folders')
    parser.add_argument('--pretrained_path', type=str,default = 'best__.ckpt', help='Path to pretrained checkpoint')
    parser.add_argument('--save_path', type=str, default=None, help='Destination path for adapted checkpoint')
    parser.add_argument('--arch', type=str, default='ir_101', choices=['ir_18', 'ir_34', 'ir_50', 'ir_101', 'ir_se_50'])
    parser.add_argument('--img_size', type=int, default=112, choices=[112, 224])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--swap_color_channel', dest='swap_color_channel', action='store_true', help='Enable BGR swap')
    parser.add_argument('--no_swap_color_channel', dest='swap_color_channel', action='store_false', help='Disable BGR swap')
    parser.set_defaults(swap_color_channel=True)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--bn_passes', type=int, default=1, help='Number of passes for AdaBN re-estimation')
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_alpha', type=float, default=8.0)
    parser.add_argument('--lora_last_blocks', type=int, default=6, help='Apply LoRA to last N residual blocks')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--freeze_bn_after_adbn', action='store_true', help='Keep BN stats frozen during LoRA training')
    parser.add_argument('--merge_lora', action='store_true', help='Merge LoRA weights into convolution weights before saving')
    parser.set_defaults(merge_lora=True)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device.startswith('cpu') else 'cpu')

    backbone = build_backbone(args.arch, args.img_size)
    checkpoint = torch.load(args.pretrained_path, map_location='cpu')
    missing, unexpected = load_checkpoint(backbone, checkpoint)
    if missing:
        print('[WARN] Missing keys encountered during checkpoint load.')
    if unexpected:
        print('[WARN] Unexpected keys encountered during checkpoint load.')

    bn_loader = make_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        swap_color=args.swap_color_channel,
        seed=args.seed,
        shuffle=False,
    )
    run_adabn(backbone, bn_loader, device=device, num_passes=args.bn_passes)

    if args.freeze_bn_after_adbn:
        set_batchnorm_eval(backbone)

    target_prefixes = determine_target_prefixes(backbone, args.lora_last_blocks)
    replaced = replace_conv_with_lora(
        backbone,
        target_prefixes=target_prefixes,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    print(f'[INFO] Injected LoRA into {len(replaced)} conv layers.')

    train_loader = make_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        swap_color=args.swap_color_channel,
        seed=args.seed,
        shuffle=True,
    )

    train_lora(
        backbone,
        data_loader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        grad_clip=args.grad_clip,
        freeze_bn=args.freeze_bn_after_adbn,
    )

    if args.merge_lora:
        merge_lora_modules(backbone)
        print('[INFO] LoRA updates merged into convolution weights.')

    save_path = args.save_path
    if save_path is None:
        base, ext = os.path.splitext(args.pretrained_path)
        suffix = '_adabn_lora'
        save_path = f'{base}{suffix}{ext or ".ckpt"}'

    updated_ckpt = copy.deepcopy(checkpoint)
    if isinstance(updated_ckpt, dict):
        if 'backbone' in updated_ckpt:
            updated_ckpt['backbone'] = backbone.state_dict()
        elif 'state_dict' in updated_ckpt:
            # Only update backbone entries if prefixed with model.
            state_dict = updated_ckpt['state_dict']
            new_backbone_state = backbone.state_dict()
            for key in list(state_dict.keys()):
                trimmed = key.replace('model.', '')
                if trimmed in new_backbone_state:
                    state_dict[key] = new_backbone_state[trimmed]
        else:
            updated_ckpt = backbone.state_dict()
    else:
        updated_ckpt = backbone.state_dict()

    torch.save(updated_ckpt, save_path)
    print(f'[INFO] Saved adapted checkpoint to {save_path}')


if __name__ == '__main__':
    main()
