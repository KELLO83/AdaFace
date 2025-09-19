import argparse
import copy
import os
from glob import glob
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode as I
from tqdm import tqdm

from PIL import Image
import numpy as np

from dataset.image_folder_dataset import CustomImageFolderDataset, cv2_loader
import net


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
    return FlatImageDataset(root=root, transform=transform, swap_color=swap_color, output_dir=output_dir)


class FlatImageDataset(Dataset):
    """Dataset that reads all images in a directory without class sub-folders."""

    def __init__(self, root: str, transform, swap_color: bool, output_dir: str):
        self.root = root
        self.transform = transform
        self.swap_color = swap_color
        self.output_dir = output_dir
        self.paths: List[str] = []
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp',
                    '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP']
        for pattern in patterns:
            self.paths.extend(glob(os.path.join(root, pattern)))
        self.paths.sort()
        if not self.paths:
            raise FileNotFoundError(f"No image files found in {root} (patterns: {patterns})")

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


def build_backbone(arch: str, img_size: int) -> nn.Module:
    if arch == 'ir_18':
        return net.IR_18((img_size, img_size))
    if arch == 'ir_34':
        return net.IR_34((img_size, img_size))
    if arch == 'ir_50':
        return net.IR_50((img_size, img_size))
    if arch == 'ir_101':
        return net.IR_101((img_size, img_size))
    if arch == 'ir_se_50':
        return net.IR_SE_50((img_size, img_size))
    raise ValueError(f"Unsupported architecture: {arch}")


def build_loader(data_dir: str,
                 batch_size: int,
                 num_workers: int,
                 img_size: int,
                 swap_color: bool,
                 seed: int) -> DataLoader:
    torch.manual_seed(seed)

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((img_size, img_size), interpolation=I.BILINEAR, antialias=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    imgs_dir = os.path.join(data_dir, 'imgs')
    root = imgs_dir if os.path.isdir(imgs_dir) else data_dir

    dataset = build_dataset_for_bn(
        root=root,
        transform=transform,
        swap_color=swap_color,
        output_dir=data_dir,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def enable_bn_updates(model: nn.Module):
    """Put BatchNorm modules into training mode while keeping others in eval."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train()
            for param in module.parameters():
                param.requires_grad = False
        elif isinstance(module, nn.Dropout):
            module.eval()
    for param in model.parameters():
        param.requires_grad = False


def extract_bn_statistics(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    bn_states = {}
    for key, tensor in state_dict.items():
        if key.endswith('running_mean') or key.endswith('running_var') or key.endswith('num_batches_tracked'):
            bn_states[key] = tensor.clone().detach().cpu()
    return bn_states


def patch_checkpoint(original_ckpt: Dict, updated_bn: Dict[str, torch.Tensor]) -> Dict:
    patched = copy.deepcopy(original_ckpt)

    def update_block(block: Dict[str, torch.Tensor]):
        for key in list(block.keys()):
            base_key = key
            if key.startswith('model.'):
                base_key = key[len('model.'):]
            if base_key in updated_bn:
                block[key] = updated_bn[base_key]

    if isinstance(patched, dict):
        if 'state_dict' in patched and isinstance(patched['state_dict'], dict):
            update_block(patched['state_dict'])
        if 'backbone' in patched and isinstance(patched['backbone'], dict):
            update_block(patched['backbone'])
    return patched


def main():
    parser = argparse.ArgumentParser(description='Adaptive BatchNorm statistics re-estimation for AdaFace backbones')
    parser.add_argument('--data_dir', type=str, default = 'adabn_data', help='Dataset root directory (class sub-folders expected)')
    parser.add_argument('--pretrained_path', type=str, default = 'best.ckpt', help='Path to the pretrained checkpoint')
    parser.add_argument('--save_path', type=str, default=None, help='Output path for the BN-updated checkpoint')
    parser.add_argument('--arch', type=str, default='ir_101', choices=['ir_18', 'ir_34', 'ir_50', 'ir_101', 'ir_se_50'])
    parser.add_argument('--img_size', type=int, default=112, choices=[112, 224])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    parser.add_argument('--num_passes', type=int, default=10, help='How many times to iterate over the dataset')
    parser.add_argument('--swap_color_channel', dest='swap_color_channel', action='store_true', help='Apply BGR swap used during training preprocessing')
    parser.add_argument('--no_swap_color_channel', dest='swap_color_channel', action='store_false', help='Keep images in RGB order')
    parser.set_defaults(swap_color_channel=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (e.g., cuda, cuda:0, cpu)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    checkpoint = torch.load(args.pretrained_path, map_location='cpu')

    backbone = build_backbone(args.arch, args.img_size)

    loaded = False
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
            missing = backbone.load_state_dict(state_dict, strict=False)
            loaded = True
        elif 'backbone' in checkpoint:
            missing = backbone.load_state_dict(checkpoint['backbone'], strict=False)
            loaded = True
    if not loaded:
        missing = backbone.load_state_dict(checkpoint, strict=False)

    if missing.missing_keys:
        print(f"Warning: missing keys during load: {missing.missing_keys}")
    if missing.unexpected_keys:
        print(f"Warning: unexpected keys during load: {missing.unexpected_keys}")

    enable_bn_updates(backbone)
    backbone.to(device)

    loader = build_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        swap_color=args.swap_color_channel,
        seed=args.seed,
    )

    processed_batches = 0
    skipped_batches = 0
    torch.set_grad_enabled(False)
    for pass_idx in range(args.num_passes):
        pbar = tqdm(loader, desc=f"BN pass {pass_idx + 1}/{args.num_passes}")
        for images, _ in pbar:
            if images.size(0) < 2:
                skipped_batches += 1
                continue  # skip tiny batches that break BatchNorm updates
            images = images.to(device, non_blocking=True)
            backbone(images)
            processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(
            "No batches with at least 2 samples were processed. Increase dataset size or reduce batch size."
        )
    if skipped_batches > 0:
        print(f"Skipped {skipped_batches} mini-batches with fewer than 2 samples.")

    updated_state = backbone.state_dict()
    updated_bn = extract_bn_statistics(updated_state)

    patched_ckpt = patch_checkpoint(checkpoint, updated_bn)

    save_path = args.save_path
    if save_path is None:
        base, ext = os.path.splitext(args.pretrained_path)
        save_path = f"{base}_adbn{ext or '.ckpt'}"

    torch.save(patched_ckpt, save_path)
    print(f"Saved BN-updated checkpoint to {save_path}")


if __name__ == '__main__':
    main()
