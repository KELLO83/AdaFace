import argparse
import os
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from torch.cuda import amp
import torchvision.transforms.v2 as v2
import cv2
import torchvision
import numpy as np
from torchvision.transforms import InterpolationMode as I

import net
from net import Backbone
import head as head_lib
from dataset.image_folder_dataset import CustomImageFolderDataset

from utils.polynomialLRWarmup import PolynomialLRWarmup
from torch.optim.lr_scheduler import CosineAnnealingLR

import logging

def split_parameters(module: nn.Module) -> Tuple[list, list]:
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay


def build_dataloaders(data_dir: str,
                      batch_size: int,
                      num_workers: int,
                      val_split: float,
                      low_res_aug: float,
                      crop_aug: float,
                      photo_aug: float,
                      swap_color: bool,
                      output_dir: str,
                      img_size: int = 112,
                      seed: int = 42) -> Tuple[DataLoader, DataLoader, int]:

    # AdaFace normalization expects mean=0.5, std=0.5 on BGR-order images (dataset does BGR swap)

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((img_size, img_size)), 
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        v2.RandomHorizontalFlip(p=0.5),
         v2.RandomApply([v2.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
                shear=None,
                interpolation=I.BILINEAR,
                fill=0.5
                )], p=0.1),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        v2.RandomErasing(p=0.1, scale=(0.02, 0.10) , value=0), 
     ])


    # Accept either <data_dir> or <data_dir>/imgs as root
    imgs_dir = os.path.join(data_dir, 'imgs')
    root = imgs_dir if os.path.isdir(imgs_dir) else data_dir

    # build a base dataset to determine classes and length
    # Augmentation parameters are set to 0 here as it's only used for metadata.
    base_dataset = CustomImageFolderDataset(
        root=root,
        transform=None,
        low_res_augmentation_prob=0.0,
        crop_augmentation_prob=0.0,
        photometric_augmentation_prob=0.0,
        swap_color_channel=False,  # Not used, but set to default for clarity
        output_dir=output_dir,
    )

    class_num = len(base_dataset.classes)
    n_total = len(base_dataset)

    if val_split > 0.0:
        val_len = int(n_total * val_split)
        train_len = n_total - val_len
        torch.manual_seed(seed)
        perm = torch.randperm(n_total).tolist()
        train_indices, val_indices = perm[:train_len], perm[train_len:]

        # instantiate separate datasets for train/val to use different transforms
        train_dataset = CustomImageFolderDataset(
            root=root,
            transform=train_transform,
            low_res_augmentation_prob=low_res_aug,
            crop_augmentation_prob=crop_aug,
            photometric_augmentation_prob=photo_aug,
            swap_color_channel=swap_color,
            output_dir=output_dir,
        )


        val_transform = v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Resize((img_size, img_size)),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        val_dataset = CustomImageFolderDataset(
            root=root,
            transform=val_transform,
            low_res_augmentation_prob=0.0,
            crop_augmentation_prob=0.0,
            photometric_augmentation_prob=0.0,
            swap_color_channel=swap_color,
            output_dir=output_dir,
        )

        # reduce datasets to splits via Subset
        from torch.utils.data import Subset
        train_set = Subset(train_dataset, train_indices)
        val_set = Subset(val_dataset, val_indices)
    else:
        train_dataset = CustomImageFolderDataset(
            root=root,
            transform=train_transform,
            low_res_augmentation_prob=low_res_aug,
            crop_augmentation_prob=crop_aug,
            photometric_augmentation_prob=photo_aug,
            swap_color_channel=swap_color,
            output_dir=output_dir,
        )
        train_set, val_set = train_dataset, None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_set is not None else None

    return train_loader, val_loader, class_num


def load_pretrained_backbone(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')

    state_dict = None
    if isinstance(ckpt, dict):
        if 'backbone' in ckpt:
            state_dict = ckpt['backbone']
            logging.info("Loading weights from 'backbone' key in checkpoint.")

        elif 'state_dict' in ckpt:
            raw_sd = ckpt['state_dict']
            if any(k.startswith('model.') for k in raw_sd.keys()):
                state_dict = {k.replace('model.', ''): v for k, v in raw_sd.items() if k.startswith('model.')}
                logging.info("Loading weights from 'state_dict' key and removing 'model.' prefix.")
            else:
                state_dict = raw_sd
                logging.info("Loading weights from 'state_dict' key.")
                
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state_dict = ckpt
                logging.info("Loading weights from a raw state dictionary checkpoint.")
        else:
            raise RuntimeError(f"Unsupported checkpoint format. Keys found: {list(ckpt.keys())}")
        
    elif isinstance(ckpt, (dict, OrderedDict)):
            state_dict = ckpt
            logging.info("Loading weights from a raw state dictionary file.")
    else:
        raise RuntimeError(f"Unsupported checkpoint file type: {type(ckpt)}")

    if state_dict is None:
        raise RuntimeError("Could not find a valid state dictionary to load.")

    result = model.load_state_dict(state_dict, strict=False)
    
    logging.info(f"Loaded pretrained backbone from {ckpt_path}")
    logging.info(f"-> Missing keys: {len(result.missing_keys)}")
    if result.missing_keys:
        logging.info(f"   (Examples: {result.missing_keys[:5]})")
    logging.info(f"-> Unexpected keys: {len(result.unexpected_keys)}")
    if result.unexpected_keys:
        logging.info(f"   (Examples: {result.unexpected_keys[:5]})")


def weight_freeze(model: nn.Module, num_unfreeze_blocks: int = 3):

    for param in model.parameters():
        param.requires_grad = False

    unfrozen_layers = []
    
    last_block_idx = -1
    if hasattr(model, 'body') and isinstance(model.body, nn.Sequential):
        last_block_idx = len(model.body) - 1

    unfreeze_targets = ['output_layer'] 
    if last_block_idx != -1 and num_unfreeze_blocks > 0:
        start_idx = max(0, last_block_idx - num_unfreeze_blocks + 1)
        for block_idx in range(start_idx, last_block_idx + 1):
            unfreeze_targets.append(f'body.{block_idx}')


    for name, param in model.named_parameters():
        if any(name.startswith(target) for target in unfreeze_targets):
            param.requires_grad = True
            
            if name.startswith('body'):
                base_name = '.'.join(name.split('.')[:2]) # e.g. body.46
            else:
                base_name = 'output_layer'

            if base_name not in unfrozen_layers:
                unfrozen_layers.append(base_name)

    backbone_params = sum(p.numel() for p in model.parameters())
    trainable_backbone_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('==' * 30)
    logging.info(f"Unfrozen layers: {', '.join(sorted(unfrozen_layers))}")
    logging.info(f"Model total params: {backbone_params:,}")
    logging.info(f"Model trainable params: {trainable_backbone_params:,}")
    logging.info(f"Trainable params percentage: {100 * trainable_backbone_params / backbone_params:.2f}%")
    logging.info('==' * 30)


def train_one_epoch(backbone, adaface_head, loader, criterion, optimizer, device, scaler: amp.GradScaler, epoch_idx: int, total_epochs: int, use_amp: bool = True):
    backbone.train()
    adaface_head.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, total=len(loader), desc=f"Train {epoch_idx+1}/{total_epochs}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = backbone(images)
            if isinstance(out, (tuple, list)):
                embeddings, norms = out
            else:
                raw = out
                norms = torch.norm(raw, p=2, dim=1, keepdim=True)
                embeddings = raw / norms
            logits = adaface_head(embeddings, norms, labels)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        pbar.set_postfix({
            'loss': f"{(running_loss/max(total,1)):.4f}",
            'acc': f"{(100.0*correct/max(total,1)):.2f}%"
        })

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(backbone, adaface_head, loader, criterion, device, use_amp: bool = True):
    if loader is None:
        return 0.0, 0.0
    backbone.eval()
    adaface_head.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = backbone(images)
            if isinstance(out, (tuple, list)):
                embeddings, norms = out
            else:
                raw = out
                norms = torch.norm(raw, p=2, dim=1, keepdim=True)
                embeddings = raw / norms
            logits = adaface_head(embeddings, norms, labels)
            loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return running_loss / total, correct / total


def data_check(loader: DataLoader, is_bgr: bool):
    """
    Visualizes a batch of images from the DataLoader to check augmentations.
    """
    logging.info("Fetching a batch of data to visualize...")
    # Get one batch of images and labels

    
    it = iter(loader)

    # Make a grid of images for easy visualization
    for images, labels in it:
        grid = torchvision.utils.make_grid(images, nrow=8)

        # Un-normalize from [-1, 1] range back to [0, 1] range
        grid = grid * 0.5 + 0.5
        grid = torch.clamp(grid, 0, 1)

        # Convert to numpy array for display (H, W, C)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()

        # Scale to [0, 255] and convert to uint8
        grid_np = (grid_np * 255).astype(np.uint8)

        # If the source tensor was RGB, convert to BGR for cv2 display
        if not is_bgr:
            grid_np = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)

        # Display the image
        cv2.imshow('Data Check - Press any key to exit', grid_np)
        key = cv2.waitKey(0)
        if key == 27:
            logging.info("ESC key pressed. Exiting data check.")
            break
    
    exit(0)
        

def main(args):

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model + Head
    # build backbone with dynamic input size (supported: 112 or 224)
    assert args.img_size in [112, 224], 'img_size must be 112 or 224 to match backbone shapes'

    def build_backbone_with_size(arch: str, img_size: int):
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
        raise ValueError(f'Unsupported arch: {arch}')

    backbone = build_backbone_with_size(args.arch, args.img_size).to(device)
    if args.pretrained :
        if args.pretrained_path and os.path.isfile(args.pretrained_path):
            load_pretrained_backbone(backbone, args.pretrained_path)
            weight_freeze(backbone, args.num_unfreeze_blocks)
        else:
            print(f"{args.pretrained_path} is not a file. Starting from scratch.")

    # Data
    train_loader, val_loader, class_num = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        low_res_aug=args.low_res_augmentation_prob,
        crop_aug=args.crop_augmentation_prob,
        photo_aug=args.photometric_augmentation_prob,
        swap_color=args.swap_color_channel,
        output_dir=args.output_dir,
        img_size=args.img_size,
        seed=args.seed,
    )

    # --- Data Check ---
    if args.data_check:
        data_check(train_loader, is_bgr=args.swap_color_channel)
        return  # Exit after checking

    adaface_head = head_lib.build_head(
        head_type=args.head,
        embedding_size=512,
        class_num=class_num,
        m=args.m,
        h=args.h,
        t_alpha=args.t_alpha,
        s=args.s,
    ).to(device)

    # --- W&B Initialization ---
    if args.use_wandb:
        import wandb
        wandb.init(
            project="AdaFace",
            name=args.wandb_name,
            config=args
        )

    if args.use_wandb:
        logging.info("Watching models with W&B.")
        wandb.watch((backbone, adaface_head), log='all', log_freq=100)


    paras_wo_bn, paras_only_bn = split_parameters(backbone)
    optimizer = optim.AdamW([
        {'params': paras_wo_bn + [adaface_head.kernel], 'weight_decay': args.weight_decay},
        {'params': paras_only_bn , 'weight_decay': 0.0}
    ], lr=args.lr, weight_decay=0.0)


    # scheduler = PolynomialLRWarmup(
    #     optimizer, warmup_iters=10, total_iters=args.epochs, power=1.0 , limit_lr = 1e-5
    # )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    scaler = torch.amp.GradScaler()

    use_amp = device.type == 'cuda'
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(backbone, adaface_head, train_loader, criterion, optimizer, device, scaler, epoch, args.epochs, use_amp=use_amp)
        val_loss, val_acc = evaluate(backbone, adaface_head, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | Val Loss {val_loss:.4f} Acc {val_acc*100:.2f}%")

        # --- W&B Logging ---
        if args.use_wandb:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': current_lr
            })

        # Save checkpoints
        state = {
            'epoch': epoch + 1,
            'backbone': backbone.state_dict(),
            'head': adaface_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'class_num': class_num,
        }
        if args.save_all or (epoch + 1 == args.epochs) or epoch % 3 ==0:
            torch.save(state, os.path.join(args.output_dir, f'epoch_{epoch+1}.ckpt'))

        if val_loader is not None and val_acc > best_acc:
            best_acc = val_acc
            torch.save(state, os.path.join(args.output_dir, 'best.ckpt'))

    # --- W&B Finish ---
    if args.use_wandb:
        wandb.finish()


def parser():
    parser = argparse.ArgumentParser(description='AdaFace Training (ImageFolder)')
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/KOR_DATA/high_resolution_train_data_640', help='Path to dataset root or root/imgs with class folders')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Output directory')
    parser.add_argument('--arch', type=str, default='ir_101', choices=['ir_18','ir_34','ir_50','ir_101','ir_se_50'], help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction for validation split from training set')
    parser.add_argument('--pretrained', action='store_true' , help='pretrained')
    parser.add_argument('--pretrained_path', type=str, default='', help='Path to pretrained AdaFace ckpt')
    parser.add_argument('--num_unfreeze_blocks', type=int, default=3, help='Number of final blocks to unfreeze for fine-tuning.')
    parser.add_argument('--save_all', action='store_true')
    parser.add_argument('--data_check', action='store_true', help='Run data loader check and exit.')

    # W&B related arguments
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging.')
    parser.add_argument('--wandb_name', type=str, default='adaface_ir101', help='Name for the W&B run.')

    # AdaFace head params
    parser.add_argument('--head', type=str, default='adaface', choices=['adaface','arcface','cosface'])
    parser.add_argument('--m', type=float, default=0.4)
    parser.add_argument('--h', type=float, default=0.333)
    parser.add_argument('--s', type=float, default=64.0)
    parser.add_argument('--t_alpha', type=float, default=0.01)
    
    # augmentations
    parser.add_argument('--low_res_augmentation_prob', type=float, default=0.05)
    parser.add_argument('--crop_augmentation_prob', type=float, default=0)
    parser.add_argument('--photometric_augmentation_prob', type=float, default=0.05)
    parser.add_argument('--swap_color_channel', default=True)

    parser.add_argument('--img_size', type=int, default=112, help='Input image size (height and width)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    parser.set_defaults(
        pretrained = True,
    )

    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser()
    main(args)