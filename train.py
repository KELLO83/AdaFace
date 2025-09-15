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

import net
import head as head_lib
from dataset.image_folder_dataset import CustomImageFolderDataset

from utils.polynomialLRWarmup import PolynomialLRWarmup
import torchvision.transforms.v2 as v2

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
                      img_size: int = 112) -> Tuple[DataLoader, DataLoader, int]:

    img_size = img_size

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((img_size, img_size)), 
        v2.TrivialAugmentWide(), 
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.3, scale=(0.02, 0.15) , value=0), 
     ])


    # Accept either <data_dir> or <data_dir>/imgs as root
    imgs_dir = os.path.join(data_dir, 'imgs')
    root = imgs_dir if os.path.isdir(imgs_dir) else data_dir

    # build a base dataset to determine classes and length
    base_dataset = CustomImageFolderDataset(
        root=root,
        transform=None,
        low_res_augmentation_prob=low_res_aug,
        crop_augmentation_prob=crop_aug,
        photometric_augmentation_prob=photo_aug,
        swap_color_channel=swap_color,
        output_dir=output_dir,
    )

    class_num = len(base_dataset.classes)
    n_total = len(base_dataset)

    if val_split > 0.0:
        val_len = int(n_total * val_split)
        train_len = n_total - val_len
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
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model_state = {k.replace('model.', ''): v for k, v in state.items() if 'model.' in k or k in model.state_dict()}
    result = model.load_state_dict(model_state, strict=False)
    print(f"Loaded pretrained backbone from {ckpt_path}")
    print(f"Missing keys: {result.missing_keys}")
    print(f"Unexpected keys: {result.unexpected_keys}")


def train_one_epoch(backbone, adaface_head, loader, criterion, optimizer, device):
    backbone.train()
    adaface_head.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = backbone(images)
        if isinstance(out, (tuple, list)):
            embeddings, norms = out
        else:
            raw = out
            norms = torch.norm(raw, p=2, dim=1, keepdim=True)
            embeddings = raw / norms
        logits = adaface_head(embeddings, norms, labels)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(backbone, adaface_head, loader, criterion, device):
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


def main():
    parser = argparse.ArgumentParser(description='AdaFace Training (ImageFolder)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root or root/imgs with class folders')
    parser.add_argument('--output_dir', type=str, default='experiments/adaface_custom', help='Output directory')
    parser.add_argument('--arch', type=str, default='ir_50', choices=['ir_18','ir_34','ir_50','ir_101','ir_se_50'], help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction for validation split from training set')
    parser.add_argument('--start_from_model_statedict', type=str, default='', help='Path to pretrained AdaFace ckpt')
    parser.add_argument('--save_all', action='store_true')

    # AdaFace head params
    parser.add_argument('--head', type=str, default='adaface', choices=['adaface','arcface','cosface'])
    parser.add_argument('--m', type=float, default=0.4)
    parser.add_argument('--h', type=float, default=0.333)
    parser.add_argument('--s', type=float, default=64.0)
    parser.add_argument('--t_alpha', type=float, default=0.01)
    
    # augmentations
    parser.add_argument('--low_res_augmentation_prob', type=float, default=0.0)
    parser.add_argument('--crop_augmentation_prob', type=float, default=0.0)
    parser.add_argument('--photometric_augmentation_prob', type=float, default=0.0)
    parser.add_argument('--swap_color_channel', action='store_true')

    parser.add_argument('--img_size', type=int, default=112, help='Input image size (height and width)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    )

    # Model + Head
    backbone = net.build_model(args.arch).to(device)
    if args.start_from_model_statedict:
        load_pretrained_backbone(backbone, args.start_from_model_statedict)

    adaface_head = head_lib.build_head(
        head_type=args.head,
        embedding_size=512,
        class_num=class_num,
        m=args.m,
        h=args.h,
        t_alpha=args.t_alpha,
        s=args.s,
    ).to(device)

    # Optimizer and scheduler (match train_val.py style)
    paras_wo_bn, paras_only_bn = split_parameters(backbone)
    optimizer = optim.AdamW([
        {'params': paras_wo_bn + [adaface_head.kernel], 'weight_decay': args.weight_decay},
        {'params': paras_only_bn , 'weight_decay': 0.0}
    ], lr=args.lr, weight_decay=0.0)


    scheduler = PolynomialLRWarmup(
        optimizer, warmup_iters=10, total_iters=args.epochs, power=1.0 , limit_lr = 1e-5
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(backbone, adaface_head, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(backbone, adaface_head, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | Val Loss {val_loss:.4f} Acc {val_acc*100:.2f}%")

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

if __name__ == '__main__':
    main()
