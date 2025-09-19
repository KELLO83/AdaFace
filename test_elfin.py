import argparse
import glob
import itertools
import logging
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import net
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from natsort import natsorted
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode as I


SUPPORTED_BACKBONES = {
    'ir_18': net.IR_18,
    'ir_34': net.IR_34,
    'ir_50': net.IR_50,
    'ir_101': net.IR_101,
    'ir_se_50': net.IR_SE_50,
}

MODEL_REGISTRY = {
    'adaface_ir101_webface12m': {
        'arch': 'ir_101',
        'img_size': 112,
        'weight_path': os.path.join('experiments', 'best.ckpt'),
    },
    'adaface_ir101_webface4m': {
        'arch': 'ir_101',
        'img_size': 112,
        'weight_path': None,
    },
    'adaface_ir101_ms1mv3': {
        'arch': 'ir_101',
        'img_size': 112,
        'weight_path': None,
    },
    'adaface_ir50_casia': {
        'arch': 'ir_50',
        'img_size': 112,
        'weight_path': None,
    },
}


def build_backbone(arch: str, img_size: int) -> torch.nn.Module:
    if arch not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone arch '{arch}'. Supported: {sorted(SUPPORTED_BACKBONES)}")
    if img_size not in (112, 224):
        raise ValueError("img_size must be 112 or 224")
    constructor = SUPPORTED_BACKBONES[arch]
    return constructor((img_size, img_size))


def resolve_model_config(model: str, arch: Optional[str], img_size: Optional[int], weight_path: Optional[str]) -> Tuple[str, int, Optional[str]]:
    model_cfg = MODEL_REGISTRY.get(model, {})
    resolved_arch = arch or model_cfg.get('arch')
    resolved_img_size = img_size or model_cfg.get('img_size', 112)
    resolved_weight = weight_path or model_cfg.get('weight_path')

    if resolved_arch is None:
        raise ValueError("Backbone architecture is undefined. Provide --arch or use a registered model name.")

    return resolved_arch, resolved_img_size, resolved_weight


def load_backbone_checkpoint(backbone: torch.nn.Module, ckpt_path: str, map_location: str = 'cpu') -> None:
    if not ckpt_path:
        raise ValueError("Checkpoint path is empty. Provide a valid --weight_path or registered model entry.")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=map_location)
    state_dict = None

    if isinstance(ckpt, dict):
        if 'backbone' in ckpt:
            state_dict = ckpt['backbone']
        elif 'state_dict' in ckpt:
            raw = ckpt['state_dict']
            if any(k.startswith('model.') for k in raw):
                state_dict = {k.replace('model.', ''): v for k, v in raw.items() if k.startswith('model.')}
            else:
                state_dict = raw
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
    elif isinstance(ckpt, OrderedDict):
        state_dict = ckpt

    if state_dict is None:
        raise RuntimeError(f"Unsupported checkpoint format. Keys: {list(ckpt)[:5]}")

    load_result = backbone.load_state_dict(state_dict, strict=False)
    logging.info("Loaded checkpoint from %s", ckpt_path)
    if load_result.missing_keys:
        logging.info("Missing keys: %s", load_result.missing_keys)
    if load_result.unexpected_keys:
        logging.info("Unexpected keys: %s", load_result.unexpected_keys)


def build_eval_transform(img_size: int) -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((img_size, img_size), interpolation=I.BILINEAR),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return image


@torch.inference_mode()
def extract_embedding(backbone: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = image_tensor.unsqueeze(0).to(device)
    output = backbone(tensor)
    if isinstance(output, (tuple, list)) and output:
        embedding = output[0]
    else:
        embedding = output
    return embedding.squeeze(0).detach()


def collect_paths(pattern: str) -> List[str]:
    return natsorted(glob.glob(pattern))


def compute_similarities(
    backbone: torch.nn.Module,
    reference_embedding: torch.Tensor,
    paths: Iterable[str],
    label: int,
    transform: v2.Compose,
    device: torch.device,
) -> Tuple[List[float], List[int]]:
    sims: List[float] = []
    labels: List[int] = []

    for path in paths:
        image = load_image(path)
        image_tensor = transform(image)
        embedding = extract_embedding(backbone, image_tensor, device)
        similarity = F.cosine_similarity(reference_embedding.unsqueeze(0), embedding.unsqueeze(0), dim=1)
        sims.append(float(similarity.item()))
        labels.append(label)
        logging.debug("%s -> similarity %.4f", path, sims[-1])

    return sims, labels


def compute_statistics(similarities: np.ndarray) -> Dict[str, float]:
    stats = {
        'min': float(np.min(similarities)),
        'p10': float(np.percentile(similarities, 10)),
        'median': float(np.median(similarities)),
        'p90': float(np.percentile(similarities, 90)),
        'max': float(np.max(similarities)),
        'mean': float(np.mean(similarities.astype(np.float64))),
    }
    stats['std'] = float(np.std(similarities.astype(np.float64))) if similarities.size > 1 else float('nan')
    return stats


def summarize_results(similarities: Sequence[float], labels: Sequence[int], target_reference: str) -> Dict[str, object]:
    sims_np = np.asarray(similarities)
    labels_np = np.asarray(labels)
    pos_sims = sims_np[labels_np == 1]
    neg_sims = sims_np[labels_np == 0]

    if sims_np.size == 0:
        raise ValueError("No similarities computed. Check input image lists.")

    fpr, tpr, thresholds = roc_curve(labels_np, sims_np)
    roc_auc = auc(fpr, tpr)

    frr = 1 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - frr)))
    eer = float(fpr[eer_index])
    eer_threshold = float(thresholds[eer_index])

    predictions = (sims_np >= eer_threshold).astype(int)
    cm = confusion_matrix(labels_np, predictions)

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        'sims_np': sims_np,
        'labels_np': labels_np,
        'pos_sims': pos_sims,
        'neg_sims': neg_sims,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'confusion_matrix': cm,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'far': far,
        'frr_rate': frr_rate,
        'positive_stats': compute_statistics(pos_sims) if pos_sims.size else None,
        'negative_stats': compute_statistics(neg_sims) if neg_sims.size else None,
        'reference': target_reference,
    }


def render_plots(results: Dict[str, object], output_path: str) -> None:
    fpr = results['fpr']
    tpr = results['tpr']
    roc_auc = results['roc_auc']
    eer = results['eer']
    eer_idx = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    eer_threshold = results['eer_threshold']
    sims_np = results['sims_np']
    labels_np = results['labels_np']
    cm = results['confusion_matrix']

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[eer_idx], tpr[eer_idx], color='red', s=80, zorder=5, label=f'EER = {eer:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    positive_similarities = sims_np[labels_np == 1]
    negative_similarities = sims_np[labels_np == 0]
    plt.hist(positive_similarities, bins=30, alpha=0.7, color='green', label=f'Same Person (n={len(positive_similarities)})', density=True)
    plt.hist(negative_similarities, bins=30, alpha=0.7, color='red', label=f'Different Person (n={len(negative_similarities)})', density=True)
    plt.axvline(eer_threshold, color='black', linestyle='--', linewidth=2, label=f'EER Threshold: {eer_threshold:.3f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Similarity Distribution')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar(im)
    classes = ['Different Person', 'Same Person']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = cm[i, j]
        percentage = cm_normalized[i, j]
        plt.text(j, i, f'{value}\n({percentage:.2%})', ha="center", va="center", color="white" if percentage > 0.5 else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info("Saved evaluation plots to %s", output_path)


def write_report(results: Dict[str, object], output_path: str) -> None:
    lines = []
    lines.append(f"Reference image: {results['reference']}")
    lines.append(f"Total samples: {len(results['sims_np'])}")
    lines.append(f"Positive samples: {len(results['pos_sims'])}")
    lines.append(f"Negative samples: {len(results['neg_sims'])}")
    lines.append("")
    lines.append("--- Performance ---")
    lines.append(f"ROC-AUC: {results['roc_auc']:.4f}")
    lines.append(f"EER: {results['eer']:.4f} (threshold {results['eer_threshold']:.4f})")
    lines.append(f"Accuracy: {results['accuracy']:.4f}")
    lines.append(f"Precision: {results['precision']:.4f}")
    lines.append(f"Recall (TPR): {results['recall']:.4f}")
    lines.append(f"Specificity (TNR): {results['specificity']:.4f}")
    lines.append(f"F1-Score: {results['f1_score']:.4f}")
    lines.append(f"FAR: {results['far']:.4f}")
    lines.append(f"FRR: {results['frr_rate']:.4f}")
    lines.append("")
    lines.append("--- Confusion Matrix ---")
    lines.append(f"TP: {results['tp']}")
    lines.append(f"TN: {results['tn']}")
    lines.append(f"FP: {results['fp']}")
    lines.append(f"FN: {results['fn']}")

    if results['positive_stats'] is not None:
        lines.append("")
        lines.append("--- Positive similarity stats ---")
        for key, value in results['positive_stats'].items():
            lines.append(f"{key}: {value:.4f}" if np.isfinite(value) else f"{key}: N/A")

    if results['negative_stats'] is not None:
        lines.append("")
        lines.append("--- Negative similarity stats ---")
        for key, value in results['negative_stats'].items():
            lines.append(f"{key}: {value:.4f}" if np.isfinite(value) else f"{key}: N/A")

    with open(output_path, 'w') as report:
        report.write('\n'.join(lines))
    logging.info("Saved evaluation report to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Elfin face verification test harness')
    parser.add_argument('--model', type=str, default='adaface_ir101_webface12m', choices=list(MODEL_REGISTRY.keys()), help='Predefined model configuration to use')
    parser.add_argument('--arch', type=str, default=None, choices=list(SUPPORTED_BACKBONES.keys()), help='Override backbone architecture')
    parser.add_argument('--img_size', type=int, default=112, choices=[112, 224], help='Override input image size')
    parser.add_argument('--weight_path', type=str, default='best_adabn_lora_merged.ckpt', help='Checkpoint file to load')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for inference (e.g. cpu, cuda, cuda:0)')
    parser.add_argument('--reference', type=str, default=None, help='(옵션) 기본값은 frr_detected/0.jpg 참조 이미지입니다.')
    parser.add_argument('--positive_glob', type=str, default='frr_detected/*.jpg', help='Glob pattern for positive samples (same identity)')
    parser.add_argument('--negative_glob', type=str, default='far_detected/*.jpg', help='Glob pattern for negative samples (different identity)')
    parser.add_argument('--report_path', type=str, default='similarity_test_elfin.txt', help='Where to store the text report (set empty to skip)')
    parser.add_argument('--plot_path', type=str, default='face_recognition_evaluation.png', help='Where to store plots (set empty to skip)')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile')
    parser.add_argument('--dry_run', action='store_true', help='Skip heavy plotting/report generation (for smoke tests)')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='[%(levelname)s] %(message)s')

    arch, img_size, weight_path = resolve_model_config(args.model, args.arch, args.img_size, args.weight_path)
    positive_paths = collect_paths(args.positive_glob)
    negative_paths = collect_paths(args.negative_glob)

    if not positive_paths:
        raise ValueError(f"No positive images found for pattern: {args.positive_glob}")
    if not negative_paths:
        raise ValueError(f"No negative images found for pattern: {args.negative_glob}")

    reference_path = os.path.join('frr_detected', '0.jpg')
    if args.reference and os.path.abspath(args.reference) != os.path.abspath(reference_path):
        logging.warning("--reference 옵션은 무시됩니다. 기본 참조 이미지(%s)를 사용합니다.", reference_path)
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(f"Reference image not found: {reference_path}")

    # Exclude the reference image from positive samples even if the glob picked it up
    positive_paths = [p for p in positive_paths if os.path.abspath(p) != os.path.abspath(reference_path)]

    logging.info("Reference image: %s", reference_path)
    logging.info("Positive samples: %d", len(positive_paths))
    logging.info("Negative samples: %d", len(negative_paths))

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA unavailable. Falling back to CPU.")
        device = torch.device('cpu')

    backbone = build_backbone(arch, img_size)

    if weight_path is None:
        raise ValueError("No checkpoint specified. Provide --weight_path or use a registered model with predefined weights.")
    load_backbone_checkpoint(backbone, weight_path, map_location='cpu')

    backbone = backbone.to(device)
    if not args.no_compile and hasattr(torch, 'compile'):
        backbone = torch.compile(backbone)
    backbone.eval()

    transform = build_eval_transform(img_size)
    reference_embedding = extract_embedding(backbone, transform(load_image(reference_path)), device)
    reference_embedding = reference_embedding.detach()

    similarities: List[float] = []
    labels: List[int] = []

    pos_sims, pos_labels = compute_similarities(backbone, reference_embedding, positive_paths, 1, transform, device)
    similarities.extend(pos_sims)
    labels.extend(pos_labels)

    neg_sims, neg_labels = compute_similarities(backbone, reference_embedding, negative_paths, 0, transform, device)
    similarities.extend(neg_sims)
    labels.extend(neg_labels)

    results = summarize_results(similarities, labels, reference_path)

    logging.info("Total comparisons: %d", len(similarities))
    logging.info("ROC-AUC: %.4f | EER: %.4f | Accuracy: %.4f", results['roc_auc'], results['eer'], results['accuracy'])
    logging.info("TP=%d TN=%d FP=%d FN=%d", results['tp'], results['tn'], results['fp'], results['fn'])

    if not args.dry_run:
        if args.plot_path:
            render_plots(results, args.plot_path)
        if args.report_path:
            write_report(results, args.report_path)


if __name__ == '__main__':
    main(parse_args())
