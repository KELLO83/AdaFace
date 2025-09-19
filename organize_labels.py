import argparse
import re
import shutil
from pathlib import Path

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
LABEL_PATTERN = re.compile(r'^([A-Za-z]+)')

def infer_label(filename: str) -> str | None:
    stem = Path(filename).stem
    match = LABEL_PATTERN.match(stem)
    if not match:
        return None
    return match.group(1)

def move_images(src_root: Path, dst_root: Path, dry_run: bool = False):
    moved = 0
    skipped = 0
    dst_root.mkdir(parents=True, exist_ok=True)

    for path in sorted(src_root.iterdir()):
        if path.is_dir():
            if path == dst_root:
                continue
            # optionally flatten nested directories
            move_images(path, dst_root, dry_run=dry_run)
            continue

        if path.suffix.lower() not in SUPPORTED_EXTS:
            skipped += 1
            continue

        label = infer_label(path.name)
        if label is None:
            print(f"[WARN] Unable to infer label from '{path.name}', skipping")
            skipped += 1
            continue

        label_dir = dst_root / label
        if not dry_run:
            label_dir.mkdir(parents=True, exist_ok=True)

        dest_path = label_dir / path.name
        if dest_path.exists():
            # Avoid overwriting: append counter
            counter = 1
            new_name = f"{dest_path.stem}_{counter}{dest_path.suffix}"
            new_path = label_dir / new_name
            while new_path.exists():
                counter += 1
                new_name = f"{dest_path.stem}_{counter}{dest_path.suffix}"
                new_path = label_dir / new_name
            dest_path = new_path

        print(f"[{'DRY' if dry_run else 'MOVE'}] {path} -> {dest_path}")
        if not dry_run:
            shutil.move(str(path), str(dest_path))
        moved += 1

    return moved, skipped

def main():
    parser = argparse.ArgumentParser(description="Organize adabn_data images into label folders based on filename prefix")
    parser.add_argument('--data_dir', type=Path, default=Path('adabn_data'), help='Directory containing unlabeled images')
    parser.add_argument('--output_subdir', type=str, default='label', help='Subfolder name to store organized images')
    parser.add_argument('--dry_run', action='store_true', help='Only print planned moves without modifying files')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_dir = data_dir / args.output_subdir

    moved, skipped = move_images(data_dir, output_dir, dry_run=args.dry_run)
    print(f"Done. moved={moved}, skipped={skipped}")

if __name__ == '__main__':
    main()
