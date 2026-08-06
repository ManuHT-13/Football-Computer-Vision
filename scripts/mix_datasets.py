import shutil
from pathlib import Path

list_to_split = {
    Path("datasets/gamestate-2024-yolo/train_stride40.txt"): "train",
    Path("datasets/gamestate-2024-yolo/val_stride41.txt"): "valid",
    Path("datasets/gamestate-2024-yolo/test_stride37.txt"): "test",
}

dst_root = Path("datasets/fieldvision-soccer")

for list_file, dst_split in list_to_split.items():
    dst_images = dst_root / dst_split / "images"
    dst_labels = dst_root / dst_split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    if not list_file.exists():
        print(f"Couldnt find {list_file}")
        continue

    copied, missing_img, missing_label = 0, 0, 0

    with open(list_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        img_path = Path(line)

        label_path = Path(str(img_path).replace("\\images\\", "\\labels\\")
                                        .replace("/images/", "/labels/")).with_suffix(".txt")

        if not img_path.exists():
            missing_img += 1
            print(f"  {img_path} not found")
            continue

        if not label_path.exists():
            missing_label += 1
            print(f"  {label_path} not found")
            continue

        dst_img_path = dst_images / img_path.name
        dst_label_path = dst_labels / label_path.name

        shutil.copy2(img_path, dst_img_path)
        shutil.copy2(label_path, dst_label_path)
        copied += 1

    print(f"[{list_file.name} -> {dst_split}] copied: {copied}, " f"missing images: {missing_img}, missing labels: {missing_label}")
