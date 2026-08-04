"""
YOLO desired format:
dataset/
    images/
        train/
            SNGS-001_000001.jpg
            ...
        val/
        test/
        challenge/
    labels/
        train/
            SNGS-001_000001.txt
            ...
        val/
        test/
        challenge/
    data.yaml

gamestate-2024 dataset COLO format:
    SNGS-001/
        img1/
            000001.jpg
            000002.jpg
            ...
        Labels-GameState.json

Labels-GameState.json format:
    root (DICT) -> 4 keys
        'info' (DICT) -> 16 keys
            'version' = '1.3' (type: str)
            'game_id' = '4' (type: str)
            'id' = '060' (type: str)
            'num_tracklets' = '26' (type: str)
            'action_position' = '895' (type: str)
            ... and 11 more keys
        'images' (LIST) -> 750 items
            'images'[0] (sample) (DICT) -> 10 keys
            'is_labeled' = True (type: bool)
            'image_id' = '1060000001' (type: str)
            'file_name' = '000001.jpg' (type: str)
            'height' = 1080 (type: int)
            'width' = 1920 (type: int)
            ... and 5 more keys
            ... and 749 more items with same structure
        'annotations' (LIST) -> 14290 items
            'annotations'[0] (sample) (DICT) -> 9 keys
            'id' = '1060000001' (type: str)
            'image_id' = '1060000001' (type: str)
            'track_id' = 1 (type: int)
            'supercategory' = 'object' (type: str)
            'category_id' = 1 (type: int)
            ... and 4 more keys
            ... and 14289 more items with same structure
        'categories' (LIST) -> 7 items
            'categories'[0] (sample) (DICT) -> 3 keys
            'supercategory' = 'object' (type: str)
            'id' = 1 (type: int)
            'name' = 'player' (type: str)
            ... and 6 more items with same structure
"""

import json
import shutil
import argparse
from pathlib import Path

CLASSES_MAP = {
    "player": 0,
    "goalkeeper": 1,
    "referee": 2,
    "ball": 3,
}



def convert_split(src_root, dst_root, split):
    out_images = Path(dst_root) / "images" / split
    out_labels = Path(dst_root) / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([d for d in Path(src_root).iterdir() if d.is_dir()])

    for seq_dir in seq_dirs:
        labels_path = seq_dir / "Labels-GameState.json"
        img_dir = seq_dir / "img1"

        if not labels_path.exists():
            if img_dir.exists():
                for img_path in sorted(img_dir.glob("*.jpg")):
                    dst_name = f"{seq_dir.name}_{img_path.stem}.jpg"
                    shutil.copy2(img_path, out_images / dst_name)
            continue

        with open(labels_path, "r") as f:
            data = json.load(f)

        categories_by_id = {c["id"]: c["name"] for c in data["categories"]}
        images_by_id = {img["image_id"]: img for img in data["images"]}

        anns_by_image = {}
        for ann in data["annotations"]:
            if categories_by_id.get(ann["category_id"]) not in CLASSES_MAP:
                continue
            if "bbox_image" not in ann:
                continue
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        for image_id, img_info in images_by_id.items():
            file_name = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]
            src_img_path = seq_dir / "img1" / file_name

            if not src_img_path.exists():
                continue

            stem = Path(file_name).stem
            dst_name = f"{seq_dir.name}_{stem}"

            lines = []
            for ann in anns_by_image.get(image_id, []):
                class_name = categories_by_id[ann["category_id"]]
                class_id = CLASSES_MAP[class_name]

                bbox = ann["bbox_image"]
                x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                x_center = min(max(x_center, 0), 1)
                y_center = min(max(y_center, 0), 1)
                w_norm = min(max(w_norm, 0), 1)
                h_norm = min(max(h_norm, 0), 1)

                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            shutil.copy2(src_img_path, out_images / f"{dst_name}.jpg")
            with open(out_labels / f"{dst_name}.txt", "w") as f:
                f.write("\n".join(lines))


def write_yaml(dst_root):
    yaml_path = Path(dst_root) / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {Path(dst_root).resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("names:\n")
        for name, class_id in sorted(CLASSES_MAP.items(), key=lambda x: x[1]):
            f.write(f"  {class_id}: {name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "challenge"])
    args = parser.parse_args()

    convert_split(args.src, args.dst, args.split)
    write_yaml(args.dst)