import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = ROOT / "datasets" / "raw" / "soccernet-calibration-2023"
DEFAULT_DST = ROOT / "datasets" / "prepared" / "soccernet-calibration-2023-hrnet-lines"
SPLITS = {"train": "train", "valid": "val", "test": "test"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
METADATA_FILES = {"match_info.json", "per_match_info.json"}

CLASS_NAMES = [
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Circle central",
    "Circle left",
    "Circle right",
    "Goal left crossbar",
    "Goal left post left ",
    "Goal left post right",
    "Goal right crossbar",
    "Goal right post left",
    "Goal right post right",
    "Goal unknown",
    "Line unknown",
    "Middle line",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Side line top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
]
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}


def find_image(folder, stem):
    for extension in IMAGE_EXTENSIONS:
        image_path = folder / f"{stem}{extension}"
        if image_path.exists():
            return image_path
    return None


def read_polylines(annotation_path):
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {annotation_path}")

    polylines = {}
    for name, points in data.items():
        if name not in CLASS_IDS:
            raise ValueError(f"Unknown line class {name!r} in {annotation_path}")
        if isinstance(points, dict):
            points = [points]
        if not isinstance(points, list):
            raise ValueError(f"Expected a point list for {name!r} in {annotation_path}")

        parsed = []
        for point in points:
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                raise ValueError(f"Invalid point for {name!r} in {annotation_path}")
            x, y = float(point["x"]), float(point["y"])
            if not np.isfinite([x, y]).all():
                raise ValueError(f"Non-finite point for {name!r} in {annotation_path}")
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            parsed.append((x, y))
        if parsed:
            polylines[name] = parsed
    return polylines


def make_line_maps(polylines, width, height, thickness):
    maps = np.zeros((len(CLASS_NAMES), height, width), dtype=np.uint8)
    for name, points in polylines.items():
        coords = np.array(
            [[round(x * (width - 1)), round(y * (height - 1))] for x, y in points],
            dtype=np.int32,
        )
        target = maps[CLASS_IDS[name]]
        if len(coords) == 1:
            cv2.circle(target, tuple(coords[0]), max(1, thickness // 2), 255, -1)
        else:
            cv2.polylines(target, [coords], False, 255, thickness, cv2.LINE_AA)
    return maps


def convert_split(src_root, dst_root, source_split, target_split, width, height, thickness, limit):
    source_dir = src_root / source_split
    image_dir = dst_root / "images" / target_split
    target_dir = dst_root / "line_maps" / target_split
    annotation_paths = sorted(
        path for path in source_dir.glob("*.json") if path.name not in METADATA_FILES
    )
    if not annotation_paths:
        raise FileNotFoundError(f"No JSON annotations found in {source_dir}")
    if limit is not None:
        annotation_paths = annotation_paths[:limit]

    image_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    for annotation_path in annotation_paths:
        image_path = find_image(source_dir, annotation_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"No matching image found for {annotation_path}")

        polylines = read_polylines(annotation_path)
        shutil.copy2(image_path, image_dir / image_path.name)
        maps = make_line_maps(polylines, width, height, thickness)
        np.savez_compressed(target_dir / f"{annotation_path.stem}.npz", line_maps=maps)

    return len(annotation_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SoccerNet field-line annotations into fixed-channel HRNet line maps."
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--input-width", type=int, default=960)
    parser.add_argument("--input-height", type=int, default=540)
    parser.add_argument("--output-stride", type=int, default=4)
    parser.add_argument("--thickness", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output_stride <= 0 or args.input_width % args.output_stride or args.input_height % args.output_stride:
        parser.error("Input dimensions must be divisible by a positive output stride")
    if args.thickness <= 0:
        parser.error("--thickness must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.src.resolve() == args.dst.resolve():
        parser.error("Source and destination directories must be different")
    if args.dst.exists() and any(args.dst.iterdir()) and not args.resume:
        parser.error(f"Destination is not empty: {args.dst}")

    output_width = args.input_width // args.output_stride
    output_height = args.input_height // args.output_stride
    args.dst.mkdir(parents=True, exist_ok=True)

    counts = {}
    for source_split, target_split in SPLITS.items():
        counts[target_split] = convert_split(
            args.src,
            args.dst,
            source_split,
            target_split,
            output_width,
            output_height,
            args.thickness,
            args.limit,
        )

    metadata = {
        "task": "multi_label_field_line_heatmaps",
        "num_classes": len(CLASS_NAMES),
        "input_size": [args.input_width, args.input_height],
        "output_size": [output_width, output_height],
        "output_stride": args.output_stride,
        "classes": CLASS_NAMES,
        "splits": counts,
    }
    (args.dst / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(CLASS_NAMES)} line-map channels at {output_width}x{output_height} to {args.dst}")
    for split, count in counts.items():
        print(f"{split}: {count} images")


if __name__ == "__main__":
    main()