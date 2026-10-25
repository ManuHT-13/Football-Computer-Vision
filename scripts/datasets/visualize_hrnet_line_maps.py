import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets" / "prepared" / "soccernet-calibration-2023-hrnet-lines"
DEFAULT_OUTPUT = DEFAULT_DATASET / "visualizations"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
LEGEND_WIDTH = 520
LEGEND_ROW_HEIGHT = 30


def find_image(image_dir, stem):
    for extension in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{stem}{extension}"
        if image_path.exists():
            return image_path
    return None


def line_color(channel, channel_count):
    hue = int(channel * 180 / max(channel_count, 1))
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def render_preview(split, sample_path, dataset_dir, output_dir, class_names, alpha):
    image_path = find_image(dataset_dir / "images" / split, sample_path.stem)
    if image_path is None:
        raise FileNotFoundError(f"No image found for {sample_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image {image_path}")

    with np.load(sample_path) as archive:
        line_maps = archive["line_maps"]
    if line_maps.ndim != 3 or line_maps.shape[0] != len(class_names):
        raise ValueError(f"Unexpected line-map shape {line_maps.shape} in {sample_path}")

    height, width = image.shape[:2]
    active_channels = [
        channel
        for channel, line_map in enumerate(line_maps)
        if np.any(line_map)
    ]
    overlay = image.copy()
    for channel in active_channels:
        resized = cv2.resize(line_maps[channel], (width, height), interpolation=cv2.INTER_NEAREST)
        mask = resized > 0
        color = np.asarray(line_color(channel, len(class_names)), dtype=np.float32)
        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1 - alpha) + color * alpha
        ).astype(np.uint8)

    legend_height = max(height, 70 + len(active_channels) * LEGEND_ROW_HEIGHT)
    preview = np.zeros((legend_height, width + LEGEND_WIDTH, 3), dtype=np.uint8)
    preview[:height, :width] = overlay
    legend = preview[:, width:]
    cv2.putText(
        legend,
        f"{split}: {image_path.name}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    for row, channel in enumerate(active_channels):
        y = 58 + row * LEGEND_ROW_HEIGHT
        color = line_color(channel, len(class_names))
        cv2.rectangle(legend, (20, y - 13), (42, y + 7), color, -1)
        cv2.putText(
            legend,
            class_names[channel].strip(),
            (56, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}_preview.png"
    if not cv2.imwrite(str(output_path), preview):
        raise OSError(f"Could not write preview to {output_path}")
    return output_path, image_path.name, len(active_channels)


def main():
    parser = argparse.ArgumentParser(
        description="Save a random line-map overlay from each dataset split."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.85)
    args = parser.parse_args()

    if not 0 < args.alpha <= 1:
        parser.error("--alpha must be greater than 0 and at most 1")

    metadata_path = args.dataset / "metadata.json"
    if not metadata_path.exists():
        parser.error(f"Dataset metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    class_names = metadata.get("classes")
    if not isinstance(class_names, list) or not class_names:
        parser.error(f"No line classes found in {metadata_path}")

    rng = random.Random(args.seed)
    for split in SPLITS:
        map_dir = args.dataset / "line_maps" / split
        samples = sorted(map_dir.glob("*.npz"))
        if not samples:
            parser.error(f"No line maps found for split {split}: {map_dir}")
        sample_path = rng.choice(samples)
        output_path, image_name, active_count = render_preview(
            split, sample_path, args.dataset, args.out, class_names, args.alpha
        )
        print(f"{split}: {image_name}, {active_count} line classes -> {output_path}")


if __name__ == "__main__":
    main()