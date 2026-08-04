import argparse
from pathlib import Path
from collections import defaultdict

CLASS_NAMES = {
    0: "player",
    1: "goalkeeper",
    2: "referee",
    3: "ball",
}


def get_sequence_prefix(stem):
    return stem.rsplit("_", 1)[0]


def build_list(src_root, split, frame_stride, out_txt):
    images_dir = Path(src_root) / "images" / split

    by_sequence = defaultdict(list)
    for img_path in sorted(images_dir.glob("*.jpg")):
        seq = get_sequence_prefix(img_path.stem)
        by_sequence[seq].append(img_path)

    selected = []
    total_src = 0

    for seq, img_paths in by_sequence.items():
        img_paths = sorted(img_paths, key=lambda p: p.stem)
        total_src += len(img_paths)
        for i, img_path in enumerate(img_paths):
            if i % frame_stride == 0:
                selected.append(img_path.resolve())

    with open(out_txt, "w") as f:
        for p in selected:
            f.write(f"{p}\n")

    print(f"{split}: {total_src} -> {len(selected)} imagenes (stride={frame_stride})")
    print(f"Lista guardada en {out_txt}")


def write_yaml(src_root, yaml_path, splits_txt):
    with open(yaml_path, "w") as f:
        f.write(f"path: {Path(src_root).resolve()}\n")
        for split, txt_name in splits_txt.items():
            f.write(f"{split}: {txt_name}\n")
        f.write("\nnames:\n")
        for class_id, name in sorted(CLASS_NAMES.items()):
            f.write(f"  {class_id}: {name}\n")
    print(f"data.yaml creado en {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test", "challenge"])
    parser.add_argument("--yaml_name", default="data_stride.yaml")
    args = parser.parse_args()

    splits_txt = {}
    for split in args.splits:
        txt_name = f"{split}_stride{args.frame_stride}.txt"
        out_txt = Path(args.src) / txt_name
        build_list(args.src, split, args.frame_stride, out_txt)
        splits_txt[split] = txt_name

    yaml_path = Path(args.src) / args.yaml_name
    write_yaml(args.src, yaml_path, splits_txt)