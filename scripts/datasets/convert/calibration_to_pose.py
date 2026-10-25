import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "soccernet-calibration-2023"
OUT = ROOT / "datasets" / "soccernet-calibration-2023-pose"

SPLITS = {"train": "train", "valid": "val", "test": "test"}
IMG_EXTS = (".jpg", ".jpeg", ".png")
PAD = 0.01
MIN_SIZE = 0.03


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    lines = {}
    for name, pts in data.items():
        if isinstance(pts, dict):
            pts = [pts]
        if not isinstance(pts, list):
            continue
        pts = [
            (min(max(float(p["x"]), 0.0), 1.0), min(max(float(p["y"]), 0.0), 1.0))
            for p in pts
            if isinstance(p, dict) and "x" in p and "y" in p
        ]
        if pts:
            lines[name] = pts
    return lines


def span(values):
    lo, hi = min(values) - PAD, max(values) + PAD
    if hi - lo < MIN_SIZE:
        mid = (lo + hi) / 2
        lo, hi = mid - MIN_SIZE / 2, mid + MIN_SIZE / 2
    lo, hi = max(lo, 0.0), min(hi, 1.0)
    return (lo + hi) / 2, hi - lo


def label_row(cls_id, pts, k):
    cx, w = span([p[0] for p in pts])
    cy, h = span([p[1] for p in pts])
    row = [str(cls_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for x, y in pts:
        row += [f"{x:.6f}", f"{y:.6f}", "2"]
    row += ["0", "0", "0"] * (k - len(pts))
    return " ".join(row)


def find_image(folder, stem):
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    data = {}
    names = set()
    k = 0
    for folder in SPLITS:
        for jp in sorted((SRC / folder).glob("*.json")):
            lines = read_lines(jp)
            data[(folder, jp.stem)] = lines
            names.update(lines)
            for pts in lines.values():
                k = max(k, len(pts))

    classes = sorted(names)
    ids = {n: i for i, n in enumerate(classes)}

    for folder, split in SPLITS.items():
        img_dir = OUT / "images" / split
        lbl_dir = OUT / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for (f, stem), lines in data.items():
            if f != folder:
                continue
            img = find_image(SRC / folder, stem)
            if img is None:
                continue
            shutil.copy2(img, img_dir / img.name)
            rows = [label_row(ids[n], pts, k) for n, pts in lines.items()]
            (lbl_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            count += 1
        print(split, count)

    yaml = ["train: images/train", "val: images/val", "test: images/test", f"kpt_shape: [{k}, 3]", "names:"]
    yaml += [f"  {i}: {json.dumps(n)}" for n, i in ids.items()]
    (OUT / "data.yaml").write_text("\n".join(yaml) + "\n")
    print(len(classes), "clases,", k, "keypoints")


if __name__ == "__main__":
    main()