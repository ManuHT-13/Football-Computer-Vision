import os
from pathlib import Path

remap = {
    0: 3,
    1: 1,
    2: 0,
    3: 2,
}

fieldvision_dir = Path("datasets/fieldvision-soccer")
splits = ["test", "train", "valid"]

for split in splits:
    print(split)
    labels_dir = fieldvision_dir / split / "labels"
    for label_file in labels_dir.glob("*.txt"):
        lines = label_file.read_text().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            old_id = int(parts[0])
            parts[0] = str(remap[old_id])
            new_lines.append(" ".join(parts))
        label_file.write_text("\n".join(new_lines) + "\n")