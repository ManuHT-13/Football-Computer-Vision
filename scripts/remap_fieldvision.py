import os
from pathlib import Path

remap = {
    1: 3,  
    2: 1,  
    3: 0,  
    4: 2,  
    
}

dataset_dir = Path("datasets/work-football")
splits = ["test", "train", "valid"]

dropped_lines = 0
dropped_files = 0

for split in splits:
    print(split)
    labels_dir = dataset_dir / split / "labels"
    if not labels_dir.exists():
        print(f"  (no existe {labels_dir}, skip)")
        continue

    for label_file in labels_dir.glob("*.txt"):
        lines = label_file.read_text().splitlines()
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            old_id = int(parts[0])

            if old_id == 0:
                dropped_lines += 1
                continue

            if old_id not in remap:
                raise ValueError(f"unknown class id {old_id} in {label_file}")

            parts[0] = str(remap[old_id])
            new_lines.append(" ".join(parts))

        if new_lines:
            label_file.write_text("\n".join(new_lines) + "\n")
        else:
            label_file.write_text("")
            dropped_files += 1

print(f"\nDropped lines: {dropped_lines}")
print(f"Dropped files: {dropped_files}")