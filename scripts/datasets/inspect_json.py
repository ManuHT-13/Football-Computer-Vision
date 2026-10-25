import json
import sys

def explore(obj, name="root", level=0, max_depth=6):
    indent = "  " * level

    if level > max_depth:
        print(f"{indent}... max depth reached")
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{indent}{name} (DICT) -> {len(keys)} keys")

        shown = 0
        for key in keys:
            if shown >= 5:
                remaining = len(keys) - 5
                if remaining > 0:
                    print(f"{indent}  ... and {remaining} more keys")
                break
            explore(obj[key], f"'{key}'", level + 1, max_depth)
            shown += 1

    elif isinstance(obj, list):
        print(f"{indent}{name} (LIST) -> {len(obj)} items")

        if len(obj) == 0:
            print(f"{indent}  [empty]")
        else:
            explore(obj[0], f"{name}[0] (sample)", level + 1, max_depth)
            if len(obj) > 1:
                print(f"{indent}  ... and {len(obj) - 1} more items with same structure")

    else:
        value = repr(obj)
        if len(value) > 60:
            value = value[:60] + "..."
        print(f"{indent}{name} = {value} (type: {type(obj).__name__})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_json.py <path_to_json>")
        sys.exit(1)

    file_path = sys.argv[1]

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    explore(data)