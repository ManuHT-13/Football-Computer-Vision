import argparse
import configparser
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class TrackIdentity:
    key: str
    role: str
    team: str
    jersey: str
    identity_type: str


@dataclass(frozen=True)
class Observation:
    sequence: str
    frame: int
    track_id: int
    box: tuple
    identity: TrackIdentity


def read_ini(path):
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(path, encoding="utf-8"):
        raise RuntimeError(f"Could not read config file: {path}")
    section = parser["Sequence"] if parser.has_section("Sequence") else parser.defaults()
    return section


def parse_identity(description, game_id, sequence, track_id):
    normalized = description.strip().lower()
    if "goalkeeper" in normalized:
        role = "goalkeeper"
    elif "player" in normalized:
        role = "player"
    else:
        return None

    prefix, separator, jersey = description.partition(";")
    jersey = jersey.strip()
    team = ""
    prefix_lower = prefix.lower()
    if "team left" in prefix_lower:
        team = "left"
    elif "team right" in prefix_lower:
        team = "right"

    if separator and jersey.isdecimal() and team:
        jersey = str(int(jersey))
        key = f"game:{game_id}:team:{team}:jersey:{jersey}"
        identity_type = "game_team_jersey"
    else:
        key = f"game:{game_id}:sequence:{sequence}:track:{track_id}"
        identity_type = "sequence_tracklet"

    return TrackIdentity(key, role, team, jersey, identity_type)


def even_indices(count, limit):
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [count // 2]
    return [
        math.floor(index * (count - 1) / (limit - 1) + 0.5)
        for index in range(limit)
    ]


def read_sequence(
    sequence_dir, raw_split, val_game_ids, min_box_height, max_crops_per_tracklet
):
    game_info_path = sequence_dir / "gameinfo.ini"
    seq_info_path = sequence_dir / "seqinfo.ini"
    gt_path = sequence_dir / "gt" / "gt.txt"
    for path in (game_info_path, seq_info_path, gt_path):
        if not path.is_file():
            raise RuntimeError(f"Required SoccerNet file is missing: {path}")

    game_info = read_ini(game_info_path)
    sequence_info = read_ini(seq_info_path)
    game_id = game_info.get("gameID", "").strip()
    if not game_id:
        raise RuntimeError(f"Missing gameID in {game_info_path}")

    if raw_split == "test":
        split = "test"
    else:
        split = "val" if game_id in val_game_ids else "train"

    image_dir = sequence_dir / sequence_info.get("imDir", "img1")
    image_ext = sequence_info.get("imExt", ".jpg")
    tracklet_descriptions = {}
    for key, value in game_info.items():
        if key.startswith("trackletid_"):
            try:
                track_id = int(key[len("trackletid_"):])
            except ValueError as exc:
                raise RuntimeError(f"Invalid tracklet key {key!r} in {game_info_path}") from exc
            identity = parse_identity(value, game_id, sequence_dir.name, track_id)
            if identity is not None:
                tracklet_descriptions[track_id] = identity

    tracklets = defaultdict(list)
    with gt_path.open("r", encoding="utf-8", newline="") as gt_file:
        for line_number, row in enumerate(csv.reader(gt_file), start=1):
            if len(row) < 7:
                raise RuntimeError(f"Malformed GT row at {gt_path}:{line_number}")
            try:
                frame, track_id = int(row[0]), int(row[1])
                x, y, width, height = (float(value) for value in row[2:6])
                marked = int(float(row[6]))
            except ValueError as exc:
                raise RuntimeError(f"Invalid GT value at {gt_path}:{line_number}") from exc

            identity = tracklet_descriptions.get(track_id)
            if identity is None or marked != 1 or height < min_box_height:
                continue
            if frame < 1 or width <= 0 or height <= 0:
                continue
            tracklets[track_id].append((frame, x, y, width, height))

    selected = []
    for track_id, observations in tracklets.items():
        observations.sort(key=lambda item: item[0])
        for index in even_indices(len(observations), max_crops_per_tracklet):
            frame, x, y, width, height = observations[index]
            selected.append(
                (
                    split,
                    image_dir,
                    image_ext,
                    Observation(
                        sequence_dir.name,
                        frame,
                        track_id,
                        (x, y, width, height),
                        tracklet_descriptions[track_id],
                    ),
                )
            )
    return game_id, selected


def crop_box(box, image_width, image_height):
    x, y, width, height = box
    x1 = max(0, math.floor(x))
    y1 = max(0, math.floor(y))
    x2 = min(image_width, math.ceil(x + width))
    y2 = min(image_height, math.ceil(y + height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def prepare_dataset(args):
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise RuntimeError(f"Source dataset directory does not exist: {source}")
    if output == source or source in output.parents or output in source.parents:
        raise RuntimeError("Output and source directories must not overlap")
    if output.exists() and not output.is_dir():
        raise RuntimeError(f"Output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output}. Choose a new path to avoid mixing datasets."
        )

    train_dir = source / "train"
    test_dir = source / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise RuntimeError(f"Expected train/ and test/ directories under {source}")

    sequences = []
    train_games = set()
    test_games = set()
    for raw_split, split_dir in (("train", train_dir), ("test", test_dir)):
        for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            game_id, selected = read_sequence(
                sequence_dir,
                raw_split,
                set(args.val_game_ids),
                args.min_box_height,
                args.max_crops_per_tracklet,
            )
            (test_games if raw_split == "test" else train_games).add(game_id)
            sequences.extend(selected)

    val_game_ids = set(args.val_game_ids)
    if not val_game_ids:
        raise RuntimeError("At least one validation game ID is required")
    missing_val_games = val_game_ids - train_games
    overlapping_test_games = val_game_ids & test_games
    if missing_val_games:
        raise RuntimeError(
            f"Validation game IDs are not present in the raw train split: {sorted(missing_val_games)}"
        )
    if overlapping_test_games:
        raise RuntimeError(
            f"Validation games also occur in the raw test split: {sorted(overlapping_test_games)}"
        )

    selected_by_split_identity = defaultdict(list)
    for split, image_dir, image_ext, observation in sequences:
        selected_by_split_identity[(split, observation.identity.key)].append(
            (image_dir, image_ext, observation)
        )

    output.mkdir(parents=True, exist_ok=True)
    image_root = output / "images"
    image_root.mkdir()
    manifests = []
    classes = []
    summary = {
        "source": str(source),
        "validation_game_ids": sorted(val_game_ids),
        "train_game_ids": sorted(train_games - val_game_ids),
        "test_game_ids": sorted(test_games),
        "min_box_height": args.min_box_height,
        "max_crops_per_tracklet": args.max_crops_per_tracklet,
        "min_crops_per_identity": args.min_crops_per_identity,
        "splits": {},
    }

    for split in ("train", "val", "test"):
        identity_keys = sorted(
            key for candidate_split, key in selected_by_split_identity if candidate_split == split
        )
        split_images = image_root / split
        split_images.mkdir()
        identity_count = 0
        crop_count = 0

        for candidate_id, identity_key in enumerate(identity_keys):
            candidates = selected_by_split_identity[(split, identity_key)]
            identity = candidates[0][2].identity
            identity_id = f"id_{candidate_id:05d}"
            identity_dir = split_images / identity_id
            identity_dir.mkdir()
            identity_manifest = []

            for image_dir, image_ext, observation in candidates:
                image_path = image_dir / f"{observation.frame:06d}{image_ext}"
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                bounds = crop_box(
                    observation.box, image.shape[1], image.shape[0]
                )
                if bounds is None:
                    continue
                x1, y1, x2, y2 = bounds
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                filename = f"{observation.sequence}_{observation.frame:06d}_{observation.track_id}.jpg"
                crop_path = identity_dir / filename
                if not cv2.imwrite(str(crop_path), crop):
                    raise RuntimeError(f"Could not write player crop: {crop_path}")
                identity_manifest.append(
                    {
                        "split": split,
                        "identity_id": identity_id,
                        "identity_key": identity_key,
                        "image": crop_path.relative_to(output).as_posix(),
                        "sequence": observation.sequence,
                        "game_id": identity_key.split(":")[1],
                        "track_id": observation.track_id,
                        "frame": observation.frame,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "role": observation.identity.role,
                        "team": observation.identity.team,
                        "jersey": observation.identity.jersey,
                        "identity_type": observation.identity.identity_type,
                    }
                )

            if len(identity_manifest) < args.min_crops_per_identity:
                for row in identity_manifest:
                    (output / row["image"]).unlink()
                identity_dir.rmdir()
                continue

            identity_count += 1
            crop_count += len(identity_manifest)
            manifests.extend(identity_manifest)
            classes.append(
                {
                    "split": split,
                    "identity_id": identity_id,
                    "identity_key": identity_key,
                    "role": identity.role,
                    "team": identity.team,
                    "jersey": identity.jersey,
                    "identity_type": identity.identity_type,
                    "num_crops": len(identity_manifest),
                }
            )

        summary["splits"][split] = {
            "identities": identity_count,
            "crops": crop_count,
        }

    if not summary["splits"]["train"]["identities"]:
        raise RuntimeError("No train identities with enough valid crops were produced")

    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as manifest_file:
        fieldnames = list(manifests[0]) if manifests else []
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(manifests)
    with (output / "classes.csv").open("w", encoding="utf-8", newline="") as classes_file:
        fieldnames = list(classes[0]) if classes else []
        writer = csv.DictWriter(classes_file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(classes)
    with (output / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")

    for split, values in summary["splits"].items():
        print(
            f"{split}: {values['identities']} identities, {values['crops']} crops"
        )
    print(f"Prepared dataset: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SoccerNet tracking crops as identity folders for OSNet metric learning."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets") / "raw" / "soccernet-tracking-2023",
        help="SoccerNet tracking dataset root containing train/ and test/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets") / "prepared" / "soccernet-reid",
        help="Output directory (must be empty or not exist).",
    )
    parser.add_argument(
        "--val-game-ids",
        nargs="+",
        default=["9"],
        help="Whole raw-train game IDs held out for validation (default: 9).",
    )
    parser.add_argument(
        "--max-crops-per-tracklet",
        type=positive_int,
        default=32,
        help="Maximum evenly time-spaced crops from each annotated tracklet.",
    )
    parser.add_argument(
        "--min-box-height",
        type=positive_int,
        default=32,
        help="Ignore boxes shorter than this many source-image pixels.",
    )
    parser.add_argument(
        "--min-crops-per-identity",
        type=positive_int,
        default=2,
        help="Minimum valid crops required for an identity to support triplets.",
    )
    args = parser.parse_args()
    prepare_dataset(args)


if __name__ == "__main__":
    main()
