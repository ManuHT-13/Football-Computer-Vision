"""Assign teams to tracked players from shirt colors. Select clear, non-overlapping crops
across each track and build torso color histograms while discounting grass; PCA and
weighted clustering then group similar tracks. Leave uncertain assignments and outliers
unassigned, and write results to teams.csv and tracks_teams.csv. An annotated video and
clustering summary are optional."""

from pathlib import Path

import cv2
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from pipeline_paths import resolve_output_dir

GRASS_H = (35, 85)
GRASS_S = 40
TORSO_Y = (0.15, 0.55)
TORSO_X = (0.2, 0.8)
HIST_BINS = (12, 4, 4)
MIN_PIXELS = 30
OVERLAP_IOU = 0.2
PCA_DIMS = 8
TEAM_COLORS = {0: (255, 128, 0), 1: (0, 0, 255), 2: (0, 200, 0)}
GRAY = (160, 160, 160)


def open_cap(sample_dir):
    cap = cv2.VideoCapture(sample_dir)
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open {sample_dir}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, fps, w, h


def l2n(x):
    return x / (np.linalg.norm(x) + 1e-9)


def iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def mark_clean(df):
    clean = np.ones(len(df), bool)
    for _, g in df.groupby("frame"):
        if len(g) < 2:
            continue
        b = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        iou = iou_matrix(b, b)
        np.fill_diagonal(iou, 0)
        clean[g.index.to_numpy()] = iou.max(1) < OVERLAP_IOU
    return df.assign(clean=clean)


def select_samples(df, per_track, min_conf, min_height):
    d = df[(df["conf"] >= min_conf) & ((df["y2"] - df["y1"]) >= min_height) & df["clean"]]
    needed = {}
    counts = {}
    for tid, g in d.groupby("track_id"):
        g = g.sort_values("frame")
        pick = np.unique(np.linspace(0, len(g) - 1, min(per_track, len(g))).round().astype(int))
        counts[tid] = len(pick)
        for r in g.iloc[pick].itertuples():
            needed.setdefault(int(r.frame), []).append((tid, np.array([r.x1, r.y1, r.x2, r.y2], dtype=float)))
    return needed, counts


def torso_feature(frame, box):
    fh, fw = frame.shape[:2]
    x1 = int(max(box[0], 0))
    y1 = int(max(box[1], 0))
    x2 = int(min(box[2], fw))
    y2 = int(min(box[3], fh))
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 8:
        return None

    crop = frame[
        int(y1 + TORSO_Y[0] * bh): int(y1 + TORSO_Y[1] * bh),
        int(x1 + TORSO_X[0] * bw): int(x1 + TORSO_X[1] * bw),
    ]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    grass = (hue >= GRASS_H[0]) & (hue <= GRASS_H[1]) & (sat >= GRASS_S)
    keep = ~grass
    if keep.sum() < MIN_PIXELS:
        return None

    hist = cv2.calcHist([hsv], [0, 1, 2], keep.astype(np.uint8), list(HIST_BINS), [0, 180, 0, 256, 0, 256]).ravel()
    return np.sqrt(hist / (hist.sum() + 1e-9))


def extract_features(cap, needed, first, last):
    feats = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for idx in range(first, last + 1):
        if idx not in needed:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        for tid, box in needed[idx]:
            f = torso_feature(frame, box)
            if f is not None:
                feats.setdefault(tid, []).append(f)
    return feats


def box_color(team, label):
    if team >= 0:
        return TEAM_COLORS.get(team, GRAY)
    if label == "referee":
        return (0, 255, 255)
    if label == "goalkeeper":
        return (255, 0, 255)
    return GRAY


def render(sample, df, out_path, first, last):
    cap, fps, w, h = open_cap(sample)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    groups = {f: g for f, g in df.groupby("frame")}
    for idx in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        g = groups.get(idx)
        if g is not None:
            for r in g.itertuples():
                p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
                color = box_color(int(r.team), r.track_label)
                cv2.rectangle(frame, p1, p2, color, 2)
                cv2.putText(frame, str(r.track_id), (p1[0], max(p1[1] - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, str(idx), (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--player-labels", nargs="+", default=["player"])
    parser.add_argument("--crops-per-track", type=int, default=15)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--min-height", type=int, default=40)
    parser.add_argument("--min-margin", type=float, default=0.1)
    parser.add_argument("--outlier-z", type=float, default=3.0)
    parser.add_argument("--out", default="out")
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    out_dir = resolve_output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.tracks)
    if "track_label" not in df.columns:
        df["track_label"] = df.groupby("track_id")["label"].transform(lambda s: s.mode().iloc[0])
    first, last = int(df["frame"].min()), int(df["frame"].max())

    df = mark_clean(df)
    needed, _ = select_samples(df, args.crops_per_track, args.min_conf, args.min_height)
    if not needed:
        raise RuntimeError("No valid crops found. Lower --min-conf or --min-height")

    cap, fps, w, h = open_cap(args.sample)
    feats = extract_features(cap, needed, first, last)
    cap.release()

    meta = df.drop_duplicates("track_id").set_index("track_id")["track_label"]
    ids = [t for t in meta.index if t in feats and meta[t] in args.player_labels]
    if len(ids) <= args.k:
        raise RuntimeError(f"Only {len(ids)} tracks have valid color features for k={args.k}")

    F = np.array([l2n(np.mean(feats[t], axis=0)) for t in ids])
    weights = np.array([len(feats[t]) for t in ids], dtype=float)

    n_comp = min(PCA_DIMS, F.shape[0] - 1, F.shape[1])
    X = PCA(n_components=n_comp, random_state=0).fit_transform(F)
    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(X, sample_weight=weights)
    labels = km.labels_
    dist = km.transform(X)

    d_own = dist[np.arange(len(ids)), labels]
    order = np.sort(dist, axis=1)
    margin = (order[:, 1] - order[:, 0]) / (order[:, 1] + order[:, 0] + 1e-9)
    med = np.median(d_own)
    mad = np.median(np.abs(d_own - med)) * 1.4826 + 1e-9
    outlier = d_own > med + args.outlier_z * mad
    doubt = margin < args.min_margin

    team = labels.copy()
    team[outlier | doubt] = -1
    sil = silhouette_score(X, labels)

    table = pd.DataFrame({"track_id": meta.index, "track_label": meta.values})
    table["team"] = -1
    table["team_conf"] = 0.0
    table["n_crops"] = table["track_id"].map(lambda t: len(feats.get(t, [])))
    pos = {t: i for i, t in enumerate(ids)}
    in_players = table["track_id"].isin(pos)
    table.loc[in_players, "team"] = table.loc[in_players, "track_id"].map(lambda t: int(team[pos[t]]))
    table.loc[in_players, "team_conf"] = table.loc[in_players, "track_id"].map(lambda t: float(margin[pos[t]]))
    table.to_csv(out_dir / "teams.csv", index=False)

    merged = df.drop(columns="clean").merge(table[["track_id", "team", "team_conf"]], on="track_id", how="left")
    merged.to_csv(out_dir / "tracks_teams.csv", index=False)

    if args.render:
        render(args.sample, merged, out_dir / "teams.mp4", first, last)

    no_data = int(((table["track_label"].isin(args.player_labels)) & (table["n_crops"] == 0)).sum())
    print(f"Total tracks: {len(meta)}")
    print(f"Player tracks with valid color features: {len(ids)}")
    print(f"Crops used: {int(weights.sum())} (average {weights.mean():.1f} per track)")
    print(f"Silhouette score: {sil:.3f}")
    for k in range(args.k):
        print(f"Team {k}: {int((team == k).sum())} tracks")
    print(f"Tracks left unassigned as outliers: {int(outlier.sum())}")
    print(f"Tracks left unassigned with margin < {args.min_margin}: {int((doubt & ~outlier).sum())}")
    print(f"Player tracks without valid crops: {no_data}")
    print(table[~table["track_label"].isin(args.player_labels)]["track_label"].value_counts().to_string())


if __name__ == "__main__":
    main()