"""Turn ball detections into one consistent position per frame. Filter candidates by pitch
location, overlap with people, static objects and plausible motion. An optional homography
allows path checks in pitch coordinates and gap interpolation that accounts for camera
movement. Save the selected track and interpolation flags to ball.csv, with an optional
video showing accepted and rejected candidates."""

from pathlib import Path
from collections import deque

import cv2
import argparse
import numpy as np
import pandas as pd

from pipeline_paths import resolve_output_dir

GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
TRAIL = 15
DRAW_CONF = 0.3

PITCH_W = 105.0
PITCH_H = 68.0

LINK_BONUS = 0.5        
JUMP_PEN = 0.5      

PEOPLE_CONF = 0.3
STATIC_CELL = 2.0   
STATIC_STD = 0.6  


def open_cap(sample_dir):
    cap = cv2.VideoCapture(sample_dir)
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open {sample_dir}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, fps, w, h, frame_count


def load_homographies(path):
    hom = pd.read_csv(path)
    hcols = [f"h{i}" for i in range(9)]
    hom = hom.dropna(subset=hcols)
    return {int(f): v.reshape(3, 3) for f, v in zip(hom["frame"], hom[hcols].to_numpy())}


def to_pitch(Hm, pts):
    """Map image coordinates to pitch coordinates (meters)."""
    p = np.c_[pts, np.ones(len(pts))] @ Hm.T
    den = np.where(np.abs(p[:, 2:3]) < 1e-9, 1e-9, p[:, 2:3])
    return p[:, :2] / den


def to_image(Hm, pts):
    """Map pitch coordinates (meters) to image coordinates."""
    p = np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(Hm).T
    den = np.where(np.abs(p[:, 2:3]) < 1e-9, 1e-9, p[:, 2:3])
    return p[:, :2] / den


def add_pitch(c, Hs):
    """Add pitch coordinates px and py; use NaN when a frame has no homography."""
    px = np.full(len(c), np.nan)
    py = np.full(len(c), np.nan)
    xy = c[["cx", "cy"]].to_numpy()
    for f, idx in c.groupby("frame").indices.items():
        Hm = Hs.get(int(f))
        if Hm is None:
            continue
        p = to_pitch(Hm, xy[idx])
        px[idx], py[idx] = p[:, 0], p[:, 1]
    return c.assign(px=px, py=py)


def outside_mask(c, margin):
    """Find candidates outside the pitch, such as those on stands or advertising boards."""
    px, py = c["px"].to_numpy(), c["py"].to_numpy()
    with np.errstate(invalid="ignore"):
        out = (px < -margin) | (px > PITCH_W + margin) | (py < -margin) | (py > PITCH_H + margin)
    return out & ~np.isnan(px)


def body_mask(c, people, frac, max_conf):
    """Find candidates inside the upper part of a player box, such as shirts or heads."""
    out = np.zeros(len(c), bool)
    cx, cy = c["cx"].to_numpy(), c["cy"].to_numpy()
    conf, fr = c["conf"].to_numpy(), c["frame"].to_numpy()
    for i in range(len(c)):
        if conf[i] >= max_conf:
            continue
        b = people.get(int(fr[i]))
        if b is None:
            continue
        hit = ((cx[i] >= b[:, 0]) & (cx[i] <= b[:, 2]) & (cy[i] >= b[:, 1])
               & (cy[i] <= b[:, 1] + frac * (b[:, 3] - b[:, 1])))
        out[i] = hit.any()
    return out


def static_mask(c, fps, static_s):
    """Find candidate clusters that remain still for several seconds, such as logos or penalty spots."""
    out = np.zeros(len(c), bool)
    ok = c["px"].notna().to_numpy()
    if static_s <= 0 or not ok.any():
        return out
    cells = pd.DataFrame({
        "gx": np.floor(c["px"].to_numpy()[ok] / STATIC_CELL).astype(int),
        "gy": np.floor(c["py"].to_numpy()[ok] / STATIC_CELL).astype(int),
        "frame": c["frame"].to_numpy()[ok],
        "px": c["px"].to_numpy()[ok],
        "py": c["py"].to_numpy()[ok],
    })
    st = cells.groupby(["gx", "gy"]).agg(n=("frame", "nunique"), sx=("px", "std"), sy=("py", "std"))
    st["sx"], st["sy"] = st["sx"].fillna(0), st["sy"].fillna(0)
    flagged = st[(st["n"] > static_s * fps) & (st["sx"] < STATIC_STD) & (st["sy"] < STATIC_STD)].index
    if len(flagged) == 0:
        return out
    key = pd.MultiIndex.from_arrays([cells["gx"], cells["gy"]])
    out[ok] = key.isin(flagged)
    return out


def best_path(frames, xy, pxy, conf, v_img, v_pitch, window):
    """Choose at most one candidate per frame, balancing confidence and path consistency.

    Candidates within ``window`` frames can share a path only if their implied speed is plausible.
    Use meters when both have homographies, otherwise use pixels. A path can restart at no cost
    after the window has elapsed.
    """
    n = len(frames)
    dp = np.zeros(n)
    prev = np.full(n, -1)
    run_best, run_arg = -np.inf, -1
    ptr = 0
    for i in range(n):
        while ptr < i and frames[ptr] < frames[i] - window:
            if dp[ptr] > run_best:
                run_best, run_arg = dp[ptr], ptr
            ptr += 1
        lo = int(np.searchsorted(frames, frames[i] - window, side="left"))
        hi = int(np.searchsorted(frames, frames[i], side="left"))

        best, arg = 0.0, -1
        if run_arg >= 0 and run_best > best:
            best, arg = run_best, run_arg
        if hi > lo:
            gap = (frames[i] - frames[lo:hi]).astype(float)
            with np.errstate(invalid="ignore"):
                d_img = np.hypot(*(xy[lo:hi] - xy[i]).T) / gap / v_img
                d_pit = np.hypot(*(pxy[lo:hi] - pxy[i]).T) / gap / v_pitch
                sp = np.where(np.isnan(d_pit), d_img, d_pit)
                val = dp[lo:hi] + LINK_BONUS - JUMP_PEN * sp
                val[sp > 1.0] = -np.inf
            j = int(val.argmax())
            if val[j] > best:
                best, arg = float(val[j]), lo + j
        dp[i] = conf[i] + best
        prev[i] = arg

    keep = np.zeros(n, bool)
    end = int(dp.argmax()) if n else -1
    while end >= 0:
        keep[end] = True
        end = int(prev[end])
    return keep


def reject_isolated(frames, window):
    if window <= 0:
        return np.ones(len(frames), dtype=bool)
    lo = np.searchsorted(frames, frames - window, side="left")
    hi = np.searchsorted(frames, frames + window, side="right")
    return (hi - lo) > 1


def fill_short_gaps(track, max_gap, Hs):
    """Fill short gaps. With a homography, interpolate on the pitch and project back into each frame
    so camera motion does not shift the estimated position."""
    missing = track["x"].isna()
    run_id = (~missing).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    filled = track[["x", "y"]].interpolate(method="linear", limit_area="inside")
    fill_mask = missing & (run_len <= max_gap) & filled["x"].notna()
    track.loc[fill_mask, ["x", "y"]] = filled.loc[fill_mask]
    track["interpolated"] = fill_mask

    if Hs:
        xs, ys = track["x"].to_numpy().copy(), track["y"].to_numpy().copy()
        pxv, pyv = track["px"].to_numpy(), track["py"].to_numpy()
        fm = fill_mask.to_numpy()
        frames = track.index.to_numpy()
        n = len(track)
        i = 0
        while i < n:
            if not fm[i]:
                i += 1
                continue
            j = i
            while j < n and fm[j]:
                j += 1
            a, b = i - 1, j
            if a >= 0 and b < n and not (np.isnan(pxv[a]) or np.isnan(pxv[b])):
                if all(int(frames[k]) in Hs for k in range(i, j)):
                    for k in range(i, j):
                        t = (k - a) / (b - a)
                        p = np.array([[pxv[a] + t * (pxv[b] - pxv[a]), pyv[a] + t * (pyv[b] - pyv[a])]])
                        q = to_image(Hs[int(frames[k])], p)[0]
                        if np.isfinite(q).all():
                            xs[k], ys[k] = q
            i = j
        track["x"], track["y"] = xs, ys
    return track


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--homography", default=None)
    parser.add_argument("--min-conf", type=float, default=0.15)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--max-speed", type=float, default=80.0)
    parser.add_argument("--max-speed-ms", type=float, default=45.0)
    parser.add_argument("--max-gap", type=float, default=0.5)
    parser.add_argument("--isolation", type=int, default=3)
    parser.add_argument("--pitch-margin", type=float, default=8.0)
    parser.add_argument("--body-frac", type=float, default=0.55)
    parser.add_argument("--body-conf", type=float, default=0.7)
    parser.add_argument("--static-s", type=float, default=8.0)
    parser.add_argument("--out", default="out")
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    out_dir = resolve_output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    det = pd.read_csv(args.detections)
    first, last = int(det["frame"].min()), int(det["frame"].max())

    ball = det[det["label"] == "ball"].copy()
    if ball.empty:
        raise RuntimeError("No ball detections found in the CSV")
    ball["cx"] = (ball["x1"] + ball["x2"]) / 2
    ball["cy"] = (ball["y1"] + ball["y2"]) / 2

    ball = ball[ball["conf"] >= args.min_conf]
    if ball.empty:
        raise RuntimeError(f"No ball detections with confidence >= {args.min_conf}")
    cands = (ball.sort_values(["frame", "conf"], ascending=[True, False])
             .groupby("frame").head(args.candidates).reset_index(drop=True))

    cap, fps, w, h, frame_count = open_cap(args.sample)
    max_gap = int(round(fps * args.max_gap))

    Hs = load_homographies(args.homography) if args.homography else {}
    cands = add_pitch(cands, Hs)
    n_cands = len(cands)

    people = {f: g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
              for f, g in det[(det["label"] != "ball") & (det["conf"] >= PEOPLE_CONF)].groupby("frame")}

    reason = np.full(len(cands), "", dtype=object)
    masks = [("outside pitch", outside_mask(cands, args.pitch_margin))]
    if args.body_frac > 0:
        masks.append(("inside player box", body_mask(cands, people, args.body_frac, args.body_conf)))
    masks.append(("static object", static_mask(cands, fps, args.static_s)))
    for name, m in masks:
        m = m & (reason == "")
        reason[m] = name
    filtered = cands[reason != ""].assign(reason=reason[reason != ""])
    cands = cands[reason == ""].reset_index(drop=True)
    if cands.empty:
        raise RuntimeError("All candidates were filtered out. Try --body-frac 0 --static-s 0")

    path = best_path(
        cands["frame"].to_numpy(),
        cands[["cx", "cy"]].to_numpy(),
        cands[["px", "py"]].to_numpy(),
        cands["conf"].to_numpy(),
        args.max_speed,
        args.max_speed_ms / fps,
        max(max_gap, 1),
    )
    off_path = cands[~path].assign(reason="path")
    chosen = cands[path]

    supported = reject_isolated(chosen["frame"].to_numpy(), args.isolation)
    isolated = chosen[~supported].assign(reason="isolated")
    kept = chosen[supported].set_index("frame")
    if kept.empty:
        raise RuntimeError("All detections were isolated. Try --isolation 0")
    rejected = pd.concat([filtered, off_path, isolated])

    track = kept[["cx", "cy", "conf", "px", "py"]].rename(columns={"cx": "x", "cy": "y"})
    track = track.reindex(pd.RangeIndex(first, last + 1, name="frame"))
    track = fill_short_gaps(track, max_gap, Hs)
    track[["x", "y", "conf", "interpolated"]].reset_index().to_csv(out_dir / "ball.csv", index=False)

    if args.render:
        writer = cv2.VideoWriter(
            str(out_dir / "ball_track.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        shown = rejected[rejected["conf"] >= DRAW_CONF]
        rejected_by_frame = {f: g for f, g in shown.groupby("frame")}
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        trail = deque(maxlen=TRAIL)

        for idx in range(first, last + 1):
            ok, frame = cap.read()
            if not ok:
                break

            row = track.loc[idx]
            status = "no ball"
            if not np.isnan(row["x"]):
                center = (int(row["x"]), int(row["y"]))
                color = ORANGE if row["interpolated"] else GREEN
                status = "interpolated" if row["interpolated"] else "detected"
                trail.append(center)
                if len(trail) > 1:
                    cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, color, 2)
                cv2.circle(frame, center, 8, color, 2)

            if idx in rejected_by_frame:
                for _, r in rejected_by_frame[idx].iterrows():
                    cv2.drawMarker(
                        frame, (int(r["cx"]), int(r["cy"])), RED, cv2.MARKER_TILTED_CROSS, 16, 2
                    )

            cv2.putText(
                frame, f"{idx} {status}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            writer.write(frame)
        writer.release()

    cap.release()

    n = len(track)
    print(f"Frames: {n} ({first}-{last})")
    print(f"Ball candidates: {n_cands} (homography: {'yes' if Hs else 'no'})")
    for name, count in filtered["reason"].value_counts().items():
        print(f"Rejected by filter ({name}): {count}")
    print(f"Rejected by path selection: {len(off_path)}")
    print(f"Rejected as isolated: {len(isolated)}")
    print(f"Detected: {int(track['conf'].notna().sum())}")
    print(f"Interpolated: {int(track['interpolated'].sum())}")
    print(f"Frames without a ball: {int(track['x'].isna().sum())}")


if __name__ == "__main__":
    main()  