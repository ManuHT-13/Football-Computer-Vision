"""Link frame-by-frame person detections into player, goalkeeper and referee tracks. A
Kalman-filter tracker associates detections using box overlap and, when enabled, appearance
features and pitch position. Camera-motion compensation and scene-cut handling help
maintain identities through movement and cuts. Short, weak or nearly stationary tracks
are filtered out; results and removed detections go to tracks.csv and tracks_dropped.csv,
with an optional video showing the retained tracks."""

from pathlib import Path

import cv2
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pipeline_paths import resolve_output_dir

GMC_WIDTH = 640
GMC_MARGIN = 0.05
MIN_POINTS = 10

LOW_COST = 0.5
FRESH_COST = 0.7
WP = 1 / 20
WV = 1 / 160

ALPHA = 0.9          
EMB_CONF = 0.5       
EMB_OVERLAP = 0.3  
LAMBDA = 0.5        
APP_GATE = 0.5  
RELINK_GATE = 0.35  
RELINK_MARGIN = 0.05
ARCHIVE_MAX = 60
GRASS_H = (35, 85)
GRASS_S = 40
HIST_MIN_PIXELS = 20

PITCH_W = 105.0
PITCH_H = 68.0
PITCH_MARGIN = 5.0 
PG_BASE = 3.0 
PG_GROW = 0.15    
PG_MAX = 12.0
EXTRAP_MAX = 25    
VEL_ALPHA = 0.3  
VMAX_PF = 0.5      
W_IOU, W_POS, W_APP = 0.3, 0.3, 0.4  
RELINK_W_APP = 0.6  
RELINK_PITCH_M = 30.0

MIN_HITS = 3        
DUP_IOU = 0.7      


def open_cap(sample_dir):
    cap = cv2.VideoCapture(sample_dir)
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open {sample_dir}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, fps, w, h, frame_count


def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


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


class HistEmbedder:
    """Compute separate Hellinger HSV histograms for the shirt and shorts, excluding grass."""

    regions = ((0.15, 0.45), (0.45, 0.65)) 
    bins = (16, 8)
    dim = 2 * 16 * 8

    def __call__(self, frame, boxes):
        fh, fw = frame.shape[:2]
        out = np.zeros((len(boxes), self.dim))
        n = self.bins[0] * self.bins[1]
        for k, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, fw), min(y2, fh)
            bw, bh = x2 - x1, y2 - y1
            if bw < 4 or bh < 8:
                continue
            parts = []
            for ya, yb in self.regions:
                crop = frame[y1 + int(ya * bh): y1 + int(yb * bh), x1 + int(0.2 * bw): x2 - int(0.2 * bw)]
                if crop.size == 0:
                    parts.append(np.zeros(n))
                    continue
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                grass = (hsv[..., 0] >= GRASS_H[0]) & (hsv[..., 0] <= GRASS_H[1]) & (hsv[..., 1] >= GRASS_S)
                keep = (~grass).astype(np.uint8)
                if keep.sum() < HIST_MIN_PIXELS:
                    parts.append(np.zeros(n))
                    continue
                hist = cv2.calcHist([hsv], [0, 1], keep, list(self.bins), [0, 180, 0, 256]).ravel()
                parts.append(np.sqrt(hist / (hist.sum() + 1e-9)))
            v = np.concatenate(parts)
            out[k] = v / (np.linalg.norm(v) + 1e-9)
        return out


class OSNetEmbedder:
    """OSNet person re-identification using torchreid."""

    dim = 512

    def __init__(self, weights="", device=None):
        import torch
        from torchreid.reid.utils import FeatureExtractor

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ex = FeatureExtractor(model_name="osnet_x1_0", model_path=weights, device=device)

    def __call__(self, frame, boxes):
        fh, fw = frame.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes.astype(int):
            c = frame[max(y1, 0):min(y2, fh), max(x1, 0):min(x2, fw), ::-1]  # BGR -> RGB
            crops.append(c if c.size else np.zeros((64, 32, 3), np.uint8))
        return l2n(self.ex(crops).cpu().numpy())


def build_embedder(kind, weights):
    if kind == "hist":
        return HistEmbedder()
    if kind == "osnet":
        return OSNetEmbedder(weights)
    return None


def xyxy_to_xyah(box):
    w = box[2] - box[0]
    h = max(box[3] - box[1], 1.0)
    return np.array([box[0] + w / 2, box[1] + h / 2, w / h, h])


def xyah_to_xyxy(z):
    w = z[2] * z[3]
    return np.array([z[0] - w / 2, z[1] - z[3] / 2, z[0] + w / 2, z[1] + z[3] / 2])


F = np.eye(8)
F[:4, 4:] = np.eye(4)
H = np.eye(4, 8)


def kalman_initiate(z):
    h = z[3]
    std = np.array([2 * WP * h, 2 * WP * h, 1e-2, 2 * WP * h, 10 * WV * h, 10 * WV * h, 1e-5, 10 * WV * h])
    return np.r_[z, np.zeros(4)], np.diag(std ** 2)


def kalman_predict(mean, cov):
    h = mean[3]
    std = np.array([WP * h, WP * h, 1e-2, WP * h, WV * h, WV * h, 1e-5, WV * h])
    return F @ mean, F @ cov @ F.T + np.diag(std ** 2)


def kalman_update(mean, cov, z):
    h = mean[3]
    R = np.diag(np.array([WP * h, WP * h, 1e-1, WP * h]) ** 2)
    S = H @ cov @ H.T + R
    K = np.linalg.solve(S, H @ cov).T
    return mean + K @ (z - H @ mean), cov - K @ S @ K.T


def iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def iou_cost(tracks, boxes):
    if not tracks or len(boxes) == 0:
        return np.zeros((len(tracks), len(boxes)))
    return 1 - iou_matrix(np.array([t.xyxy() for t in tracks]), boxes)


def associate(cost, thresh):
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    rows, cols = linear_sum_assignment(np.where(cost > thresh, 1e6, cost))
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    used_r = {r for r, _ in matches}
    used_c = {c for _, c in matches}
    return (
        matches,
        [i for i in range(n) if i not in used_r],
        [j for j in range(m) if j not in used_c],
    )


def nms(boxes, confs, thr=0.7):
    """Class-agnostic NMS prevents one person from creating duplicate player/referee tracks."""
    iou = iou_matrix(boxes, boxes)
    sup = np.zeros(len(boxes), bool)
    keep = []
    for i in np.argsort(-confs):
        if sup[i]:
            continue
        keep.append(i)
        sup |= iou[i] > thr
    return np.array(keep, dtype=int)


class Track:
    def __init__(self, box, conf, det_idx, frame, feat=None, pos=None):
        self.mean, self.cov = kalman_initiate(xyxy_to_xyah(box))
        self.conf = conf
        self.det_idx = det_idx
        self.last_frame = frame
        self.state = "new"
        self.id = 0
        self.hits = 1
        self.feat = feat
        self.pos = pos         
        self.pos_frame = frame
        self.vel = np.zeros(2)    

    def predict(self):
        if self.state != "tracked":
            self.mean[7] = 0
        self.mean, self.cov = kalman_predict(self.mean, self.cov)

    def update(self, box, conf, det_idx, frame, feat=None, pos=None):
        self.mean, self.cov = kalman_update(self.mean, self.cov, xyxy_to_xyah(box))
        self.conf = conf
        self.det_idx = det_idx
        self.last_frame = frame
        self.state = "tracked"
        self.hits += 1
        if feat is not None:
            self.feat = l2n(ALPHA * self.feat + (1 - ALPHA) * feat) if self.feat is not None else feat
        if pos is not None:
            if self.pos is not None and 0 < frame - self.pos_frame <= 5:
                v = (pos - self.pos) / (frame - self.pos_frame)
                self.vel = (1 - VEL_ALPHA) * self.vel + VEL_ALPHA * v
                sp = float(np.hypot(*self.vel))
                if sp > VMAX_PF:
                    self.vel = self.vel * (VMAX_PF / sp)
            self.pos, self.pos_frame = pos, frame

    def pred_pos(self, frame):
        if self.pos is None:
            return None
        return self.pos + self.vel * min(frame - self.pos_frame, EXTRAP_MAX)

    def xyxy(self):
        return xyah_to_xyxy(self.mean[:4])


class ByteTracker:
    def __init__(self, high, low, min_iou, max_lost, next_id=1, archive=None,
                 new_conf=0.4, min_hits=MIN_HITS, relink_window=90):
        self.high = high
        self.low = low
        self.match_cost = 1 - min_iou
        self.max_lost = max_lost
        self.next_id = next_id
        self.archive = archive or [] 
        self.retired = {}          
        self.new_conf = new_conf
        self.min_hits = min_hits
        self.relink_window = relink_window
        self.relinked = 0
        self.frame = 0
        self.tracks = []

    def snapshot(self):
        """Return active, recently closed, and unclaimed archived tracks for scene-cut relinking."""
        cur = {t.id: (t.id, t.feat, t.pos, t.vel.copy(), self.frame - t.last_frame)
               for t in self.tracks
               if t.id and t.feat is not None and t.state in ("tracked", "lost")}
        ret = {i: (i, f, p, v, self.frame - lf)
               for i, (_, f, p, v, lf) in self.retired.items() if i not in cur}
        old = [(i, f, p, v, a + self.frame)
               for i, f, p, v, a in self.archive if i not in cur and i not in ret]
        return (old + list(ret.values()) + list(cur.values()))[-ARCHIVE_MAX:]

    def pitch_dist(self, tracks, pitch):
        """Measure distances in meters from each predicted track position to each detection."""
        P = np.full((len(tracks), 2), np.nan)
        age = np.zeros(len(tracks))
        for i, t in enumerate(tracks):
            p = t.pred_pos(self.frame)
            if p is not None:
                P[i] = p
                age[i] = self.frame - t.pos_frame
        d = np.hypot(P[:, None, 0] - pitch[None, :, 0], P[:, None, 1] - pitch[None, :, 1])
        return d, age

    def cost(self, tracks, boxes, feats, pitch=None):
        c = iou_cost(tracks, boxes)
        if c.size == 0 or (feats is None and pitch is None):
            return c

        app, valid = c, np.zeros(c.shape, bool)
        if feats is not None:
            tf = np.array([t.feat for t in tracks])
            valid = (np.linalg.norm(tf, axis=1) > 0)[:, None] & (np.linalg.norm(feats, axis=1) > 0)[None, :]
            app = np.where(valid, 1 - tf @ feats.T, c)  # Fall back to IoU when no embedding is available.

        with np.errstate(invalid="ignore"):
            if pitch is not None:
                d, age = self.pitch_dist(tracks, pitch)
                gate = np.minimum(PG_BASE + PG_GROW * age, PG_MAX)[:, None]
                pv = ~np.isnan(d)
                pn = np.where(pv, np.clip(d / gate, 0, 1), c)
                fused = np.where(pv, W_IOU * c + W_POS * pn + W_APP * app, LAMBDA * c + (1 - LAMBDA) * app)
                geo_ok = (c <= self.match_cost) | (pv & (d <= gate)) 
            else:
                fused = LAMBDA * c + (1 - LAMBDA) * app
                geo_ok = c <= self.match_cost
            fused[~geo_ok | ((app > APP_GATE) & valid)] = 1e6
        return fused

    def assign_ids(self, newly):
        if not newly:
            return
        free = set(range(len(newly)))
        if self.archive and self.frame <= self.relink_window and newly[0].feat is not None:
            A = np.array([a[1] for a in self.archive])
            app = 1 - np.array([t.feat for t in newly]) @ A.T
            cost = app.copy()
            for r, t in enumerate(newly):
                if t.pos is None:
                    continue
                for c, (_, _, pos, vel, age) in enumerate(self.archive):
                    if pos is None:
                        continue
                    pred = pos + vel * min(age + self.frame, EXTRAP_MAX)
                    d = float(np.hypot(*(t.pos - pred)))
                    cost[r, c] = RELINK_W_APP * app[r, c] + (1 - RELINK_W_APP) * min(d / RELINK_PITCH_M, 1.0)
            matches, _, _ = associate(cost, RELINK_GATE)
            used = set()
            for r, c in matches:
                second = np.partition(cost[r], 1)[1] if cost.shape[1] > 1 else 1.0
                if second - cost[r, c] < RELINK_MARGIN: 
                    continue
                newly[r].id = self.archive[c][0]
                used.add(c)
                free.discard(r)
                self.relinked += 1
            self.archive = [a for i, a in enumerate(self.archive) if i not in used]
        for r in sorted(free):
            newly[r].id = self.next_id
            self.next_id += 1

    def update(self, boxes, confs, feats=None, pitch=None):
        self.frame += 1
        use_app = feats is not None
        high = np.where(confs >= self.high)[0]
        low = np.where((confs >= self.low) & (confs < self.high))[0]

        if len(boxes) > 1:
            ov = iou_matrix(boxes, boxes)
            np.fill_diagonal(ov, 0)
            clean = (confs >= EMB_CONF) & (ov.max(1) < EMB_OVERLAP)
        else:
            clean = confs >= EMB_CONF

        def f(i):
            return feats[i] if use_app and clean[i] else None

        def pp(i):
            if pitch is None or not np.isfinite(pitch[i]).all():
                return None
            return pitch[i]

        pool = [t for t in self.tracks if t.state in ("tracked", "lost")]
        fresh = [t for t in self.tracks if t.state == "new"]
        for t in pool + fresh:
            t.predict()
        newly = []

        fh = feats[high] if use_app else None
        fp = pitch[high] if pitch is not None else None
        matches, un_pool, un_high = associate(self.cost(pool, boxes[high], fh, fp), self.match_cost)
        for r, c in matches:
            pool[r].update(boxes[high[c]], confs[high[c]], high[c], self.frame, f(high[c]), pp(high[c]))

        remaining = [pool[i] for i in un_pool if pool[i].state == "tracked"]
        matches, un_rem, _ = associate(iou_cost(remaining, boxes[low]), LOW_COST)
        for r, c in matches:
            remaining[r].update(boxes[low[c]], confs[low[c]], low[c], self.frame, None, pp(low[c]))
        for i in un_rem:
            remaining[i].state = "lost"

        rest = high[un_high]
        matches, un_fresh, un_rest = associate(iou_cost(fresh, boxes[rest]), FRESH_COST)
        for r, c in matches:
            t = fresh[r]
            t.update(boxes[rest[c]], confs[rest[c]], rest[c], self.frame, f(rest[c]), pp(rest[c]))
            if t.hits >= self.min_hits:
                newly.append(t)
            else:
                t.state = "new"
        for i in un_fresh:
            fresh[i].state = "removed"

        active = np.array([t.xyxy() for t in self.tracks if t.last_frame == self.frame]).reshape(-1, 4)
        for c in un_rest:
            i = rest[c]
            if confs[i] < self.new_conf:
                continue
            if len(active) and iou_matrix(boxes[i][None], active).max() > DUP_IOU:
                continue
            self.tracks.append(Track(boxes[i], confs[i], i, self.frame, feats[i] if use_app else None, pp(i)))

        self.assign_ids(newly)

        for t in self.tracks:
            if t.state == "lost" and self.frame - t.last_frame > self.max_lost:
                t.state = "removed"
                if t.id and t.feat is not None:
                    self.retired[t.id] = (t.id, t.feat, t.pos, t.vel.copy(), t.last_frame)
        self.tracks = [t for t in self.tracks if t.state != "removed"]

        return [
            (t.id, t.det_idx)
            for t in self.tracks
            if t.state == "tracked" and t.last_frame == self.frame
        ]


def estimate_motion(prev_gray, gray, prev_boxes):
    hh, ww = prev_gray.shape
    mask = np.zeros((hh, ww), np.uint8)
    my, mx = int(hh * GMC_MARGIN), int(ww * GMC_MARGIN)
    mask[my:hh - my, mx:ww - mx] = 255
    for x1, y1, x2, y2 in prev_boxes.astype(int):
        mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0

    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=3, mask=mask)
    if pts is None or len(pts) < MIN_POINTS:
        return None, None

    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    ok = status.ravel() == 1
    if ok.sum() < MIN_POINTS:
        return None, 0.0

    M, inliers = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None, 0.0
    return M, float(inliers.sum()) / len(pts)


def to_full(M, scale):
    full = np.eye(3)
    full[:2, :2] = M[:, :2]
    full[:2, 2] = M[:, 2] / scale
    return full


def to_reference(boxes, T):
    if len(boxes) == 0:
        return boxes
    x1, y1, x2, y2 = boxes.T
    xs = np.stack([x1, x2, x2, x1])
    ys = np.stack([y1, y1, y2, y2])
    rx = T[0, 0] * xs + T[0, 1] * ys + T[0, 2]
    ry = T[1, 0] * xs + T[1, 1] * ys + T[1, 2]
    return np.stack([rx.min(0), ry.min(0), rx.max(0), ry.max(0)], axis=1)


def color_for(track_id):
    hue = (track_id * 47) % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def filter_tracks(df, min_len, min_conf, min_move):
    """Discard short, low-confidence, or nearly stationary tracks."""
    df = df.assign(h=df.y2 - df.y1)
    st = df.groupby("track_id").agg(
        n=("frame", "size"), conf=("conf", "mean"), h=("h", "median"),
        sx=("rx", "std"), sy=("ry", "std"),
    )
    st["move"] = np.hypot(st.sx.fillna(0), st.sy.fillna(0)) / st.h
    bad = (st.n < min_len) | (st.conf < min_conf) | ((st.move < min_move) & (st.conf < 0.6))
    drop = set(st.index[bad])
    mask = df.track_id.isin(drop)
    return df[~mask].drop(columns="h"), df[mask].drop(columns="h"), len(drop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--homography", default=None)
    parser.add_argument("--labels", nargs="+", default=["player", "goalkeeper", "referee"])
    parser.add_argument("--high-conf", type=float, default=0.25)
    parser.add_argument("--low-conf", type=float, default=0.1)
    parser.add_argument("--new-conf", type=float, default=0.4)
    parser.add_argument("--min-iou", type=float, default=0.2)
    parser.add_argument("--min-hits", type=int, default=MIN_HITS)
    parser.add_argument("--buffer", type=float, default=2.0)
    parser.add_argument("--no-gmc", action="store_true")
    parser.add_argument("--cut-ratio", type=float, default=0.25)
    parser.add_argument("--reid", choices=["none", "hist", "osnet"], default="hist")
    parser.add_argument("--reid-weights", default="")
    parser.add_argument("--relink-s", type=float, default=3.0)
    parser.add_argument("--min-track-len", type=int, default=15)
    parser.add_argument("--min-track-conf", type=float, default=0.45)
    parser.add_argument("--min-move", type=float, default=0.05)
    parser.add_argument("--out", default="out")
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    out_dir = resolve_output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    det = pd.read_csv(args.detections)
    first, last = int(det["frame"].min()), int(det["frame"].max())
    det = det[det["label"].isin(args.labels) & (det["conf"] >= args.low_conf)]
    if det.empty:
        raise RuntimeError(f"No detections for {args.labels} with confidence >= {args.low_conf}")
    groups = {f: g.reset_index(drop=True) for f, g in det.groupby("frame")}

    Hs = load_homographies(args.homography) if args.homography else {}
    if args.homography and not Hs:
        print("Warning: homography.csv contains no valid homographies; tracking will use image coordinates only")

    cap, fps, w, h, frame_count = open_cap(args.sample)
    max_lost = int(round(fps * args.buffer))
    scale = min(1.0, GMC_WIDTH / w)
    embedder = build_embedder(args.reid, args.reid_weights)
    use_video = (not args.no_gmc) or args.render or embedder is not None

    def new_tracker(next_id, archive=None):
        return ByteTracker(args.high_conf, args.low_conf, args.min_iou, max_lost, next_id,
                           archive, args.new_conf, args.min_hits, int(fps * args.relink_s))

    writer = None
    if args.render:
        writer = cv2.VideoWriter(
            str(out_dir / "tracks.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
    if use_video:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    tracker = new_tracker(1)
    T = np.eye(3)
    prev_gray = None
    prev_boxes = np.zeros((0, 4))
    cuts = []
    rows = []
    relinked = 0
    n_pitch = 0

    for idx in range(first, last + 1):
        frame = None
        if use_video:
            ok, frame = cap.read()
            if not ok:
                break

        sub = groups.get(idx)
        if sub is None:
            boxes = np.zeros((0, 4))
            confs = np.zeros(0)
            labels = np.array([], dtype=object)
        else:
            boxes = sub[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            confs = sub["conf"].to_numpy(dtype=float)
            labels = sub["label"].to_numpy()
            if len(boxes) > 1:
                keep = nms(boxes, confs)
                boxes, confs, labels = boxes[keep], confs[keep], labels[keep]

        cut = False
        if not args.no_gmc:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if scale < 1.0:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if prev_gray is not None:
                M, ratio = estimate_motion(prev_gray, gray, prev_boxes * scale)
                if ratio is not None and ratio < args.cut_ratio:
                    cut = True
                elif M is not None:
                    T = T @ np.linalg.inv(to_full(M, scale))
            prev_gray = gray
            prev_boxes = boxes

        if cut:
            cuts.append(idx)
            T = np.eye(3)
            relinked += tracker.relinked
            tracker = new_tracker(tracker.next_id, tracker.snapshot())

        feats = None
        if embedder is not None:
            feats = np.zeros((len(boxes), embedder.dim))
            hi = confs >= args.high_conf
            if hi.any():
                feats[hi] = embedder(frame, boxes[hi])

        pitch = None
        if Hs:
            pitch = np.full((len(boxes), 2), np.nan)
            Hm = Hs.get(idx)
            if Hm is not None and len(boxes):
                foot = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
                p = to_pitch(Hm, foot)
                inside = ((p[:, 0] >= -PITCH_MARGIN) & (p[:, 0] <= PITCH_W + PITCH_MARGIN)
                          & (p[:, 1] >= -PITCH_MARGIN) & (p[:, 1] <= PITCH_H + PITCH_MARGIN))
                pitch[inside] = p[inside]
                n_pitch += int(inside.sum())

        ref = to_reference(boxes, T)
        matched = tracker.update(ref, confs, feats, pitch)

        for tid, i in matched:
            x1, y1, x2, y2 = boxes[i]
            rows.append((idx, tid, labels[i], x1, y1, x2, y2, confs[i],
                         (ref[i, 0] + ref[i, 2]) / 2, ref[i, 3]))

        if writer is not None:
            for tid, i in matched:
                x1, y1, x2, y2 = boxes[i].astype(int)
                color = color_for(tid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, str(tid), (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            status = f"{idx}  scene cut" if cut else f"{idx}"
            cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    relinked += tracker.relinked

    if not rows:
        raise RuntimeError("The tracker did not produce any tracks")

    df = pd.DataFrame(
        rows, columns=["frame", "track_id", "label", "x1", "y1", "x2", "y2", "conf", "rx", "ry"]
    )
    n_raw = df["track_id"].nunique()
    df, dropped, n_drop = filter_tracks(df, args.min_track_len, args.min_track_conf, args.min_move)
    dropped.to_csv(out_dir / "tracks_dropped.csv", index=False)
    if df.empty:
        raise RuntimeError("The track filter removed everything. Lower --min-track-len, --min-track-conf, or --min-move")

    df["track_label"] = df.groupby("track_id")["label"].transform(lambda s: s.mode().iloc[0])
    df.to_csv(out_dir / "tracks.csv", index=False)

    lengths = df.groupby("track_id").size()
    per_frame = df.groupby("frame").size()
    print(f"Frames: {last - first + 1} ({first}-{last})")
    print(f"Pitch positions: {'yes' if Hs else 'no'}" + (f" ({n_pitch} detections projected)" if Hs else ""))
    print(f"Tracks before filtering: {n_raw}, removed: {n_drop}")
    print(f"Distinct tracks: {len(lengths)}")
    print(f"Tracks shorter than 1 s: {int((lengths < fps).sum())}")
    print(f"Active tracks per frame: average {per_frame.mean():.1f}, max {int(per_frame.max())}")
    print(df.drop_duplicates("track_id")["track_label"].value_counts().to_string())
    print(f"Scene cuts: {len(cuts)} {cuts[:10]}, relinked IDs: {relinked}")


if __name__ == "__main__":
    main()