"""Combine tracked players, team assignments, ball positions and per-frame homographies
into an annotated overview video. Project player foot points onto a pitch minimap, with
optional overlays for team territories, remembered positions, ball trails, possession,
distance and speed. Goalkeepers without a team can be assigned from their pitch positions;
per-track statistics can also be saved to stats.csv alongside overview.mp4."""

from pathlib import Path
from collections import deque

import cv2
import argparse
import numpy as np
import pandas as pd

from pipeline_paths import resolve_output_dir

PITCH_W = 105.0
PITCH_H = 68.0
PITCH_MARGIN = 3.0

MAP_PAD = 4.0
MAP_ALPHA = 0.85
MAP_BG = (45, 110, 45)
MAP_LINE = (230, 230, 230)
VORONOI_ALPHA = 0.4

VORONOI_STEP = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX


SMOOTH_S = 0.4
MAX_GAP = 5
MAX_KMH = 40.0
POSS_DIST = 2.5

TRAIL = 15
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)

MEMORY_TAU = 2.0
MAX_MS = 8.0
VEL_ALPHA = 0.2

TEAM_COLORS = {0: (255, 128, 0), 1: (0, 0, 255), 2: (0, 200, 0)}
GRAY = (160, 160, 160)

FOOT_LABELS = {"player", "goalkeeper", "referee"}
ELLIPSE_START = 320
ELLIPSE_END = 580


def open_cap(sample_dir):
    cap = cv2.VideoCapture(sample_dir)
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open {sample_dir}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, fps, w, h, frame_count


def box_color(team, label):
    if team >= 0:
        return TEAM_COLORS.get(team, GRAY)
    if label == "referee":
        return (0, 255, 255)
    if label == "goalkeeper":
        return (255, 0, 255)
    return GRAY


def draw_foot_ellipse(frame, x1, x2, y2, color):
    cx = int((x1 + x2) / 2)
    cy = int(y2)
    ax = max(int((x2 - x1) * 0.65), 8)
    ay = max(int(ax * 0.3), 3)

    cv2.ellipse(frame, (cx, cy), (ax, ay), 0, ELLIPSE_START, ELLIPSE_END, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (ax, ay), 0, ELLIPSE_START, ELLIPSE_END, color, 1, cv2.LINE_AA)


def draw_tag(frame, text, x_center, y_top, color):
    fh, fw = frame.shape[:2]
    scale, th, pad = 0.5, 1, 5
    (tw, tht), _ = cv2.getTextSize(text, FONT, scale, th)
    bw = tw + 2 * pad
    bh = tht + 2 * pad

    x1 = int(x_center - bw / 2)
    x1 = max(0, min(x1, fw - bw))
    y1 = max(0, int(y_top - 8 - bh))
    x2 = x1 + bw
    y2 = y1 + bh

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(frame, text, (x1 + pad, y2 - pad), FONT, scale, (0, 0, 0), th, cv2.LINE_AA)


def to_pitch(H, pts):
    """Map image coordinates to pitch coordinates (meters) using homography.csv."""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    den = np.where(np.abs(p[:, 2:3]) < 1e-9, 1e-9, p[:, 2:3])
    return p[:, :2] / den


def inside(x, y, m=PITCH_MARGIN):
    return (x >= -m) & (x <= PITCH_W + m) & (y >= -m) & (y <= PITCH_H + m)


def pitch_polylines():
    lines = [
        np.array([[0, 0], [PITCH_W, 0], [PITCH_W, PITCH_H], [0, PITCH_H], [0, 0]], dtype=float),
        np.array([[PITCH_W / 2, 0], [PITCH_W / 2, PITCH_H]], dtype=float),
    ]
    t = np.linspace(0, 2 * np.pi, 73)
    lines.append(np.stack([PITCH_W / 2 + 9.15 * np.cos(t), PITCH_H / 2 + 9.15 * np.sin(t)], axis=1))
    for x0, sgn in ((0.0, 1.0), (PITCH_W, -1.0)):
        lines.append(np.array([[x0, 13.84], [x0 + sgn * 16.5, 13.84], [x0 + sgn * 16.5, 54.16], [x0, 54.16]], dtype=float))
        lines.append(np.array([[x0, 24.84], [x0 + sgn * 5.5, 24.84], [x0 + sgn * 5.5, 43.16], [x0, 43.16]], dtype=float))
    a = np.arccos(5.5 / 9.15)
    t = np.linspace(-a, a, 25)
    lines.append(np.stack([11 + 9.15 * np.cos(t), 34 + 9.15 * np.sin(t)], axis=1))
    lines.append(np.stack([94 - 9.15 * np.cos(t), 34 + 9.15 * np.sin(t)], axis=1))
    return lines


class Memory:
    """Keep each player's last position and estimate it while the player is out of view.

    Fill each team up to team_size with the most recently seen players.
    """

    def __init__(self, fps, team_size, memory_s):
        self.fps = fps
        self.team_size = team_size
        self.max_age = memory_s * fps
        self.last = {}

    def update(self, idx, players):
        for tid, label, team, x, y in players:
            if team < 0:
                continue
            r = self.last.get(tid)
            vx = vy = 0.0
            if r is not None and 0 < idx - r["frame"] <= MAX_GAP:
                dt = (idx - r["frame"]) / self.fps
                vx = (1 - VEL_ALPHA) * r["vx"] + VEL_ALPHA * (x - r["x"]) / dt
                vy = (1 - VEL_ALPHA) * r["vy"] + VEL_ALPHA * (y - r["y"]) / dt
                sp = np.hypot(vx, vy)
                if sp > MAX_MS:
                    vx, vy = vx * MAX_MS / sp, vy * MAX_MS / sp
            self.last[tid] = dict(frame=idx, x=x, y=y, vx=vx, vy=vy, team=team, label=label)

    def ghosts(self, idx, players):
        seen = {p[0] for p in players}
        vis = {}
        for p in players:
            if p[2] >= 0:
                vis[p[2]] = vis.get(p[2], 0) + 1
        out = []
        for team in {r["team"] for r in self.last.values()}:
            missing = self.team_size - vis.get(team, 0)
            if missing <= 0:
                continue
            cand = sorted(
                ((idx - r["frame"], tid, r) for tid, r in self.last.items()
                 if r["team"] == team and tid not in seen and 0 < idx - r["frame"] <= self.max_age),
                key=lambda c: c[0],
            )
            for age, tid, r in cand[:missing]:
                t = age / self.fps
                d = MEMORY_TAU * (1 - np.exp(-t / MEMORY_TAU))
                x = float(np.clip(r["x"] + r["vx"] * d, 0, PITCH_W))
                y = float(np.clip(r["y"] + r["vy"] * d, 0, PITCH_H))
                out.append((tid, r["label"], team, x, y))
        return out


class MiniMap:
    def __init__(self, width):
        self.s = width / (PITCH_W + 2 * MAP_PAD)
        self.w = int(width)
        self.h = int(round((PITCH_H + 2 * MAP_PAD) * self.s))
        self.bg = np.full((self.h, self.w, 3), MAP_BG, np.uint8)
        self.polys = [self.px(p) for p in pitch_polylines()]

        self.x0 = self.y0 = int(round(MAP_PAD * self.s))
        self.x1 = self.x0 + int(round(PITCH_W * self.s))
        self.y1 = self.y0 + int(round(PITCH_H * self.s))
        gw = max(1, (self.x1 - self.x0) // VORONOI_STEP)
        gh = max(1, (self.y1 - self.y0) // VORONOI_STEP)
        self.gx, self.gy = np.meshgrid(
            (np.arange(gw) + 0.5) / gw * PITCH_W,
            PITCH_H - (np.arange(gh) + 0.5) / gh * PITCH_H,
        )

    def px(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        return np.c_[(pts[:, 0] + MAP_PAD) * self.s, (PITCH_H - pts[:, 1] + MAP_PAD) * self.s].round().astype(np.int32)

    def voronoi(self, img, pts, teams):
        ids = np.unique(teams)
        if len(ids) < 2:
            return {}
        d = (self.gx[..., None] - pts[:, 0]) ** 2 + (self.gy[..., None] - pts[:, 1]) ** 2
        own = teams[d.argmin(-1)]
        layer = np.zeros((*own.shape, 3), np.uint8)
        for t in ids:
            layer[own == t] = TEAM_COLORS.get(int(t), GRAY)
        layer = cv2.resize(layer, (self.x1 - self.x0, self.y1 - self.y0), interpolation=cv2.INTER_NEAREST)
        roi = img[self.y0:self.y1, self.x0:self.x1]
        img[self.y0:self.y1, self.x0:self.x1] = cv2.addWeighted(layer, VORONOI_ALPHA, roi, 1 - VORONOI_ALPHA, 0)
        return {int(t): float((own == t).mean()) for t in ids}

    def draw(self, players, ball, owner, voronoi, trail=None, color=GREEN, ghosts=None):
        """Draw players as (track_id, label, team, x, y) in meters; ball is (x, y) or None."""
        img = self.bg.copy()
        share = {}
        vp = [p for p in players + (ghosts or []) if p[2] >= 0]
        if voronoi and vp:
            share = self.voronoi(img, np.array([[p[3], p[4]] for p in vp]), np.array([p[2] for p in vp]))
        for poly in self.polys:
            cv2.polylines(img, [poly], False, MAP_LINE, 1, cv2.LINE_AA)

        for tid, label, team, x, y in players:
            c = tuple(int(v) for v in self.px([[x, y]])[0])
            cv2.circle(img, c, 5, box_color(team, label), -1, cv2.LINE_AA)
            cv2.circle(img, c, 5, (0, 0, 0), 1, cv2.LINE_AA)
            if tid == owner:
                cv2.circle(img, c, 8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, str(tid), (c[0] + 6, c[1] - 6), FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        for tid, label, team, x, y in ghosts or []:
            c = tuple(int(v) for v in self.px([[x, y]])[0])
            cv2.circle(img, c, 5, box_color(team, label), 1, cv2.LINE_AA)

        if trail is not None and len(trail) > 1:
            cv2.polylines(img, [self.px(trail)], False, color, 1, cv2.LINE_AA)

        if ball is not None:
            c = tuple(int(v) for v in self.px([ball])[0])
            cv2.circle(img, c, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, c, 4, (0, 0, 0), 1, cv2.LINE_AA)

        for i, (t, p) in enumerate(sorted(share.items())):
            cv2.putText(img, f"voronoi {t}: {p:.0%}", (6, 14 + 14 * i), FONT, 0.45, TEAM_COLORS.get(t, GRAY), 1, cv2.LINE_AA)
        return img


def compute_stats(df, fps):
    """Add per-track speed (km/h) and cumulative distance (m) from pitch positions."""
    win = max(3, int(round(fps * SMOOTH_S)))
    parts = []
    for _, g in df.dropna(subset=["px"]).groupby("track_id"):
        g = g.copy()
        sm = g[["px", "py"]].rolling(win, center=True, min_periods=1).mean()
        gap = g["frame"].diff()
        step = np.hypot(sm["px"].diff(), sm["py"].diff())
        v = step / (gap / fps)
        bad = (gap > MAX_GAP) | (v * 3.6 > MAX_KMH)
        g["speed"] = v.where(~bad).rolling(win, min_periods=1).mean() * 3.6
        g["dist"] = step.where(~bad, 0.0).fillna(0.0).cumsum()
        parts.append(g[["speed", "dist"]])

    if not parts:
        return df.assign(speed=np.nan, dist=0.0)
    out = df.join(pd.concat(parts))
    out["dist"] = out.groupby("track_id")["dist"].ffill().fillna(0.0)
    return out


def ball_owner(df, ball_pitch):
    """Map each frame to the nearest player within POSS_DIST of the ball."""
    owner = {}
    cand = df[(df["track_label"] != "referee") & df["px"].notna()]
    for f, g in cand.groupby("frame"):
        b = ball_pitch.get(int(f))
        if b is None:
            continue
        d = np.hypot(g["px"].to_numpy() - b[0], g["py"].to_numpy() - b[1])
        i = int(d.argmin())
        if d[i] < POSS_DIST:
            owner[int(f)] = int(g["track_id"].iloc[i])
    return owner


def assign_goalkeepers(df):
    """Assign each goalkeeper to the team whose average pitch position is closest.

    Goalkeepers are excluded from team clustering and usually stay in their own half.
    """
    pl = df[(df["team"] >= 0) & (df["track_label"] == "player") & df["px"].notna()]
    mean_x = pl.groupby("team")["px"].mean()
    if len(mean_x) < 2:
        return df, 0
    gk = df[(df["team"] < 0) & (df["track_label"] == "goalkeeper") & df["px"].notna()]
    for tid, g in gk.groupby("track_id"):
        df.loc[df["track_id"] == tid, "team"] = int((mean_x - g["px"].mean()).abs().idxmin())
    return df, gk["track_id"].nunique()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--teams", default=None)
    parser.add_argument("--homography", required=True)
    parser.add_argument("--ball", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--map-width", type=int, default=None)
    parser.add_argument("--voronoi", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--ball-trail", action="store_true")
    parser.add_argument("--trail-len", type=int, default=TRAIL)
    parser.add_argument("--team-size", type=int, default=11)
    parser.add_argument("--no-gk-team", action="store_true")
    parser.add_argument("--memory-s", type=float, default=8.0)
    parser.add_argument("--out", default="out")

    args = parser.parse_args()

    out_dir = resolve_output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.tracks).sort_values(["frame", "track_id"]).reset_index(drop=True)
    if "track_label" not in df.columns:
        df["track_label"] = df.groupby("track_id")["label"].transform(lambda s: s.mode().iloc[0])
    if args.teams:
        tm = pd.read_csv(args.teams)[["track_id", "team"]]
        df = df.drop(columns=[c for c in ("team", "team_conf") if c in df.columns]).merge(tm, on="track_id", how="left")
    if "team" not in df.columns:
        print("Warning: no team assignments found. Pass --teams teams.csv or use tracks_teams.csv; team colors and Voronoi areas will be unavailable.")
        df["team"] = -1
    df["team"] = df["team"].fillna(-1).astype(int)

    hom = pd.read_csv(args.homography)
    hcols = [f"h{i}" for i in range(9)]
    hom = hom.dropna(subset=hcols)
    Hs = {int(f): v.reshape(3, 3) for f, v in zip(hom["frame"], hom[hcols].to_numpy())}
    if not Hs:
        raise RuntimeError("homography.csv contains no valid homographies")

    ball_img = {}
    ball_interp = set()
    if args.ball:
        b = pd.read_csv(args.ball).dropna(subset=["x", "y"])
        ball_img = {int(f): (float(x), float(y)) for f, x, y in zip(b["frame"], b["x"], b["y"])}
        if "interpolated" in b.columns:
            ball_interp = set(b.loc[b["interpolated"].astype(bool), "frame"].astype(int))
    ball_pitch = {}
    for f, xy in ball_img.items():
        H = Hs.get(f)
        if H is None:
            continue
        p = to_pitch(H, np.array([xy]))[0]
        if inside(p[0], p[1]):
            ball_pitch[f] = (float(p[0]), float(p[1]))

    cap, fps, w, h, frame_count = open_cap(args.sample)
    start = max(args.start_frame, int(df["frame"].min()))
    if frame_count > 0 and start >= frame_count:
        raise RuntimeError(f"--start-frame {start} is out of range ({frame_count} frames)")
    end = int(df["frame"].max())
    if frame_count > 0:
        end = min(end, frame_count - 1)
    if args.frames is not None:
        end = min(end, start + args.frames - 1)
    if start > end:
        raise RuntimeError(f"Invalid frame range: {start}-{end}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    df = df[(df["frame"] >= start) & (df["frame"] <= end)].reset_index(drop=True)
    foot = np.c_[(df["x1"] + df["x2"]) / 2, df["y2"]]
    px = np.full(len(df), np.nan)
    py = np.full(len(df), np.nan)
    for f, idx in df.groupby("frame").indices.items():
        H = Hs.get(int(f))
        if H is None:
            continue
        p = to_pitch(H, foot[idx])
        px[idx], py[idx] = p[:, 0], p[:, 1]
    ok = inside(px, py)
    df["px"] = np.where(ok, px, np.nan)
    df["py"] = np.where(ok, py, np.nan)

    n_gk = 0
    if not args.no_gk_team:
        df, n_gk = assign_goalkeepers(df)

    owner = {}
    if args.stats:
        df = compute_stats(df, fps)
        owner = ball_owner(df, ball_pitch)

    team_of = df.drop_duplicates("track_id").set_index("track_id")["team"].to_dict()
    groups = {f: g for f, g in df.groupby("frame")}

    mm = MiniMap(args.map_width or int(0.35 * w))
    ox = (w - mm.w) // 2
    oy = max(h - mm.h - 10, 0)

    writer = cv2.VideoWriter(str(out_dir / "overview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    poss = {}
    trail = deque(maxlen=args.trail_len)
    trail_map = deque(maxlen=args.trail_len)
    memory = Memory(fps, args.team_size, args.memory_s) if args.voronoi and args.memory_s > 0 else None
    n_h = 0
    n = 0

    for idx in range(start, end + 1):
        okr, frame = cap.read()
        if not okr:
            break
        n += 1

        g = groups.get(idx)
        H = Hs.get(idx)
        n_h += H is not None
        players = []

        if g is not None:
            for r in g.itertuples():
                color = box_color(r.team, r.track_label)
                text = str(r.track_id)
                if args.stats and not np.isnan(r.px):
                    text += f" {r.dist:.0f}m"
                    if not np.isnan(r.speed):
                        text += f" {r.speed:.1f}km/h"
                if r.track_label in FOOT_LABELS:
                    draw_foot_ellipse(frame, r.x1, r.x2, r.y2, color)
                    draw_tag(frame, text, (r.x1 + r.x2) / 2, r.y1, color)
                else:
                    p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
                    cv2.rectangle(frame, p1, p2, color, 2)
                    cv2.putText(frame, text, (p1[0], max(p1[1] - 5, 12)), FONT, 0.5, color, 2)
                if not np.isnan(r.px):
                    players.append((int(r.track_id), r.track_label, int(r.team), float(r.px), float(r.py)))

        ball_color = ORANGE if idx in ball_interp else GREEN
        if idx in ball_img:
            center = (int(ball_img[idx][0]), int(ball_img[idx][1]))
            if args.ball_trail:
                trail.append(center)
                if len(trail) > 1:
                    cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, ball_color, 2)
            cv2.circle(frame, center, 8, ball_color, 2)
        if idx in ball_pitch:
            trail_map.append(ball_pitch[idx])

        ghosts = []
        if memory is not None:
            memory.update(idx, players)
            ghosts = memory.ghosts(idx, players)

        own = owner.get(idx)
        if own is not None and team_of.get(own, -1) >= 0:
            poss[team_of[own]] = poss.get(team_of[own], 0) + 1

        if H is not None:
            small = mm.draw(players, ball_pitch.get(idx), own, args.voronoi, list(trail_map) if args.ball_trail else None, ball_color, ghosts)
            roi = frame[oy:oy + mm.h, ox:ox + mm.w]
            frame[oy:oy + mm.h, ox:ox + mm.w] = cv2.addWeighted(small, MAP_ALPHA, roi, 1 - MAP_ALPHA, 0)

        status = f"{idx}" + ("" if H is not None else "  no homography")
        cv2.putText(frame, status, (10, 24), FONT, 0.7, (255, 255, 255), 2)
        if args.stats and sum(poss.values()) > 0:
            tot = sum(poss.values())
            txt = "  ".join(f"{t}: {c / tot:.0%}" for t, c in sorted(poss.items()))
            cv2.putText(frame, f"possession  {txt}", (10, 48), FONT, 0.6, (255, 255, 255), 2)
        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Rendered frames: {n} ({start}-{start + n - 1})")
    print(f"Frames with minimap: {n_h} ({100 * n_h / max(n, 1):.1f}%)")
    print(f"Track rows projected onto the pitch: {int(df['px'].notna().sum())} of {len(df)}")
    print(f"Goalkeepers assigned to a team: {n_gk}")

    if args.stats:
        n_pos = df.dropna(subset=["px"]).groupby("track_id").size()
        summ = df.groupby("track_id").agg(
            track_label=("track_label", "first"),
            team=("team", "first"),
            frames=("frame", "size"),
            distance_m=("dist", "max"),
            max_speed_kmh=("speed", "max"),
        )
        summ["avg_speed_kmh"] = summ["distance_m"] / (n_pos.reindex(summ.index) / fps) * 3.6
        counts = pd.Series(list(owner.values()), dtype=int).value_counts()
        summ["possession_s"] = counts.reindex(summ.index).fillna(0) / fps
        summ.round(2).to_csv(out_dir / "stats.csv")
        print(summ.sort_values("distance_m", ascending=False).head(5).round(1).to_string())
        if poss:
            tot = sum(poss.values())
            print("Possession by team: " + ", ".join(f"{t}: {c / tot:.1%}" for t, c in sorted(poss.items())))


if __name__ == "__main__":
    main()