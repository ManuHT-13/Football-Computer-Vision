"""Run a YOLO model over a selected range of a football video and save each detection's
frame, class, bounding box and confidence to detections.csv. Optional video rendering
highlights people with foot-level ellipses and labels, and draws boxes for other classes.
Confidence thresholds and image size can be adjusted from the command line."""

from pathlib import Path

from ultralytics import YOLO
import cv2
import argparse
import pandas as pd

from pipeline_paths import resolve_output_dir

COLORS = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 165, 255), (255, 0, 0)]

ELLIPSE_COLORS = {
    "player": (255, 180, 0),
    "goalkeeper": (0, 220, 255),
    "referee": (40, 140, 255),
}


def open_cap(sample_dir):
    cap = cv2.VideoCapture(sample_dir)
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open {sample_dir}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, fps, w, h, frame_count


def draw_foot_ellipse(frame, x1, x2, y2, color):
    cx = int((x1 + x2) / 2)
    cy = int(y2)
    ax = max(int((x2 - x1) * 0.65), 8)
    ay = max(int(ax * 0.3), 3)

    start, end = 320, 580
    cv2.ellipse(frame, (cx, cy), (ax, ay), 0, start, end, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (ax, ay), 0, start, end, color, 1, cv2.LINE_AA)


def draw_tag(frame, text, x_center, y_top, color):
    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, th, pad = 0.5, 1, 5
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    bw = tw + 2 * pad
    bh = tht + 2 * pad

    x1 = int(x_center - bw / 2)
    x1 = max(0, min(x1, fw - bw))
    y1 = max(0, int(y_top - 8 - bh))
    x2 = x1 + bw
    y2 = y1 + bh

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(frame, text, (x1 + pad, y2 - pad), font, scale, (0, 0, 0), th, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--out", default="out")
    parser.add_argument("--draw-conf", type=float, default=0.25)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--imgsz", type=int, default=None)

    args = parser.parse_args()

    out_dir = resolve_output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    class_names = model.names

    predict_kwargs = {"conf": args.conf, "verbose": False}
    if args.imgsz is not None:
        predict_kwargs["imgsz"] = args.imgsz

    cap, fps, w, h, frame_count = open_cap(args.sample)
    if frame_count > 0 and args.start_frame >= frame_count:
        raise RuntimeError(f"--start-frame {args.start_frame} is out of range ({frame_count} frames)")
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    available = frame_count - args.start_frame if frame_count > 0 else args.frames
    n_frames = min(args.frames, available)

    writer = None
    if args.render:
        writer = cv2.VideoWriter(
            str(out_dir / "sample_detections.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

    rows = []
    processed = 0
    for i in range(n_frames):
        idx = args.start_frame + i
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, **predict_kwargs)[0]
        boxes = res.boxes
        for (x1, y1, x2, y2), conf, cls in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
        ):
            label = class_names[int(cls)]
            rows.append((idx, label, x1, y1, x2, y2, conf))

            if args.render and conf >= args.draw_conf:
                key = label.lower()
                if key in ELLIPSE_COLORS:
                    color = ELLIPSE_COLORS[key]
                    draw_foot_ellipse(frame, x1, x2, y2, color)
                    draw_tag(frame, f"{label.capitalize()} {conf:.2f}", (x1 + x2) / 2, y1, color)
                else:
                    color = COLORS[int(cls) % len(COLORS)]
                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    cv2.rectangle(frame, p1, p2, color, 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (p1[0], max(p1[1] - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

        if writer is not None:
            writer.write(frame)
        processed += 1

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows, columns=["frame", "label", "x1", "y1", "x2", "y2", "conf"])
    df.to_csv(out_dir / "detections.csv", index=False)

    print(f"Frames processed: {processed}")
    print(f"FPS: {fps:.3f}  resolution: {w}x{h}  image size: {args.imgsz or 'model default'}")
    print(df.groupby("label")["conf"].agg(["count", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()