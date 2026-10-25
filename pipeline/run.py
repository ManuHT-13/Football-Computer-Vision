"""Run selected football-video processing stages in order, passing each stage the files
produced by the previous ones. Assemble command-line options for detection, pitch
calibration, tracking, team assignment, ball tracking and rendering, with support for
per-stage arguments, skipping existing outputs and optional visualizations. Missing
scripts or failed stages stop the run; timings and expected output files are reported
at the end."""

from pathlib import Path

import sys
import time
import shlex
import argparse
import subprocess

from pipeline_paths import resolve_output_dir

STAGES = ["detect", "homography", "track", "teams", "ball", "render"]

# Script for each stage
SCRIPTS = {
    "detect": "detect.py",
    "homography": "homography.py",
    "track": "track.py",
    "teams": "teams.py",
    "ball": "ball.py",
    "render": "render.py",
}

# Necessary outputs
OUTPUTS = {
    "detect": "detections.csv",
    "homography": "homography.csv",
    "track": "tracks.csv",
    "teams": "teams.csv",
    "ball": "ball.csv",
    "render": "overview.mp4",
}


def build(stage, args, out):
    s, o = str(args.sample), str(out)
    rng = ["--start-frame", str(args.start_frame)]
    if args.frames is not None:
        rng += ["--frames", str(args.frames)]
    dbg = ["--render"] if args.render_stages else []

    def f(name):
        return str(out / name)

    hom = ["--homography", f("homography.csv")] if (out / OUTPUTS["homography"]).exists() else []

    if stage == "detect":
        return ["--model", args.model, "--sample", s, *rng, "--out", o, *dbg]
    if stage == "homography":
        return ["--sample", s, "--weights", args.weights, "--tracks", f("detections.csv"), *rng, "--out", o, *dbg]
    if stage == "track":
        return ["--detections", f("detections.csv"), "--sample", s, *hom, "--out", o, *dbg]
    if stage == "teams":
        return ["--tracks", f("tracks.csv"), "--sample", s, "--k", str(args.k), "--out", o, *dbg]
    if stage == "ball":
        return ["--detections", f("detections.csv"), "--sample", s, *hom, "--out", o, *dbg]

    cmd = ["--sample", s, "--tracks", f("tracks.csv"), "--homography", f("homography.csv"), *rng, "--out", o]
    if (out / OUTPUTS["teams"]).exists():
        cmd += ["--teams", f("teams.csv")]
    if (out / OUTPUTS["ball"]).exists():
        cmd += ["--ball", f("ball.csv")]
    for flag, on in (("--voronoi", args.voronoi), ("--stats", args.stats), ("--ball-trail", args.ball_trail)):
        if on:
            cmd.append(flag)
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--render-stages", action="store_true")
    parser.add_argument("--voronoi", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--ball-trail", action="store_true")
    parser.add_argument("--scripts-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--out", default="out")

    for stage in STAGES:
        parser.add_argument(f"--{stage}-args", default="")

    args = parser.parse_args()

    out = resolve_output_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(args.scripts_dir)
    stages = [s for s in STAGES if s in args.stages]

    missing = [SCRIPTS[s] for s in stages if not (scripts_dir / SCRIPTS[s]).exists()]
    if missing:
        raise RuntimeError(f"Missing scripts in {scripts_dir}: {', '.join(missing)}")
    if "detect" in stages and not args.model:
        raise RuntimeError("The detect stage requires --model")
    if "homography" in stages and not args.weights:
        raise RuntimeError("The homography stage requires --weights")

    t_all = time.time()
    for stage in stages:
        target = out / OUTPUTS[stage]
        if args.skip_existing and target.exists():
            print(f"[{stage}] {target.name} already exists, skipping")
            continue

        cmd = [sys.executable, str(scripts_dir / SCRIPTS[stage]), *build(stage, args, out), *shlex.split(getattr(args, f"{stage}_args"))]
        print(f"\n[{stage}] {' '.join(shlex.quote(c) for c in cmd)}")
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(f"Stage '{stage}' failed (exit code {e.returncode})")
        print(f"[{stage}] finished in {time.time() - t0:.1f} s")

    print(f"\nPipeline finished in {time.time() - t_all:.1f} s -> {out}")
    for stage in stages:
        mark = "ok" if (out / OUTPUTS[stage]).exists() else "missing"
        print(f"  {OUTPUTS[stage]}: {mark}")


if __name__ == "__main__":
    main()