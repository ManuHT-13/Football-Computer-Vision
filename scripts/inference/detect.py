from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--sample", required=True)
parser.add_argument("--dest", default="results")
parser.add_argument("--name", default="predict")
parser.add_argument("--imgsz", type=int, default=960)
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--save_txt", action="store_true")

args = parser.parse_args()

model = YOLO(args.model)

results = model.predict(
    source=args.sample,
    imgsz=args.imgsz,
    conf=args.conf,
    save=True,
    save_txt=args.save_txt,
    project=args.dest,
    name=args.name,
    exist_ok=True,
)

