from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--sample", required=True)
parser.add_argument("--dest", default="results")
parser.add_argument("--name", default="predict")

args = parser.parse_args()

model = YOLO(args.model)

result = model.predict(source=args.sample, save=True, project=args.dest, name=args.name)