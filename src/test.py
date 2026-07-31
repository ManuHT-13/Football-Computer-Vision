from ultralytics import YOLO

model = YOLO("yolo11l.pt")

result = model.predict(source="data/gol_iniesta.mp4", save=True)