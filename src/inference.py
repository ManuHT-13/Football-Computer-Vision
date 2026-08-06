from ultralytics import YOLO

model = YOLO("best.pt")

result = model.predict(source="datasets/video.mp4", save=True)