from ultralytics import YOLO

model = YOLO("kaggle/runs/detect/soccernet_finetune/yolo11s_stride3_960/weights/best.pt") 
metrics = model.val(data="datasets/gamestate-2024-yolo/data_stride3.yaml", split="test", workers=0)
print(metrics.box.maps)  