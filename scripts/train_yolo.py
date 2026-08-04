import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/gamestate-2024-yolo/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="gamestate-yolo11")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    model = YOLO(args.model)

    project_dir = str(Path("runs/detect").resolve())
    print(f"TensorBoard: tensorboard --logdir {project_dir}/{args.name}")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        name=args.name,
        resume=args.resume,
        workers=args.workers,
        project=project_dir,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        val=True,
        plots=True,
    )

    metrics = model.val(data=args.data, imgsz=args.imgsz, split="test")
    print(metrics.box.map)
    print(metrics.box.map50)


if __name__ == "__main__":
    main()