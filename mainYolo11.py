from ultralytics import YOLO
import torch

model = YOLO("custom/yolo11.yaml")

checkPointFolder = 'yolo11n_cpu'
device = 'cpu'
ampOn = False
if torch.cuda.is_available():
    checkPointFolder = 'yolo11n_gpu'
    device = 'cuda'
    ampOn = True

model.train(
    data="coco128.yaml",
    epochs=300,
    imgsz=640,
    batch=4,
    device=device,
    workers=4,
    project="runs/detect",
    name=checkPointFolder,
    exist_ok=True,
    amp=ampOn,
    patience=50,
    save_period=50,
    plots=True,
    close_mosaic=10,
    cache=False,
    verbose=True
)