from ultralytics import YOLO
import torch

model = YOLO("custom/yolo11_custom.yaml")

checkPointFolder = 'yolo11n_gerf_allcoco_cpu'
device = 'cpu'
ampOn = False
if torch.cuda.is_available():
    checkPointFolder = 'yolo11n_gerf_allcoco_gpu'
    device = 'cuda'
    ampOn = True
    print("Training on GPU")
else:
    print("Training on CPU")

model.train(
    data="coco.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
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
    cache='disk',
    verbose=True
)