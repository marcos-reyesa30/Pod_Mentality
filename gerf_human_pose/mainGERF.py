from ultralytics import YOLO
import torch

model = YOLO("custom/yolo11-gerfpose.yaml")

checkPointFolder = 'yolo11n_gerfpose_cpu'
device = 'cpu'
ampOn = False
if torch.cuda.is_available():
    checkPointFolder = 'yolo11n_gerfpose_gpu'
    device = 'cuda'
    ampOn = True
    print("Training on GPU")
else:
    print("Training on CPU")

model.train(
    data="coco-pose.yaml",  # COCO pose dataset
    epochs=20,
    imgsz=640,
    batch=16,               # Adjust based on GPU
    device=device,
    amp=ampOn,
    workers=4,
    fraction=0.1,
    
    # Pose-specific settings
    pose=True,
    kobj=1.0,               # Keypoint objectness weight
    
    # Standard settings
    patience=50,
    save_period=5,
    project="runs/pose",
    name=checkPointFolder,
    exist_ok=True,
    verbose=True
)