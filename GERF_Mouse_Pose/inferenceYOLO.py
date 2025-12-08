from ultralytics import YOLO
import torch
from pathlib import Path

# ----------------------------------------------
# 1. Load the Yolo11 Model
# ----------------------------------------------
model_path = 'runs/pose/yolo11n_mouse_cpu/weights/best.pt'  # Path to your best.pt file
if torch.cuda.is_available():
    model_path = 'runs/pose/yolo11n_mouse_gpu/weights/best.pt'
model = YOLO(model_path)
print("Running Normal Yolo")

# ----------------------------------------------
# 2. Get First 50 JPG Images (Sorted)
# ----------------------------------------------
# Define the directory containing test images
test_dir = Path('datasets/mouse-pose/images/val')

# Get all .jpg images in the directory
all_images = list(test_dir.glob('*.jpg'))

# Sort the images by filename and take first 50
image_paths = sorted(all_images)[:50]

print(f"Found {len(all_images)} total images in {test_dir}")
print(f"Running inference on first {len(image_paths)} images")

# ----------------------------------------------
# 3. Run Inference on First 50 Images
# ----------------------------------------------
for i, image_path in enumerate(image_paths):
    print(f"Processing [{i+1}/{len(image_paths)}]: {image_path.name}")
    
    # Run inference on the image
    results = model.predict(image_path)

    # Check if results is a list (for multiple images) or a single result object
    if isinstance(results, list):
        for res in results:
            #res.show()  # Show the result
            res.save()  # Save the result
    else:
        #results.show()  # Show the result
        results.save()  # Save the result

# ----------------------------------------------
# 4. Process Completion
# ----------------------------------------------
print(f"Inference complete on {len(image_paths)} images. Results saved.")
