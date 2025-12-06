from ultralytics import YOLO
import torch

# ----------------------------------------------
# 1. Load the YOLOv5 Model
# ----------------------------------------------
model_path = 'runs/detect/yolo11n_cpu/weights/best.pt'  # Path to your best.pt file
if torch.cuda.is_available():
    model_path = 'runs/detect/yolo11n_gpu/weights/best.pt'
model = YOLO(model_path)

# ----------------------------------------------
# 2. Define the Input Images (Select Only 2-3 Images)
# ----------------------------------------------
# Update this with the actual paths to the images you want to use
image_paths = [
    'datasets/coco128/images/train2017/000000000384.jpg',  # Path to first image
    'datasets/coco128/images/train2017/000000000387.jpg',  # Path to second image
    'datasets/coco128/images/train2017/000000000389.jpg'   # Path to third image
]

# ----------------------------------------------
# 3. Run Inference on the Selected Images
# ----------------------------------------------
for image_path in image_paths:
    print(f"Processing: {image_path}")
    
    # Run inference on the image (predict method will return a list of results)
    results = model.predict(image_path)

    # Check if results is a list (for multiple images) or a single result object
    if isinstance(results, list):
        for res in results:  # Loop through the list and process each result
            #res.show()  # Show the result (image with bounding boxes)
            res.save()  # Optionally save the result
    else:
        #results.show()  # Show the result (image with bounding boxes)
        results.save()  # Optionally save the result

# ----------------------------------------------
# 4. Process Completion
# ----------------------------------------------
print("Inference complete. Results saved.")