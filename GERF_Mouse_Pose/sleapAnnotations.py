import json
import numpy as np
from pathlib import Path

def coco_to_yolo_pose(coco_json_path, output_labels_dir):
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image_id to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    print(f"Converting {len(image_info)} images...")
    converted = 0
    
    # Process each image
    for image_id, img_data in image_info.items():
        img_width = img_data['width']
        img_height = img_data['height']
        img_filename = img_data['file_name']
        
        # Get label filename (same name as image but .txt)
        label_filename = Path(img_filename).stem + '.txt'
        label_path = output_labels_dir / label_filename
        
        # Get annotations for this image
        anns = annotations_by_image.get(image_id, [])
        
        if not anns:
            # Create empty label file
            label_path.touch()
            continue
        
        # Convert each annotation to YOLO format
        yolo_lines = []
        for ann in anns:
            # Class index (0 for mouse, since we only have 1 class)
            class_idx = 0  # ann['category_id'] - 1 if multiple classes
            
            # Bounding box (convert from [x, y, w, h] to normalized)
            bbox = ann['bbox']
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height
            
            # Keypoints (already in [x1, y1, v1, x2, y2, v2, ...] format)
            keypoints = ann['keypoints']
            
            # Normalize keypoints
            normalized_kpts = []
            for i in range(0, len(keypoints), 3):
                x = keypoints[i] / img_width if keypoints[i] > 0 else 0
                y = keypoints[i+1] / img_height if keypoints[i+1] > 0 else 0
                v = keypoints[i+2]  # Visibility (0, 1, or 2)
                normalized_kpts.extend([x, y, v])
            
            # Create YOLO format line
            # Format: class x_center y_center width height kpt1_x kpt1_y kpt1_v ...
            line_parts = [class_idx, x_center, y_center, width, height] + normalized_kpts
            yolo_line = ' '.join(map(str, line_parts))
            yolo_lines.append(yolo_line)
        
        # Write to label file
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        converted += 1
        if converted % 100 == 0:
            print(f"  Converted {converted}/{len(image_info)} images")
    
    print(f"Conversion complete!")
    print(f"   Converted {converted} label files")
    print(f"   Output: {output_labels_dir}")


# Convert training set
print("Converting training set...")
coco_to_yolo_pose(
    coco_json_path='datasets/mouse-pose/images/train/annotations.json',
    output_labels_dir='datasets/mouse-pose/labels/train'
)

# Convert validation set
print("\nConverting validation set...")
coco_to_yolo_pose(
    coco_json_path='datasets/mouse-pose/images/val/annotations.json',
    output_labels_dir='datasets/mouse-pose/labels/val'
)

print("\nDataset is now ready for YOLO training")