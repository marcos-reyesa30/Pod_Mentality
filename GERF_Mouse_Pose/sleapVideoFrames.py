import sleap_io as sio
import numpy as np
import json
from pathlib import Path
import cv2

def sleapToCoco(pkg_file, output_dir):    
    labels = sio.load_slp(pkg_file)
    
    output_dir = Path(output_dir)
    img_dir = output_dir
    img_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(labels)} labeled frames")
    print(f"Videos: {len(labels.videos)}")
    
    skeleton = labels.skeletons[0]
    keypoint_names = [node.name for node in skeleton.nodes]
    num_keypoints = len(keypoint_names)
    
    print(f"Skeleton: {num_keypoints} keypoints")
    print(f"Keypoints: {keypoint_names}")
    
    coco_data = {
        'info': {
            'description': 'SLEAP Dataset converted to COCO format',
            'version': '1.0',
        },
        'images': [],
        'annotations': [],
        'categories': [{
            'id': 1,
            'name': 'mouse',
            'keypoints': keypoint_names,
            'skeleton': []
        }]
    }
    
    annotation_id = 0
    image_id = 0
    
    # Process each labeled frame directly
    for lf in labels:
        try:
            # Get frame image directly from labeled frame
            frame = lf.image
            
            if frame is None:
                print(f"Skipping frame {lf.frame_idx} (no image)")
                continue
            
            # Ensure it's a numpy array
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            
            height, width = frame.shape[:2]
            
            # Save image
            img_filename = f"img_{image_id:08d}.jpg"
            img_path = img_dir / img_filename
            
            # Handle color conversion
            if len(frame.shape) == 2:
                # Grayscale
                cv2.imwrite(str(img_path), frame)
            elif frame.shape[2] == 3:
                # RGB to BGR for OpenCV
                cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(img_path), frame)
            
            # Add image entry
            coco_data['images'].append({
                'id': image_id,
                'file_name': img_filename,
                'width': width,
                'height': height,
                'frame_idx': lf.frame_idx,
            })
            
            # Process instances in this frame
            for instance in lf.instances:
                keypoints = []
                visible_points = []
                
                # Access points as structured arrays
                # point[0] = [x, y], point[1] = visible, point[2] = complete, point[3] = name
                for point in instance.points:
                    coords = point[0]  # [x, y]
                    is_visible = point[1]  # visibility boolean
                    
                    x, y = coords[0], coords[1]
                    
                    if not np.isnan(x) and not np.isnan(y) and is_visible:
                        x, y = float(x), float(y)
                        keypoints.extend([x, y, 2])  # 2 = visible
                        visible_points.append([x, y])
                    else:
                        keypoints.extend([0, 0, 0])  # 0 = not labeled
                
                # Skip instances with no visible points
                if not visible_points:
                    continue
                
                # Calculate bounding box from visible keypoints
                visible_points = np.array(visible_points)
                x_min, y_min = visible_points.min(axis=0)
                x_max, y_max = visible_points.max(axis=0)
                
                # Add padding
                padding = 10
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(width, x_max + padding)
                y_max = min(height, y_max + padding)
                
                bbox = [float(x_min), float(y_min), 
                       float(x_max - x_min), float(y_max - y_min)]
                area = bbox[2] * bbox[3]
                
                # Add annotation
                coco_data['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'keypoints': keypoints,
                    'num_keypoints': len(visible_points),
                    'bbox': bbox,
                    'area': float(area),
                    'iscrowd': 0,
                })
                
                annotation_id += 1
            
            image_id += 1
            
            if image_id % 100 == 0:
                print(f"  Processed {image_id} images, {annotation_id} annotations")
                
        except Exception as e:
            print(f"Error on frame {lf.frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save COCO JSON
    json_path = output_dir / "annotations.json"
    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"  Images: {image_id}")
    print(f"  Annotations: {annotation_id}")
    print(f"  Average annotations per image: {annotation_id/image_id if image_id > 0 else 0:.2f}")
    print(f"  Saved to: {output_dir}")
    print(f"  Keypoints: {num_keypoints}")
    print(f"  Names: {keypoint_names}")
    
    return coco_data, keypoint_names, num_keypoints

# Convert train sleap package
coco_data, kpt_names, num_kpts = sleapToCoco(
    "datasets/train.pkg.slp",
    "datasets/mouse-pose/images/train"
)

print(f"1. Create YAML config:")
print(f"   nc: 1")
print(f"   kpt_shape: [{num_kpts}, 3]")
print(f"   names: ['mouse']")
print(f"\n2. Keypoint names: {kpt_names}")

# Convert validate sleap package
coco_data, kpt_names, num_kpts = sleapToCoco(
    "datasets/val.pkg.slp",
    "datasets/mouse-pose/images/val"
)

print(f"1. Create YAML config:")
print(f"   nc: 1")
print(f"   kpt_shape: [{num_kpts}, 3]")
print(f"   names: ['mouse']")
print(f"\n2. Keypoint names: {kpt_names}")