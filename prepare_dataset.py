#!/usr/bin/env python3
"""
Prepare egg detection dataset for RT-DETR training
Converts YOLO format to COCO format and creates train/val splits
"""

import os
import json
import random
from pathlib import Path
from PIL import Image
import shutil
from typing import List, Dict, Tuple

def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Read YOLO format labels (class_id, x_center, y_center, width, height)"""
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append((class_id, x_center, y_center, width, height))
    return labels

def yolo_to_coco_bbox(yolo_bbox: Tuple[float, float, float, float], 
                      img_width: int, img_height: int) -> List[float]:
    """Convert YOLO bbox to COCO format [x, y, width, height]"""
    x_center, y_center, width, height = yolo_bbox
    
    # Convert from normalized to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Convert from center to top-left corner
    x = x_center_px - width_px / 2
    y = y_center_px - height_px / 2
    
    return [x, y, width_px, height_px]

def create_coco_dataset(image_dir: Path, label_dir: Path, 
                       split_ratio: float = 0.8, seed: int = 42) -> Tuple[Dict, Dict]:
    """Create COCO format datasets for train and validation"""
    
    random.seed(seed)
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")
    
    # Categories for egg detection (COCO format needs id starting from 1)
    categories = [
        {"id": 1, "name": "egg", "supercategory": "object"}
    ]
    
    def create_coco_annotations(file_list: List[str], split_name: str) -> Dict:
        """Create COCO format annotations for a split"""
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        annotation_id = 0
        
        for idx, img_file in enumerate(file_list):
            img_path = image_dir / img_file
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            label_path = label_dir / label_file
            
            # Get image dimensions
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                continue
            
            # Add image info
            image_info = {
                "id": idx,
                "file_name": img_file,
                "width": img_width,
                "height": img_height
            }
            coco_data["images"].append(image_info)
            
            # Read and convert labels if they exist
            if label_path.exists():
                yolo_labels = read_yolo_labels(label_path)
                
                for class_id, x_center, y_center, width, height in yolo_labels:
                    # Convert YOLO to COCO bbox
                    coco_bbox = yolo_to_coco_bbox(
                        (x_center, y_center, width, height),
                        img_width, img_height
                    )
                    
                    # Calculate area
                    area = coco_bbox[2] * coco_bbox[3]
                    
                    # Add annotation (convert class_id 0 to 1 for COCO format)
                    annotation = {
                        "id": annotation_id,
                        "image_id": idx,
                        "category_id": class_id + 1,  # Convert 0 to 1
                        "bbox": coco_bbox,
                        "area": area,
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(file_list)} {split_name} images")
        
        return coco_data
    
    # Create annotations for train and val
    print("\nCreating train annotations...")
    train_coco = create_coco_annotations(train_files, "train")
    
    print("\nCreating val annotations...")
    val_coco = create_coco_annotations(val_files, "val")
    
    return train_coco, val_coco, train_files, val_files

def setup_dataset_structure(base_dir: Path, source_image_dir: Path,
                          train_files: List[str], val_files: List[str]):
    """Setup directory structure for RT-DETR training"""
    
    # Create directory structure (base_dir is already the dataset directory)
    dataset_dir = base_dir
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    annotations_dir = dataset_dir / "annotations"
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreated dataset structure at {dataset_dir}")
    
    # Copy images to respective directories
    print("\nCopying train images...")
    for idx, img_file in enumerate(train_files):
        src = source_image_dir / img_file
        dst = train_dir / img_file
        shutil.copy2(src, dst)
        if (idx + 1) % 100 == 0:
            print(f"Copied {idx + 1}/{len(train_files)} train images")
    
    print("\nCopying val images...")
    for idx, img_file in enumerate(val_files):
        src = source_image_dir / img_file
        dst = val_dir / img_file
        shutil.copy2(src, dst)
        if (idx + 1) % 100 == 0:
            print(f"Copied {idx + 1}/{len(val_files)} val images")
    
    return annotations_dir

def main():
    # Paths
    source_dir = Path("/active-1/Omlet_IP/formatted_egg_detection")
    image_dir = source_dir / "images"
    label_dir = source_dir / "labels"
    base_dir = Path("/active-1/Omlet_IP/coco_egg_dataset")
    
    # Create COCO format annotations
    train_coco, val_coco, train_files, val_files = create_coco_dataset(
        image_dir, label_dir, split_ratio=0.85
    )
    
    # Setup dataset structure
    annotations_dir = setup_dataset_structure(
        base_dir, image_dir, train_files, val_files
    )
    
    # Save COCO annotations
    train_ann_path = annotations_dir / "instances_train.json"
    val_ann_path = annotations_dir / "instances_val.json"
    
    with open(train_ann_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    print(f"\nSaved train annotations to {train_ann_path}")
    
    with open(val_ann_path, 'w') as f:
        json.dump(val_coco, f, indent=2)
    print(f"Saved val annotations to {val_ann_path}")
    
    # Print statistics
    print("\n" + "="*50)
    print("Dataset Preparation Complete!")
    print("="*50)
    print(f"Train: {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations")
    print(f"Val: {len(val_coco['images'])} images, {len(val_coco['annotations'])} annotations")
    print(f"Dataset location: {base_dir}")

if __name__ == "__main__":
    main()