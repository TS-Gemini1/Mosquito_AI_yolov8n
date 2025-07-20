from ultralytics import YOLO
import torch
from pathlib import Path

def create_data_yaml():
    """Create data.yaml for D1.5 dataset with 2 classes"""
    yaml_content = """# D1.5 Mosquito Egg Dataset - 2 Classes
path: D:/AI/datasets/D1.5
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: small_egg    # < 3px width or < 7px height
  1: large_egg    # >= 4px width and >= 8px height
"""
    
    yaml_path = Path("D:/AI/datasets/D1.5/data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path

def convert_labels_to_two_classes():
    """Convert existing single-class labels to two-class based on size"""
    dataset_path = Path("D:/AI/datasets/D1.5")
    
    print("🔄 Converting labels to 2-class system...")
    
    total_converted = 0
    small_eggs = 0
    large_eggs = 0
    
    for split in ['train', 'val', 'test']:
        labels_dir = dataset_path / 'labels' / split
        if not labels_dir.exists():
            continue
            
        for label_file in labels_dir.glob('*.txt'):
            new_lines = []
            
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = parts
                        
                        # Convert normalized coordinates to pixels (assuming 640x640 tiles)
                        width_px = float(width) * 640
                        height_px = float(height) * 640
                        
                        # Apply your size classification logic
                        if width_px < 3 or height_px < 7:
                            new_class_id = 0  # small_egg
                            small_eggs += 1
                        else:
                            new_class_id = 1  # large_egg
                            large_eggs += 1
                        
                        new_lines.append(f"{new_class_id} {x_center} {y_center} {width} {height}")
                        total_converted += 1
            
            # Write back the converted labels
            with open(label_file, 'w') as f:
                for line in new_lines:
                    f.write(line + '\n')
    
    print(f" Converted {total_converted} annotations:")
    print(f"   Small eggs (class 0): {small_eggs}")
    print(f"   Large eggs (class 1): {large_eggs}")
    print(f"   Ratio: {small_eggs/(small_eggs+large_eggs)*100:.1f}% small, {large_eggs/(small_eggs+large_eggs)*100:.1f}% large")
    
    return small_eggs, large_eggs

def train_yolo():
    """Simple training with 2-class system"""
    
    # Convert labels to 2-class system
    small_count, large_count = convert_labels_to_two_classes()
    
    # Create data.yaml
    data_yaml = create_data_yaml()
    
    # Check CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    print(" Starting 2-class training (small vs large eggs)...")
    
    # Training with lessons learned + 2-class system
    results = model.train(
        data=str(data_yaml),
        
        # Basic settings
        epochs=250,           # Optimal based on your graphs
        batch=4,              # RTX 4060 Ti safe batch size
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',
        
        # IoU fix for close targets
        iou=0.6,              # Higher to prevent deleting close eggs
        conf=0.001,
        
        # Conservative augmentation for tiny objects
        mosaic=0.1,
        mixup=0.0,
        scale=0.1,
        degrees=10,
        
        # Loss weights for tiny objects
        box=7.5,
        cls=1.0,              # Slightly higher for 2-class classification
        dfl=1.5,
        
        # Optimizer
        lr0=0.01,
        lrf=0.01,
        
        # Memory settings
        cache=False,
        amp=True,
        workers=2,
        patience=30,
        
        # Output
        project='D:/AI/runs',
        name='mosquito_eggs_2class',  # Different name for 2-class
        plots=True,
    )
    
    # Final validation
    val_results = model.val(
        data=str(data_yaml),
        conf=0.20,
        iou=0.6,
        max_det=300,
        imgsz=640,
    )
    
    # Print results for both classes
    if hasattr(val_results, 'box') and val_results.box is not None:
        metrics = val_results.box
        print(f"\n FINAL RESULTS (2-CLASS SYSTEM):")
        print(f"Overall mAP50: {metrics.map50:.3f}")
        print(f"Overall mAP50-95: {metrics.map:.3f}")
        print(f"Overall Precision: {metrics.mp:.3f}")
        print(f"Overall Recall: {metrics.mr:.3f}")
        
        # Per-class metrics if available
        if hasattr(metrics, 'ap_class_index') and len(metrics.ap_class_index) == 2:
            print(f"\n PER-CLASS BREAKDOWN:")
            print(f"Small eggs (class 0): {small_count} annotations")
            print(f"Large eggs (class 1): {large_count} annotations")
    
    print(f"\n 2-class training complete!")
    print(f"Results: D:/AI/runs/detect/mosquito_eggs_2class/")
    
    return results

if __name__ == "__main__":
    # Check dataset exists
    dataset_path = Path("D:/AI/datasets/D1.5")
    if not dataset_path.exists():
        print(f" Dataset not found: {dataset_path}")
        exit(1)
    
    print(" Mosquito Egg Training - D1.5 Dataset (2-Class System)")
    print("=" * 55)
    print(" Classes: 0=small_egg (<4x7px), 1=large_egg (≥4x7px)")
    print()
    
    # Start training
    train_yolo()