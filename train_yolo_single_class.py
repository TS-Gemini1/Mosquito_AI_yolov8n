from ultralytics import YOLO
import torch
from pathlib import Path

def create_data_yaml():
    """Create data.yaml for D1.5 dataset with single class"""
    yaml_content = """# D1.5 Mosquito Egg Dataset - Single Class
path: /mnt/c/Users/theod/Documents/CODE/AI9/D1.5
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: egg    # All mosquito eggs regardless of size
"""
    
    yaml_path = Path("/mnt/c/Users/theod/Documents/CODE/AI9/D1.5/data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f" Created data.yaml at: {yaml_path}")
    return yaml_path

def convert_labels_to_single_class():
    """Convert any existing multi-class labels to single class (0)"""
    dataset_path = Path("/mnt/c/Users/theod/Documents/CODE/AI9/D1.5")
    
    print(" Converting all labels to single class (0 = egg)...")
    
    total_converted = 0
    
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
                        # Change any class_id to 0 (single class)
                        class_id, x_center, y_center, width, height = parts
                        new_lines.append(f"0 {x_center} {y_center} {width} {height}")
                        total_converted += 1
            
            # Write back the converted labels
            with open(label_file, 'w') as f:
                for line in new_lines:
                    f.write(line + '\n')
    
    print(f" Converted {total_converted} annotations to single class")
    return total_converted

def train_yolo():
    """Training with single class system"""
    
    # Convert labels to single class
    total_annotations = convert_labels_to_single_class()
    
    # Create data.yaml
    data_yaml = create_data_yaml()
    
    # Check CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    
    # Load model
    model = YOLO('yolov8n.pt')
    print(f"Model device: {model.device}")
    
    print(" Starting single-class training (all eggs as class 0)...")
    
    # Training with your proven settings
    results = model.train(
        data=str(data_yaml),
        
        # Basic settings
        epochs=300,           # Optimal based on your experience
        batch=4,              # RTX 4060 Ti safe batch size
        imgsz=640,            # Standard tile size
        device=0 if torch.cuda.is_available() else 'cpu',
        
        # IoU and confidence settings
        iou=0.6,              # Higher to prevent deleting close eggs
        conf=0.001,           # Very low for tiny objects
        
        # Conservative augmentation for tiny objects
        mosaic=0.1,           # Light mosaic
        mixup=0.0,            # No mixup for tiny objects
        scale=0.1,            # Minimal scaling
        degrees=10,           # Light rotation
        translate=0.1,        # Light translation
        
        # Loss weights optimized for tiny objects
        box=7.5,              # High box loss for localization
        cls=0.5,              # Standard classification (single class)
        dfl=1.5,              # Distribution focal loss
        
        # Optimizer settings
        lr0=0.01,             # Learning rate
        lrf=0.01,             # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        
        # Memory and performance
        cache=False,          # No caching for memory safety
        amp=True,             # Mixed precision
        workers=2,            # Conservative workers
        patience=30,          # Early stopping patience
        save_period=25,       # Save checkpoints
        val=True,
        plots=True,
        
        # Output
        project='/mnt/c/Users/theod/Documents/CODE/AI9/runs',
        name='d15_single_class_eggs',
    )
    
    print("Training complete!")
    print(f"Results saved to: /mnt/c/Users/theod/Documents/CODE/AI9/runs/detect/d15_single_class_eggs/")
    
    # Final validation
    print("🔍 Running final validation...")
    val_results = model.val(
        data=str(data_yaml),
        conf=0.20,            # Higher confidence for final evaluation
        iou=0.6,              # Same as training
        max_det=300,         # Allow many detections
        imgsz=640,
    )
    
    # Print results
    if hasattr(val_results, 'box') and val_results.box is not None:
        metrics = val_results.box
        print(f"\n FINAL RESULTS (SINGLE-CLASS SYSTEM):")
        print(f"mAP50: {metrics.map50:.3f}")
        print(f"mAP50-95: {metrics.map:.3f}")
        print(f"Precision: {metrics.mp:.3f}")
        print(f"Recall: {metrics.mr:.3f}")
        print(f"Total annotations: {total_annotations}")
        
    return results

if __name__ == "__main__":
    # Check dataset exists (WSL path)
    dataset_path = Path("/mnt/c/Users/theod/Documents/CODE/AI9/D1.5")
    if not dataset_path.exists():
        print(f" Dataset not found: {dataset_path}")
        exit(1)
    
    # Start training
    train_yolo()