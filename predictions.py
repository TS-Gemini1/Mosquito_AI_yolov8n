from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def visualize_predictions(model_path, test_images_dir, num_images=12, conf_threshold=0.1):
    """
    Visualize model predictions on test images
    """
    # Load trained model
    model = YOLO(model_path)
    
    # Get test images - check multiple formats
    test_images_dir = Path(test_images_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(test_images_dir.glob(ext))
    
    if len(image_files) == 0:
        print(f"No image files found in {test_images_dir}")
        return None
    
    # Select images to visualize
    if num_images is None:
        selected_images = image_files
    else:
        selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    # Create output directory
    output_dir = Path("C:/Users/theod/Desktop/APPENDIX/predictions_visualization")
    output_dir.mkdir(exist_ok=True)
    
    total_detections = 0
    images_with_detections = 0
    
    for i, img_path in enumerate(selected_images):
        if i % 50 == 0 and i > 0:
            print(f"Progress: {i+1}/{len(selected_images)}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Make prediction
        results = model(img, conf=conf_threshold, iou=0.4, max_det=500)
        
        # Override class names for display
        if results[0].names is not None:
            results[0].names[0] = 'egg'
        
        # Draw predictions on image
        annotated_img = results[0].plot(
            conf=True,
            labels=True,
            boxes=True,
            line_width=2,
            font_size=12
        )
        
        # Get detection info
        detections = results[0].boxes
        num_detections = len(detections) if detections is not None else 0
        
        if num_detections > 0:
            images_with_detections += 1
            total_detections += num_detections
        
        # Add title with detection count
        title_img = np.zeros((50, annotated_img.shape[1], 3), dtype=np.uint8)
        title_text = f"{img_path.name} - {num_detections} detections"
        cv2.putText(title_img, title_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Combine title and image
        final_img = np.vstack([title_img, annotated_img])
        
        # Save result
        output_path = output_dir / f"prediction_{i+1:04d}_{img_path.name}"
        cv2.imwrite(str(output_path), final_img)
    
    # Print summary
    print(f"\nResults saved to: {output_dir}")
    print(f"Images processed: {len(selected_images)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(selected_images):.2f}")
    
    return output_dir

def create_summary_grid(predictions_dir, grid_size=(3, 3)):
    """
    Create a grid showing multiple predictions in one image
    """
    predictions_dir = Path(predictions_dir)
    pred_images = list(predictions_dir.glob("prediction_*.jpg"))
    
    if len(pred_images) == 0:
        print("No prediction images found for grid")
        return
    
    # Select images for grid
    max_images = grid_size[0] * grid_size[1]
    selected = pred_images[:max_images]
    
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
    axes = axes.flatten()
    
    for i, img_path in enumerate(selected):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(img_path.name, fontsize=8)
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(selected), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save grid
    grid_path = predictions_dir / "prediction_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Prediction grid saved to: {grid_path}")

def test_different_thresholds(model_path, test_image_path, thresholds=[0.05, 0.1, 0.2, 0.3, 0.5]):
    """
    Test the same image with different confidence thresholds
    """
    model = YOLO(model_path)
    img = cv2.imread(str(test_image_path))
    
    if img is None:
        print(f"Could not load image: {test_image_path}")
        return
    
    output_dir = Path("C:/Users/theod/Desktop/APPENDIX/threshold_comparison")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Testing different confidence thresholds on {Path(test_image_path).name}")
    
    for threshold in thresholds:
        results = model(img, conf=threshold, iou=0.4, max_det=500)
        annotated_img = results[0].plot(conf=True, labels=True, boxes=True)
        
        # Get detection count
        detections = results[0].boxes
        num_detections = len(detections) if detections is not None else 0
        
        # Add threshold info to image
        title_img = np.zeros((40, annotated_img.shape[1], 3), dtype=np.uint8)
        title_text = f"Conf: {threshold} - Detections: {num_detections}"
        cv2.putText(title_img, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        final_img = np.vstack([title_img, annotated_img])
        
        # Save
        output_path = output_dir / f"conf_{threshold}_{num_detections}det.jpg"
        cv2.imwrite(str(output_path), final_img)
        
        print(f"Confidence {threshold}: {num_detections} detections")

def interactive_threshold_test(model_path, test_image_path):
    """
    Interactive threshold adjustment with slider for a single image
    """
    model = YOLO(model_path)
    img = cv2.imread(str(test_image_path))
    
    if img is None:
        print(f"Could not load image: {test_image_path}")
        return
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)
    
    # Initial threshold
    initial_threshold = 0.25
    
    # Function to update the display
    def update_display(threshold):
        ax.clear()
        
        # Run detection with current threshold
        results = model(img, conf=threshold, iou=0.4, max_det=500)
        
        # Override class names for display
        if results[0].names is not None:
            results[0].names[0] = 'egg'
        
        # Get annotated image
        annotated_img = results[0].plot(conf=True, labels=True, boxes=True)
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Get detection count
        detections = results[0].boxes
        num_detections = len(detections) if detections is not None else 0
        
        # Display the image
        ax.imshow(annotated_img_rgb)
        ax.set_title(f'Confidence Threshold: {threshold:.2f} - Detections: {num_detections}', fontsize=14)
        ax.axis('off')
        
        fig.canvas.draw_idle()
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Confidence', 0.01, 0.99, valinit=initial_threshold, valstep=0.01)
    
    # Connect slider to update function
    slider.on_changed(update_display)
    
    # Initial display
    update_display(initial_threshold)
    
    # Add instructions
    plt.figtext(0.5, 0.02, 'Adjust the slider to change confidence threshold', 
                ha='center', fontsize=10, style='italic')
    
    plt.show()

def interactive_threshold_test_multi(model_path, test_images_dir):
    """
    Interactive threshold adjustment with navigation between multiple images
    """
    model = YOLO(model_path)
    
    # Get all test images
    test_images_dir = Path(test_images_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(test_images_dir.glob(ext))
    
    if not image_files:
        print("No images found in test directory")
        return
    
    # Sort for consistent ordering
    image_files = sorted(image_files)
    
    # State variables
    current_index = 0
    current_threshold = 0.25
    current_iou = 0.45
    
    # Create figure with fixed size and layout
    fig = plt.figure(figsize=(14, 10))
    
    # Define layout - ensure consistent spacing
    img_height = 0.70  # 70% for image (reduced to make room for 2nd slider)
    
    # Image display area
    ax = plt.axes([0.05, 0.25, 0.90, img_height])
    
    def update_display():
        ax.clear()
        
        # Load current image
        img_path = image_files[current_index]
        img = cv2.imread(str(img_path))
        
        if img is None:
            ax.text(0.5, 0.5, f'Could not load: {img_path.name}', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Run detection with both thresholds
        results = model(img, conf=current_threshold, iou=current_iou, max_det=500)
        
        # Override class names
        if results[0].names is not None:
            results[0].names[0] = 'egg'
        
        # Get annotated image
        annotated_img = results[0].plot(conf=True, labels=True, boxes=True)
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Get detection count
        detections = results[0].boxes
        num_detections = len(detections) if detections is not None else 0
        
        # Display
        ax.imshow(annotated_img_rgb)
        ax.set_title(f'Image {current_index + 1}/{len(image_files)}: {img_path.name}\n' + 
                    f'Confidence: {current_threshold:.2f} | IoU: {current_iou:.2f} | Detections: {num_detections}', 
                    fontsize=14, pad=10)
        ax.axis('off')
        
        fig.canvas.draw_idle()
    
    def on_threshold_change(val):
        nonlocal current_threshold
        current_threshold = val
        update_display()
    
    def on_iou_change(val):
        nonlocal current_iou
        current_iou = val
        update_display()
    
    def next_image(event):
        nonlocal current_index
        current_index = (current_index + 1) % len(image_files)
        update_display()
    
    def prev_image(event):
        nonlocal current_index
        current_index = (current_index - 1) % len(image_files)
        update_display()
    
    # Create UI elements with fixed positions
    # Confidence slider
    ax_conf_slider = plt.axes([0.25, 0.12, 0.50, 0.03])
    conf_slider = Slider(ax_conf_slider, 'Confidence', 0.01, 0.99, valinit=current_threshold, valstep=0.01)
    conf_slider.on_changed(on_threshold_change)
    
    # IoU slider
    ax_iou_slider = plt.axes([0.25, 0.07, 0.50, 0.03])
    iou_slider = Slider(ax_iou_slider, 'IoU', 0.01, 0.99, valinit=current_iou, valstep=0.01)
    iou_slider.on_changed(on_iou_change)
    
    # Navigation buttons
    ax_prev = plt.axes([0.15, 0.02, 0.15, 0.04])
    ax_next = plt.axes([0.70, 0.02, 0.15, 0.04])
    btn_prev = Button(ax_prev, '← Previous')
    btn_next = Button(ax_next, 'Next →')
    btn_prev.on_clicked(prev_image)
    btn_next.on_clicked(next_image)
    
    # Keyboard navigation
    def on_key(event):
        if event.key == 'left':
            prev_image(None)
        elif event.key == 'right':
            next_image(None)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    update_display()
    
    # Instructions
    plt.figtext(0.5, 0.17, 'Use arrow keys or buttons to navigate | Adjust sliders for thresholds', 
                ha='center', fontsize=10, style='italic')
    
    # Set window title
    fig.canvas.manager.set_window_title('YOLO Detection Threshold Testing')
    
    plt.show()

if __name__ == "__main__":
    # Paths
    model_path = "C:/Users/theod/Desktop/APPENDIX/Results/mosquito_eggs_2class/weights/best.pt"
    test_images_dir = "C:/Users/theod/Documents/CODE/AI9/D1.5/images/test"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found at: {model_path}")
        exit(1)
    
    # Check if test images exist
    if not Path(test_images_dir).exists():
        print(f"Test images not found at: {test_images_dir}")
        exit(1)
    
    # Run visualizations with interactive mode
    predictions_dir = visualize_predictions(
        model_path=model_path,
        test_images_dir=test_images_dir,
        num_images=None,  # Process all images
        conf_threshold=0.1
    )
    
    # Create summary grid only if predictions were made
    if predictions_dir is not None:
        create_summary_grid(predictions_dir, grid_size=(3, 4))
    
    # Always run interactive threshold test with navigation
    interactive_threshold_test_multi(
        model_path=model_path,
        test_images_dir=test_images_dir
    )