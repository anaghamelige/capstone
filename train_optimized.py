from ultralytics import YOLO
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
import cv2
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
import random

def create_focused_dataset_config(original_yaml, focus_classes=[0, 1, 2]):
    """
    Create a focused dataset configuration for sleeping, phone usage, and no ID card
    Args:
        original_yaml: Path to original dataset YAML
        focus_classes: List of class indices to focus on [sleeping=0, phone_usage=1, no_id_card=2]
    """
    with open(original_yaml, 'r') as file:
        config = yaml.safe_load(file)
    
    # Update class names to focus on priority anomalies
    focused_names = {
        0: 'sleeping',
        1: 'phone_usage', 
        2: 'no_id_card'
    }
    
    config['names'] = focused_names
    config['nc'] = len(focused_names)
    
    # Save focused config
    focused_yaml = original_yaml.replace('.yaml', '_focused.yaml')
    with open(focused_yaml, 'w') as f:
        yaml.dump(config, f)
    
    return focused_yaml

def enhance_classroom_dataset(data_yaml, augment_factor=3):
    """
    Enhanced augmentation specifically for classroom behavior detection
    Focus on sleeping, phone usage, and no ID card scenarios
    """
    with open(data_yaml, 'r') as file:
        config = yaml.safe_load(file)
    
    base_path = Path(config['path'])
    train_img_path = base_path / config['train']
    train_label_path = base_path / 'labels' / 'train'
    
    # Create enhanced directories
    aug_img_dir = base_path / 'images' / 'train_enhanced'
    aug_label_dir = base_path / 'labels' / 'train_enhanced'
    
    for dir_path in [aug_img_dir, aug_label_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Classroom-specific augmentations for better real-world performance
    classroom_transform = A.Compose([
        # Lighting variations (classroom lighting changes)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=1),
        ], p=0.9),
        
        # Perspective and rotation (different camera angles)
        A.OneOf([
            A.SafeRotate(limit=15, p=1),
            A.Perspective(scale=(0.02, 0.1), p=1),
        ], p=0.7),
        
        # Blur variations (motion blur, focus issues)
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1),
            A.GaussianBlur(blur_limit=5, p=1),
            A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.5), p=1),
        ], p=0.6),
        
        # Noise and quality variations
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
        ], p=0.5),
        
        # Shadows and occlusions (common in classrooms)
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.4),
        
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2))
    
    # Sleeping-specific augmentations (darker, more shadows)
    sleeping_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.1), contrast_limit=0.3, p=0.9),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=2, num_shadows_upper=3, p=0.7),
        A.SafeRotate(limit=20, p=0.6),  # People sleep in various positions
        A.GaussianBlur(blur_limit=3, p=0.4),  # Slight blur for sleeping
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2))
    
    # Phone usage augmentations (focus on small objects)
    phone_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.8),
        A.Sharpen(alpha=(0.3, 0.7), lightness=(0.7, 1.0), p=0.8),  # Enhance phone visibility
        A.SafeRotate(limit=10, p=0.5),
        A.Perspective(scale=(0.02, 0.05), p=0.6),  # Slight perspective changes
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    
    # No ID card augmentations (lighting and contrast focus)
    no_id_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(0.1, 0.4), contrast_limit=0.4, p=0.9),
        A.Sharpen(alpha=(0.2, 0.6), lightness=(0.6, 1.0), p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.6),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2))
    
    # Copy original files
    print("Copying original training files...")
    for img_file in train_img_path.glob('*.jpg'):
        shutil.copy(img_file, aug_img_dir / img_file.name)
        label_file = train_label_path / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy(label_file, aug_label_dir / label_file.name)
    
    # Process images by class with specialized augmentations
    class_images = {0: [], 1: [], 2: []}  # sleeping, phone, no_id
    
    for label_file in train_label_path.glob('*.txt'):
        img_file = train_img_path / f"{label_file.stem}.jpg"
        if not img_file.exists():
            continue
            
        with open(label_file, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 5:
                    class_id = int(values[0])
                    if class_id in [0, 1, 2]:  # Only our focus classes
                        class_images[class_id].append((img_file, label_file))
                        break
    
    # Apply class-specific augmentations
    transforms = {0: sleeping_transform, 1: phone_transform, 2: no_id_transform}
    class_names = {0: 'sleeping', 1: 'phone_usage', 2: 'no_id_card'}
    
    for class_id, images in class_images.items():
        if images:
            print(f"Augmenting {class_names[class_id]} class: {len(images)} images")
            for img_file, label_file in tqdm(images, desc=f"Processing {class_names[class_id]}"):
                process_class_images_enhanced(
                    img_file, label_file, 
                    transforms[class_id], 
                    aug_img_dir, aug_label_dir, 
                    augment_factor=augment_factor + 1,  # Extra augmentation for focus classes
                    target_class=class_id
                )
    
    # Update config for enhanced dataset
    config['train'] = 'images/train_enhanced'
    enhanced_yaml = data_yaml.replace('.yaml', '_enhanced.yaml')
    with open(enhanced_yaml, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Enhanced dataset created for focused training. Config: {enhanced_yaml}")
    return enhanced_yaml

def process_class_images_enhanced(img_file, label_file, transform, aug_img_dir, aug_label_dir, 
                                augment_factor=3, target_class=None):
    """Enhanced image processing with better validation"""
    image = cv2.imread(str(img_file))
    if image is None:
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Read and filter labels for target class
    bboxes = []
    class_labels = []
    
    with open(label_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 5:
                class_id = int(values[0])
                if target_class is None or class_id == target_class:
                    x, y, w, h = map(float, values[1:5])
                    
                    # Strict bbox validation
                    eps = 1e-6
                    x = max(eps, min(1-eps, x))
                    y = max(eps, min(1-eps, y))
                    w = max(eps, min(1-x-eps, w))
                    h = max(eps, min(1-y-eps, h))
                    
                    if w > eps and h > eps:
                        bboxes.append([x, y, w, h])
                        class_labels.append(class_id)
    
    if not bboxes:
        return
    
    # Generate augmentations
    for i in range(augment_factor):
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            # Validate augmented bboxes
            valid_bboxes = []
            valid_labels = []
            
            for box, label in zip(augmented['bboxes'], augmented['class_labels']):
                x, y, w, h = box
                eps = 1e-6
                
                x = max(eps, min(1-eps, x))
                y = max(eps, min(1-eps, y))
                w = max(eps, min(1-x-eps, w))
                h = max(eps, min(1-y-eps, h))
                
                if w > eps and h > eps:
                    valid_bboxes.append([x, y, w, h])
                    valid_labels.append(label)
            
            if valid_bboxes:
                # Save augmented image
                aug_img_name = f"{img_file.stem}_cls{target_class}_aug{i}.jpg"
                aug_label_name = f"{img_file.stem}_cls{target_class}_aug{i}.txt"
                
                cv2.imwrite(
                    str(aug_img_dir / aug_img_name),
                    cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                )
                
                # Save labels
                with open(aug_label_dir / aug_label_name, 'w') as f:
                    for box, label in zip(valid_bboxes, valid_labels):
                        f.write(f"{label} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                        
        except Exception as e:
            print(f"Augmentation error for {img_file.name}, iteration {i}: {e}")
            continue

def train_optimized_model(data_yaml, training_platform='colab'):
    """
    Optimized training for maximum mAP and confidence scores
    Args:
        data_yaml: Path to dataset configuration
        training_platform: 'colab' or 'mac' for platform-specific optimization
    """
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create focused dataset config
    focused_yaml = create_focused_dataset_config(data_yaml, focus_classes=[0, 1, 2])
    
    # Enhance dataset with class-specific augmentations
    enhanced_yaml = enhance_classroom_dataset(focused_yaml, augment_factor=4)
    
    # Platform-specific configuration
    if training_platform == 'colab':
        model_size = 'yolov8l.pt'  # Large model for better accuracy
        batch_size = 24
        workers = 8
        device = '0'  # GPU
        amp = True
        epochs = 120
    else:  # mac
        model_size = 'yolov8m.pt'  # Medium model to prevent overheating
        batch_size = 8
        workers = 4
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        amp = False  # Disable AMP on MPS
        epochs = 80
    
    # Initialize model
    model = YOLO(model_size)
    
    print(f"Training on {training_platform} with {model_size}")
    print(f"Device: {device}, Batch size: {batch_size}, Epochs: {epochs}")
    
    # Stage 1: Base training with enhanced parameters
    print("Stage 1: Base training for feature learning...")
    stage1_results = model.train(
        data=enhanced_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        workers=workers,
        device=device,
        amp=amp,
        
        # Optimized hyperparameters for classroom detection
        patience=25,
        optimizer='AdamW',
        lr0=0.002,  # Higher initial learning rate
        lrf=0.0001,  # Lower final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Enhanced loss weights for better classification
        box=8.0,  # Higher box loss for better localization
        cls=2.0,  # Much higher classification loss
        dfl=1.5,  # Distribution focal loss
        
        # Classroom-optimized augmentations
        hsv_h=0.02,  # More HSV variation for classroom lighting
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=15.0,  # More rotation for various seating positions
        translate=0.15,  # More translation
        scale=0.7,  # More scale variation
        shear=3.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,  # More mixup for better generalization
        copy_paste=0.3,
        
        # Training stability
        cos_lr=True,
        close_mosaic=15,
        label_smoothing=0.1,
        
        # Saving and monitoring
        save=True,
        save_period=10,
        plots=True,
        name='classroom_focused_stage1',
        exist_ok=True,
        pretrained=True,
        verbose=True
    )
    
    # Stage 2: Fine-tuning with higher resolution
    print("Stage 2: Fine-tuning with higher resolution for precision...")
    stage2_results = model.train(
        data=enhanced_yaml,
        epochs=40,
        imgsz=800,  # Higher resolution for fine details
        batch=max(4, batch_size // 2),  # Reduce batch for higher resolution
        workers=workers,
        device=device,
        amp=amp,
        
        # Fine-tuning parameters
        patience=15,
        optimizer='AdamW',
        lr0=0.0005,  # Much lower learning rate for fine-tuning
        lrf=0.00005,
        momentum=0.937,
        weight_decay=0.0008,
        warmup_epochs=0,  # No warmup for fine-tuning
        
        # Adjusted loss weights
        box=6.0,
        cls=3.0,  # Even higher classification emphasis
        dfl=1.5,
        
        # Reduced augmentation for fine-tuning
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=8.0,
        translate=0.1,
        scale=0.8,
        shear=1.0,
        mosaic=0.5,  # Reduced mosaic
        mixup=0.1,
        copy_paste=0.1,
        
        cos_lr=True,
        close_mosaic=5,
        label_smoothing=0.05,
        
        name='classroom_focused_stage2',
        exist_ok=True,
        resume=True  # Resume from stage 1
    )
    
    # Validation with optimized thresholds
    print("Validating with optimized confidence thresholds...")
    val_results = model.val(
        data=enhanced_yaml,
        imgsz=800,
        conf=0.15,  # Lower confidence to catch more detections
        iou=0.6,    # Balanced IoU threshold
        max_det=300,
        save_json=True,
        save_conf=True,
        plots=True,
        verbose=True
    )
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    print(f"Final mAP50-95: {val_results.box.map:.4f}")
    print(f"Final mAP50: {val_results.box.map50:.4f}")
    print(f"Final Precision: {val_results.box.mp:.4f}")
    print(f"Final Recall: {val_results.box.mr:.4f}")
    print(f"Final F1-Score: {2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr + 1e-6):.4f}")
    
    # Per-class metrics
    if hasattr(val_results.box, 'ap_class_index'):
        print("\nPer-class AP50:")
        class_names = ['sleeping', 'phone_usage', 'no_id_card']
        for i, class_name in enumerate(class_names):
            if i < len(val_results.box.ap50):
                print(f"  {class_name}: {val_results.box.ap50[i]:.4f}")
    
    print("="*60)
    
    return model

def create_inference_config(model, data_yaml):
    """Create optimized inference configuration for real-world testing"""
    os.makedirs('deployment', exist_ok=True)
    
    # Copy best model
    best_model_path = Path(model.trainer.save_dir) / 'weights/best.pt'
    shutil.copy(best_model_path, 'deployment/classroom_best.pt')
    
    # Create inference configuration with class-specific thresholds
    inference_config = {
        'model_path': 'classroom_best.pt',
        'input_size': 800,
        'class_thresholds': {
            'sleeping': {
                'conf': 0.25,  # Lower threshold for sleeping (may be subtle)
                'iou': 0.5
            },
            'phone_usage': {
                'conf': 0.35,  # Higher threshold for phone (more distinct)
                'iou': 0.6
            },
            'no_id_card': {
                'conf': 0.30,  # Medium threshold for no ID
                'iou': 0.6
            }
        },
        'default_conf': 0.25,
        'default_iou': 0.6,
        'max_detections': 100,
        'agnostic_nms': False
    }
    
    # Save configuration
    with open('deployment/inference_config.yaml', 'w') as f:
        yaml.dump(inference_config, f)
    
    # Create inference script
    inference_script = """
import torch
from ultralytics import YOLO
import yaml
import cv2
import numpy as np

def load_optimized_model(config_path='inference_config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = YOLO(config['model_path'])
    return model, config

def predict_with_confidence(model, image_path, config):
    # Load image
    image = cv2.imread(image_path)
    
    # Run inference
    results = model(image, 
                   imgsz=config['input_size'],
                   conf=config['default_conf'],
                   iou=config['default_iou'],
                   max_det=config['max_detections'])
    
    # Process results with class-specific thresholds
    processed_results = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = ['sleeping', 'phone_usage', 'no_id_card'][class_id]
                
                # Apply class-specific threshold
                class_conf_threshold = config['class_thresholds'][class_name]['conf']
                
                if confidence >= class_conf_threshold:
                    processed_results.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    })
    
    return processed_results

# Example usage
if __name__ == "__main__":
    model, config = load_optimized_model()
    results = predict_with_confidence(model, 'test_image.jpg', config)
    
    for detection in results:
        print(f"Detected {detection['class']} with confidence {detection['confidence']:.3f}")
"""
    
    with open('deployment/optimized_inference.py', 'w') as f:
        f.write(inference_script)
    
    print("Inference configuration and script saved to deployment/")

if __name__ == "__main__":
    # Configuration
    data_yaml = 'classroom_combined.yaml'  # Your combined dataset
    training_platform = 'colab'  # Change to 'mac' if training locally
    
    print("Starting optimized training for classroom behavior detection...")
    print("Focus classes: sleeping, phone_usage, no_id_card")
    print(f"Training platform: {training_platform}")
    
    # Train the optimized model
    model = train_optimized_model(data_yaml, training_platform=training_platform)
    
    # Create deployment configuration
    create_inference_config(model, data_yaml)
    
    # Export model for deployment
    print("Exporting optimized model...")
    model.export(format='onnx', optimize=True, dynamic=True)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("Files created:")
    print("- deployment/classroom_best.pt (optimized model)")
    print("- deployment/inference_config.yaml (inference settings)")
    print("- deployment/optimized_inference.py (inference script)")
    print("- Model exported to ONNX format")
    print("="*60)