#Checking the validity and correctness of annotations against their image counterparts
import os
import json
from PIL import Image, ExifTags
from tqdm import tqdm

def get_image_size_with_orientation(img_path):
    with Image.open(img_path) as img:
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif is not None:
                exif = dict(exif.items())
                if orientation in exif:
                    if exif[orientation] == 3:
                        img = img.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        img = img.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        img = img.rotate(90, expand=True)
        except AttributeError:
            pass
        return img.size

def check_annotations(dataset_dir, annotations_file):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    image_dir = os.path.join(dataset_dir, 'images', 'train' if 'train' in annotations_file else 'val')
    
    mismatches = []
    images_in_dir = set()
    images_in_annotations = set()
    
    for img_file in tqdm(os.listdir(image_dir), desc="Checking images in directory"):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            images_in_dir.add(img_file)
            img_path = os.path.join(image_dir, img_file)
            actual_dimensions = get_image_size_with_orientation(img_path)
            
            annotation = next((img for img in annotations['images'] if img['file_name'].endswith(img_file)), None)
            
            if annotation:
                expected_dimensions = (annotation['width'], annotation['height'])
                if actual_dimensions != expected_dimensions:
                    mismatches.append(f"Dimension mismatch for {img_file}: "
                                      f"Annotation: {expected_dimensions}, "
                                      f"Actual: {actual_dimensions}")
            else:
                mismatches.append(f"Image not in annotations: {img_file}")
    
    for img_info in tqdm(annotations['images'], desc="Checking images in annotations"):
        img_file = os.path.basename(img_info['file_name'])
        images_in_annotations.add(img_file)
        
        if img_file not in images_in_dir:
            mismatches.append(f"Image in annotations but not in directory: {img_file}")
    
    return mismatches, images_in_dir, images_in_annotations

dataset_dir = "dataset"
train_annotations = os.path.join(dataset_dir, "annotations", "instances_train.json")
val_annotations = os.path.join(dataset_dir, "annotations", "instances_val.json")

print("Checking training set:")
train_mismatches, train_images_in_dir, train_images_in_annotations = check_annotations(dataset_dir, train_annotations)

print("\nChecking validation set:")
val_mismatches, val_images_in_dir, val_images_in_annotations = check_annotations(dataset_dir, val_annotations)

print(f"\nTraining images in directory: {len(train_images_in_dir)}")
print(f"Training images in annotations: {len(train_images_in_annotations)}")
print(f"Training images only in directory: {len(train_images_in_dir - train_images_in_annotations)}")
print(f"Training images only in annotations: {len(train_images_in_annotations - train_images_in_dir)}")

print(f"\nValidation images in directory: {len(val_images_in_dir)}")
print(f"Validation images in annotations: {len(val_images_in_annotations)}")
print(f"Validation images only in directory: {len(val_images_in_dir - val_images_in_annotations)}")
print(f"Validation images only in annotations: {len(val_images_in_annotations - val_images_in_dir)}")

print("\nMismatches in training set:")
for mismatch in train_mismatches:
    print(mismatch)

print("\nMismatches in validation set:")
for mismatch in val_mismatches:
    print(mismatch)

print(f"\nTotal mismatches: {len(train_mismatches) + len(val_mismatches)}")
