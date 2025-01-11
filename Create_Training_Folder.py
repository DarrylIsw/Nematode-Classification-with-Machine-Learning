"""
Use this script to generate the necessary folder structure for the model training and
to split the data into training, validation, and test sets according to ratios
"""

# Immport libraries
import os
import shutil
import random

# Paths to source and destination directories (Modifiable)
source_dir = 'output-incognita-v2' # Folder that contains the source images
train_dir = 'dataset5/train/pattern_B' 
val_dir = 'dataset5/validation/pattern_B'
test_dir = 'dataset5/test/pattern_B'

# Check and create directories if they do not exist
if not all(os.path.exists(dir) for dir in [train_dir, val_dir, test_dir]):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print("Created target directories.")
else:
    print("Target directories already exist. Proceeding with partitioning.")

# Set the split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Get all image filenames from the source directory
images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
random.shuffle(images)

# Calculate the number of images for each set
total_images = len(images)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)
test_count = total_images - train_count - val_count  # Remaining for test

# Move the images to their respective directories
for i, img_name in enumerate(images):
    src_path = os.path.join(source_dir, img_name)
    
    if i < train_count:
        dst_path = os.path.join(train_dir, img_name)
    elif i < train_count + val_count:
        dst_path = os.path.join(val_dir, img_name)
    else:
        dst_path = os.path.join(test_dir, img_name)
    
    shutil.copy(src_path, dst_path)

print("Images have been successfully partitioned into training, validation, and testing folders.")
