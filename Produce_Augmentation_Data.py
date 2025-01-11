import cv2
import numpy as np
import os
import random
from glob import glob

# Function to create directories if they don't exist
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Define parent directory for augmented images
parent_dir = 'augmented_graminicola'

# Create the parent directory if it doesn't exist
create_dir(parent_dir)

# Define subdirectories for each augmentation type within the parent directory
augmentation_dirs = ['flipped_horizontal', 'flipped_vertical', 'flipped_180', 'blurred', 'noisy', 'brightness', 'contrast']
for aug_dir in augmentation_dirs:
    create_dir(os.path.join(parent_dir, aug_dir))

# Resize function
def resize_image(image, width=224, height=224):
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized

# Load images from the dataset folder, looking for both .jpg and .png files
image_paths = glob('../PROJECT BU NAB/Meloidogyne graminicola-v2/*.jpg') + glob('../PROJECT BU NAB/Meloidogyne graminicola-v2/*.png')  # Change to your dataset folder path

# Apply augmentations on each image
for img_path in image_paths:
    # Load and resize image
    image = cv2.imread(img_path)
    base_name = os.path.basename(img_path).split('.')[0]  # Get original name without extension
    resized_img = resize_image(image, 224, 224)

    # 1. Image Flipping
    # Horizontal Flip
    h_flip_img = cv2.flip(resized_img, 1)
    cv2.imwrite(f'{parent_dir}/flipped_horizontal/{base_name}_flipped_horizontal.jpg', h_flip_img)

    # Vertical Flip
    v_flip_img = cv2.flip(resized_img, 0)
    cv2.imwrite(f'{parent_dir}/flipped_vertical/{base_name}_flipped_vertical.jpg', v_flip_img)

    # 180-Degree Flip (Horizontal + Vertical Flip)
    flip_180_img = cv2.flip(h_flip_img, 0)
    cv2.imwrite(f'{parent_dir}/flipped_180/{base_name}_flipped_180.jpg', flip_180_img)

    # 2. Image Blurring (50% of images in the dataset)
    if random.random() < 0.5:
        blurred_img = cv2.GaussianBlur(resized_img, (3, 3), 1)
        cv2.imwrite(f'{parent_dir}/blurred/{base_name}_blurred.jpg', blurred_img)

    # 3. Noise Addition (White Gaussian Noise, 50% of images in the dataset)
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.15, resized_img.shape).astype(np.float32)
        noisy_img = cv2.addWeighted(resized_img.astype(np.float32), 1.0, noise, 1.0, 0).astype(np.uint8)
        cv2.imwrite(f'{parent_dir}/noisy/{base_name}_noisy.jpg', noisy_img)

    # 4. Brightness Change (Random increase from 0 to 0.3 for each image)
    brightness_factor = random.uniform(0, 0.3)
    bright_img = cv2.convertScaleAbs(resized_img, alpha=1, beta=brightness_factor * 50)  # Beta adjusts brightness
    cv2.imwrite(f'{parent_dir}/brightness/{base_name}_brightness.jpg', bright_img)

    # 5. Contrast Change (Random increase from 0 to 1 for each image)
    contrast_factor = random.uniform(0, 1)
    contrast_img = cv2.convertScaleAbs(resized_img, alpha=1 + contrast_factor, beta=0)  # Alpha adjusts contrast
    cv2.imwrite(f'{parent_dir}/contrast/{base_name}_contrast.jpg', contrast_img)

print("Augmentation completed.")
