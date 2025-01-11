# Import libaries
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Function to resize the image
def resize_image(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is not None:
        ratio = width / float(w)
        dimension = (width, int(h * ratio))
    elif height is not None:
        ratio = height / float(h)
        dimension = (int(w * ratio), height)
    else:
        return image
    resized = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
    return resized

# Function to process each image
def process_image(image_path, output_folder):
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize the image
    resized_image = resize_image(image, width=128, height=128)
    
    # Convert to Grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Apply Adaptive Threshold to Focus on Lighter Pixels
    thresholded = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    
    # Apply Weakened Dilation to Expand White Background
    kernel = np.ones((2, 1), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=1)
    
    # Save the processed image to the output folder
    filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_image_path, dilated)
    
    # Display the processed image (optional)
    # plt.imshow(dilated, cmap='gray')
    # plt.title(f"Processed Image - {filename}")
    # plt.axis('off')
    # plt.show()
    
    print(f"Processed image saved to {output_image_path}")

# Function to process images in a folder (including subfolders)
def process_images_in_folder(input_folder, output_folder):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # First, process images in all subfolders
    for root, dirs, files in os.walk(input_folder):
        # Ignore .ipynb_checkpoints folder
        if '.ipynb_checkpoints' in dirs:
            dirs.remove('.ipynb_checkpoints')
        
        # Only process images if we are in a subfolder (i.e., root != input_folder)
        if root != input_folder:
            for filename in files:
                file_path = os.path.join(root, filename)
                
                # Check if it's an image file
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    process_image(file_path, output_folder)
                    
    # Next, process images in the main folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Check if it's an image file and not a directory
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            process_image(file_path, output_folder)


# Set the input and output folder paths manually
input_folder = 'Meloidogyne graminicola'  # Replace with the path to your folder containing images
output_folder = 'output-graminicola'        # Folder for saving processed images

# Process all images in the specified folder
process_images_in_folder(input_folder, output_folder)
