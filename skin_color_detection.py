import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def detect_skin_hsv(image_path):
    """
    Detect skin in an image using HSV color space thresholding.
    Steps:
    1. Read RGB values
    2. Convert to HSV values
    3. Judge values (apply threshold)
    4. Mark results
    """
    # Step 1: Read the image (in BGR format by default)
    img = cv2.imread(image_path)
    
    # Convert from BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 2: Convert RGB to HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Extract H, S, V channels
    h = img_hsv[:, :, 0]  # Hue
    s = img_hsv[:, :, 1]  # Saturation
    v = img_hsv[:, :, 2]  # Value (brightness)
    
    # Step 3: Apply threshold for skin detection
    # The rule is similar to what's shown in the image:
    # 0.23 < s < 0.68 and 0 < h < 0.2
    # Since OpenCV scales H to 0-180 and S,V to 0-255, we need to adjust our thresholds
    h_min, h_max = 0, 36  # 0.2 * 180 = 36
    s_min, s_max = 58, 173  # 0.23 * 255 = 58, 0.68 * 255 = 173
    
    # Create a mask where skin color is detected
    skin_mask = np.zeros_like(h)
    skin_mask[(s >= s_min) & (s <= s_max) & (h >= h_min) & (h <= h_max)] = 255
    
    # Step 4: Mark the results - create output image showing detected skin
    skin_detected = cv2.bitwise_and(img_rgb, img_rgb, mask=skin_mask)
    
    return img_rgb, skin_detected, skin_mask

def main():
    # Create a directory to store example images
    if not os.path.exists('example_images'):
        os.makedirs('example_images')
        print("Please place some images in the 'example_images' folder")
        print("Then run this script again")
        return
    
    # Get all the image files in the example_images directory
    image_files = glob.glob('example_images/*.jpg') + glob.glob('example_images/*.jpeg') + glob.glob('example_images/*.png')
    
    if not image_files:
        print("No images found. Please add images to the 'example_images' folder")
        return
    
    # Process each image
    for i, image_path in enumerate(image_files):
        # Extract just the filename without the path
        file_name = os.path.basename(image_path)
        
        # Process the image
        original, skin_detected, skin_mask = detect_skin_hsv(image_path)
        
        # Create a figure to display the results
        plt.figure(figsize=(12, 4))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(original)
        plt.title(f'Original: {file_name}')
        plt.axis('off')
        
        # Plot skin mask
        plt.subplot(1, 3, 2)
        plt.imshow(skin_mask, cmap='gray')
        plt.title('Skin Mask')
        plt.axis('off')
        
        # Plot skin detected
        plt.subplot(1, 3, 3)
        plt.imshow(skin_detected)
        plt.title('Skin Detected')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()