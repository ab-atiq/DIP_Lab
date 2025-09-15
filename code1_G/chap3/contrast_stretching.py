import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import numpy as np

def contrast_stretching(image_path):
    """
    Performs linear contrast stretching on a grayscale image.
    """
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
        
    img_array = np.array(img)
    
    # Get the min and max intensity values of the image
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    
    # Perform contrast stretching using the formula
    stretched_array = (img_array - min_val) * (255.0 / (max_val - min_val))
    
    # Convert back to an image and ensure data type is correct
    stretched_img = Image.fromarray(stretched_array.astype(np.uint8))
    
    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Low-Contrast Image')
    axes[0].axis('off')
    
    axes[1].imshow(stretched_img, cmap='gray')
    axes[1].set_title('Contrast Stretched Image')
    axes[1].axis('off')
    
    plt.show()
    
    # Discussion
    print("--- Effect of Contrast Stretching ---")
    print("Contrast stretching expands the range of pixel values, mapping the darkest point in the original image to black (0) and the brightest point to white (255).")
    print("This re-mapping makes the image appear more vibrant and dynamic, as the difference between various gray levels is magnified.")
    print("It is a simple and effective method for improving the appearance of images with a narrow range of pixel intensities.")

# Example usage:
contrast_stretching('low_contrast_image.jpg')