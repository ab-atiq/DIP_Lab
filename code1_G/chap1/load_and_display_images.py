import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_and_display_images(color_path, grayscale_path):
    """
    Loads and displays a color and a grayscale image side by side.
    """
    # Load images
    color_img = Image.open(color_path)
    grayscale_img = Image.open(grayscale_path)
    
    # Ensure grayscale image is in 'L' mode
    if grayscale_img.mode != 'L':
        grayscale_img = grayscale_img.convert('L')

    # Display images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display Color Image
    axes[0].imshow(color_img)
    axes[0].set_title(f"Color Image\nChannels: {len(np.array(color_img).shape)}")
    axes[0].axis('off')
    
    # Display Grayscale Image
    axes[1].imshow(grayscale_img, cmap='gray')
    axes[1].set_title(f"Grayscale Image\nChannels: {len(np.array(grayscale_img).shape)}")
    axes[1].axis('off')
    
    plt.show()

    # Describe the differences
    print("--- Image Analysis ---")
    print(f"Color Image Dimensions: {color_img.size}")
    print(f"Grayscale Image Dimensions: {grayscale_img.size}")
    print("\nObservation:")
    print("1. Channels: The color image has three channels (RGB), while the grayscale image has only one.")
    print("2. Pixel Intensity: In the color image, each pixel is represented by a triplet of values (R, G, B), while in the grayscale image, each pixel is a single value representing brightness.")
    print("3. Appearance: The color image displays a full range of colors, while the grayscale image only displays shades of black, white, and gray.")
    
# Example usage: Replace with your actual file names
# load_and_display_images('color_image2.jpg', 'grayscale_image2.jpg')
load_and_display_images('color_image2.jpg', 'color_image2.jpg')