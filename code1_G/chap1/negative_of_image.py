import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def negative_of_image(image_path):
    """
    Generates and displays the negative of a grayscale image.
    """
    # Load the grayscale image
    original_img = Image.open(image_path).convert('L')
    
    # Convert image to numpy array for processing
    img_array = np.array(original_img)
    
    # Generate the negative image by subtracting each pixel from 255
    negative_array = 255 - img_array
    
    # Convert the numpy array back to an image
    negative_img = Image.fromarray(negative_array.astype(np.uint8))
    
    # Display original and negative images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Grayscale Image")
    axes[0].axis('off')
    
    axes[1].imshow(negative_img, cmap='gray')
    axes[1].set_title("Negative Grayscale Image")
    axes[1].axis('off')
    
    plt.show()
    
    # Explain the effect on visual perception
    print("--- Visual Perception of Negative Image ---")
    print("The negative image inverts the brightness values.")
    print("Dark areas in the original image become bright, and bright areas become dark.")
    print("This can sometimes make subtle details in shadows or highlights more visible.")

# Example usage: Replace with your actual file name
# negative_of_image('grayscale_image.jpg') # grayscale image to negative image
negative_of_image('color_image.jpg') # color image to grayscale image then convert into negative image