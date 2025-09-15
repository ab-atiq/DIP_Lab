import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def quantize_image(image_path, levels):
    """
    Quantizes a grayscale image to a specified number of gray levels.
    """
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float64)

    # Calculate the step size for uniform quantization
    step = 256.0 / levels
    
    # Apply quantization: round to the nearest quantization level
    quantized_array = np.floor(img_array / step) * step
    
    # Convert back to uint8 for image display
    quantized_img = Image.fromarray(quantized_array.astype(np.uint8))
    return quantized_img

def display_quantized_images(image_path):
    """
    Displays an image quantized to different gray levels.
    """
    levels = [2, 4, 8, 16, 256]
    
    fig, axes = plt.subplots(1, len(levels), figsize=(20, 4))
    fig.suptitle("Gray Level Quantization", fontsize=16)

    for i, level in enumerate(levels):
        if level == 256:
            # For 256 levels, just load the original image
            img = Image.open(image_path).convert('L')
            title = "Original (256 Levels)"
        else:
            img = quantize_image(image_path, level)
            title = f"{level} Levels"
        
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(title)
        axes[i].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Discussion on effects
    print("--- Effects of Quantization ---")
    print("As the number of gray levels decreases, the image loses fine tonal variations and smooth gradients.")
    print("For a low number of levels (e.g., 2 or 4), the image appears stark and blocky due to 'false contours' or 'banding'.")
    print("The 2-level image is a binary image (black and white).")
    print("The visibility of details is significantly reduced, as small intensity differences are lost in the quantization process.")
    print("Quantization is a key step in image compression, but it is a trade-off between file size and image quality.")

# Example usage: Replace with your actual grayscale image file name
display_quantized_images('grayscale_image.jpg')