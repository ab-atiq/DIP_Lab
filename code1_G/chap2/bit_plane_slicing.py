import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def bit_plane_slicing(image_path):
    """
    Extracts and displays all 8 bit-planes of a grayscale image.
    """
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Bit-Plane Slicing (MSB to LSB)", fontsize=16)

    for i in range(8):
        # Extract the i-th bit-plane
        # The expression `(img_array >> i) & 1` is a bitwise operation:
        # `>> i` shifts bits to the right, placing the i-th bit in the LSB position.
        # `& 1` performs a bitwise AND with 1, which keeps only the LSB (0 or 1).
        bit_plane = (img_array >> i) & 1
        
        # Scale the bit-plane for better visualization (0 to 255)
        scaled_plane = bit_plane * 255
        
        row = i // 4
        col = i % 4
        
        axes[row, col].imshow(scaled_plane, cmap='gray')
        axes[row, col].set_title(f"Bit Plane {i} (2^{i})")
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # Discussion on information content
    print("--- Bit-Plane Information Content ---")
    print("The higher-order bit-planes (Bit 7, 6, 5) contain the most significant visual information.")
    print("These planes represent the overall structure and general appearance of the image.")
    print("As you move to lower-order bit-planes, the images contain finer details and appear noisier.")
    print("The lowest-order bit-planes (Bit 0, 1) often resemble random noise, as they represent the least significant changes in pixel intensity.")
    print("This technique is useful for image analysis, compression, and watermarking.")
    
# Example usage:
bit_plane_slicing('grayscale_image.jpg')