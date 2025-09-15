import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def subsample_image(image_path):
    """
    Downsamples a grayscale image by factors of 2, 4, and 8 and displays the results.
    """
    # Load the grayscale image
    original_img = Image.open(image_path).convert('L')
    img_array = np.array(original_img)
    
    # Perform subsampling
    subsampled_2x = img_array[::2, ::2]
    subsampled_4x = img_array[::4, ::4]
    subsampled_8x = img_array[::8, ::8]
    
    # Convert numpy arrays back to images
    img_2x = Image.fromarray(subsampled_2x.astype(np.uint8))
    img_4x = Image.fromarray(subsampled_4x.astype(np.uint8))
    img_8x = Image.fromarray(subsampled_8x.astype(np.uint8))
    
    # Display all images
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f"Original\n({original_img.size[0]}x{original_img.size[1]})")
    axes[0].axis('off')
    
    # Subsampled by 2
    axes[1].imshow(img_2x, cmap='gray')
    axes[1].set_title(f"Subsampled by 2\n({img_2x.size[0]}x{img_2x.size[1]})")
    axes[1].axis('off')
    
    # Subsampled by 4
    axes[2].imshow(img_4x, cmap='gray')
    axes[2].set_title(f"Subsampled by 4\n({img_4x.size[0]}x{img_4x.size[1]})")
    axes[2].axis('off')
    
    # Subsampled by 8
    axes[3].imshow(img_8x, cmap='gray')
    axes[3].set_title(f"Subsampled by 8\n({img_8x.size[0]}x{img_8x.size[1]})")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

    # Discuss the changes
    print("--- Changes with Image Subsampling ---")
    print("As the subsampling factor increases, the image resolution decreases significantly.")
    print("This leads to a loss of fine details and sharpness.")
    print("The images appear more pixelated or blocky, and fine lines or textures become blurred or disappear entirely.")
    print("This demonstrates the trade-off between image size and visual information content.")

# Example usage: Replace with your actual file name
# subsample_image('grayscale_image.jpg')
subsample_image('color_image.jpg')