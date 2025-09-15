import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np

def histogram_equalization(image_path):
    """
    Plots histograms and applies histogram equalization.
    """
    try:
        # Load the grayscale image
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    # Convert to NumPy array for histogram calculation
    img_array = np.array(img)

    # Apply histogram equalization
    equalized_img = ImageOps.equalize(img)
    equalized_array = np.array(equalized_img)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Histogram and Equalization", fontsize=16)

    # Original Image
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Original Histogram
    axes[0, 1].hist(img_array.flatten(), bins=256, range=[0, 256], color='gray')
    axes[0, 1].set_title('Original Histogram')
    axes[0, 1].set_xlabel('Gray Level')
    axes[0, 1].set_ylabel('Number of Pixels')

    # Equalized Image
    axes[1, 0].imshow(equalized_img, cmap='gray')
    axes[1, 0].set_title('Equalized Image')
    axes[1, 0].axis('off')

    # Equalized Histogram
    axes[1, 1].hist(equalized_array.flatten(), bins=256, range=[0, 256], color='gray')
    axes[1, 1].set_title('Equalized Histogram')
    axes[1, 1].set_xlabel('Gray Level')
    axes[1, 1].set_ylabel('Number of Pixels')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Discussion
    print("--- Discussion on Histogram Equalization ---")
    print("The original image's histogram is likely clustered in a narrow range, indicating low contrast.")
    print("After equalization, the histogram is stretched to cover a wider range, distributing pixel intensities more uniformly.")
    print("This process enhances the overall contrast, making previously indistinguishable details in dark and bright regions more visible.")
    print("The equalized image appears sharper and more defined compared to the original.")

# Example usage:
histogram_equalization('low_contrast_image.jpg')