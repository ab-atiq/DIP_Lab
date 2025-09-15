import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def smoothing_filters(image_path):
    """
    Applies averaging filters with different kernel sizes.
    """
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    # Apply 3x3 averaging filter
    smoothed_3x3 = img.filter(ImageFilter.BoxBlur(1)) # BoxBlur(1) is equivalent to 3x3
    
    # Apply 5x5 averaging filter
    smoothed_5x5 = img.filter(ImageFilter.BoxBlur(2)) # BoxBlur(2) is equivalent to 5x5
    
    # Display images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(smoothed_3x3, cmap='gray')
    axes[1].set_title('3x3 Averaging Filter')
    axes[1].axis('off')
    
    axes[2].imshow(smoothed_5x5, cmap='gray')
    axes[2].set_title('5x5 Averaging Filter')
    axes[2].axis('off')
    
    plt.show()
    
    # Discussion
    print("--- Effect of Smoothing Filters ---")
    print("Smoothing filters, such as the averaging filter, reduce image noise by replacing each pixel value with the average of its neighbors.")
    print("As the kernel size increases (from 3x3 to 5x5), the averaging window becomes larger.")
    print("This results in a stronger blurring effect: the 5x5 filtered image is more blurred than the 3x3 filtered image.")
    print("While effective at removing noise, this process also blurs fine details and edges, as these features are also averaged out.")

# Example usage:
smoothing_filters('noisy_image2.jpg')