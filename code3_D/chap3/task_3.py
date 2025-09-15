import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def main():
    # Load a grayscale image
    # You can replace this with any image path
    image_path = 'noisy_image.jpg'
    
    # Read image in grayscale mode
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_img is None:
        # If image loading fails, create a sample image for demonstration
        print("Creating a sample image for demonstration...")
        original_img = create_sample_image()
    else:
        print(f"Successfully loaded image with shape: {original_img.shape}")
    
    # Task 1: Histogram and Histogram Equalization
    print("\n=== TASK 1: Histogram and Histogram Equalization ===")
    perform_histogram_analysis(original_img)
    
    # Task 2: Contrast Stretching
    print("\n=== TASK 2: Contrast Stretching ===")
    perform_contrast_stretching(original_img)
    
    # Task 3: Smoothing with Spatial Filters
    print("\n=== TASK 3: Smoothing with Spatial Filters ===")
    perform_smoothing(original_img)
    
    # Task 4: Sharpening with Laplacian Filter
    print("\n=== TASK 4: Sharpening with Laplacian Filter ===")
    perform_sharpening(original_img)

def create_sample_image():
    """Create a sample image with various features for demonstration"""
    # Create a 400x400 image
    size = 400
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Add a gradient background
    for i in range(size):
        img[i, :] = i * 255 // size
    
    # Add some shapes with different intensities
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)  # Bright rectangle
    cv2.circle(img, (300, 100), 50, 100, -1)  # Medium circle
    cv2.rectangle(img, (250, 250), (350, 350), 50, -1)  # Dark rectangle
    
    # Add some noise
    noise = np.random.normal(0, 15, (size, size)).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def perform_histogram_analysis(img):
    """Perform histogram analysis and equalization"""
    # Apply histogram equalization
    equalized_img = cv2.equalizeHist(img)
    
    # Calculate histograms
    hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
    
    # Display results
    plt.figure(figsize=(15, 10))
    plt.suptitle('Histogram Analysis and Equalization', fontsize=16, fontweight='bold')
    
    # Original image and histogram
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.plot(hist_original, color='black')
    plt.title('Original Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Equalized image and histogram
    plt.subplot(2, 2, 3)
    plt.imshow(equalized_img, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.plot(hist_equalized, color='black')
    plt.title('Equalized Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Discussion
    print("Histogram Equalization Effects:")
    print("- Redistributes pixel intensities to cover the full range [0, 255]")
    print("- Improves contrast by stretching the intensity distribution")
    print("- Particularly effective for images with poor contrast")
    print("- Can reveal details in dark and bright regions")
    print("- May sometimes over-enhance noise in the image")

def perform_contrast_stretching(img):
    """Perform linear contrast stretching"""
    # Find minimum and maximum intensity values
    min_val = np.min(img)
    max_val = np.max(img)
    
    print(f"Original image intensity range: [{min_val}, {max_val}]")
    
    # Apply contrast stretching
    # Formula: output = (input - min) * (255 / (max - min))
    stretched_img = ((img - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
    
    # Display results
    plt.figure(figsize=(12, 5))
    plt.suptitle('Contrast Stretching', fontsize=16, fontweight='bold')
    
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Original Image\nRange: [{min_val}, {max_val}]')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(stretched_img, cmap='gray')
    plt.title('Contrast Stretched Image\nRange: [0, 255]')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Discussion
    print("Contrast Stretching Effects:")
    print("- Linearly maps the original intensity range to the full [0, 255] range")
    print("- Improves image contrast by utilizing the complete dynamic range")
    print("- Simple but effective for images with narrow intensity distributions")
    print("- Does not change the relative distribution of intensities (unlike histogram equalization)")
    print("- Particularly useful for images where most pixels are clustered in a small range")

def perform_smoothing(img):
    """Apply averaging filters of different sizes"""
    # Apply 3x3 averaging filter
    kernel_3x3 = np.ones((3, 3), np.float32) / 9
    smoothed_3x3 = cv2.filter2D(img, -1, kernel_3x3)
    
    # Apply 5x5 averaging filter
    kernel_5x5 = np.ones((5, 5), np.float32) / 25
    smoothed_5x5 = cv2.filter2D(img, -1, kernel_5x5)
    
    # Display results
    plt.figure(figsize=(15, 5))
    plt.suptitle('Smoothing with Averaging Filters', fontsize=16, fontweight='bold')
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(smoothed_3x3, cmap='gray')
    plt.title('3x3 Averaging Filter')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(smoothed_5x5, cmap='gray')
    plt.title('5x5 Averaging Filter')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display kernel visualizations
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(kernel_3x3, cmap='viridis', interpolation='nearest')
    plt.title('3x3 Averaging Kernel')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(kernel_5x5, cmap='viridis', interpolation='nearest')
    plt.title('5x5 Averaging Kernel')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Discussion
    print("Smoothing Filter Effects:")
    print("- Averaging filters reduce noise by replacing each pixel with the average of its neighborhood")
    print("- Larger kernel sizes produce more blurring but better noise reduction")
    print("- 3x3 filter: Mild smoothing, preserves most details while reducing noise")
    print("- 5x5 filter: Strong smoothing, significantly reduces noise but also blurs edges and details")
    print("- Trade-off: Noise reduction vs. detail preservation")
    print("- Applications: Noise reduction, preprocessing for other operations")

def perform_sharpening(img):
    """Apply Laplacian filter for sharpening"""
    # Apply Gaussian blur first to reduce noise (optional but often done)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Convert to absolute values and scale to 0-255
    laplacian_abs = np.absolute(laplacian)
    laplacian_scaled = np.uint8(255 * laplacian_abs / np.max(laplacian_abs))
    
    # Sharpening: original + scaled Laplacian
    sharpened = cv2.addWeighted(img, 1.5, laplacian_scaled, -0.5, 0)
    
    # Display results
    plt.figure(figsize=(15, 10))
    plt.suptitle('Sharpening with Laplacian Filter', fontsize=16, fontweight='bold')
    
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(laplacian_scaled, cmap='gray')
    plt.title('Laplacian Filter Output (Edges)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(sharpened, cmap='gray')
    plt.title('Sharpened Image')
    plt.axis('off')
    
    # Display a zoomed-in region for better comparison
    h, w = img.shape
    roi = slice(h//3, 2*h//3), slice(w//3, 2*w//3)
    
    plt.subplot(2, 2, 4)
    plt.imshow(np.hstack([img[roi], sharpened[roi]]), cmap='gray')
    plt.title('Comparison: Original (Left) vs Sharpened (Right)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Discussion
    print("Laplacian Sharpening Effects:")
    print("- Laplacian is a second-derivative operator that highlights regions of rapid intensity change")
    print("- Sharpening enhances edges and fine details by adding the Laplacian to the original image")
    print("- Makes edges more pronounced and improves perceived clarity")
    print("- Can also amplify noise, so often combined with mild smoothing first")
    print("- Formula: Sharpened = Original + k * Laplacian (where k is negative)")
    print("- Applications: Enhancing medical images, improving text readability, general image enhancement")

if __name__ == "__main__":
    main()