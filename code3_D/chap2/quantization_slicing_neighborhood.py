import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load a grayscale image
    # You can replace this with any grayscale image path
    image_path = 'color_image.jpg'
    
    # Read image in grayscale mode
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_img is None:
        # If image loading fails, create a sample image for demonstration
        print("Creating a sample image for demonstration...")
        original_img = create_sample_image()
    else:
        print(f"Successfully loaded image with shape: {original_img.shape}")
    
    # Task 1: Gray-Level Quantization
    print("\n=== TASK 1: Gray-Level Quantization ===")
    perform_quantization(original_img)
    
    # Task 2: Bit-Plane Slicing
    print("\n=== TASK 2: Bit-Plane Slicing ===")
    perform_bit_plane_slicing(original_img)
    
    # Task 3: Pixel Neighborhood Analysis
    print("\n=== TASK 3: Pixel Neighborhood Analysis ===")
    perform_neighborhood_analysis(original_img)

def create_sample_image():
    """Create a sample image with gradients and patterns for demonstration"""
    # Create a 300x300 image with a gradient
    x = np.linspace(0, 255, 300)
    y = np.linspace(0, 255, 300)
    xx, yy = np.meshgrid(x, y)
    
    # Create gradient in one direction
    gradient = xx.astype(np.uint8)
    
    # Add some patterns
    center = (150, 150)
    y_coords, x_coords = np.indices((300, 300))
    distance = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)
    circle = (distance < 100).astype(np.uint8) * 100
    
    # Combine gradient and circle
    sample_img = np.clip(gradient + circle, 0, 255).astype(np.uint8)
    
    return sample_img

def perform_quantization(img):
    """Reduce image to different gray levels and display results"""
    levels = [2, 4, 8, 16, 256]
    quantized_images = []
    
    for level in levels:
        if level == 256:
            # Original image (no quantization)
            quantized = img.copy()
        else:
            # Calculate quantization factor
            factor = 256 / level
            # Quantize the image
            quantized = (np.floor(img / factor) * factor).astype(np.uint8)
        
        quantized_images.append(quantized)
    
    # Display results
    plt.figure(figsize=(15, 8))
    plt.suptitle('Gray-Level Quantization', fontsize=16, fontweight='bold')
    
    for i, (quantized_img, level) in enumerate(zip(quantized_images, levels)):
        plt.subplot(2, 3, i+1)
        plt.imshow(quantized_img, cmap='gray')
        plt.title(f'{level} Gray Levels')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Discussion
    print("Quantization Effects:")
    print("- Reducing gray levels decreases the number of available intensity values")
    print("- This leads to loss of smooth gradients and creates 'false contours' or 'banding'")
    print("- With fewer levels (2-4), the image becomes highly posterized")
    print("- Critical details may be lost in areas with subtle intensity variations")
    print("- The effect is most noticeable in smooth gradient regions")

def perform_bit_plane_slicing(img):
    """Extract and display all 8 bit-planes of the image"""
    # Ensure the image is 8-bit
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    # Extract each bit-plane
    bit_planes = []
    for bit in range(7, -1, -1):  # From MSB (bit 7) to LSB (bit 0)
        # Create the bit-plane by isolating the specific bit
        plane = ((img >> bit) & 1) * 255
        bit_planes.append(plane)
    
    # Display results
    plt.figure(figsize=(15, 10))
    plt.suptitle('Bit-Plane Slicing', fontsize=16, fontweight='bold')
    
    titles = [
        'Bit 7 (MSB) - Most Significant',
        'Bit 6',
        'Bit 5',
        'Bit 4',
        'Bit 3',
        'Bit 2',
        'Bit 1',
        'Bit 0 (LSB) - Least Significant'
    ]
    
    for i, (plane, title) in enumerate(zip(bit_planes, titles)):
        plt.subplot(3, 3, i+1)
        plt.imshow(plane, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    # Add original image for comparison
    plt.subplot(3, 3, 9)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Discussion
    print("Bit-Plane Analysis:")
    print("- MSB (Bits 7-6): Contain the majority of visual information")
    print("- Middle Bits (Bits 5-3): Contain finer details and textures")
    print("- LSB (Bits 2-0): Contain noise and subtle variations")
    print("- The first 2-3 bits (MSBs) contribute about 87.5% of the image information")
    print("- Bit-plane slicing is useful for image compression and watermarking")

def perform_neighborhood_analysis(img):
    """Analyze pixel neighborhoods and their importance"""
    # Choose a pixel not on the border
    height, width = img.shape
    y, x = min(50, height-2), min(50, width-2)  # Ensure we're not on the border
    
    print(f"Analyzing pixel at position (y, x) = ({y}, {x})")
    print(f"Pixel intensity value: {img[y, x]}")
    
    # Get 4-neighbors (top, bottom, left, right)
    top = img[y-1, x]
    bottom = img[y+1, x]
    left = img[y, x-1]
    right = img[y, x+1]
    
    print("\n4-Neighbors (N4):")
    print(f"  Top: ({y-1}, {x}) = {top}")
    print(f"  Bottom: ({y+1}, {x}) = {bottom}")
    print(f"  Left: ({y}, {x-1}) = {left}")
    print(f"  Right: ({y}, {x+1}) = {right}")
    
    # Get 8-neighbors (includes diagonals)
    top_left = img[y-1, x-1]
    top_right = img[y-1, x+1]
    bottom_left = img[y+1, x-1]
    bottom_right = img[y+1, x+1]
    
    print("\n8-Neighbors (N8):")
    print(f"  Top-Left: ({y-1}, {x-1}) = {top_left}")
    print(f"  Top-Right: ({y-1}, {x+1}) = {top_right}")
    print(f"  Bottom-Left: ({y+1}, {x-1}) = {bottom_left}")
    print(f"  Bottom-Right: ({y+1}, {x+1}) = {bottom_right}")
    
    # Visualize the neighborhood
    visualize_neighborhood(img, y, x)
    
    # Discussion
    print("\nImportance of Pixel Neighborhoods:")
    print("1. Fundamental for spatial filtering operations:")
    print("   - Smoothing filters (blurring) average neighborhood values")
    print("   - Sharpening filters enhance differences from neighbors")
    print("   - Median filters replace pixel with median of neighborhood")
    print("2. Essential for edge detection:")
    print("   - Edge detectors (Sobel, Prewitt) compute gradients across neighborhoods")
    print("   - Laplacian operators measure second-order derivatives in neighborhoods")
    print("3. Crucial for morphological operations:")
    print("   - Dilation expands bright regions based on neighborhood")
    print("   - Erosion shrinks bright regions based on neighborhood")
    print("4. Key for texture analysis and feature extraction")

def visualize_neighborhood(img, y, x):
    """Create a visualization of the pixel neighborhood"""
    # Extract a 3x3 region around the pixel
    region = img[y-1:y+2, x-1:x+2].copy()
    
    # Create a color version for highlighting
    if len(img.shape) == 2:  # Grayscale
        region_color = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
    else:
        region_color = region.copy()
    
    # Highlight the center pixel in red
    region_color[1, 1] = [0, 0, 255]  # Red center
    
    # Highlight 4-neighbors in green
    region_color[0, 1] = [0, 255, 0]  # Top (green)
    region_color[2, 1] = [0, 255, 0]  # Bottom (green)
    region_color[1, 0] = [0, 255, 0]  # Left (green)
    region_color[1, 2] = [0, 255, 0]  # Right (green)
    
    # Highlight 8-neighbors (diagonals) in blue
    region_color[0, 0] = [255, 0, 0]  # Top-left (blue)
    region_color[0, 2] = [255, 0, 0]  # Top-right (blue)
    region_color[2, 0] = [255, 0, 0]  # Bottom-left (blue)
    region_color[2, 2] = [255, 0, 0]  # Bottom-right (blue)
    
    # Display the neighborhood
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.plot(x, y, 'ro', markersize=10)  # Mark the analyzed pixel
    plt.title('Image with Analyzed Pixel')
    plt.axis('on')
    
    plt.subplot(1, 2, 2)
    plt.imshow(region_color)
    plt.title('3x3 Neighborhood\n(Red=Center, Green=N4, Blue=N8)')
    
    # Add value annotations
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(region[i, j]), 
                    ha='center', va='center', 
                    color='white' if region[i, j] < 128 else 'black',
                    fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()