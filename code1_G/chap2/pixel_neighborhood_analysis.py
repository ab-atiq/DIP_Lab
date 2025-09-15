import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def pixel_neighborhood_analysis(image_path, x, y):
    """
    Analyzes and displays the 4-neighbors and 8-neighbors of a chosen pixel.
    """
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    height, width = img_array.shape

    # Check if the chosen pixel is on the boundary
    if not (0 < x < width - 1 and 0 < y < height - 1):
        print(f"Chosen pixel ({x}, {y}) is on the boundary. Choosing a new, valid pixel.")
        x, y = width // 2, height // 2 # A safer default
        print(f"Using new pixel coordinates: ({x}, {y})")

    # The chosen pixel and its value
    pixel_value = img_array[y, x]

    # Get 4-neighbors and their values
    n4_coords = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
    n4_values = [img_array[ny, nx] for nx, ny in n4_coords]
    
    # Get 8-neighbors and their values
    n8_coords = [
        (x-1, y-1), (x, y-1), (x+1, y-1),  # Top row
        (x-1, y),   (x, y),   (x+1, y),    # Middle row
        (x-1, y+1), (x, y+1), (x+1, y+1)   # Bottom row
    ]
    n8_values = [img_array[ny, nx] for nx, ny in n8_coords]

    # Print the analysis
    print(f"--- Pixel Neighborhood Analysis for pixel at ({x}, {y}) ---")
    print(f"Central Pixel Value: {pixel_value}")
    
    print("\n4-Neighbors (Top, Bottom, Left, Right):")
    for (nx, ny), val in zip(n4_coords, n4_values):
        print(f"  - ({nx}, {ny}): Value = {val}")

    print("\n8-Neighbors:")
    print("  (Top-Left, Top, Top-Right)")
    print("  (Left,     Center,   Right)")
    print("  (Bot-Left, Bot, Bot-Right)")
    n8_formatted_values = np.array(n8_values).reshape(3, 3)
    print(n8_formatted_values)

    # Explanation of concepts
    print("\n--- Importance of Pixel Neighborhoods ---")
    print("The concept of a pixel neighborhood is crucial because it allows us to analyze an image based on local relationships, not just individual pixels.")
    print("Most image processing operations, such as filtering, convolution, and transformations, operate on a pixel's neighborhood to determine its new value.")
    print("For example:")
    print("- **Blurring** uses the average or median of a pixel's neighbors to create a new value, smoothing out sharp transitions.")
    print("- **Edge detection** looks for large differences between a pixel and its neighbors to identify boundaries.")
    print("This local analysis is the foundation of many powerful image processing algorithms.")

    # Visualization
    # Create a small sub-image to display the neighborhood
    y_start, y_end = max(0, y - 2), min(height, y + 3)
    x_start, x_end = max(0, x - 2), min(width, x + 3)
    
    neighborhood_patch = img_array[y_start:y_end, x_start:x_end]

    plt.figure(figsize=(6, 6))
    plt.imshow(neighborhood_patch, cmap='gray')
    plt.title(f"Neighborhood around pixel ({x}, {y})")
    
    # Highlight the center pixel
    center_y = y - y_start
    center_x = x - x_start
    plt.scatter(center_x, center_y, color='red', s=100, label='Central Pixel', marker='o')
    plt.text(center_x + 0.5, center_y, f'({pixel_value})', color='red', fontsize=12, verticalalignment='center')
    # make neibor pixel yellow
    for (nx, ny), val in zip(n8_coords, n8_values):
        if x_start <= nx < x_end and y_start <= ny < y_end:
            plt.scatter(nx - x_start, ny - y_start, color='yellow', s=100, label='Neighbor Pixel', marker='x')
            plt.text(nx - x_start + 0.5, ny - y_start, f'({val})', color='yellow', fontsize=12, verticalalignment='center')
    
    plt.axis('off')
    plt.legend()
    plt.show()

# Example usage:
# Adjust the image path and the chosen pixel coordinates (x, y)
pixel_neighborhood_analysis('grayscale_image.jpg', x=200, y=150)