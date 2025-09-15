import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------
# TASK 1: Load and Display Images
# ----------------------------

# Load images
# Replace 'your_color_image.jpg' and 'your_grayscale_image.jpg' with your actual file paths.
color_img = cv2.imread('color_image.jpg') # Load a color image
img = Image.open('grayscale_image.jpg').convert('L')
gray_img = np.array(img)
# gray_img = cv2.imread('grayscale_image.jpg', cv2.IMREAD_GRAYSCALE) # The '0' flag loads image in grayscale mode

# Check if images were loaded successfully
if color_img is None or gray_img is None:
    print("Error: Could not load images. Please check the file paths.")
    exit()

# Convert BGR (OpenCV's default format) to RGB for correct color display with Matplotlib
color_img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

# Create a figure to display images side by side
plt.figure(figsize=(12, 6))

# Display Grayscale Image
plt.subplot(1, 2, 1) # (rows, columns, index)
plt.imshow(gray_img, cmap='gray')
plt.title('Grayscale Image (1 Channel)')
plt.axis('off')

# Display Color Image
plt.subplot(1, 2, 2)
plt.imshow(color_img_rgb)
plt.title('Color Image (3 Channels)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Describe the differences
print("Grayscale vs. Color:")
print("- A grayscale image has only 1 intensity channel. Shape:", gray_img.shape)
print("- A color image has 3 channels (R, G, B). Shape:", color_img.shape)
print("- Pixel value at (0,0) for grayscale:", gray_img[0, 0])
print("- Pixel value at (0,0) for color (B,G,R format):", color_img[0, 0])

# ----------------------------
# TASK 2: Negative of an Image
# ----------------------------

# Generate the negative
# Since the image is 8-bit, the max intensity is 255.
negative_img = 255 - gray_img

# Display original and negative
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_img, cmap='gray')
plt.title('Negative Image')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nNegative Image:")
print("- The negative inverts the intensity values. Dark areas become bright and vice-versa.")
print("- This can help in visualizing details that are otherwise too dark to see.")

# ----------------------------
# TASK 3: Image Subsampling
# ----------------------------

def subsample_image(image, factor):
    """
    Downsample an image by a given factor.
    For a factor of n, it keeps every n-th row and every n-th column.
    """
    # Use array slicing to select every 'factor'-th row and column
    subsampled = image[::factor, ::factor]
    return subsampled

# Perform subsampling
factors = [2, 4, 8]
subsampled_imgs = [subsample_image(gray_img, f) for f in factors]

# Display the results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(gray_img, cmap='gray')
plt.title(f'Original Image\nSize: {gray_img.shape}')
plt.axis('off')

# Subsampled Images
for i, (img, factor) in enumerate(zip(subsampled_imgs, factors)):
    plt.subplot(2, 2, i+2) # Plot in positions 2, 3, and 4
    plt.imshow(img, cmap='gray')
    plt.title(f'Subsampling Factor: {factor}\nSize: {img.shape}')
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\nSubsampling Effects:")
print("- As the subsampling factor increases, the image resolution (size) decreases.")
print("- Fine details and textures are lost, and the image becomes more blocky/pixelated.")
print("- Edges may appear jagged (aliasing). High factors can make the image unrecognizable.")

# This line keeps all the plots open until you close them
plt.show()

print(gray_img.dtype)  # Should print uint8