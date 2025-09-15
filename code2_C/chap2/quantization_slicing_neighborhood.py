from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load grayscale image using PIL
img_path = "grayscale_image.jpg"  # Update this if your image is elsewhere
try:
    img = Image.open(img_path).convert('L')
    gray_img = np.array(img)
except Exception as e:
    raise FileNotFoundError(f"Error: Could not load image at '{img_path}'. {e}")

# -----------------------------
# 1. Gray-Level Quantization
# -----------------------------
def quantize_image(image, levels):
    step = 256 // levels               # quantization step
    return (image // step) * step      # reduce levels

levels = [2, 4, 8, 16, 256]
quantized_imgs = [quantize_image(gray_img, L) for L in levels]

plt.figure(figsize=(12, 6))
for i, (L, img) in enumerate(zip(levels, quantized_imgs)):
    plt.subplot(1, len(levels), i+1)
    plt.imshow(img, cmap="gray")
    plt.title(f"{L} levels")
    plt.axis("off")
plt.suptitle("Gray-Level Quantization")
plt.show()

# -----------------------------
# 2. Bit-Plane Slicing
# -----------------------------
bit_planes = []
for i in range(8):  # Extract each bit (0=LSB, 7=MSB)
    plane = (gray_img >> i) & 1
    plane = plane * 255  # scale to visible intensity
    bit_planes.append(plane)

plt.figure(figsize=(12, 6))
for i, plane in enumerate(reversed(bit_planes)):  # Show MSB first
    plt.subplot(2, 4, i+1)
    plt.imshow(plane, cmap="gray")
    plt.title(f"Bit {7-i}")
    plt.axis("off")
plt.suptitle("Bit-Plane Slicing")
plt.show()

# -----------------------------
# 3. Pixel Neighborhood Analysis
# -----------------------------
# Choose a pixel (center of image)
x, y = gray_img.shape[0]//2, gray_img.shape[1]//2
pixel_value = gray_img[x, y]

# 4-neighbors (up, down, left, right)
neighbors_4 = [
    gray_img[x-1, y],   # top
    gray_img[x+1, y],   # bottom
    gray_img[x, y-1],   # left
    gray_img[x, y+1]    # right
]

# 8-neighbors (includes diagonals)
neighbors_8 = neighbors_4 + [
    gray_img[x-1, y-1], # top-left
    gray_img[x-1, y+1], # top-right
    gray_img[x+1, y-1], # bottom-left
    gray_img[x+1, y+1]  # bottom-right
]

print("Pixel location:", (x, y))
print("Pixel value:", pixel_value)
print("4-neighbors:", neighbors_4)
print("8-neighbors:", neighbors_8)

plt.figure(figsize=(6, 6))
plt.imshow(gray_img, cmap='gray')
ax = plt.gca()

# Highlight center pixel
rect = patches.Rectangle((y-0.5, x-0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)

# Highlight 8 neighbors
neighbor_coords = [
    (x-1, y),   # top
    (x+1, y),   # bottom
    (x, y-1),   # left
    (x, y+1),   # right
    (x-1, y-1), # top-left
    (x-1, y+1), # top-right
    (x+1, y-1), # bottom-left
    (x+1, y+1)  # bottom-right
]
for nx, ny in neighbor_coords:
    rect = patches.Rectangle((ny-0.5, nx-0.5), 1, 1, linewidth=2, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect)

plt.title("Center Pixel (red) and Neighbors (yellow)")
plt.axis("off")
plt.show()
