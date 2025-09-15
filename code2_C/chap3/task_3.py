from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image using PIL
img_path = "grayscale_image.jpg"  # Update this if your image is elsewhere
try:
    img = Image.open(img_path).convert('L')
    gray_img = np.array(img)
except Exception as e:
    raise FileNotFoundError(f"Error: Could not load image at '{img_path}'. {e}")

# -----------------------------
# 1. Histogram & Equalization
# -----------------------------
# Compute histogram
hist, _ = np.histogram(gray_img.flatten(), bins=256, range=[0,256])

# Histogram equalization
hist_eq_img = gray_img.copy().flatten()
hist_eq_img = np.interp(hist_eq_img, np.arange(256), np.cumsum(hist)/np.sum(hist)*255)
equalized_img = hist_eq_img.reshape(gray_img.shape).astype(np.uint8)
hist_eq, _ = np.histogram(equalized_img.flatten(), bins=256, range=[0,256])

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.imshow(gray_img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(equalized_img, cmap='gray')
plt.title("Equalized Image")
plt.axis("off")

plt.subplot(2,2,3)
plt.plot(hist, color='black')
plt.title("Original Histogram")

plt.subplot(2,2,4)
plt.plot(hist_eq, color='black')
plt.title("Equalized Histogram")
plt.show()

# -----------------------------
# 2. Contrast Stretching
# -----------------------------
min_val = np.min(gray_img)
max_val = np.max(gray_img)
stretched_img = ((gray_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(gray_img, cmap='gray')
plt.title("Original (Low Contrast)")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(stretched_img, cmap='gray')
plt.title("Contrast Stretched")
plt.axis("off")
plt.show()

# -----------------------------
# 3. Smoothing with Spatial Filters
# -----------------------------
def average_filter(img, ksize):
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    return np.clip(np.round(np.convolve(img.flatten(), kernel.flatten(), mode='same').reshape(img.shape)), 0, 255).astype(np.uint8)

avg3 = average_filter(gray_img, 3)
avg5 = average_filter(gray_img, 5)

plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(gray_img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(avg3, cmap='gray')
plt.title("3x3 Averaging")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(avg5, cmap='gray')
plt.title("5x5 Averaging")
plt.axis("off")
plt.show()

# -----------------------------
# 4. Sharpening with Laplacian Filter
# -----------------------------
laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
laplacian = np.clip(np.round(np.convolve(gray_img.flatten(), laplacian_kernel.flatten(), mode='same').reshape(gray_img.shape)), 0, 255).astype(np.uint8)

# Sharpened = original + Laplacian
sharpened = np.clip(gray_img + laplacian, 0, 255).astype(np.uint8)

plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(gray_img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian (Edges)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(sharpened, cmap='gray')
plt.title("Sharpened Image")
plt.axis("off")
plt.show()
