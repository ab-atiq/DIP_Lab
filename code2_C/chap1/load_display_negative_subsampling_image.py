from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and Display Images
# -----------------------------
# Load grayscale image
img = Image.open('grayscale_image.jpg').convert('L')
gray_img = np.array(img)

# Load color image
color_img = cv2.imread("color_image.jpg", cv2.IMREAD_COLOR)
# Convert BGR (OpenCV default) to RGB for correct display in matplotlib
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Grayscale Image")
plt.imshow(gray_img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Color Image")
plt.imshow(color_img)
plt.axis("off")

plt.show()

# -----------------------------
# 2. Negative of Image
# -----------------------------
negative_img = 255 - gray_img

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Grayscale")
plt.imshow(gray_img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Negative Image")
plt.imshow(negative_img, cmap='gray')
plt.axis("off")

plt.show()

# -----------------------------
# 3. Image Subsampling
# -----------------------------
# Downsample by factors
down_2 = gray_img[::2, ::2]   # every 2nd pixel
down_4 = gray_img[::4, ::4]   # every 4th pixel
down_8 = gray_img[::8, ::8]   # every 8th pixel

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(gray_img, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Downsample x2")
plt.imshow(down_2, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Downsample x4")
plt.imshow(down_4, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Downsample x8")
plt.imshow(down_8, cmap='gray')
plt.axis("off")

plt.show()
