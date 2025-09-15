from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load grayscale image using PIL
img_path = "grayscale_image.jpg"
img = Image.open(img_path).convert('L')
img = np.array(img)

if img.size == 0:
    raise ValueError("Image not loaded correctly or is empty.")

# -----------------------------
# 1. Fourier Transform & Spectrum
# -----------------------------
# Compute DFT
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)   # shift low-freq to center
magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Magnitude Spectrum (log scale)")
plt.axis("off")
plt.show()

# -----------------------------
# 2. Low-Pass Filtering (ILPF)
# -----------------------------
rows, cols = img.shape
crow, ccol = rows//2, cols//2
D0 = 30   # cutoff radius

# Create Ideal LPF mask
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), D0, 1, -1)

# Apply filter
lpf = dft_shift * mask
img_lpf = np.fft.ifft2(np.fft.ifftshift(lpf))
img_lpf = np.abs(img_lpf)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_lpf, cmap="gray")
plt.title("Low-Pass Filtered (Blurred)")
plt.axis("off")
plt.show()

# -----------------------------
# 3. High-Pass Filtering (IHPF)
# -----------------------------
# High-pass = 1 - LPF
mask_hpf = 1 - mask
hpf = dft_shift * mask_hpf
img_hpf = np.fft.ifft2(np.fft.ifftshift(hpf))
img_hpf = np.abs(img_hpf)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_hpf, cmap="gray")
plt.title("High-Pass Filtered (Edges)")
plt.axis("off")
plt.show()

# -----------------------------
# 4. Notch Filtering for Periodic Noise
# -----------------------------
# Add synthetic sinusoidal noise
rows, cols = img.shape
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)
sinusoidal_noise = 50 * np.sin(2*np.pi*X/15)
noisy_img = img + sinusoidal_noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# Fourier transform of noisy image
dft_noisy = np.fft.fft2(noisy_img)
dft_noisy_shift = np.fft.fftshift(dft_noisy)

# Design notch filter (remove specific frequencies)
mask_notch = np.ones((rows, cols), np.uint8)
# Example: block small regions at known noise frequencies
mask_notch[crow-5:crow+5, ccol+15-5:ccol+15+5] = 0
mask_notch[crow-5:crow+5, ccol-15-5:ccol-15+5] = 0

# Apply notch filter
filtered_dft = dft_noisy_shift * mask_notch
img_notch = np.fft.ifft2(np.fft.ifftshift(filtered_dft))
img_notch = np.abs(img_notch)

plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(noisy_img, cmap="gray")
plt.title("Noisy (Periodic)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img_notch, cmap="gray")
plt.title("After Notch Filtering")
plt.axis("off")
plt.show()
