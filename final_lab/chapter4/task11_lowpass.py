import cv2
import matplotlib.pyplot as plt
import numpy as np

gray_img = cv2.imread("../gray_image.jpg", cv2.IMREAD_GRAYSCALE)
f = np.fft.fftshift(np.fft.fft2(gray_img))
rows, cols = gray_img.shape
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (cols//2, rows//2), 30, 1, -1)
img_low = np.abs(np.fft.ifft2(np.fft.ifftshift(f * mask)))

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1); plt.title("Original"); plt.imshow(gray_img, cmap='gray'); plt.axis('off')
plt.subplot(1,2,2); plt.title("Low-pass Filtered"); plt.imshow(img_low, cmap='gray'); plt.axis('off')
plt.savefig('task11_lowpass.png'); plt.show()
