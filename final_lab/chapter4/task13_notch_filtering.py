import cv2
import matplotlib.pyplot as plt
import numpy as np

gray_img = cv2.imread("../gray_image.jpg", cv2.IMREAD_GRAYSCALE)
rows, cols = gray_img.shape
x, y = np.meshgrid(np.arange(cols), np.arange(rows))
noisy = gray_img + 50*np.sin(0.1*x)

f_noisy = np.fft.fftshift(np.fft.fft2(noisy))
notch_filter = np.ones_like(noisy)
notch_filter[rows//2-5:rows//2+5, cols//2+30:cols//2+40] = 0
notch_filter[rows//2-5:rows//2+5, cols//2-40:cols//2-30] = 0
img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(f_noisy * notch_filter)))

plt.figure(figsize=(15, 5))
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(gray_img, cmap='gray'); plt.axis('off')
plt.subplot(1,3,2); plt.title("Noisy"); plt.imshow(noisy, cmap='gray'); plt.axis('off')
plt.subplot(1,3,3); plt.title("Filtered"); plt.imshow(img_filtered, cmap='gray'); plt.axis('off')
plt.savefig('task13_notch_filtering.png'); plt.show()
