import cv2
import matplotlib.pyplot as plt
import numpy as np

gray_img = cv2.imread("../gray_image.jpg", cv2.IMREAD_GRAYSCALE)
f = np.fft.fftshift(np.fft.fft2(gray_img))
magnitude_spectrum = 20 * np.log(np.abs(f)+1)

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1); plt.title("Original"); plt.imshow(gray_img, cmap='gray'); plt.axis('off')
plt.subplot(1,2,2); plt.title("Magnitude Spectrum"); plt.imshow(magnitude_spectrum, cmap='gray'); plt.axis('off')
plt.savefig('task10_fourier.png'); plt.show()
