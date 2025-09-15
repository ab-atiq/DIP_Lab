import cv2
import matplotlib.pyplot as plt
import numpy as np

gray_img = cv2.imread("rony.jpg", cv2.IMREAD_GRAYSCALE)
stretched = ((gray_img - gray_img.min()) / (gray_img.max() - gray_img.min()) * 255).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1); plt.title("Original"); plt.imshow(gray_img, cmap='gray'); plt.axis('off')
plt.subplot(1,2,2); plt.title("Stretched"); plt.imshow(stretched, cmap='gray'); plt.axis('off')
plt.savefig('task7_contrast_stretching.png'); plt.show()
