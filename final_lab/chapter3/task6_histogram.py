import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread("../gray_image.jpg", cv2.IMREAD_GRAYSCALE)
equalized = cv2.equalizeHist(gray_img)

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1); plt.hist(gray_img.ravel(), bins=256); plt.title("Original Histogram")
plt.subplot(1,2,2); plt.hist(equalized.ravel(), bins=256); plt.title("Equalized Histogram")
plt.savefig('task6_histogram.png'); plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1); plt.title("Original"); plt.imshow(gray_img, cmap='gray'); plt.axis('off')
plt.subplot(1,2,2); plt.title("Equalized"); plt.imshow(equalized, cmap='gray'); plt.axis('off')
plt.savefig('task6_equalization.png'); plt.show()
