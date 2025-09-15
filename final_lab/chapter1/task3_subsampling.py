import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread("rony.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(8,8))
plt.subplot(2,2,1); plt.title("Original"); plt.imshow(gray_img, cmap='gray'); plt.axis('off')
plt.subplot(2,2,2); plt.title("2x"); plt.imshow(gray_img[::2, ::2], cmap='gray'); plt.axis('off')
plt.subplot(2,2,3); plt.title("4x"); plt.imshow(gray_img[::4, ::4], cmap='gray'); plt.axis('off')
plt.subplot(2,2,4); plt.title("8x"); plt.imshow(gray_img[::8, ::8], cmap='gray'); plt.axis('off')
plt.savefig('task3_subsampling.png'); plt.show()
