import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread("rony.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(15, 5))
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(gray_img, cmap='gray'); plt.axis('off')
plt.subplot(1,3,2); plt.title("3x3 Avg"); plt.imshow(cv2.blur(gray_img, (3,3)), cmap='gray'); plt.axis('off')
plt.subplot(1,3,3); plt.title("5x5 Avg"); plt.imshow(cv2.blur(gray_img, (5,5)), cmap='gray'); plt.axis('off')
plt.savefig('task8_smoothing.png'); plt.show()
