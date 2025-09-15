import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread("rony.jpg", cv2.IMREAD_GRAYSCALE)
negative_img = 255 - gray_img

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.title("Original");
plt.imshow(gray_img, cmap='gray');
plt.axis('off')

plt.subplot(1, 2, 2);
plt.title("Negative");
plt.imshow(negative_img, cmap='gray');
plt.axis('off')
plt.savefig('task2_negative.png');
plt.show()
