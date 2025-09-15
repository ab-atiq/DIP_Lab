import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread("rony.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(12,6))
for i in range(8):
    plt.subplot(2,4,i+1); plt.title(f"Bit {7-i}")
    plt.imshow(((gray_img >> (7-i)) & 1)*255, cmap='gray'); plt.axis('off')
plt.savefig('task5_bitplane.png'); plt.show()
