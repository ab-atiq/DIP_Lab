import cv2
import matplotlib.pyplot as plt
import numpy as np

gray_img = cv2.imread("rony.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(12,6))
for i, levels in enumerate([2, 4, 8, 16,32, 256], 1):
    q_img = (np.floor(gray_img / (256/levels)) * (256/levels)).astype(np.uint8)
    plt.subplot(1, 6, i); plt.title(f"{levels} levels"); plt.imshow(q_img, cmap='gray'); plt.axis('off')
plt.savefig('task4_quantization.png'); plt.show()
