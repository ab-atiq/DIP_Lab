import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread("rony.jpg", cv2.IMREAD_GRAYSCALE)
color_img = cv2.cvtColor(cv2.imread("rony.jpg"), cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.title("Grayscale"); plt.imshow(gray_img, cmap='gray'); plt.axis('off')
plt.subplot(1, 2, 2); plt.title("Color"); plt.imshow(color_img); plt.axis('off')
plt.savefig('task1_load_display.png'); plt.show()
