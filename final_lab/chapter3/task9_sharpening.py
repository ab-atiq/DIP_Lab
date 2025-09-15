import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread("rony.jpg", cv2.IMREAD_GRAYSCALE)
laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
sharpened = cv2.convertScaleAbs(gray_img - laplacian)

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1); plt.title("Original"); plt.imshow(gray_img, cmap='gray'); plt.axis('off')
plt.subplot(1,2,2); plt.title("Sharpened"); plt.imshow(sharpened, cmap='gray'); plt.axis('off')
plt.savefig('task9_sharpening.png'); plt.show()
