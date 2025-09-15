import numpy as np
import cv2

# Create a simple grayscale image
gray_img = np.zeros((200, 200), dtype=np.uint8)
gray_img[50:150, 50:150] = 128  # Gray square
gray_img[75:125, 75:125] = 255  # White square inside
cv2.imwrite("gray_image.jpg", gray_img)

# Create a simple color image
color_img = np.zeros((200, 200, 3), dtype=np.uint8)
color_img[50:150, 50:150] = [255, 0, 0]  # Red square
color_img[75:125, 75:125] = [0, 255, 0]  # Green square inside
cv2.imwrite("color_image.jpg", color_img)

print("Sample images created successfully!")
