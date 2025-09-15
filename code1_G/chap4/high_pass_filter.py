import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def apply_ihpf(image_path, D0):
    """
    Applies an Ideal High-Pass Filter (IHPF) to an image.
    """
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
        
    img_array = np.array(img, dtype=np.float64)
    M, N = img_array.shape

    # 1. Compute the 2D Fourier Transform
    f_transform = np.fft.fft2(img_array)
    f_shift = np.fft.fftshift(f_transform)

    # 2. Create the Ideal High-Pass Filter mask
    P = M // 2
    Q = N // 2
    H = np.ones((M, N))
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - P)**2 + (v - Q)**2)
            if D <= D0:
                H[u, v] = 0

    # 3. Apply the filter
    f_shift_filtered = f_shift * H

    # 4. Perform the Inverse Fourier Transform
    f_inverse_shift = np.fft.ifftshift(f_shift_filtered)
    img_filtered = np.fft.ifft2(f_inverse_shift)
    img_filtered = np.abs(img_filtered)

    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(img_filtered, cmap='gray')
    axes[1].set_title(f'High-Pass Filtered (D0={D0})')
    axes[1].axis('off')

    plt.show()

    # Discussion
    print("--- Effect of High-Pass Filtering ---")
    print("High-pass filtering removes low-frequency components, which are related to the general brightness and smooth areas of the image.")
    print("The result is an image that only contains high-frequency information, such as edges and fine details.")
    print("This process effectively enhances or 'sharpens' the image, but it can also amplify noise, which is also a high-frequency component.")

# Example usage: Choose a suitable cutoff frequency (e.g., 20)
apply_ihpf('grayscale_image.jpg', D0=20)