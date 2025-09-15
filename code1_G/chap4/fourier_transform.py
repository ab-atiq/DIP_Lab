import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def fourier_transform(image_path):
    """
    Computes and displays the 2D Fourier Transform magnitude spectrum.
    """
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
        
    img_array = np.array(img, dtype=np.float64)

    # 1. Compute the 2D Fourier Transform
    f_transform = np.fft.fft2(img_array)

    # 2. Shift the zero-frequency component to the center of the spectrum
    f_shift = np.fft.fftshift(f_transform)

    # 3. Calculate the magnitude spectrum and apply logarithmic scaling
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))

    # Display the original image and its magnitude spectrum
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Grayscale Image')
    axes[0].axis('off')

    axes[1].imshow(magnitude_spectrum, cmap='gray')
    axes[1].set_title('Magnitude Spectrum (Log-Scaled)')
    axes[1].axis('off')

    plt.show()

    # Explanation
    print("--- Analysis of the Magnitude Spectrum ---")
    print("The center of the spectrum represents low frequencies, corresponding to the smooth, large-scale structures and overall brightness of the image.")
    print("The outer parts of the spectrum represent high frequencies, corresponding to edges, textures, and noise.")
    print("A sharp image will have energy spread out towards the edges, while a blurry image will have a more concentrated center.")

# Example usage:
fourier_transform('grayscale_image.jpg')