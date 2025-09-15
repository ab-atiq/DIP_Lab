import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift

def create_periodic_noise(M, N, A, u0, v0):
    """Generates a sinusoidal pattern for periodic noise."""
    x = np.arange(M)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)
    noise = A * np.sin(2 * np.pi * (u0 * X / M + v0 * Y / N))
    return noise.T  # Transpose to match image shape

def apply_notch_filter(image_path):
    """
    Applies a notch filter to remove periodic noise.
    """
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
        
    img_array = np.array(img, dtype=np.float64)
    M, N = img_array.shape

    # 1. Add synthetic periodic noise
    # Adjust amplitude (A) and frequency (u0, v0) for desired effect
    noise_freq_x, noise_freq_y = 50, 20
    noise = create_periodic_noise(M, N, 20, noise_freq_x, noise_freq_y)
    noisy_img_array = img_array + noise
    
    # 2. Compute the Fourier Transform of the noisy image
    f_transform_noisy = fft2(noisy_img_array)
    f_shift_noisy = fftshift(f_transform_noisy)
    magnitude_spectrum_noisy = 20 * np.log(np.abs(f_shift_noisy) + 1)

    # 3. Create the Notch Filter mask
    H = np.ones((M, N))
    
    # Coordinates of noise spikes (and their symmetric counterparts)
    # The spikes are at (P-u0, Q-v0) and (P+u0, Q+v0) in the shifted spectrum.
    P, Q = M // 2, N // 2
    
    # Define a small region around each spike to zero out
    notch_radius = 5
    
    # Notch out the positive frequency spike
    for u in range(M):
        for v in range(N):
            dist1 = np.sqrt((u - (P + noise_freq_x))**2 + (v - (Q + noise_freq_y))**2)
            dist2 = np.sqrt((u - (P - noise_freq_x))**2 + (v - (Q - noise_freq_y))**2)
            if dist1 < notch_radius or dist2 < notch_radius:
                H[u, v] = 0

    # 4. Apply the filter
    f_shift_filtered = f_shift_noisy * H

    # 5. Perform the Inverse Fourier Transform
    f_inverse_shift = ifftshift(f_shift_filtered)
    img_filtered = ifft2(f_inverse_shift)
    img_filtered = np.abs(img_filtered)

    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(noisy_img_array, cmap='gray')
    axes[0, 0].set_title('Image with Periodic Noise')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(magnitude_spectrum_noisy, cmap='gray')
    axes[0, 1].set_title('Spectrum of Noisy Image')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(img_filtered, cmap='gray')
    axes[1, 0].set_title('Filtered Image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(20 * np.log(np.abs(f_shift_filtered)+1), cmap='gray')
    axes[1, 1].set_title('Filtered Spectrum')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    # Discussion
    print("--- Notch Filtering for Periodic Noise ---")
    print("Periodic noise in the spatial domain appears as distinct, symmetrical bright spots in the frequency spectrum.")
    print("A notch filter works by creating 'notches' or zeros in the frequency spectrum at the exact locations of these noise spikes.")
    print("By setting these specific frequency components to zero, the filter effectively removes the periodic pattern when the image is transformed back to the spatial domain.")
    print("This is a highly precise method for removing specific types of noise without affecting the rest of the image content significantly.")

# Example usage:
apply_notch_filter('grayscale_image.jpg')