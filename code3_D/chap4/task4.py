import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def main():
    # Load a grayscale image
    image_path = 'grayscale_image.jpg'
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_img is None:
        print("Creating a sample image for demonstration...")
        original_img = create_sample_image()
    else:
        print(f"Successfully loaded image with shape: {original_img.shape}")
    
    # Resize if image is too large for better visualization
    if max(original_img.shape) > 512:
        original_img = cv2.resize(original_img, (512, 512))
    
    # Task 1: Fourier Transform and Spectrum Visualization
    print("\n=== TASK 1: Fourier Transform and Spectrum Visualization ===")
    fft_shift, magnitude_spectrum = perform_fourier_transform(original_img)
    
    # Task 2: Low-Pass Filtering
    print("\n=== TASK 2: Low-Pass Filtering in Frequency Domain ===")
    perform_low_pass_filtering(original_img, fft_shift)
    
    # Task 3: High-Pass Filtering
    print("\n=== TASK 3: High-Pass Filtering in Frequency Domain ===")
    perform_high_pass_filtering(original_img, fft_shift)
    
    # Task 4: Notch Filtering
    print("\n=== TASK 4: Notch Filtering for Periodic Noise Removal ===")
    perform_notch_filtering(original_img)

def create_sample_image():
    """Create a sample image with clear features for frequency domain analysis"""
    size = 512
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Add a rectangle (low frequency component)
    cv2.rectangle(img, (150, 150), (350, 350), 200, -1)
    
    # Add some text (high frequency components)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'IP', (200, 256), font, 2, 100, 3, cv2.LINE_AA)
    
    # Add a gradient (varying frequencies)
    for i in range(size):
        img[i, :] = np.clip(img[i, :] + (i * 50 // size), 0, 255)
    
    return img

def perform_fourier_transform(img):
    """Compute and display the 2D Fourier Transform"""
    # Convert image to float32 for FFT
    f = np.float32(img)
    
    # Compute the 2D Fourier Transform
    fft = np.fft.fft2(f)
    
    # Shift the zero frequency component to the center
    fft_shift = np.fft.fftshift(fft)
    
    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(fft_shift)
    
    # Apply logarithmic scaling for better visualization
    log_spectrum = np.log(1 + magnitude_spectrum)
    
    # Display results
    plt.figure(figsize=(15, 6))
    plt.suptitle('Fourier Transform and Frequency Spectrum', fontsize=16, fontweight='bold')
    
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(log_spectrum, cmap='viridis')
    plt.title('Magnitude Spectrum (Log Scale)')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Frequency component explanation
    print("Frequency Components Significance:")
    print("- Low frequencies (center of spectrum): Represent smooth areas, overall shape, and gradual intensity changes")
    print("- High frequencies (edges of spectrum): Represent fine details, edges, noise, and rapid intensity changes")
    print("- Bright spots in spectrum: Correspond to dominant spatial frequencies in the image")
    print("- The orientation of features in spectrum corresponds to the orientation of edges in the spatial domain")
    
    return fft_shift, magnitude_spectrum

def create_ideal_lowpass_filter(shape, cutoff):
    """Create an ideal low-pass filter"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2  # center coordinates
    
    # Create a mask with 1's within cutoff radius and 0's elsewhere
    mask = np.zeros((rows, cols), np.float32)
    y, x = np.ogrid[:rows, :cols]
    
    # Calculate distance from center
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Apply cutoff
    mask[distance <= cutoff] = 1
    
    return mask

def perform_low_pass_filtering(img, fft_shift):
    """Apply ideal low-pass filtering"""
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create low-pass filters with different cutoffs
    cutoff_small = 30
    cutoff_large = 70
    
    mask_small = create_ideal_lowpass_filter((rows, cols), cutoff_small)
    mask_large = create_ideal_lowpass_filter((rows, cols), cutoff_large)
    
    # Apply filters
    fft_filtered_small = fft_shift * mask_small
    fft_filtered_large = fft_shift * mask_large
    
    # Inverse FFT
    img_filtered_small = np.fft.ifft2(np.fft.ifftshift(fft_filtered_small)).real
    img_filtered_large = np.fft.ifft2(np.fft.ifftshift(fft_filtered_large)).real
    
    # Display results
    plt.figure(figsize=(18, 12))
    plt.suptitle('Ideal Low-Pass Filtering', fontsize=16, fontweight='bold')
    
    # Original image and spectrum
    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    magnitude_spectrum = np.abs(fft_shift)
    log_spectrum = np.log(1 + magnitude_spectrum)
    
    plt.subplot(3, 3, 2)
    plt.imshow(log_spectrum, cmap='viridis')
    plt.title('Original Spectrum')
    plt.axis('off')
    
    # Small cutoff filter
    plt.subplot(3, 3, 4)
    plt.imshow(mask_small, cmap='gray')
    plt.title(f'ILPF Mask (Cutoff={cutoff_small})')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    filtered_spectrum_small = np.log(1 + np.abs(fft_filtered_small))
    plt.imshow(filtered_spectrum_small, cmap='viridis')
    plt.title('Filtered Spectrum')
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.imshow(img_filtered_small, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')
    
    # Large cutoff filter
    plt.subplot(3, 3, 7)
    plt.imshow(mask_large, cmap='gray')
    plt.title(f'ILPF Mask (Cutoff={cutoff_large})')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    filtered_spectrum_large = np.log(1 + np.abs(fft_filtered_large))
    plt.imshow(filtered_spectrum_large, cmap='viridis')
    plt.title('Filtered Spectrum')
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.imshow(img_filtered_large, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Low-Pass Filtering Effects:")
    print("- Attenuates high-frequency components while preserving low frequencies")
    print("- Results in image blurring/smoothing")
    print("- Smaller cutoff radius = more aggressive filtering = more blurring")
    print("- Removes noise and fine details but also blurs edges")
    print("- Ideal filters create ringing artifacts (Gibbs phenomenon) due to sharp cutoff")

def create_ideal_highpass_filter(shape, cutoff):
    """Create an ideal high-pass filter"""
    # High-pass filter is 1 - low-pass filter
    lowpass = create_ideal_lowpass_filter(shape, cutoff)
    highpass = 1 - lowpass
    return highpass

def perform_high_pass_filtering(img, fft_shift):
    """Apply ideal high-pass filtering"""
    rows, cols = img.shape
    
    # Create high-pass filters with different cutoffs
    cutoff_small = 10
    cutoff_large = 30
    
    mask_small = create_ideal_highpass_filter((rows, cols), cutoff_small)
    mask_large = create_ideal_highpass_filter((rows, cols), cutoff_large)
    
    # Apply filters
    fft_filtered_small = fft_shift * mask_small
    fft_filtered_large = fft_shift * mask_large
    
    # Inverse FFT
    img_filtered_small = np.fft.ifft2(np.fft.ifftshift(fft_filtered_small)).real
    img_filtered_large = np.fft.ifft2(np.fft.ifftshift(fft_filtered_large)).real
    
    # Display results
    plt.figure(figsize=(18, 12))
    plt.suptitle('Ideal High-Pass Filtering', fontsize=16, fontweight='bold')
    
    # Original image and spectrum
    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    magnitude_spectrum = np.abs(fft_shift)
    log_spectrum = np.log(1 + magnitude_spectrum)
    
    plt.subplot(3, 3, 2)
    plt.imshow(log_spectrum, cmap='viridis')
    plt.title('Original Spectrum')
    plt.axis('off')
    
    # Small cutoff filter
    plt.subplot(3, 3, 4)
    plt.imshow(mask_small, cmap='gray')
    plt.title(f'IHPF Mask (Cutoff={cutoff_small})')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    filtered_spectrum_small = np.log(1 + np.abs(fft_filtered_small))
    plt.imshow(filtered_spectrum_small, cmap='viridis')
    plt.title('Filtered Spectrum')
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.imshow(img_filtered_small, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')
    
    # Large cutoff filter
    plt.subplot(3, 3, 7)
    plt.imshow(mask_large, cmap='gray')
    plt.title(f'IHPF Mask (Cutoff={cutoff_large})')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    filtered_spectrum_large = np.log(1 + np.abs(fft_filtered_large))
    plt.imshow(filtered_spectrum_large, cmap='viridis')
    plt.title('Filtered Spectrum')
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.imshow(img_filtered_large, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("High-Pass Filtering Effects:")
    print("- Attenuates low-frequency components while preserving high frequencies")
    print("- Enhances edges and fine details")
    print("- Smaller cutoff radius = more aggressive filtering = stronger edge enhancement")
    print("- Removes overall intensity information (DC component) resulting in dark images")
    print("- Useful for edge detection, sharpening, and feature extraction")

def add_periodic_noise(img):
    """Add synthetic periodic noise to the image"""
    rows, cols = img.shape
    noisy_img = img.copy().astype(np.float32)
    
    # Create sinusoidal noise patterns
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Add multiple frequency components
    noise1 = 50 * np.sin(2 * np.pi * x / 30 + 2 * np.pi * y / 40)
    noise2 = 30 * np.cos(2 * np.pi * x / 20 + 2 * np.pi * y / 25)
    noise3 = 20 * np.sin(2 * np.pi * x / 15)
    
    # Combine noise patterns
    total_noise = noise1 + noise2 + noise3
    
    # Add noise to image
    noisy_img += total_noise
    
    # Clip to valid range and convert back to uint8
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return noisy_img, total_noise

def create_notch_filter(shape, frequencies, bandwidth=5):
    """Create a notch filter to remove specific frequency components"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    
    for freq in frequencies:
        u, v = freq
        # Create a mask that blocks frequencies around (u, v) and (-u, -v)
        y, x = np.ogrid[:rows, :cols]
        
        # Distance from the frequency point and its symmetric counterpart
        dist1 = np.sqrt((x - (ccol + u))**2 + (y - (crow + v))**2)
        dist2 = np.sqrt((x - (ccol - u))**2 + (y - (crow - v))**2)
        
        # Apply notch (set to 0 within bandwidth)
        mask[(dist1 <= bandwidth) | (dist2 <= bandwidth)] = 0
    
    return mask

def perform_notch_filtering(img):
    """Perform notch filtering to remove periodic noise"""
    # Add periodic noise
    noisy_img, noise_pattern = add_periodic_noise(img)
    
    # Compute FFT of noisy image
    f = np.float32(noisy_img)
    fft_noisy = np.fft.fft2(f)
    fft_shift_noisy = np.fft.fftshift(fft_noisy)
    magnitude_spectrum_noisy = np.log(1 + np.abs(fft_shift_noisy))
    
    # Identify noise frequencies (these would typically be identified from the spectrum)
    # For this example, we'll use known frequencies from our noise pattern
    noise_frequencies = [(50, 40), (40, 30), (30, 0)]  # Approximate frequencies
    
    # Create notch filter
    notch_mask = create_notch_filter(img.shape, noise_frequencies, bandwidth=10)
    
    # Apply notch filter
    fft_filtered = fft_shift_noisy * notch_mask
    
    # Inverse FFT
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fft_filtered)).real
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
    
    # Display results
    plt.figure(figsize=(18, 12))
    plt.suptitle('Notch Filtering for Periodic Noise Removal', fontsize=16, fontweight='bold')
    
    # Original and noisy images
    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(noisy_img, cmap='gray')
    plt.title('Image with Periodic Noise')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(noise_pattern, cmap='viridis')
    plt.title('Noise Pattern')
    plt.axis('off')
    
    # Spectra
    fft_shift_original = np.fft.fftshift(np.fft.fft2(np.float32(img)))
    magnitude_spectrum_original = np.log(1 + np.abs(fft_shift_original))
    
    plt.subplot(3, 3, 4)
    plt.imshow(magnitude_spectrum_original, cmap='viridis')
    plt.title('Original Spectrum')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(magnitude_spectrum_noisy, cmap='viridis')
    plt.title('Noisy Spectrum')
    plt.axis('off')
    
    # Filter and filtered spectrum
    plt.subplot(3, 3, 6)
    plt.imshow(notch_mask, cmap='gray')
    plt.title('Notch Filter Mask')
    plt.axis('off')
    
    magnitude_spectrum_filtered = np.log(1 + np.abs(fft_filtered))
    plt.subplot(3, 3, 7)
    plt.imshow(magnitude_spectrum_filtered, cmap='viridis')
    plt.title('Filtered Spectrum')
    plt.axis('off')
    
    # Filtered image
    plt.subplot(3, 3, 8)
    plt.imshow(img_filtered, cmap='gray')
    plt.title('Filtered Image (Noise Removed)')
    plt.axis('off')
    
    # Difference
    plt.subplot(3, 3, 9)
    difference = cv2.absdiff(img, img_filtered)
    plt.imshow(difference, cmap='viridis')
    plt.title('Difference (Original vs Filtered)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Notch Filtering Effects:")
    print("- Selectively removes specific frequency components while preserving others")
    print("- Effective for removing periodic noise patterns (stripes, waves, etc.)")
    print("- Works by creating 'notches' in the frequency domain at noise frequencies")
    print("- Requires identification of noise frequencies from the spectrum")
    print("- Applications: Removing scanning artifacts, moiré patterns, and other periodic noise")

if __name__ == "__main__":
    main()