import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from skimage.transform import rotate
from scipy.fftpack import rfft, irfft
from PIL import Image
import argparse

def ramp_filter(ffts):
    """Applies a ramp filter in the frequency domain."""
    ramp = np.floor(np.arange(0.5, ffts.shape[1] // 2 + 0.1, 0.5))
    return ffts * ramp

def hamming_ramp_filter(ffts):
    """Applies a ramp filter with Hamming window."""
    ramp = np.floor(np.arange(0.5, ffts.shape[1] // 2 + 0.1, 0.5))
    hamming = np.hamming(len(ramp))
    return ffts * ramp * hamming

def hann_ramp_filter(ffts):
    """Applies a ramp filter with Hann window."""
    ramp = np.floor(np.arange(0.5, ffts.shape[1] // 2 + 0.1, 0.5))
    # "Hann window" and "Hanning window" are the same thing apparently
    # https://numpy.org/doc/stable/reference/generated/numpy.hanning.html
    hann = np.hanning(len(ramp))
    return ffts * ramp * hann

def reconstruct_image(sinogram, filter_fn=None):
    """Reconstructs an image from a sinogram using backprojection."""
    sinogram_fft = rfft(sinogram, axis=1)

    if filter_fn:
        sinogram_fft = filter_fn(sinogram_fft)

    filtered_sinogram = irfft(sinogram_fft, axis=1)

    steps, M = filtered_sinogram.shape
    laminogram = np.zeros((M, M))
    dTheta = 180.0 / steps
    for i in range(steps):
        temp = np.tile(filtered_sinogram[i], (M, 1))
        bp = rotate(temp, dTheta * i)
        laminogram += bp

    return laminogram


def crop_image(image, width_ratio, height_ratio):
    """Crops the image from the center using the correct width-to-height ratio formula."""
    h, w = image.shape[:2]

    # Compute the correct scaling factor
    scaling_factor = w / np.sqrt(width_ratio ** 2 + height_ratio ** 2)

    # Compute the new dimensions
    new_w = int(round(scaling_factor * width_ratio))
    new_h = int(round(scaling_factor * height_ratio))

    # Ensure new dimensions do not exceed original size
    new_w = min(new_w, w)
    new_h = min(new_h, h)

    # Compute crop coordinates (centered)
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2

    return image[start_y:start_y + new_h, start_x:start_x + new_w]


def normalize_image(img):
    """Normalize image to 8-bit range (0-255)."""
    chi, clo = img.max(), img.min()
    chnorm = 255 * (img - clo) / (chi - clo)
    return np.floor(chnorm).astype('uint8')


def main():
    parser = argparse.ArgumentParser(description="Reconstruct an image from a sinogram.")
    parser.add_argument("sinogram_path", type=str, help="Path to the sinogram image file.")
    args = parser.parse_args()

    # Load sinogram
    sinogram = iio.imread(args.sinogram_path)
    metadata = iio.immeta(args.sinogram_path)

    # Extract aspect ratio from metadata
    aspect_ratio_str = metadata.get("AspectRatio", "1:1")  # Default to 1:1 if missing
    width_ratio, height_ratio = map(int, aspect_ratio_str.split(':'))

    # Print the extracted aspect ratio
    print(f"Extracted Aspect Ratio from Metadata: {aspect_ratio_str} ({width_ratio}:{height_ratio})")

    # Define filters
    filters = {
        "No Filter": None,
        "Ramp Filter": ramp_filter,
        "Hamming Ramp Filter": hamming_ramp_filter,
        "Hann Ramp Filter": hann_ramp_filter
    }

    # Process each color channel
    reconstructed_images = {}
    channels = ['Red', 'Green', 'Blue']
    
    for filter_name, filter_func in filters.items():
        recon_channels = []
        for i, color in enumerate(channels):
            recon = reconstruct_image(sinogram[:, :, i], filter_func)  # Reconstruct
            recon = crop_image(recon, width_ratio, height_ratio)  # Crop
            recon = normalize_image(recon)  # Normalize
            recon_channels.append(recon)
        
        reconstructed_images[filter_name] = np.dstack(recon_channels)  # Stack RGB channels

    # Save images
    for filter_name, image in reconstructed_images.items():
        Image.fromarray(image).save(f'recon_{filter_name.lower().replace(" ", "_")}.png')
    
    # Display results
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for ax, (filter_name, image) in zip(axes, reconstructed_images.items()):
        ax.imshow(image)
        ax.set_title(filter_name)
        ax.axis('off')
    
    plt.show()


if __name__ == "__main__":
    main()
