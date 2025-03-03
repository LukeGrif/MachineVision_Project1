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

    # Split into RGB channels
    red_sinogram, green_sinogram, blue_sinogram = sinogram[:, :, 0], sinogram[:, :, 1], sinogram[:, :, 2]

    # Reconstruction without filter
    recon_no_filter_red = reconstruct_image(red_sinogram)
    recon_no_filter_green = reconstruct_image(green_sinogram)
    recon_no_filter_blue = reconstruct_image(blue_sinogram)

    # Reconstruction with ramp filter
    recon_ramp_red = reconstruct_image(red_sinogram, ramp_filter)
    recon_ramp_green = reconstruct_image(green_sinogram, ramp_filter)
    recon_ramp_blue = reconstruct_image(blue_sinogram, ramp_filter)

    # Reconstruction with Hamming ramp filter
    recon_hamming_red = reconstruct_image(red_sinogram, hamming_ramp_filter)
    recon_hamming_green = reconstruct_image(green_sinogram, hamming_ramp_filter)
    recon_hamming_blue = reconstruct_image(blue_sinogram, hamming_ramp_filter)

    # Crop images using the corrected function
    recon_no_filter_red = crop_image(recon_no_filter_red, width_ratio, height_ratio)
    recon_no_filter_green = crop_image(recon_no_filter_green, width_ratio, height_ratio)
    recon_no_filter_blue = crop_image(recon_no_filter_blue, width_ratio, height_ratio)

    recon_ramp_red = crop_image(recon_ramp_red, width_ratio, height_ratio)
    recon_ramp_green = crop_image(recon_ramp_green, width_ratio, height_ratio)
    recon_ramp_blue = crop_image(recon_ramp_blue, width_ratio, height_ratio)

    recon_hamming_red = crop_image(recon_hamming_red, width_ratio, height_ratio)
    recon_hamming_green = crop_image(recon_hamming_green, width_ratio, height_ratio)
    recon_hamming_blue = crop_image(recon_hamming_blue, width_ratio, height_ratio)

    # Normalize results
    recon_no_filter = np.dstack((
        normalize_image(recon_no_filter_red),
        normalize_image(recon_no_filter_green),
        normalize_image(recon_no_filter_blue)
    ))

    recon_ramp = np.dstack((
        normalize_image(recon_ramp_red),
        normalize_image(recon_ramp_green),
        normalize_image(recon_ramp_blue)
    ))

    recon_hamming = np.dstack((
        normalize_image(recon_hamming_red),
        normalize_image(recon_hamming_green),
        normalize_image(recon_hamming_blue)
    ))

    # Save images
    Image.fromarray(recon_no_filter).save('recon_no_filter.png')
    Image.fromarray(recon_ramp).save('recon_ramp.png')
    Image.fromarray(recon_hamming).save('recon_hamming.png')

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(recon_no_filter)
    axes[0].set_title("No Filter")
    axes[1].imshow(recon_ramp)
    axes[1].set_title("Ramp Filter")
    axes[2].imshow(recon_hamming)
    axes[2].set_title("Hamming Ramp Filter")
    plt.show()


if __name__ == "__main__":
    main()
