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


def crop_image_using_metadata(image, aspect_ratio):
    """Crops the given image to match the aspect ratio."""
    h, w = image.shape[:2]
    target_w = w
    target_h = int(w / aspect_ratio)

    if target_h > h:
        target_h = h
        target_w = int(h * aspect_ratio)

    start_x = (w - target_w) // 2
    start_y = (h - target_h) // 2

    return image[start_y:start_y + target_h, start_x:start_x + target_w]


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
    aspect_ratio_str = metadata.get("AspectRatio")
    width_ratio, height_ratio = map(int, aspect_ratio_str.split(':'))
    original_aspect_ratio = width_ratio / height_ratio

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

    # Crop images using aspect ratio metadata
    recon_no_filter_red = crop_image_using_metadata(recon_no_filter_red, original_aspect_ratio)
    recon_no_filter_green = crop_image_using_metadata(recon_no_filter_green, original_aspect_ratio)
    recon_no_filter_blue = crop_image_using_metadata(recon_no_filter_blue, original_aspect_ratio)

    recon_ramp_red = crop_image_using_metadata(recon_ramp_red, original_aspect_ratio)
    recon_ramp_green = crop_image_using_metadata(recon_ramp_green, original_aspect_ratio)
    recon_ramp_blue = crop_image_using_metadata(recon_ramp_blue, original_aspect_ratio)

    recon_hamming_red = crop_image_using_metadata(recon_hamming_red, original_aspect_ratio)
    recon_hamming_green = crop_image_using_metadata(recon_hamming_green, original_aspect_ratio)
    recon_hamming_blue = crop_image_using_metadata(recon_hamming_blue, original_aspect_ratio)

    # Normalize results
    recon_no_filter_red = normalize_image(recon_no_filter_red)
    recon_no_filter_green = normalize_image(recon_no_filter_green)
    recon_no_filter_blue = normalize_image(recon_no_filter_blue)

    recon_ramp_red = normalize_image(recon_ramp_red)
    recon_ramp_green = normalize_image(recon_ramp_green)
    recon_ramp_blue = normalize_image(recon_ramp_blue)

    recon_hamming_red = normalize_image(recon_hamming_red)
    recon_hamming_green = normalize_image(recon_hamming_green)
    recon_hamming_blue = normalize_image(recon_hamming_blue)

    # Combine RGB channels
    recon_no_filter = np.dstack((recon_no_filter_red, recon_no_filter_green, recon_no_filter_blue))
    recon_ramp = np.dstack((recon_ramp_red, recon_ramp_green, recon_ramp_blue))
    recon_hamming = np.dstack((recon_hamming_red, recon_hamming_green, recon_hamming_blue))

    # Save ramp-filtered reconstruction
    img = Image.fromarray(recon_no_filter)
    img.save('recon_no_filter.png')
    img = Image.fromarray(recon_ramp)
    img.save('recon_ramp.png')
    img = Image.fromarray(recon_hamming)
    img.save('recon_hamming.png')

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(recon_no_filter, cmap='gray')
    axes[0].set_title("No Filter")
    axes[1].imshow(recon_ramp, cmap='gray')
    axes[1].set_title("Ramp Filter")
    axes[2].imshow(recon_hamming, cmap='gray')
    axes[2].set_title("Hamming Ramp Filter")
    plt.show()

if __name__ == "__main__":
    main()
