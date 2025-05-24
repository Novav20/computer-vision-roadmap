import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- Configuration ---
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "Zenan_battle.png")
NOISE_TYPE = "gaussian"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_week3_tuesday")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Helper functions ---
def add_noise(image, noise_type="gaussian", amount=0.05, salt_vs_pepper_ratio=0.5):
    """Add specified noise to an image"""
    noisy_image = image.copy()
    if noise_type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        sigma = amount * 255
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy_image = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    elif noise_type == "salt_and_pepper":
        row, col = image.shape[:2]
        s_vs_p = salt_vs_pepper_ratio
        # Amount of salt noise
        num_salt = np.ceil(amount * image.size * s_vs_p / 3)  # Divide by 3 for channels
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[0:2]]
        noisy_image[coords[0], coords[1], :] = 255
        # Amount of pepper noise
        num_pepper = np.ceil(amount * image.size * (1 - s_vs_p) / 3)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 0
    return noisy_image


def display_images(img_dict, main_title="Image Comparison"):
    num_images = len(img_dict)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    if num_images == 1:
        cols, rows = 1, 1
    if num_images == 2:
        cols, rows = 2, 1

    plt.figure(figsize=(cols * 5, rows * 5))
    plt.suptitle(main_title, fontsize=16)
    for i, (title, img_bgr) in enumerate(img_dict.items()):
        plt.subplot(rows, cols, i + 1)
        if img_bgr is None:
            plt.title(title + " (Error)")
            plt.axis("off")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"{title}\n{img_rgb.shape[1]}x{img_rgb.shape[0]}")
        plt.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# --- Block 1: Implement 2D Convolution ---
def convolve2d_numpy(image, kernel, border_type="reflect"):
    """
    Performs 2D convolution using NumPy.
    Handles grayscale and color images.
    border_type: 'zero', 'reflect', 'replicate'
    """
    if image.ndim == 2:  # Grayscale
        image = image[:, :, np.newaxis]  # Add channel dimension

    k_h, k_w = kernel.shape
    img_h, img_w, img_ch = image.shape

    # Calculate padding size
    pad_h_half = k_h // 2
    pad_w_half = k_w // 2

    # Apply padding
    if border_type == "zero":
        padded_image = np.pad(
            image,
            ((pad_h_half, pad_h_half), (pad_w_half, pad_w_half), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    elif border_type == "reflect":
        padded_image = np.pad(
            image,
            ((pad_h_half, pad_h_half), (pad_w_half, pad_w_half), (0, 0)),
            mode="reflect",
        )
    elif border_type == "replicate":
        padded_image = np.pad(
            image,
            ((pad_h_half, pad_h_half), (pad_w_half, pad_w_half), (0, 0)),
            mode="edge",
        )
    else:
        raise ValueError("Unsupported border_type")

    output_image = np.zeros_like(image, dtype=np.float32)

    # Flip kernel for convolution (NumPy's way)
    # For cross-correlation (which filter2D effectively does), don't flip.
    # For true convolution, flip kernel.
    kernel_flipped = np.flipud(np.fliplr(kernel))

    for c in range(img_ch):  # Iterate over channels
        for y in range(img_h):
            for x in range(img_w):
                # Extract region of interest (ROI) from padded image
                roi = padded_image[y : y + k_h, x : x + k_w, c]
                # Apply convolution
                output_image[y, x, c] = np.sum(
                    roi * kernel_flipped
                )  # or kernel for cross-correlation

    if img_ch == 1:  # If original was grayscale, remove added dimension
        output_image = output_image.squeeze(axis=2)

    return np.clip(output_image, 0, 255).astype(np.uint8)


# --- Main Execution ---
if __name__ == "__main__":
    # Load image
    try:
        img_original = cv2.imread(IMAGE_PATH)
        if img_original is None:
            # Fallback for environments where sample paths might not be found
            print(
                f"Warning: Could not load {IMAGE_PATH}. Trying default OpenCV sample 'lena.png'"
            )
            img_original = cv2.imread(cv2.samples.findFile("lena.png"))
            if img_original is None:
                raise FileNotFoundError("Could not load any test image.")
    except Exception as e:
        print(f"Error loading image: {e}")
        # Create a dummy image if loading fails
        img_original = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.putText(
            img_original,
            "Image Load Failed",
            (30, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        print("Using a dummy image for demonstration.")

    # Add noise
    img_noisy = add_noise(
        img_original,
        noise_type=NOISE_TYPE,
        amount=0.05 if NOISE_TYPE == "gaussian" else 0.05,
    )
    cv2.imwrite(os.path.join(OUTPUT_DIR, "0_original.png"), img_original)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_noisy.png"), img_noisy)

    images_to_show = {"Original": img_original, "Noisy": img_noisy}

    print("--- Block 1: Applying Filters with Custom Convolution ---")

    # Define Filters
    # Box Filter (3x3)
    box_kernel_3x3 = np.ones((3, 3), np.float32) / 9.0
    # Gaussian Filter (sigma=1, kernel size typically ~6*sigma)
    # OpenCV's getGaussianKernel creates 1D, then outer product for 2D
    ksize_gauss_s1 = int(6 * 1 + 1)  # e.g. 7x7 for sigma=1
    if ksize_gauss_s1 % 2 == 0:
        ksize_gauss_s1 += 1  # ensure odd
    gauss_kernel_1d_s1 = cv2.getGaussianKernel(ksize_gauss_s1, 1)
    gaussian_kernel_s1 = np.outer(gauss_kernel_1d_s1, gauss_kernel_1d_s1)

    # Gaussian Filter (sigma=2)
    ksize_gauss_s2 = int(6 * 2 + 1)  # e.g. 13x13 for sigma=2
    if ksize_gauss_s2 % 2 == 0:
        ksize_gauss_s2 += 1
    gauss_kernel_1d_s2 = cv2.getGaussianKernel(ksize_gauss_s2, 2)
    gaussian_kernel_s2 = np.outer(gauss_kernel_1d_s2, gauss_kernel_1d_s2)

    # Apply Box Filter with custom convolve2d_numpy
    print("Applying Box Filter (3x3) with convolve2d_numpy...")
    start_time = time.time()
    img_box_filtered_numpy = convolve2d_numpy(img_noisy, box_kernel_3x3)
    time_box_numpy = time.time() - start_time
    images_to_show["Box Filter (NumPy)"] = img_box_filtered_numpy
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "2_box_filtered_numpy.png"), img_box_filtered_numpy
    )
    print(f"  Done in {time_box_numpy:.4f}s")

    # Apply Gaussian Filter (sigma=1) with custom convolve2d_numpy
    print("Applying Gaussian Filter (sigma=1) with convolve2d_numpy...")
    start_time = time.time()
    img_gauss_s1_numpy = convolve2d_numpy(img_noisy, gaussian_kernel_s1)
    time_gauss_s1_numpy = time.time() - start_time
    images_to_show["Gaussian sigma=1 (NumPy)"] = img_gauss_s1_numpy
    cv2.imwrite(os.path.join(OUTPUT_DIR, "3_gaussian_s1_numpy.png"), img_gauss_s1_numpy)
    print(f"  Done in {time_gauss_s1_numpy:.4f}s")

    # Apply Gaussian Filter (sigma=2) with custom convolve2d_numpy
    print("Applying Gaussian Filter (sigma=2) with convolve2d_numpy...")
    start_time = time.time()
    img_gauss_s2_numpy = convolve2d_numpy(img_noisy, gaussian_kernel_s2)
    time_gauss_s2_numpy = time.time() - start_time
    images_to_show["Gaussian sigma=2 (NumPy)"] = img_gauss_s2_numpy
    cv2.imwrite(os.path.join(OUTPUT_DIR, "4_gaussian_s2_numpy.png"), img_gauss_s2_numpy)
    print(f"  Done in {time_gauss_s2_numpy:.4f}s")

    print("\n--- Block 2: Benchmarking and OpenCV Comparison ---")
    # OpenCV's filter2D (for general convolution)
    print("Applying Box Filter (3x3) with cv2.filter2D...")
    start_time = time.time()
    # For cv2.filter2D, the kernel is not flipped (it does cross-correlation)
    # If we want true convolution that matches our numpy implementation, we should provide the flipped kernel.
    # However, for symmetric kernels like box and Gaussian, it doesn't matter.
    img_box_filtered_cv = cv2.filter2D(
        img_noisy, -1, box_kernel_3x3
    )  # ddepth=-1 means same as source
    time_box_cv = time.time() - start_time
    images_to_show["Box Filter (OpenCV)"] = img_box_filtered_cv
    cv2.imwrite(os.path.join(OUTPUT_DIR, "5_box_filtered_cv.png"), img_box_filtered_cv)
    print(
        f"  Done in {time_box_cv:.4f}s. NumPy vs OpenCV time: {time_box_numpy:.4f}s vs {time_box_cv:.4f}s"
    )

    # OpenCV's GaussianBlur (optimized for Gaussian)
    print("Applying Gaussian Filter (sigma=1) with cv2.GaussianBlur...")
    start_time = time.time()
    img_gauss_s1_cv = cv2.GaussianBlur(img_noisy, (ksize_gauss_s1, ksize_gauss_s1), 1)
    time_gauss_s1_cv = time.time() - start_time
    images_to_show["Gaussian sigma=1 (OpenCV)"] = img_gauss_s1_cv
    cv2.imwrite(os.path.join(OUTPUT_DIR, "6_gaussian_s1_cv.png"), img_gauss_s1_cv)
    print(
        f"  Done in {time_gauss_s1_cv:.4f}s. NumPy vs OpenCV time: {time_gauss_s1_numpy:.4f}s vs {time_gauss_s1_cv:.4f}s"
    )

    print("Applying Gaussian Filter (sigma=2) with cv2.GaussianBlur...")
    start_time = time.time()
    img_gauss_s2_cv = cv2.GaussianBlur(img_noisy, (ksize_gauss_s2, ksize_gauss_s2), 2)
    time_gauss_s2_cv = time.time() - start_time
    images_to_show["Gaussian sigma=2 (OpenCV)"] = img_gauss_s2_cv
    cv2.imwrite(os.path.join(OUTPUT_DIR, "7_gaussian_s2_cv.png"), img_gauss_s2_cv)
    print(
        f"  Done in {time_gauss_s2_cv:.4f}s. NumPy vs OpenCV time: {time_gauss_s2_numpy:.4f}s vs {time_gauss_s2_cv:.4f}s"
    )

    # Visualize all results
    display_images(images_to_show, "Linear Filtering: NumPy vs OpenCV")

    print(f"\nResults saved in '{OUTPUT_DIR}'.")
    print(
        "Note: NumPy convolution is implemented for clarity and is much slower than OpenCV's optimized C/C++ versions."
    )
