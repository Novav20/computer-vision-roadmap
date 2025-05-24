import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configruation ---
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "Crono.png")
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "output_downsampled_images")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

DOWNSAMPLE_FACTORS = [2, 4, 8]


# --- Helper function for display ---
def display_images(image_dict, main_title="Downsampling comparison"):
    """Display images in a grid."""
    num_images = len(image_dict)
    if num_images == 0:
        return
    # Determine grid size (aims for a somewhat square grid)
    cols = np.ceil(np.sqrt(num_images)).astype(int)
    rows = np.ceil(num_images / cols).astype(int)
    plt.figure(figsize=(cols * 4, rows * 4))
    plt.suptitle(main_title, fontsize=16)
    for i, (title, img_bgr) in enumerate(image_dict.items()):
        plt.subplot(rows, cols, i + 1)
        if img_bgr is None:
            plt.text(0.5, 0.5, "Image N/A", ha="center", va="center")
            plt.title(title + "Error")
            plt.axis("off")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"{title}\n{img_rgb.shape[1]}x{img_rgb.shape[0]}")  # WxH
        plt.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to make space for suptitle
    plt.show()


# --- Downsampling functions ---
def downsample_naive_slicing(image, factor):
    """
    Downsamples by simply taking every Nth pixel.
    This is prone to aliasing as it does not pre-filter.
    """
    if factor <= 0:
        return image
    return image[::factor, ::factor, :]


def downsample_with_blur_then_resize(image, factor, kernel_size=(5, 5)):
    """
    Downsamples by first applying a Gaussian blur (low-pass filter)
    and then resizing using OpenCV's resize with INTER_LINEAR.
    This helps mitigate aliasing.
    """
    if factor <= 0:
        return image
    # Calculate target dimensions
    new_width = image.shape[1] // factor
    new_height = image.shape[0] // factor

    # Apply Gaussian blur before resizing
    # The sigma for GaussianBlur is often related to the downsampling factor.
    # A simple heuristic: sigma = factor / 2.0
    # However, a fixed kernel size might be easier to start with.
    # If factor is large, a larger sigma or kernel size might be needed.
    sigma_x = factor / 2.0  # Heuristic
    sigma_y = factor / 2.0
    # Ensure kernel_size is odd
    k_w = kernel_size[0] if kernel_size[0] % 2 != 0 else kernel_size[0] + 1
    k_h = kernel_size[1] if kernel_size[1] % 2 != 0 else kernel_size[1] + 1
    blurred_image = cv2.GaussianBlur(image, (k_w, k_h), sigmaX=sigma_x, sigmaY=sigma_y)

    # Resize using INTER_LINEAR (a form of bilinear interpolation)
    # OpenCV's resize with INTER_AREA is generally recommended for decimation (shrinking)
    # as it performs pixel area relation, inherently handling some anti-aliasing.
    # Let's compare with INTER_LINEAR after blur vs INTER_AREA directly.
    downsampled_image = cv2.resize(
        blurred_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    return downsampled_image


def downsample_with_inter_area(image, factor):
    """
    Downsamples using OpenCV's resize with INTER_AREA.
    This method is generally good for shrinking images as it considers pixel area relation.
    """
    if factor <= 0:
        return image
    new_width = image.shape[1] // factor
    new_height = image.shape[0] // factor
    downsampled_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return downsampled_image


if __name__ == "__main__":
    # Load test image
    original_image = cv2.imread(TEST_IMAGE_PATH)
    if original_image is None:
        print(f"Failed to load image from {TEST_IMAGE_PATH}")
        exit(1)

    print(f"Original image loaded: {original_image.shape[1]}x{original_image.shape[0]}")

    images_to_display = {"Original": original_image.copy()}
    saved_images_info = ["Original"]

    for factor in DOWNSAMPLE_FACTORS:
        print(f"\n--- Downsampling by factor {factor} ---")

        # Method 1: Naive slicing (expect aliasing)
        ds_naive = downsample_naive_slicing(original_image, factor)
        title_naive = f"Naiver x{factor}"
        images_to_display[title_naive] = ds_naive
        cv2.imwrite(
            os.path.join(OUTPUT_DIRECTORY, f"downsampled_naive_x{factor}.png"), ds_naive
        )
        saved_images_info.append(title_naive)
        print(f"Naive slicing result: {ds_naive.shape[1]}x{ds_naive.shape[0]}")

        #  Method 2: Blur then INTER_LINEAR Resize (Better Anti-Aliasing)
        # Adjust kernel size based on factor if needed, or use sigma heuristic with fixed kernel
        # For larger factors, a larger blur might be needed.
        # k_size = (3,3) if factor <=2 else (5,5) if factor <=4 else (7,7)
        ds_blur_resize = downsample_with_blur_then_resize(original_image, factor)
        title_blur = f"Blur+Resize x{factor}"
        images_to_display[title_blur] = ds_blur_resize
        cv2.imwrite(
            os.path.join(OUTPUT_DIRECTORY, f"downsampled_blur_resize_x{factor}.png"), ds_blur_resize
        )
        saved_images_info.append(title_blur)
        print(
            f"Blur+INTER_LINEAR result: {ds_blur_resize.shape[1]}x{ds_blur_resize.shape[0]}"
        )

        # Method 3: INTER_AREA Resize (Good for decimation)
        ds_inter_area = downsample_with_inter_area(original_image, factor)
        title_inter_area = f"INTER_AREA x{factor}"
        images_to_display[title_inter_area] = ds_inter_area
        cv2.imwrite(
            os.path.join(OUTPUT_DIRECTORY, f"downsampled_inter_area_x{factor}.png"), ds_inter_area
        )
        saved_images_info.append(title_inter_area)
        print(f"INTER_AREA result: {ds_inter_area.shape[1]}x{ds_inter_area.shape[0]}")

    # Display all images
    print("\n--- Displaying results ---")
    display_images(images_to_display)

    print(f"\nDownsampled images saved in '{OUTPUT_DIRECTORY}' directory.")
    print(
        "Observe the differences, especially in areas with fine detail or sharp edges."
    )
    print(
        "The 'Naive' method should show more aliasing (jagged lines, Moiré patterns)."
    )
    print("'Blur+Resize' and 'INTER_AREA' should produce smoother results.")
