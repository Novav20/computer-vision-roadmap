import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# You can use one of the downsampled images from the previous exercise,
# or a small image you have.
SOURCE_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "crono_ds_x8.png"
)  # Path to the downsampled image
UPSAMPLE_FACTOR = 4  # How much to magnify the small image
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_upsampled_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Helper Function for Display (same as before or simplified) ---
def display_images_upsampling(image_dict, main_title="Upsampling Comparison"):
    num_images = len(image_dict)
    if num_images == 0:
        return
    cols = num_images
    rows = 1
    if num_images > 3:  # Adjust layout if many images
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))

    plt.figure(figsize=(cols * 5, rows * 5))
    plt.suptitle(main_title, fontsize=16)
    for i, (title, img_bgr) in enumerate(image_dict.items()):
        plt.subplot(rows, cols, i + 1)
        if img_bgr is None:
            plt.text(0.5, 0.5, "Image N/A", ha="center", va="center")
            plt.title(title + " (Error)")
            plt.axis("off")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"{title}\n{img_rgb.shape[1]}x{img_rgb.shape[0]}")
        plt.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# --- Upsampling Functions (using OpenCV's resize for convenience) ---


def upsample_nearest_neighbor(image, factor):
    """Upsamples using Nearest Neighbor interpolation."""
    if factor <= 1:
        return image
    new_width = int(image.shape[1] * factor)
    new_height = int(image.shape[0] * factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)


def upsample_bilinear(image, factor):
    """Upsamples using Bilinear interpolation."""
    if factor <= 1:
        return image
    new_width = int(image.shape[1] * factor)
    new_height = int(image.shape[0] * factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


# OpenCV also offers cv2.INTER_CUBIC (Bicubic) and cv2.INTER_LANCZOS4 for higher quality upsampling
def upsample_bicubic(image, factor):
    """Upsamples using Bicubic interpolation."""
    if factor <= 1:
        return image
    new_width = int(image.shape[1] * factor)
    new_height = int(image.shape[0] * factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


# --- Main Script ---
if __name__ == "__main__":
    # Load a source image and create a small version to upsample
    source_downsampled_image = cv2.imread(SOURCE_IMAGE_PATH)
    if source_downsampled_image is None:
        print(f"Error: Could not load source image '{SOURCE_IMAGE_PATH}'")
        exit()

    images_to_display = {"Small Base": source_downsampled_image.copy()}

    # --- Upsample using Nearest Neighbor ---
    print(f"\n--- Upsampling by factor {UPSAMPLE_FACTOR} ---")
    upsampled_nn = upsample_nearest_neighbor(source_downsampled_image, UPSAMPLE_FACTOR)
    title_nn = f"Nearest Neighbor x{UPSAMPLE_FACTOR}"
    images_to_display[title_nn] = upsampled_nn
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"upsampled_nearest_neighbor_x{UPSAMPLE_FACTOR}.png"),
        upsampled_nn,
    )
    print(f"Nearest Neighbor result: {upsampled_nn.shape[1]}x{upsampled_nn.shape[0]}")

    # --- Upsample using Bilinear Interpolation ---
    upsampled_bilinear = upsample_bilinear(source_downsampled_image, UPSAMPLE_FACTOR)
    title_bilinear = f"Bilinear x{UPSAMPLE_FACTOR}"
    images_to_display[title_bilinear] = upsampled_bilinear
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"upsampled_bilinear_x{UPSAMPLE_FACTOR}.png"),
        upsampled_bilinear,
    )
    print(
        f"Bilinear result: {upsampled_bilinear.shape[1]}x{upsampled_bilinear.shape[0]}"
    )

    # --- Optional: Upsample using Bicubic Interpolation for comparison ---
    upsampled_bicubic = upsample_bicubic(source_downsampled_image, UPSAMPLE_FACTOR)
    title_bicubic = f"Bicubic x{UPSAMPLE_FACTOR}"
    images_to_display[title_bicubic] = upsampled_bicubic  # Add to display if desired
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"upsampled_bicubic_x{UPSAMPLE_FACTOR}.png"),
        upsampled_bicubic,
    )
    print(f"Bicubic result: {upsampled_bicubic.shape[1]}x{upsampled_bicubic.shape[0]}")

    # Display all images together
    print("\nDisplaying images...")
    display_images_upsampling(images_to_display)

    print(f"\nUpsampled images saved in '{OUTPUT_DIR}' directory.")
    print("Observe the differences:")
    print(
        "- Nearest Neighbor: Blocky appearance, 'pixelated' look. Preserves sharp edges but creates visible blocks."
    )
    print("- Bilinear: Smoother transitions, less blocky, but can blur sharp edges.")
    print(
        "- Bicubic (if shown): Often a good compromise, sharper than bilinear but less blocky than nearest."
    )
