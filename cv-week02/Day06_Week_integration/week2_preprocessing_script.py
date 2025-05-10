import cv2
import numpy as np
import os

# --- Configuration ---
INPUT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "esp32_test_image.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_week2_preprocessing")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pipeline parameters
DOWNSAMPLE_FACTOR = 2
# If upsampling after downsampling (e.g., to restore original size for some comparisons)
# If UPSAMPLE_FACTOR = DOWNSAMPLE_FACTOR, it will attempt to restore original size.
# If UPSAMPLE_FACTOR = 1, no upsampling after downsampling.
UPSAMPLE_FACTOR_AFTER_DOWN = 1  # Let's keep it 1 to process the downsampled image.
# Change to DOWNSAMPLE_FACTOR to go back to original size.

# Color space for luminance equalization: 'HSV' or 'LAB'
LUMINANCE_EQ_COLOR_SPACE = "LAB"  # LAB is generally preferred for perceptual luminance


#  --- Helper Functions from previous days (simplified for script use) ---
def downsample_image(image, factor):
    """Downsamples using INTER_AREA."""
    if factor <= 0:
        return image
    new_width = image.shape[1] // factor
    new_height = image.shape[0] // factor
    downsampled_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return downsampled_image


def upsample_image(image, factor, method=cv2.INTER_LINEAR):
    """Upsamples using specified interpolation method."""
    if factor <= 1:
        return image
    new_width = int(image.shape[1] * factor)
    new_height = int(image.shape[0] * factor)
    return cv2.resize(image, (new_width, new_height), interpolation=method)


def equalize_luminance_channel(image_bgr, color_space_choice="LAB"):
    """
    Converts image to specified color space, equalizes luminance channel and converts back to BGR.
    """
    processed_image_bgr = None
    luminance_channel_original = None
    luminance_channel_equalized = None

    if color_space_choice == "HSV":
        img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        luminance_channel_original = v.copy()
        v_eq = cv2.equalizeHist(v)  # Equalize the V (Value/Brightness) channel
        luminance_channel_equalized = v_eq.copy()
        img_hsv_eq = cv2.merge((h, s, v_eq))
        processed_image_bgr = cv2.cvtColor(img_hsv_eq, cv2.COLOR_HSV2BGR)
    elif color_space_choice == "LAB":
        # For LAB, it's often better to work with float32 for L* in [0,100]
        # However, cv2.equalizeHist expects uint8 input.
        # So we convert to LAB, equalize L (which is scaled 0-255), then convert back.
        img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        luminance_channel_original = l_channel.copy()
        l_eq = cv2.equalizeHist(l_channel)  # Equalize the L (Lightness) channel
        luminance_channel_equalized = l_eq.copy()
        img_lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        processed_image_bgr = cv2.cvtColor(img_lab_eq, cv2.COLOR_Lab2BGR)
    else:
        print(
            f"Error: Unsupported color space '{color_space_choice}' for luminance equalization."
        )
        return image_bgr, None, None  # Return original if error

    return processed_image_bgr, luminance_channel_original, luminance_channel_equalized


# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting Week 2 Preprocessing Pipeline Script ---")

    # 1. Load an ESP32-CAM test image (or any test image)
    img_original = cv2.imread(INPUT_IMAGE_PATH)
    if img_original is None:
        print(f"Error: Could not load image '{INPUT_IMAGE_PATH}'")
        exit()
    print(f"1. Original image loaded: {img_original.shape[1]}x{img_original.shape[0]}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "0_original.png"), img_original)
    current_img = img_original.copy()

    # 2. Apply downsampling
    if DOWNSAMPLE_FACTOR > 1:
        img_downsampled = downsample_image(current_img, DOWNSAMPLE_FACTOR)
        print(
            f"2. Downsampled by factor {DOWNSAMPLE_FACTOR}: {img_downsampled.shape[1]}x{img_downsampled.shape[0]}"
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"1_downsampled_x{DOWNSAMPLE_FACTOR}.png"),
            img_downsampled,
        )
        current_img = img_downsampled
    else:
        print("2. Downsampling skipped (factor <= 1).")

    # (Optional) Apply upsampling (e.g., if you wanted to restore size or target a new size)
    if UPSAMPLE_FACTOR_AFTER_DOWN > 1:
        # Example: Upsample back to original size if DOWNSAMPLE_FACTOR was > 1
        # target_upsample_factor = DOWNSAMPLE_FACTOR if DOWNSAMPLE_FACTOR > 1 else 1
        # We use the defined UPSAMPLE_FACTOR_AFTER_DOWN
        img_upsampled = upsample_image(
            current_img, UPSAMPLE_FACTOR_AFTER_DOWN, method=cv2.INTER_CUBIC
        )
        print(
            f"3. Upsampled by factor {UPSAMPLE_FACTOR_AFTER_DOWN}: {img_upsampled.shape[1]}x{img_upsampled.shape[0]}"
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"2_upsampled_x{UPSAMPLE_FACTOR_AFTER_DOWN}.png"),
            img_upsampled,
        )
        current_img = img_upsampled
    else:
        print("3. Upsampling skipped (factor <= 1).")

    # Save the image state before color transforms for easier comparison in notebook
    img_before_color_eq = current_img.copy()
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "3_img_before_color_eq.png"), img_before_color_eq
    )

    # 4. Convert to chosen color space & Perform histogram equalization on luminance channel
    print(
        f"4. Performing luminance equalization using {LUMINANCE_EQ_COLOR_SPACE} space..."
    )
    img_luminance_equalized, lum_orig, lum_eq = equalize_luminance_channel(
        current_img, LUMINANCE_EQ_COLOR_SPACE
    )

    if lum_orig is not None and lum_eq is not None:
        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR, f"4a_luminance_original_{LUMINANCE_EQ_COLOR_SPACE}.png"
            ),
            lum_orig,
        )
        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR, f"4b_luminance_equalized_{LUMINANCE_EQ_COLOR_SPACE}.png"
            ),
            lum_eq,
        )
        print(f"   Original and equalized luminance channels saved.")

    if img_luminance_equalized is not None:
        print(
            f"   Luminance equalization complete. Image shape: {img_luminance_equalized.shape[1]}x{img_luminance_equalized.shape[0]}"
        )
        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                f"5_final_luminance_equalized_{LUMINANCE_EQ_COLOR_SPACE}.png",
            ),
            img_luminance_equalized,
        )
        current_img = img_luminance_equalized
    else:
        print("   Luminance equalization failed or skipped.")

    print(f"\n--- Pipeline Complete. Results saved in '{OUTPUT_DIR}' ---")

    # Optional: Display final image (if running in an environment that supports it)
    # cv2.imshow("Original", img_original)
    # cv2.imshow(f"Final Output ({LUMINANCE_EQ_COLOR_SPACE} Equalized)", current_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
