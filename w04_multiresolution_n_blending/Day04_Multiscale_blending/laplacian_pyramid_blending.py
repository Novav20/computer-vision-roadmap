# laplacian_pyramid_blending.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
IMAGE_A_PATH = os.path.join(
    os.path.dirname(__file__), "live.png"
)  # Replace with your first image
IMAGE_B_PATH = os.path.join(
    os.path.dirname(__file__), "dead.png"
)  # Replace with your second image
# Ensure images A and B are of the same size for simple vertical/horizontal blending
# If not, they need to be resized to the same dimensions before processing.

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_week4_thursday_blending")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_PYRAMID_LEVELS = 5  # Number of pyramid levels


# --- Helper Function for Display ---
def display_results(img_dict, main_title="Image Blending Results", cols=3):
    num_images = len(img_dict)
    if num_images == 0:
        return
    rows = int(np.ceil(num_images / cols))

    fig_width = cols * 4
    fig_height = rows * 4 + 0.5

    plt.figure(figsize=(fig_width, fig_height))
    plt.suptitle(main_title, fontsize=16)
    for i, (title, img_bgr) in enumerate(img_dict.items()):
        plt.subplot(rows, cols, i + 1)
        if img_bgr is None:
            plt.title(title + "\n(N/A)")
            plt.axis("off")
            continue

        if img_bgr.ndim == 2 or img_bgr.shape[2] == 1:  # Grayscale or mask
            plt.imshow(img_bgr.squeeze(), cmap="gray")
        else:  # Color
            plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# --- Pyramid Functions (from Tuesday, slightly adapted for BGR) ---
def build_gaussian_pyramid_bgr(image, levels):
    if image is None:
        return []
    pyramid = [image.copy()]
    current_level_img = image.copy()
    for _ in range(levels - 1):
        downsampled = cv2.pyrDown(current_level_img)
        pyramid.append(downsampled)
        current_level_img = downsampled
    return pyramid


def build_laplacian_pyramid_bgr(gaussian_pyramid):
    if not gaussian_pyramid:
        return []
    laplacian_pyramid = []
    num_levels = len(gaussian_pyramid)
    for i in range(num_levels - 1):
        G_i = gaussian_pyramid[i]
        G_i_plus_1 = gaussian_pyramid[i + 1]
        expanded_G_i_plus_1 = cv2.pyrUp(G_i_plus_1)
        h_gi, w_gi = G_i.shape[:2]
        expanded_G_i_plus_1_resized = cv2.resize(expanded_G_i_plus_1, (w_gi, h_gi))
        L_i = cv2.subtract(
            G_i.astype(np.float32), expanded_G_i_plus_1_resized.astype(np.float32)
        )
        laplacian_pyramid.append(L_i)
    laplacian_pyramid.append(
        gaussian_pyramid[-1].astype(np.float32).copy()
    )  # Ensure last is float
    return laplacian_pyramid


def reconstruct_from_laplacian_pyramid_bgr(laplacian_pyramid):
    if not laplacian_pyramid:
        return None
    current_reconstructed = laplacian_pyramid[-1].copy()  # Already float
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        expanded_current = cv2.pyrUp(current_reconstructed)
        L_i = laplacian_pyramid[i]  # Already float
        h_li, w_li = L_i.shape[:2]
        expanded_current_resized = cv2.resize(expanded_current, (w_li, h_li))
        current_reconstructed = cv2.add(L_i, expanded_current_resized)
    reconstructed_image = np.clip(current_reconstructed, 0, 255).astype(np.uint8)
    return reconstructed_image


# --- Block 1: Multi-Scale Blending Implementation ---
def blend_images_laplacian(img_a_bgr, img_b_bgr, mask_0to1_float, levels):
    """
    Blends two images using Laplacian pyramids and a mask.
    Args:
        img_a_bgr (np.ndarray): First input image (BGR, uint8).
        img_b_bgr (np.ndarray): Second input image (BGR, uint8, same size as img_a).
        mask_0to1_float (np.ndarray): Grayscale mask (float32, range 0-1, same size as img_a).
                                      Values closer to 1 take from img_a, closer to 0 from img_b.
        levels (int): Number of pyramid levels.
    Returns:
        np.ndarray: The blended image (BGR, uint8).
    """
    if not (
        img_a_bgr.shape == img_b_bgr.shape == mask_0to1_float.shape[:2] + (3,)
        if mask_0to1_float.ndim == 2
        else mask_0to1_float.shape
    ):  # mask can be 2D or 3D
        print("Error: Image A, B, and Mask must have compatible dimensions.")
        print(
            f"  A: {img_a_bgr.shape}, B: {img_b_bgr.shape}, Mask: {mask_0to1_float.shape}"
        )
        if mask_0to1_float.ndim == 2:
            print("  (Mask should be HxW, images HxWx3)")
        return None

    # 1. Build Gaussian pyramid for the mask (GM)
    # Ensure mask is float for pyramid operations if it isn't already, and possibly 3-channel if images are color
    if (
        mask_0to1_float.ndim == 2
    ):  # If mask is grayscale, replicate to 3 channels for blending BGR images
        mask_pyramid_input = cv2.cvtColor(mask_0to1_float, cv2.COLOR_GRAY2BGR)
    else:
        mask_pyramid_input = mask_0to1_float.copy()

    GM = build_gaussian_pyramid_bgr(
        mask_pyramid_input.astype(np.float32), levels
    )  # Mask pyramid needs to be float

    # 2. Build Laplacian pyramids for images A and B (LA, LB)
    gp_A = build_gaussian_pyramid_bgr(img_a_bgr, levels)
    gp_B = build_gaussian_pyramid_bgr(img_b_bgr, levels)
    LA = build_laplacian_pyramid_bgr(gp_A)
    LB = build_laplacian_pyramid_bgr(gp_B)

    # 3. Blend corresponding levels of Laplacian pyramids (LS)
    LS = []  # Blended Laplacian pyramid
    for la, lb, gm in zip(LA, LB, GM):
        # Ensure gm is correctly broadcastable or shaped if la, lb are color
        # gm is already float and 3-channel from build_gaussian_pyramid_bgr if input was made 3-channel
        ls_level = gm * la + (1.0 - gm) * lb
        LS.append(ls_level)

    # 4. Collapse the blended Laplacian pyramid to get the final result
    blended_image = reconstruct_from_laplacian_pyramid_bgr(LS)
    return blended_image


# --- Main Execution ---
if __name__ == "__main__":
    # Load images
    img_A = cv2.imread(IMAGE_A_PATH)
    img_B = cv2.imread(IMAGE_B_PATH)

    if img_A is None or img_B is None:
        print(
            f"Error: Could not load one or both images: '{IMAGE_A_PATH}', '{IMAGE_B_PATH}'"
        )
        # Create dummy images if loading fails
        img_A = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(img_A, (128, 128), 100, (0, 0, 255), -1)
        img_B = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(img_B, (28, 28), (228, 228), (0, 255, 0), -1)
        print("Using dummy images for A and B.")

    # Ensure images are the same size (resize if necessary - simple resize here)
    if img_A.shape != img_B.shape:
        print("Images are different sizes. Resizing B to match A.")
        img_B = cv2.resize(img_B, (img_A.shape[1], img_A.shape[0]))

    h, w = img_A.shape[:2]

    print(f"--- Block 1: Testing Basic Laplacian Pyramid Blending (Vertical Split) ---")
    # Create a simple vertical split mask (binary initially, then normalized float)
    mask_binary = np.zeros((h, w), dtype=np.uint8)
    mask_binary[:, : w // 2] = 255  # Left half from image A
    mask_float = mask_binary.astype(np.float32) / 255.0

    blended_img_sharp_mask = blend_images_laplacian(
        img_A, img_B, mask_float, DEFAULT_PYRAMID_LEVELS
    )

    if blended_img_sharp_mask is not None:
        display_results(
            {
                "Image A": img_A,
                "Image B": img_B,
                "Mask (Binary)": mask_binary,
                "Blended (Sharp Mask)": blended_img_sharp_mask,
            },
            "Laplacian Blending with Sharp Mask",
            cols=2,
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "01_blended_sharp_mask.png"),
            blended_img_sharp_mask,
        )
    else:
        print("Blending with sharp mask failed.")

    print(f"\n--- Block 2: Varying Mask Smoothness & Pyramid Levels ---")
    # 1. Vary mask smoothness
    mask_smooth_weak = cv2.GaussianBlur(mask_float, (25, 25), 0)  # Weak blur
    mask_smooth_strong = cv2.GaussianBlur(mask_float, (101, 101), 0)  # Strong blur
    # Ensure masks are still [0,1] after blur if needed (GaussianBlur on float should keep range if input is [0,1])
    mask_smooth_weak = np.clip(mask_smooth_weak, 0, 1)
    mask_smooth_strong = np.clip(mask_smooth_strong, 0, 1)

    blended_img_smooth_weak = blend_images_laplacian(
        img_A, img_B, mask_smooth_weak, DEFAULT_PYRAMID_LEVELS
    )
    blended_img_smooth_strong = blend_images_laplacian(
        img_A, img_B, mask_smooth_strong, DEFAULT_PYRAMID_LEVELS
    )

    if blended_img_smooth_weak is not None and blended_img_smooth_strong is not None:
        results_smoothness = {
            "Mask (Binary)": mask_binary,
            "Blended (Sharp Mask)": blended_img_sharp_mask,
            "Mask (Weak Blur)": (mask_smooth_weak * 255).astype(np.uint8),
            "Blended (Weak Blur)": blended_img_smooth_weak,
            "Mask (Strong Blur)": (mask_smooth_strong * 255).astype(np.uint8),
            "Blended (Strong Blur)": blended_img_smooth_strong,
        }
        display_results(
            results_smoothness,
            "Effect of Mask Smoothness (Levels=" + str(DEFAULT_PYRAMID_LEVELS) + ")",
            cols=2,
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "02_blended_weak_blur_mask.png"),
            blended_img_smooth_weak,
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "03_blended_strong_blur_mask.png"),
            blended_img_smooth_strong,
        )
        print("Observations on mask smoothness:")
        print(
            "- Sharp mask: Can result in a slightly visible seam, especially if low-frequency content differs greatly."
        )
        print("- Weakly blurred mask: Reduces the seam, transition is softer.")
        print(
            "- Strongly blurred mask: Very smooth transition, but might create a wider 'ghosting' or overly blended area if the blur is too wide for the content."
        )
        print(
            "  The Laplacian pyramid blending inherently handles different frequencies with appropriately scaled masks, so even a sharp input mask M0 leads to smooth blending of low frequencies."
        )
    else:
        print("Blending with smooth masks failed.")

    # 2. Experiment with number of pyramid levels (using the strongly blurred mask)
    levels_to_test = [
        1,
        3,
        DEFAULT_PYRAMID_LEVELS,
        DEFAULT_PYRAMID_LEVELS + 2,
    ]  # Depth of pyramid
    # Max levels depends on image size: min(H,W) / 2^levels > some_min_size (e.g., 1)
    max_possible_levels = int(np.log2(min(h, w)))

    results_levels = {
        "Original A": img_A,
        "Original B": img_B,
        "Mask (Strong Blur)": (mask_smooth_strong * 255).astype(np.uint8),
    }
    print(
        "\nExperimenting with number of pyramid levels (using strongly blurred mask)..."
    )
    for levels in levels_to_test:
        if levels > max_possible_levels:
            print(
                f"Skipping levels={levels}, max possible for image size is approx {max_possible_levels}"
            )
            continue
        print(f"  Blending with {levels} levels...")
        blended_img = blend_images_laplacian(img_A, img_B, mask_smooth_strong, levels)
        if blended_img is not None:
            results_levels[f"Blended ({levels} levels)"] = blended_img
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"04_blended_levels_{levels}.png"), blended_img
            )
        else:
            print(f"  Blending failed for {levels} levels.")
            results_levels[f"Blended ({levels} levels)"] = np.zeros_like(
                img_A
            )  # Placeholder

    display_results(
        results_levels,
        "Effect of Number of Pyramid Levels (Strongly Blurred Mask)",
        cols=3,
    )
    print("Observations on number of levels:")
    print(
        "- Too few levels (e.g., 1): Behaves closer to direct alpha blending with the mask for that single scale; might not handle large low-frequency differences well, seam might be visible."
    )
    print(
        "- Sufficient levels (e.g., 3-5 for typical images): Achieves smooth blending as low frequencies are handled by very blurred masks at coarser levels."
    )
    print(
        "- Too many levels: Diminishing returns. The top levels of pyramid become very small. Can sometimes introduce minor artifacts if top levels are noisy or processed incorrectly. The number of levels is limited by image size (cannot downsample indefinitely)."
    )

    print(f"\nAll outputs saved in '{OUTPUT_DIR}'.")