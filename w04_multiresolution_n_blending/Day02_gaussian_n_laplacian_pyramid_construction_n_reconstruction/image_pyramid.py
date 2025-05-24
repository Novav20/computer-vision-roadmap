# image_pyramids.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

# --- Configuration ---
IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'lucca.png')
PYRAMID_DEPTH = 4 # Number of levels (G0, G1, G2, G3 means depth 3 if G0 is level 0)
                  # If depth=4, we'll have G0, G1, G2, G3. L0, L1, L2, L3 (where L3=G3)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_week4_tuesday_pyramids')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Function for Displaying Pyramid Levels ---
def display_pyramid_levels(pyramid_levels_dict: dict, main_title="Pyramid Levels", cols=None):
    """Displays a list of pyramid levels (images)."""
    num_images = len(pyramid_levels_dict)
    if num_images == 0: return

    if cols is None:
        cols = num_images # Display horizontally by default if not specified
    rows = int(np.ceil(num_images / cols))
    
    # Adjust figure size based on number of images and columns
    fig_width = cols * 3.5
    fig_height = rows * 3.5 + (0.5 if main_title else 0)


    plt.figure(figsize=(fig_width, fig_height))
    if main_title:
        plt.suptitle(main_title, fontsize=14)
    
    for i, (title, img_level) in enumerate(pyramid_levels_dict.items()):
        plt.subplot(rows, cols, i + 1)
        if img_level is None:
            plt.title(title + "\n(N/A)")
            plt.axis('off')
            continue
            
        if img_level.ndim == 3 and img_level.shape[2] == 3 : # Color
            plt.imshow(cv2.cvtColor(img_level, cv2.COLOR_BGR2RGB))
        elif img_level.ndim == 2 or (img_level.ndim == 3 and img_level.shape[2] == 1): # Grayscale
            plt.imshow(img_level.squeeze(), cmap='gray') # Squeeze in case of (H,W,1)
        else:
            plt.imshow(img_level.astype(np.uint8)) # Attempt to display other types

        plt.title(f"{title}\n{img_level.shape[1]}x{img_level.shape[0]}")
        plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95 if main_title else 1.0])
    plt.show()

# --- Block 1: Pyramid Construction ---

def build_gaussian_pyramid(image, levels):
    """
    Builds a Gaussian pyramid of a specified number of levels.
    Args:
        image (np.ndarray): The input image (BGR or Grayscale).
        levels (int): The number of pyramid levels to generate (G0, G1, ..., G_levels-1).
                      So, levels=4 means G0, G1, G2, G3.
    Returns:
        list: A list of images representing the Gaussian pyramid, starting with G0 (original).
    """
    if image is None: return []
    pyramid = [image.copy()] # G0 is the original image
    current_level_img = image.copy()
    for _ in range(levels - 1): # Need levels-1 reduce operations
        # cv2.pyrDown already applies Gaussian blurring before downsampling
        downsampled = cv2.pyrDown(current_level_img)
        pyramid.append(downsampled)
        current_level_img = downsampled
    return pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    """
    Builds a Laplacian pyramid from a given Gaussian pyramid.
    Args:
        gaussian_pyramid (list): A list of images from G0 to Gn.
    Returns:
        list: A list of images representing the Laplacian pyramid (L0, L1, ..., Ln-1, Ln=Gn).
    """
    if not gaussian_pyramid: return []
    
    laplacian_pyramid = []
    num_levels = len(gaussian_pyramid)

    for i in range(num_levels - 1):
        # EXPAND(G_{i+1}) : Upsample G_{i+1} then blur
        # cv2.pyrUp upsamples and blurs. Ensure its output size matches G_i.
        # Sometimes pyrUp output can be 1 pixel off from G_i due to odd/even dimensions.
        # We need to resize expanded_G_i_plus_1 to match G_i's shape exactly for subtraction.
        
        G_i = gaussian_pyramid[i]
        G_i_plus_1 = gaussian_pyramid[i+1]
        
        expanded_G_i_plus_1 = cv2.pyrUp(G_i_plus_1)
        
        # Ensure shapes match for subtraction by resizing/cropping expanded if needed
        # This handles cases where pyrUp might result in slightly different dimensions
        # than the original G_i due to the halving/doubling process.
        h_gi, w_gi = G_i.shape[:2]
        expanded_G_i_plus_1_resized = cv2.resize(expanded_G_i_plus_1, (w_gi, h_gi))

        # L_i = G_i - EXPAND(G_{i+1})
        # Difference images can have negative values, so use float for calculation
        # and then be prepared to handle for display (e.g., normalize or add offset).
        L_i = cv2.subtract(G_i.astype(np.float32), expanded_G_i_plus_1_resized.astype(np.float32))
        laplacian_pyramid.append(L_i)
        
    # The last level of the Laplacian pyramid is the smallest Gaussian level
    laplacian_pyramid.append(gaussian_pyramid[-1].copy()) # Ln = Gn
    return laplacian_pyramid

# --- Block 2: Pyramid Reconstruction & Error Analysis ---

def reconstruct_from_laplacian_pyramid(laplacian_pyramid):
    """
    Reconstructs the original image from its Laplacian pyramid.
    Args:
        laplacian_pyramid (list): Laplacian pyramid (L0, ..., Ln).
    Returns:
        np.ndarray: The reconstructed image.
    """
    if not laplacian_pyramid: return None
    
    # Start with the smallest level (Ln = Gn)
    current_reconstructed = laplacian_pyramid[-1].copy().astype(np.float32)
    
    # Iterate from Ln-1 down to L0
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        # EXPAND(current_reconstructed which is G_{i+1})
        expanded_current = cv2.pyrUp(current_reconstructed)
        
        L_i = laplacian_pyramid[i].astype(np.float32) # The current Laplacian level
        
        # Ensure shapes match for addition
        h_li, w_li = L_i.shape[:2]
        expanded_current_resized = cv2.resize(expanded_current, (w_li, h_li))
        
        # G_i = L_i + EXPAND(G_{i+1})
        current_reconstructed = cv2.add(L_i, expanded_current_resized)
        
    # Clip and convert to uint8
    reconstructed_image = np.clip(current_reconstructed, 0, 255).astype(np.uint8)
    return reconstructed_image


# --- Main Execution ---
if __name__ == "__main__":
    # Load image
    img_original_bgr = cv2.imread(IMAGE_PATH)
    if img_original_bgr is None:
        img_original_bgr = cv2.imread(cv2.samples.findFile('lena.png')) # Fallback
        if img_original_bgr is None:
            raise FileNotFoundError(f"Could not load test image: {IMAGE_PATH} or fallback.")

    # For simplicity in observing Laplacian values, let's use grayscale for pyramids
    img_original_gray = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2GRAY)
    print(f"Original image loaded: {img_original_gray.shape}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "0_original_gray.png"), img_original_gray)

    print(f"\n--- Block 1: Building Pyramids (Depth={PYRAMID_DEPTH}) ---")
    # Build Gaussian Pyramid
    gaussian_pyr = build_gaussian_pyramid(img_original_gray, levels=PYRAMID_DEPTH)
    print(f"Gaussian Pyramid built with {len(gaussian_pyr)} levels.")
    gp_display_dict = {f"G{i}": level for i, level in enumerate(gaussian_pyr)}
    display_pyramid_levels(gp_display_dict, "Gaussian Pyramid Levels", cols=PYRAMID_DEPTH)
    for i, level_img in enumerate(gaussian_pyr):
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"1_gaussian_level_{i}.png"), level_img)

    # Build Laplacian Pyramid
    laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)
    print(f"Laplacian Pyramid built with {len(laplacian_pyr)} levels.")
    
    # For displaying Laplacian levels, they often have negative values.
    # We can normalize them or add an offset for visualization.
    # Displaying L_i + 128 for visualization (shifting [-128, 127] to [0, 255] approx)
    lp_display_dict = {}
    for i, level_img_float in enumerate(laplacian_pyr):
        if i < len(laplacian_pyr) -1 : # L0 to Ln-1 are difference images
            # Normalize to 0-255 for display: (val - min) / (max - min) * 255
            min_val, max_val = np.min(level_img_float), np.max(level_img_float)
            if max_val == min_val: # Avoid division by zero if flat
                display_level = np.full_like(level_img_float, 128, dtype=np.uint8)
            else:
                display_level = ((level_img_float - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            lp_display_dict[f"L{i} (Normalized)"] = display_level
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"2_laplacian_level_{i}_normalized.png"), display_level)
        else: # Ln = Gn
            lp_display_dict[f"L{i} (Same as G{i})"] = level_img_float.astype(np.uint8) # Already uint8
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"2_laplacian_level_{i}_Gn.png"), level_img_float.astype(np.uint8))

    display_pyramid_levels(lp_display_dict, "Laplacian Pyramid Levels", cols=PYRAMID_DEPTH)


    print("\n--- Block 2: Reconstructing from Laplacian Pyramid ---")
    img_reconstructed = reconstruct_from_laplacian_pyramid(laplacian_pyr)
    if img_reconstructed is not None:
        print(f"Image reconstructed. Shape: {img_reconstructed.shape}")
        cv2.imwrite(os.path.join(OUTPUT_DIR, "3_reconstructed_from_laplacian.png"), img_reconstructed)

        # Measure Reconstruction Error
        mse_val = mse(img_original_gray, img_reconstructed)
        psnr_val = psnr(img_original_gray, img_reconstructed, data_range=255)
        
        print(f"\nReconstruction Error Analysis:")
        print(f"  Mean Squared Error (MSE): {mse_val:.4f}")
        print(f"  Peak Signal-to-Noise Ratio (PSNR): {psnr_val:.2f} dB")

        # Display original vs reconstructed
        display_pyramid_levels({
            "Original Grayscale": img_original_gray,
            "Reconstructed Image": img_reconstructed
        }, "Original vs. Reconstructed", cols=2)

        print("\nObservations:")
        print("- Reconstruction is typically very good (high PSNR, low MSE) if no quantization or info loss occurs during pyramid ops.")
        print("- Minor differences can arise due to floating point precision in intermediate L_i levels and resizing during EXPAND.")
        print("- If L_i levels were heavily processed (e.g., quantized for compression), reconstruction quality would degrade.")
        print("- The `cv2.subtract` and `cv2.add` for Laplacian levels should ideally handle values outside [0,255] by using float types before clipping the final result.")
    else:
        print("Reconstruction failed.")
        
    print(f"\nProcessing complete. Outputs saved in '{OUTPUT_DIR}'")