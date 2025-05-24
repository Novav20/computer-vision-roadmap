# object_sorter_model/object_sorter_model/preprocessing/filtering.py
import cv2
import numpy as np

def apply_gaussian_blur(image, kernel_size_wh=(5, 5), sigma_x=1.5):
    """Applies Gaussian blur to an image."""
    if image is None: return None
    k_w = kernel_size_wh[0] if kernel_size_wh[0] % 2 != 0 else kernel_size_wh[0] + 1
    k_h = kernel_size_wh[1] if kernel_size_wh[1] % 2 != 0 else kernel_size_wh[1] + 1
    try:
        return cv2.GaussianBlur(image, (k_w, k_h), sigmaX=sigma_x)
    except cv2.error as e:
        print(f"Error in GaussianBlur: {e}")
        return image # return original on error

def apply_median_blur(image, kernel_size=5):
    """Applies Median blur to an image."""
    if image is None: return None
    k_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    if k_size <= 1: k_size = 3
    try:
        return cv2.medianBlur(image, k_size)
    except cv2.error as e:
        print(f"Error in medianBlur: {e}")
        return image

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Applies Bilateral filter to an image."""
    if image is None: return None
    try:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    except cv2.error as e:
        print(f"Error in bilateralFilter: {e}")
        return image

def apply_box_blur(image, kernel_size_wh=(3,3)):
    """Applies Box blur (mean filter) to an image."""
    if image is None: return None
    k_w = kernel_size_wh[0] if kernel_size_wh[0] > 0 else 1
    k_h = kernel_size_wh[1] if kernel_size_wh[1] > 0 else 1
    try:
        return cv2.blur(image, (k_w, k_h))
    except cv2.error as e:
        print(f"Error in box blur (cv2.blur): {e}")
        return image

def apply_denoising_filter(frame, filter_type='none', params=None):
    """
    Applies a chosen denoising filter to an input frame.

    Args:
        frame (np.ndarray): The input image (BGR format).
        filter_type (str): Type of filter to apply.
                           Options: 'none', 'gaussian', 'median', 'bilateral', 'box'.
        params (dict, optional): Dictionary of parameters for the chosen filter.
                                 See individual apply_* functions for param examples.

    Returns:
        np.ndarray: The processed frame, or original frame if filter_type is 'none' or unknown.
    """
    if frame is None:
        print("Error: Input frame to apply_denoising_filter is None.")
        return None
    if params is None:
        params = {}

    processed_frame = frame.copy()

    if filter_type.lower() == 'gaussian':
        k_size = params.get('kernel_size_wh', (5, 5))
        sigma = params.get('sigma_x', 1.5)
        processed_frame = apply_gaussian_blur(processed_frame, kernel_size_wh=k_size, sigma_x=sigma)
    elif filter_type.lower() == 'median':
        k_size = params.get('kernel_size', 5)
        processed_frame = apply_median_blur(processed_frame, kernel_size=k_size)
    elif filter_type.lower() == 'bilateral':
        d_val = params.get('d', 9)
        sc_val = params.get('sigma_color', 75)
        ss_val = params.get('sigma_space', 75)
        processed_frame = apply_bilateral_filter(processed_frame, d=d_val, sigma_color=sc_val, sigma_space=ss_val)
    elif filter_type.lower() == 'box':
        k_size = params.get('kernel_size_wh', (3,3))
        processed_frame = apply_box_blur(processed_frame, kernel_size_wh=k_size)
    elif filter_type.lower() == 'none':
        pass # No filtering
    else:
        print(f"Warning: Unknown filter_type '{filter_type}' in apply_denoising_filter. Returning original frame.")
        return frame # Return original frame, not copy

    return processed_frame