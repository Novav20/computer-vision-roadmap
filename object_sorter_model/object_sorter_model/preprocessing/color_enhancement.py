# object_sorter_model/object_sorter_model/preprocessing/color_enhancement.py
import cv2
import numpy as np

def equalize_luminance_channel(image_bgr, color_space_choice='LAB'):
    """
    Converts an image to the specified color space (HSV or LAB),
    equalizes its luminance channel, and converts it back to BGR.

    Args:
        image_bgr (np.ndarray): Input BGR image.
        color_space_choice (str, optional): The color space to use for luminance
                                            equalization. Options: 'HSV', 'LAB'.
                                            Defaults to 'LAB'.

    Returns:
        tuple: (processed_bgr_image, original_luminance_channel, equalized_luminance_channel)
               Returns (original_image_bgr, None, None) if an error occurs or unsupported
               color space is chosen.
    """
    if image_bgr is None:
        print("Error: Input image_bgr to equalize_luminance_channel is None.")
        return None, None, None # Or raise error

    processed_image_bgr = image_bgr.copy() # Default to original if something fails
    luminance_channel_original = None
    luminance_channel_equalized = None

    try:
        if color_space_choice.upper() == 'HSV':
            img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(img_hsv)
            luminance_channel_original = v.copy()
            v_eq = cv2.equalizeHist(v)
            luminance_channel_equalized = v_eq.copy()
            img_hsv_eq = cv2.merge([h, s, v_eq])
            processed_image_bgr = cv2.cvtColor(img_hsv_eq, cv2.COLOR_HSV2BGR)
        elif color_space_choice.upper() == 'LAB':
            # cv2.equalizeHist expects uint8 input for the channel.
            # So, we convert to LAB, split, equalize L (which is uint8 scaled version of L*), merge, convert back.
            img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            luminance_channel_original = l_channel.copy()
            l_eq = cv2.equalizeHist(l_channel)
            luminance_channel_equalized = l_eq.copy()
            img_lab_eq = cv2.merge([l_eq, a_channel, b_channel])
            processed_image_bgr = cv2.cvtColor(img_lab_eq, cv2.COLOR_Lab2BGR)
        else:
            print(f"Error: Unsupported color_space_choice '{color_space_choice}'. Options are 'HSV' or 'LAB'.")
            return image_bgr, None, None
    except cv2.error as e:
        print(f"OpenCV error during luminance equalization with {color_space_choice}: {e}")
        return image_bgr, None, None # Return original on error

    return processed_image_bgr, luminance_channel_original, luminance_channel_equalized

# You can add other color enhancement functions here in the future,
# e.g., gamma_correction, contrast_stretching, etc.