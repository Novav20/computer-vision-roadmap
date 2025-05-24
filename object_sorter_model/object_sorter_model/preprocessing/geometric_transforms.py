# object_sorter_model/object_sorter_model/preprocessing/geometric_transforms.py
import numpy as np
import cv2


def get_perspective_transform_matrix(src_points, output_width, output_height):
    """
    Calculates the perspective transform (homography) matrix.

    Args:
        src_points (np.ndarray): 4 source points (4x2 float32 array) in the input image
                                 (e.g., from the undistorted camera view).
                                 Order typically: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        output_width (int): Desired width of the warped rectangular output image.
        output_height (int): Desired height of the warped rectangular output image.

    Returns:
        np.ndarray: The 3x3 perspective transformation matrix, or None if error.
        tuple: The output dimensions (width, height) used for the destination points.
    """
    if not isinstance(src_points, np.ndarray) or src_points.shape != (4, 2):
        print("Error: src_points must be a 4x2 NumPy float32 array.")
        return None, (0, 0)
    if not (
        isinstance(output_width, int)
        and output_width > 0
        and isinstance(output_height, int)
        and output_height > 0
    ):
        print("Error: output_width and output_height must be positive integers.")
        return None, (0, 0)

    # Define standard destination points for the rectangular output
    dst_points = np.float32(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ]
    )

    try:
        # getPerspectiveTransform requires exactly 4 points for src and dst
        homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    except cv2.error as e:
        print(f"Error calculating perspective transform: {e}")
        return None, (output_width, output_height)

    if (
        homography_matrix is None
    ):  # Should be caught by try-except but good to double check
        print(
            "Error: Could not compute homography matrix (cv2.getPerspectiveTransform returned None)."
        )
        return None, (output_width, output_height)

    return homography_matrix, (output_width, output_height)


def warp_frame_perspective(frame, homography_matrix, output_size_wh):
    """
    Applies perspective warp to a frame.

    Args:
        frame (np.ndarray): The input image (BGR).
        homography_matrix (np.ndarray): The 3x3 perspective transformation matrix.
        output_size_wh (tuple): The desired (width, height) of the output warped image.

    Returns:
        np.ndarray: The warped image, or None if an error occurs.
    """
    if frame is None:
        print("Error: Input frame for perspective warp is None.")
        return None
    if homography_matrix is None:
        print("Error: Homography matrix is None. Cannot warp.")
        return None
    if not (
        isinstance(output_size_wh, tuple)
        and len(output_size_wh) == 2
        and output_size_wh[0] > 0
        and output_size_wh[1] > 0
    ):
        print(
            "Error: output_size_wh must be a tuple of (positive_width, positive_height)."
        )
        return None

    try:
        warped_image = cv2.warpPerspective(frame, homography_matrix, output_size_wh)
        return warped_image
    except cv2.error as e:
        print(f"Error during cv2.warpPerspective: {e}")
        return None


def downsample_image(image, factor):
    """
    Downsamples an image using INTER_AREA for good quality.

    Args:
        image (np.ndarray): Input image.
        factor (int): Factor by which to downsample (e.g., 2 for half size).
                      Must be > 0. If 1 or less, original image is returned.

    Returns:
        np.ndarray: Downsampled image.
    """
    if not isinstance(factor, int) or factor <= 1:
        if factor != 1:  # Only warn if not explicitly asking for original size
            print(f"Warning: Invalid downsample factor {factor}. Returning original image.")
        return image
    if image is None:
        print("Error: Input image to downsample_image is None.")
        return None

    new_width = image.shape[1] // factor
    new_height = image.shape[0] // factor

    if new_width < 1 or new_height < 1:
        print(
            f"Warning: Downsample factor {factor} results in zero or negative dimensions. Returning original image."
        )
        return image

    try:
        downsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        return downsampled_image
    except cv2.error as e:
        print(f"Error during cv2.resize (downsample): {e}")
        return image  # Or None, depending on desired error handling


def upsample_image(image, factor, interpolation_method=cv2.INTER_LINEAR):
    """
    Upsamples an image using a specified interpolation method.

    Args:
        image (np.ndarray): Input image.
        factor (int or float): Factor by which to upsample (e.g., 2 for double size).
                               Must be > 0. If 1 or less, original image is returned.
        interpolation_method (int, optional): OpenCV interpolation flag.
                                              Defaults to cv2.INTER_LINEAR.

    Returns:
        np.ndarray: Upsampled image.
    """
    if factor <= 1:
        if factor != 1:
            print(f"Warning: Invalid upsample factor {factor}. Returning original image.")
        return image
    if image is None:
        print("Error: Input image to upsample_image is None.")
        return None

    new_width = int(image.shape[1] * factor)
    new_height = int(image.shape[0] * factor)

    if new_width < 1 or new_height < 1:  # Should not happen with factor > 1 unless original is 0-dim
        print(
            f"Warning: Upsample factor {factor} results in very small/invalid dimensions. Returning original image."
        )
        return image

    try:
        upsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=interpolation_method
        )
        return upsampled_image
    except cv2.error as e:
        print(f"Error during cv2.resize (upsample): {e}")
        return image  # Or None
