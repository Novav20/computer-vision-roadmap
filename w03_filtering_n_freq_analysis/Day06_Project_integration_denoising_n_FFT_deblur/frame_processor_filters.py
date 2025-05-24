import cv2
import numpy as np
import time # For potential performance metrics if needed

# --- Filter Implementations (Re-usable functions) ---

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma_x=1.5):
    """Applies Gaussian blur to an image."""
    # Ensure kernel_size dimensions are odd
    k_w = kernel_size[0] if kernel_size[0] % 2 != 0 else kernel_size[0] + 1
    k_h = kernel_size[1] if kernel_size[1] % 2 != 0 else kernel_size[1] + 1
    return cv2.GaussianBlur(image, (k_w, k_h), sigmaX=sigma_x)

def apply_median_blur(image, kernel_size=5):
    """Applies Median blur to an image."""
    # Ensure kernel_size is odd and > 1
    k_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    if k_size <= 1: k_size = 3
    return cv2.medianBlur(image, k_size)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Applies Bilateral filter to an image."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_box_blur(image, kernel_size=(3,3)):
    """Applies Box blur (mean filter) to an image."""
    # Ensure kernel_size dimensions are positive
    k_w = kernel_size[0] if kernel_size[0] > 0 else 1
    k_h = kernel_size[1] if kernel_size[1] > 0 else 1
    return cv2.blur(image, (k_w, k_h))


# --- Main Frame Processing Function ---

def process_frame_with_filter(frame, filter_type='none', params=None):
    """
    Applies a chosen filter to an input frame.

    Args:
        frame (np.ndarray): The input image (BGR format).
        filter_type (str): Type of filter to apply.
                           Options: 'none', 'gaussian', 'median', 'bilateral', 'box'.
        params (dict, optional): Dictionary of parameters for the chosen filter.
                                 Example for 'gaussian': {'kernel_size': (7,7), 'sigma_x': 2.0}
                                 Example for 'median': {'kernel_size': 7}
                                 Example for 'bilateral': {'d': 15, 'sigma_color': 80, 'sigma_space': 80}
                                 Example for 'box': {'kernel_size': (5,5)}

    Returns:
        np.ndarray: The processed frame.
    """
    if params is None:
        params = {}

    processed_frame = frame.copy() # Work on a copy

    if filter_type == 'gaussian':
        k_size = params.get('kernel_size', (5, 5))
        sigma = params.get('sigma_x', 1.5)
        processed_frame = apply_gaussian_blur(processed_frame, kernel_size=k_size, sigma_x=sigma)
    elif filter_type == 'median':
        k_size = params.get('kernel_size', 5)
        processed_frame = apply_median_blur(processed_frame, kernel_size=k_size)
    elif filter_type == 'bilateral':
        d_val = params.get('d', 9)
        sc_val = params.get('sigma_color', 75)
        ss_val = params.get('sigma_space', 75)
        processed_frame = apply_bilateral_filter(processed_frame, d=d_val, sigma_color=sc_val, sigma_space=ss_val)
    elif filter_type == 'box':
        k_size = params.get('kernel_size', (3,3))
        processed_frame = apply_box_blur(processed_frame, kernel_size=k_size)
    elif filter_type == 'none':
        pass # No filtering
    else:
        print(f"Warning: Unknown filter_type '{filter_type}'. Returning original frame.")

    return processed_frame

# --- Example Usage (Conceptual for a loop) ---
if __name__ == "__main__":
    # This section is for demonstration and testing the process_frame_with_filter function.
    # In a real RPi5 loop, you'd get 'current_frame' from the ESP32-CAM.

    # Create a dummy frame or load a test image
    try:
        # Try to load a common sample image if available
        # sample_image_path = cv2.samples.findFile('lena.png')
        sample_image_path = 'cv-w03-filtering_n_freq_analysis/Day04_median_n_bilateral_filters/Marie.png'
        current_frame = cv2.imread(sample_image_path)
        if current_frame is None:
            raise FileNotFoundError
    except:
        print("Image not found, using a dummy noisy image for testing.")
        current_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # Add some synthetic noise
        noise = np.random.randint(-30, 30, current_frame.shape, dtype=np.int16)
        current_frame = np.clip(current_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


    # --- Runtime Switch Simulation ---
    # In a real application, this could come from a UI, command line, or config file
    active_filter_type = 'bilateral' # Change this to test: 'gaussian', 'median', 'box', 'none'
    filter_params = {}

    if active_filter_type == 'gaussian':
        filter_params = {'kernel_size': (7, 7), 'sigma_x': 2.0}
    elif active_filter_type == 'median':
        filter_params = {'kernel_size': 7}
    elif active_filter_type == 'bilateral':
        filter_params = {'d': 9, 'sigma_color': 50, 'sigma_space': 50} # Adjusted for potentially noisy frame
    elif active_filter_type == 'box':
        filter_params = {'kernel_size': (5,5)}

    print(f"Original frame shape: {current_frame.shape}")
    print(f"Simulating frame processing with filter: '{active_filter_type}'")
    if filter_params:
        print(f"Using parameters: {filter_params}")

    start_time = time.time()
    filtered_frame = process_frame_with_filter(current_frame, active_filter_type, filter_params)
    processing_time = time.time() - start_time

    print(f"Filtered frame shape: {filtered_frame.shape}")
    print(f"Processing time: {processing_time:.4f} seconds")

    # In a real application, 'filtered_frame' would be used for further processing or display.
    # For this script, we'll just show them if OpenCV's highgui is available.
    try:
        cv2.imshow("Original Frame", current_frame)
        cv2.imshow(f"Filtered Frame ({active_filter_type})", filtered_frame)
        print("\nDisplaying images. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"\nCould not display images (headless environment?): {e}")
        print("To see visual results, run in an environment with GUI support or save the images:")
        # cv2.imwrite("original_frame_test.png", current_frame)
        # cv2.imwrite(f"filtered_frame_{active_filter_type}_test.png", filtered_frame)

    print("\nScript demonstration finished.")