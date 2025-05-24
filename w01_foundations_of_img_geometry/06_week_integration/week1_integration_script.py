import numpy as np
import cv2
import os

# --- Configuration ---
CALIBRATION_FILE = 'camera_calibration_data.npz'
TEST_IMAGE_FILE = 'esp32_test_image.jpg' # Image captured by ESP32-CAM or similar

OUTPUT_DIR = 'output_week1' # Directory to save results
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Define the Warp Transformation ---
# These points define the region in the *undistorted* image you want to warp
# to a rectangular, top-down view. 
# TODO: You might need to find these coordinates
# by displaying the undistorted image first and clicking points, or estimating.
# Example: Four corners of a rectangular area on the conveyor belt seen in perspective.
# Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] - Order matters! (e.g., TL, TR, BR, BL)
# *** Adjust these points based on your TEST_IMAGE_FILE after undistortion! ***
UNDISTORTED_SRC_POINTS = np.float32([
    [150, 100],  # Top-Left corner of ROI in undistorted image
    [450, 110],  # Top-Right corner
    [500, 350],  # Bottom-Right corner
    [100, 340]   # Bottom-Left corner
])

# TODO: Define the desired output rectangle size and shape for the warped region
# Often chosen to match the aspect ratio of the real-world object/area
OUTPUT_WIDTH = 300
OUTPUT_HEIGHT = 250
TARGET_DST_POINTS = np.float32([
    [0, 0],                     # Target Top-Left
    [OUTPUT_WIDTH - 1, 0],      # Target Top-Right
    [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1], # Target Bottom-Right
    [0, OUTPUT_HEIGHT - 1]      # Target Bottom-Left
])

# --- Helper Functions ---
def load_calibration_data(filepath):
    """Loads camera matrix, distortion coeffs, and undistortion maps."""
    try:
        with np.load(filepath) as data:
            mtx = data['mtx']
            dist = data['dist']
            new_camera_mtx = data['new_camera_mtx']
            roi = data['roi']
            map1 = data['map1']
            map2 = data['map2']
            img_size = tuple(data['img_size'])
        print(f"Calibration data loaded successfully from '{filepath}'")
        return mtx, dist, new_camera_mtx, roi, map1, map2, img_size
    except FileNotFoundError:
        print(f"Error: Calibration file not found at '{filepath}'")
        return None
    except KeyError as e:
        print(f"Error: Missing key {e} in calibration file '{filepath}'")
        return None

def load_image(filepath):
    """Loads an image using OpenCV."""
    img = cv2.imread(filepath)
    if img is None:
        print(f"Error: Could not load image from '{filepath}'")
        return None
    print(f"Image loaded successfully from '{filepath}'")
    return img

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Week 1 Integration ---")

    # 1. Load Calibration Data
    calib_data = load_calibration_data(CALIBRATION_FILE)
    if calib_data is None:
        exit()
    mtx, dist, new_camera_mtx, roi, map1, map2, img_size = calib_data

    # 2. Load the ESP32-CAM Test Image
    original_img = load_image(TEST_IMAGE_FILE)
    if original_img is None:
        exit()

    # Verify image size matches calibration data (optional but good practice)
    h, w = original_img.shape[:2]
    if (w, h) != img_size:
        print(f"Warning: Test image size {(w,h)} differs from calibration image size {img_size}. Results may be inaccurate.")
        # You might resize the test image here if needed, but calibration is specific to an image size.
        # original_img = cv2.resize(original_img, img_size)

    # 3. Apply Undistortion
    # Use cv2.remap with the precomputed maps for efficiency
    print("Applying undistortion...")
    undistorted_img = cv2.remap(original_img, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Optional: Crop the image using the ROI obtained during calibration
    # This removes black areas introduced by undistortion if alpha=1 was used
    x, y, w_roi, h_roi = roi
    undistorted_cropped = undistorted_img[y:y+h_roi, x:x+w_roi]
    print(f"Undistortion applied. ROI cropping: {(x, y, w_roi, h_roi)}")
    # We will use the *full* undistorted image for warping,
    # assuming the SRC_POINTS are within the valid area.

    # 4. Calculate the Perspective Warp (Homography)
    # This warp corrects the perspective based on points defined relative
    # to the *undistorted* camera view.
    print("Calculating perspective transform (Homography)...")
    homography_matrix, status = cv2.findHomography(UNDISTORTED_SRC_POINTS, TARGET_DST_POINTS)
    # Alternative using getPerspectiveTransform if findHomography isn't needed (requires exactly 4 points)
    # homography_matrix = cv2.getPerspectiveTransform(UNDISTORTED_SRC_POINTS, TARGET_DST_POINTS)

    if homography_matrix is None:
        print("Error: Could not compute homography matrix. Check source/destination points.")
        exit()
    print("Homography Matrix:\n", homography_matrix)

    # 5. Apply the Perspective Warp to the Undistorted Image
    print(f"Applying perspective warp to size ({OUTPUT_WIDTH}, {OUTPUT_HEIGHT})...")
    final_rectified_img = cv2.warpPerspective(undistorted_img, homography_matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    # 6. Save and Display Results
    print("Saving output images...")
    original_path = os.path.join(OUTPUT_DIR, '1_original.jpg')
    undistorted_path = os.path.join(OUTPUT_DIR, '2_undistorted.jpg')
    undistorted_cropped_path = os.path.join(OUTPUT_DIR, '3_undistorted_cropped.jpg')
    final_path = os.path.join(OUTPUT_DIR, '4_final_rectified.jpg')

    cv2.imwrite(original_path, original_img)
    cv2.imwrite(undistorted_path, undistorted_img)
    cv2.imwrite(undistorted_cropped_path, undistorted_cropped) # Save the cropped version too
    cv2.imwrite(final_path, final_rectified_img)
    print(f"Outputs saved to '{OUTPUT_DIR}'")

    # Display (optional)
    cv2.imshow('1. Original Image', original_img)
    cv2.imshow('2. Undistorted Image', undistorted_img)
    # cv2.imshow('3. Undistorted Cropped', undistorted_cropped)
    cv2.imshow('4. Final Rectified Image', final_rectified_img)

    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("--- Week 1 Integration Script Finished ---")