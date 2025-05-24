# object_sorter_model/object_sorter_model/preprocessing/calibration.py
import numpy as np
import cv2
import os # For path joining, though config loading would be better

# Potentially, a configuration loader would live in utils or config module
# For now, we might hardcode the path or pass it.
DEFAULT_CALIBRATION_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'camera_calibration_data.npz') # Adjust relative path

def load_calibration_data(filepath=DEFAULT_CALIBRATION_FILE_PATH):
    """
    Loads camera matrix, distortion coefficients, and undistortion maps.

    Args:
        filepath (str): Path to the .npz calibration file.

    Returns:
        tuple: (mtx, dist, new_camera_mtx, roi, map1, map2, img_size) or None if error.
    """
    try:
        with np.load(filepath) as data:
            mtx = data['mtx']
            dist = data['dist']
            new_camera_mtx = data.get('new_camera_mtx') # Use .get for optional keys
            roi = data.get('roi')
            map1 = data['map1']
            map2 = data['map2']
            img_size_arr = data.get('img_size')
            img_size = tuple(img_size_arr) if img_size_arr is not None else None
            
            # Basic validation
            if not all([isinstance(map1, np.ndarray), isinstance(map2, np.ndarray)]):
                print(f"Error: map1 or map2 are not valid numpy arrays in {filepath}")
                return None
            if img_size is None and new_camera_mtx is None: # Need one of these to proceed
                print(f"Error: Missing img_size or new_camera_mtx in {filepath}")
                return None


        # print(f"Calibration data loaded successfully from '{filepath}'")
        return mtx, dist, new_camera_mtx, roi, map1, map2, img_size
    except FileNotFoundError:
        print(f"Error: Calibration file not found at '{filepath}'")
        return None
    except KeyError as e:
        print(f"Error: Missing essential key {e} in calibration file '{filepath}'")
        return None
    except Exception as e:
        print(f"Error loading calibration data from '{filepath}': {e}")
        return None

def undistort_frame(frame, map1, map2, roi=None, crop_to_roi=False):
    """
    Applies undistortion to a frame using precomputed remap matrices.

    Args:
        frame (np.ndarray): The input image (BGR).
        map1 (np.ndarray): The first undistortion remap matrix.
        map2 (np.ndarray): The second undistortion remap matrix.
        roi (tuple, optional): Region of Interest (x,y,w,h) from getOptimalNewCameraMatrix.
        crop_to_roi (bool): If True and roi is provided, crops the undistorted image to the ROI.

    Returns:
        np.ndarray: The undistorted (and optionally cropped) image.
    """
    if frame is None:
        print("Error: Input frame for undistortion is None.")
        return None
    if map1 is None or map2 is None:
        print("Error: Undistortion maps (map1, map2) are None.")
        return frame # Return original if maps are missing

    undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    if crop_to_roi and roi is not None and len(roi) == 4:
        x, y, w, h = roi
        if w > 0 and h > 0: # Ensure ROI dimensions are valid
            undistorted_img = undistorted_img[y:y+h, x:x+w]
        else:
            print("Warning: Invalid ROI dimensions provided for cropping. Not cropping.")
            
    return undistorted_img

# You could add a function here to generate maps if only mtx and dist are available
# def generate_undistortion_maps(mtx, dist, img_size, alpha=1.0):
#     new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_size, alpha, img_size)
#     map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, img_size, cv2.CV_32FC1)
#     return map1, map2, new_camera_mtx, roi