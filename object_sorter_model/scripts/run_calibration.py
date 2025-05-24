# object_sorter_model/scripts/run_calibration.py
import numpy as np
import cv2
import glob
import os
import argparse

# --- Default Paths and Values ---
# Determine project root assuming 'scripts' is a child of the project root
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir)) # Go up one level from 'scripts'

DEFAULT_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "calibration_images")
DEFAULT_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "object_sorter_model", "config", "camera_calibration_data.npz")
DEFAULT_CHESSBOARD_WIDTH = 9
DEFAULT_CHESSBOARD_HEIGHT = 6
DEFAULT_SQUARE_SIZE = 25.0 # Example: 25mm

def calibrate_camera(images_path, output_filepath, board_width, board_height, square_size, image_format='*.jpg'):
    """
    Performs camera calibration using chessboard images.

    Args:
        images_path (str): Path to the directory containing calibration images.
        output_filepath (str): Path where the calibration data (.npz) will be saved.
        board_width (int): Number of inner corners along the width of the chessboard.
        board_height (int): Number of inner corners along the height of the chessboard.
        square_size (float): Size of a chessboard square in real-world units (e.g., mm).
        image_format (str): Glob pattern for image files (e.g., '*.jpg', '*.png').
    """
    print("--- Starting Camera Calibration ---")
    print(f"Reading images from: {images_path}")
    print(f"Chessboard size: {board_width}x{board_height} inner corners")
    print(f"Square size: {square_size} units")
    print(f"Output file: {output_filepath}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((board_height * board_width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints_list = []
    imgpoints_list = []
    
    images = glob.glob(os.path.join(images_path, image_format))
    if not images:
        print(f"Error: No images found in '{images_path}' with format '{image_format}'.")
        return False
    print(f"Found {len(images)} images for calibration.")

    img_size = None
    processed_count = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Failed to load image {fname}")
            continue

        if img_size is None:
            img_size = img.shape[:2][::-1]
            print(f"Image size detected: {img_size}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (board_width, board_height), None)

        if ret:
            processed_count += 1
            print(f"Found corners in {os.path.basename(fname)} ({processed_count}/{len(images)})")
            objpoints_list.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints_list.append(corners2)
            
            # Optional: Draw and display corners (uncomment to debug)
            # cv2.drawChessboardCorners(img, (board_width, board_height), corners2, ret)
            # cv2.imshow('Corners Found', cv2.resize(img, (640,480)))
            # cv2.waitKey(100)
        else:
            print(f"Warning: Chessboard corners not found in {os.path.basename(fname)}")
    
    # if processed_count > 0 : cv2.destroyAllWindows()

    if not objpoints_list or not imgpoints_list or processed_count == 0:
        print("Error: Could not find corners in any suitable image. Calibration failed.")
        return False
    
    print(f"\nProcessed {processed_count} images successfully for calibration.")
    print("Running cv2.calibrateCamera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_list, imgpoints_list, img_size, None, None)

    if not ret:
        print("Error: cv2.calibrateCamera failed.")
        return False

    print("\nCalibration Successful!")
    print("Camera Matrix (K):\n", mtx)
    print("Distortion Coefficients:\n", dist)

    mean_error = 0
    for i in range(len(objpoints_list)):
        imgpoints2, _ = cv2.projectPoints(objpoints_list[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints_list[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    reprojection_error = mean_error / len(objpoints_list)
    print(f"Mean Reprojection Error: {reprojection_error:.4f} pixels")

    print("\nGenerating undistortion maps...")
    alpha = 1.0 # Retain all pixels, may result in black borders
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_size, alpha, img_size)
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, img_size, cv2.CV_32FC1)
    print("Undistortion maps generated.")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty string
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    print(f"Saving calibration data to '{output_filepath}'...")
    np.savez(output_filepath,
             mtx=mtx,
             dist=dist,
             new_camera_mtx=new_camera_mtx,
             roi=roi,
             map1=map1,
             map2=map2,
             img_size=np.array(img_size))
    print("Calibration data saved successfully.")
    print("--- Calibration Process Complete ---")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration using chessboard images.")
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR,
                        help="Path to the directory containing calibration images.")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE,
                        help="Path to save the output calibration data (.npz file).")
    parser.add_argument("--board_width", type=int, default=DEFAULT_CHESSBOARD_WIDTH,
                        help="Number of inner corners along the width of the chessboard.")
    parser.add_argument("--board_height", type=int, default=DEFAULT_CHESSBOARD_HEIGHT,
                        help="Number of inner corners along the height of the chessboard.")
    parser.add_argument("--square_size", type=float, default=DEFAULT_SQUARE_SIZE,
                        help="Size of a chessboard square in real-world units (e.g., mm).")
    parser.add_argument("--format", type=str, default='*.jpg',
                        help="Image file format glob pattern (e.g., '*.jpg', '*.png').")
    
    args = parser.parse_args()

    if not os.path.isdir(args.images_dir):
        print(f"Error: Calibration images directory not found: {args.images_dir}")
        print(f"Please create it and add chessboard images, or specify a valid path with --images_dir.")
        print(f"Example: place images in '{DEFAULT_IMAGES_DIR}'")
        exit(1)

    calibrate_camera(args.images_dir, args.output_file, args.board_width, 
                     args.board_height, args.square_size, args.format)