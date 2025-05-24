import numpy as np
import cv2
import glob
import os

# ---- Configuration ----
# Path to the directory containing calibration chessboard images
images_path = "calibration_images" 
# Output file to save calibration data
output_file = "./calibration_data.npz"
# Chessboard dimensions (number of inner corners in x and y) 
chessboard_width = 9
chessboard_height = 6
# Size of the chessboard squares (e.g., in mm, cm, or any unit).
# This only affects the scale of the translation vectors (tvecs),
# not the intrinsics (mtx) or distortion (dist).
# Set to 1.0 if you don't care about real-world units for now.
square_size = 1.0

# ---- Initialization ----
print("Starting camera calibration...")

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points like (0,0,0), (1,0,0), (2,0,0) ..., (chessboard_width-1, chessboard_height-1, 0)
objp = np.zeros((chessboard_height * chessboard_width, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_width, 0:chessboard_height].T.reshape(-1, 2)
objp = objp * square_size # Scale by square size if needed

# Arrays to store object points and image points from all the images.
objpoints_list = [] # 3d point in real world space
imgpoints_list = [] # 2d points in image plane.

# Find calibration images
images = glob.glob(os.path.join(images_path, '*.jpg')) # Adjust wildcard if using .png etc.
if not images:
    print(f"Error: No images found in directory '{images_path}'. Please check the path and file extensions.")
    exit()

print(f"Found {len(images)} images for calibration.")

# --- Process Images ---
img_size = None
processed_count = 0
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Failed to load image {fname}")
        continue

    if img_size is None:
        img_size = img.shape[:2][::-1] # Get (width, height)
        print(f"Image size detected: {img_size}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (chessboard_width, chessboard_height), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        processed_count += 1
        print(f"Found corners in {os.path.basename(fname)} ({processed_count}/{len(images)})")
        objpoints_list.append(objp)

        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints_list.append(corners2)

        # --- Optional: Draw and display the corners ---
        # cv2.drawChessboardCorners(img, (chessboard_width, chessboard_height), corners2, ret)
        # cv2.imshow('Corners Found', img)
        # cv2.waitKey(500) # Display for 0.5 seconds
    else:
        print(f"Warning: Chessboard corners not found in {os.path.basename(fname)}")

# cv2.destroyAllWindows() # Close display windows if used

if not objpoints_list or not imgpoints_list:
     print("Error: Could not find corners in any image. Calibration failed.")
     exit()

print(f"\nProcessed {processed_count} images successfully.")
print("Running calibration...")

# --- Perform Camera Calibration ---
# mtx: Camera Matrix (K)
# dist: Distortion Coefficients (k1, k2, p1, p2, k3[, k4, k5, k6])
# rvecs: Rotation vectors for each image pose
# tvecs: Translation vectors for each image pose
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_list, imgpoints_list, img_size, None, None)

if not ret:
    print("Error: Calibration failed.")
    exit()

print("\nCalibration Successful!")
print("Camera Matrix (K):\n", mtx)
print("\nDistortion Coefficients:\n", dist)

# --- Calculate Reprojection Error ---
mean_error = 0
for i in range(len(objpoints_list)):
    imgpoints2, _ = cv2.projectPoints(objpoints_list[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints_list[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
reprojection_error = mean_error / len(objpoints_list)
print(f"\nMean Reprojection Error: {reprojection_error:.4f} pixels")
# A good calibration typically has an error < 0.5 pixels.

# --- Generate Undistortion Maps ---
print("\nGenerating undistortion maps...")

# Get the optimal new camera matrix (controls cropping after undistortion)
# alpha=1: Retains all pixels from the original image, may result in black borders.
# alpha=0: Crops the image to show only valid pixels after undistortion.
alpha = 1.0 # You can try 0.0 as well
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_size, alpha, img_size)
print("Optimal New Camera Matrix:\n", new_camera_mtx)
print("Region of Interest (ROI) after cropping:\n", roi) # Format (x, y, w, h)

# Generate the mapping functions for cv2.remap()
# map1, map2 contain the coordinates (float) for each pixel in the *destination* (undistorted) image,
# telling remap where to fetch the corresponding pixel from the *source* (distorted) image.
# Using CV_32FC1 for floating-point maps.
map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, img_size, cv2.CV_32FC1)
# For stereo rectification, the R argument would be the rotation between cameras. Here it's None (or identity).

print("Undistortion maps generated.")

# --- Save Results ---
print(f"Saving calibration data to '{output_file}'...")
np.savez(output_file,
         mtx=mtx,
         dist=dist,
         new_camera_mtx=new_camera_mtx,
         roi=roi,
         map1=map1,
         map2=map2,
         img_size=np.array(img_size)) # Save img_size as numpy array too

print("Calibration data saved successfully.")
print("\n--- Calibration Process Complete ---")