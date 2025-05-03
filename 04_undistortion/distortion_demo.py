import cv2
import numpy as np

# Load distorted image
img = cv2.imread('chessboard_distorted.jpg')
h, w = img.shape[:2]

# Assume these are known from calibration
K = np.array([[800, 0, w/2],
              [0, 800, h/2],
              [0, 0, 1]])
dist = np.array([0.1, -0.25, 0, 0, 0])  # k1, k2, p1, p2, k3

# Undistort
new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1)
undistorted = cv2.undistort(img, K, dist, None, new_K)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
