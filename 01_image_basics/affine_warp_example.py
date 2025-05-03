import cv2
import numpy as np

img = cv2.imread("01_image_basics/civilization.png")
if img is None:
    print("Error: could not load image")
    exit()
rows, cols = img.shape[:2]

# Rotate 45° around the center, no scaling
center = (rows / 2, cols / 2)
angle = 45
scale = 1
M_rotate = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
print(f"Rotation matrix (2x3):\n{M_rotate}")

# Example: Define a translation matrix manually
tx = 50
ty = 20
M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
print(f"Translation matrix (2x3):\n{M_translate}")

# Let's use the rotation matrix for warping
M = M_rotate
img_affine_warped = cv2.warpAffine(img, M, (cols, rows))
cv2.imwrite("01_image_basics/output_images/affine_warped_output.jpg", img=img_affine_warped)
cv2.imshow("Original", img)
cv2.imshow("Warped image", img_affine_warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
