import cv2
import numpy as np

img = cv2.imread("01_image_basics/singing.png")
if img is None:
    print("Error: could not load image")
    exit()
rows, cols = img.shape[:2]

# Define 4 source points ( e.g. corners of a region of interest)
pts_src = np.float32([[0, 0],       # top left     
                      [cols, 0],     # top right
                      [0, rows],     # bottom left
                      [cols, rows]])     # bottom right 

# Define where those 4 points should map in the destination image
# Let's simulate a perspective view, making the top edge shorter
pts_dst = np.float32([[cols * 0.25, rows * 0.33],   # top left
                      [cols * 0.75, rows * 0.33],   # top right
                      [cols * 0.15, rows],          # bottom left
                      [cols * 0.85, rows]])         # bottom right


H = cv2.getPerspectiveTransform(pts_src, pts_dst)
print(f"Perspective transformation matrix (3x3):\n{H}")
img_perspective_warped = cv2.warpPerspective(img, H, (cols, rows))
cv2.imwrite("01_image_basics/output_images/perspective_warped_output.jpg", img=img_perspective_warped)
cv2.imshow("Original", img)
cv2.imshow("Warped image", img_perspective_warped)
cv2.waitKey(0)
cv2.destroyAllWindows()