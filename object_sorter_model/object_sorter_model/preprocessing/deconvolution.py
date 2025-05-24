# object_sorter_model/object_sorter_model/preprocessing/deconvolution.py
import cv2
import numpy as np
from scipy.signal import convolve2d # For creating example blurred images if needed

def generate_gaussian_blur_kernel(kernel_size=15, sigma=3.0):
    """Generates a 2D Gaussian blur kernel."""
    kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.astype(np.float32)
    return kernel_2d / np.sum(kernel_2d)

def get_otf(kernel_spatial, target_shape):
    """
    Computes the Optical Transfer Function (OTF) H(u,v) from a spatial kernel.
    The kernel is padded and shifted appropriately for DFT.
    """
    k_h, k_w = kernel_spatial.shape
    
    # Pad kernel to the target shape
    padded_kernel = np.zeros(target_shape, dtype=np.float32)
    
    # Place the spatial kernel at the top-left of the padded array
    # For DFT to represent convolution correctly, the (0,0) of the kernel
    # (its center) should effectively be at the (0,0) of the spatial domain before DFT.
    # This is achieved by putting the kernel at the corners after an fftshift-like operation.
    # Or, simpler for cv2.dft: place as is, then cv2.dft, then the H(u,v) will be correct
    # if the image F(u,v) is also standard cv2.dft output.
    # The most common way for OTF is to pad and then shift it such that its center moves to (0,0)
    # of the padded array, then take DFT. np.fft.ifftshift does this "center to origin" shift.
    
    # Create a padded version with kernel at center, then shift it to origin for DFT
    temp_padded = np.zeros(target_shape, dtype=np.float32)
    r_start, c_start = (target_shape[0] - k_h)//2, (target_shape[1] - k_w)//2
    temp_padded[r_start:r_start+k_h, c_start:c_start+k_w] = kernel_spatial
    
    kernel_for_otf = np.fft.ifftshift(temp_padded) # Shift center to (0,0)

    otf = cv2.dft(kernel_for_otf, flags=cv2.DFT_COMPLEX_OUTPUT)
    return otf # This is H(u,v)

def deconvolve_inverse_filter(blurred_image_float, blur_kernel_spatial, epsilon=1e-3):
    """
    Performs deconvolution using Inverse Filtering.
    Assumes blurred_image_float is normalized [0,1].
    """
    if blurred_image_float is None or blur_kernel_spatial is None:
        print("Error: Input image or kernel is None for inverse_filter.")
        return None

    # DFT of the blurred image
    dft_blurred = cv2.dft(blurred_image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # OTF of the blur kernel
    otf_kernel = get_otf(blur_kernel_spatial, blurred_image_float.shape) # H(u,v)
    
    # Complex division: G(u,v) / H(u,v)
    denom_H_mag_sq = otf_kernel[:,:,0]**2 + otf_kernel[:,:,1]**2 + epsilon
    
    F_hat_real = (dft_blurred[:,:,0] * otf_kernel[:,:,0] + \
                  dft_blurred[:,:,1] * otf_kernel[:,:,1]) / denom_H_mag_sq
                  
    F_hat_imag = (dft_blurred[:,:,1] * otf_kernel[:,:,0] - \
                  dft_blurred[:,:,0] * otf_kernel[:,:,1]) / denom_H_mag_sq

    dft_F_hat = np.dstack((F_hat_real, F_hat_imag))
    
    idft_F_hat = cv2.idft(dft_F_hat, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    
    restored_img = np.clip(idft_F_hat, 0, 1)
    return (restored_img * 255).astype(np.uint8) # Convert back to uint8

def deconvolve_wiener_filter(blurred_image_float, blur_kernel_spatial, K_wiener=0.01):
    """
    Performs deconvolution using Wiener Filtering.
    Assumes blurred_image_float is normalized [0,1].
    K_wiener is an estimate of the Noise-to-Signal Ratio (NSR).
    """
    if blurred_image_float is None or blur_kernel_spatial is None:
        print("Error: Input image or kernel is None for wiener_filter.")
        return None

    dft_blurred = cv2.dft(blurred_image_float, flags=cv2.DFT_COMPLEX_OUTPUT) # G(u,v)
    otf_kernel = get_otf(blur_kernel_spatial, blurred_image_float.shape)    # H(u,v)
    
    otf_conj_real = otf_kernel[:,:,0]
    otf_conj_imag = -otf_kernel[:,:,1] # H*(u,v)
    
    otf_mag_sq = otf_kernel[:,:,0]**2 + otf_kernel[:,:,1]**2 # |H(u,v)|^2
    
    # Wiener transfer function W = H* / (|H|^2 + K)
    W_denom = otf_mag_sq + K_wiener
    W_real = otf_conj_real / W_denom
    W_imag = otf_conj_imag / W_denom
    
    # Restored spectrum F_hat = G * W
    F_hat_real = dft_blurred[:,:,0] * W_real - dft_blurred[:,:,1] * W_imag
    F_hat_imag = dft_blurred[:,:,0] * W_imag + dft_blurred[:,:,1] * W_real
    
    dft_F_hat = np.dstack((F_hat_real, F_hat_imag))

    idft_F_hat = cv2.idft(dft_F_hat, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    
    restored_img = np.clip(idft_F_hat, 0, 1)
    return (restored_img * 255).astype(np.uint8)

# Note: For color image deconvolution, one typically processes each channel separately
# or converts to a YCbCr/LAB like space, deconvolves luminance, and recombines.
# The functions above are for grayscale float images [0,1].