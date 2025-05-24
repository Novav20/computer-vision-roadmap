# object_sorter_model/scripts/run_pipeline_test.py
import cv2
import numpy as np
import os
import sys
import argparse

# --- Add package to Python path ---
PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from object_sorter_model.preprocessing import calibration, geometric_transforms, color_enhancement, filtering, deconvolution # Added filtering, deconvolution
    from object_sorter_model.utils import config_loader
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure this script is run from a context where 'object_sorter_model' package is found.")
    print(f"Current sys.path includes: {PROJECT_ROOT}")
    exit(1)

# --- Default Paths ---
DEFAULT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "object_sorter_model", "config", "camera_params.json")
DEFAULT_TEST_IMAGE = os.path.join(PROJECT_ROOT, "data", "sample_images", "esp32_test_image.jpg")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_pipeline_test")


def run_test_pipeline(image_path, main_config_filepath, output_dir):
    """
    Runs the current version of the image processing pipeline.
    """
    print("--- Running Object Sorter Pipeline Test ---")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Main Configuration
    print(f"Loading main configuration from: {main_config_filepath}")
    main_config = config_loader.load_json_config(main_config_filepath)
    if main_config is None:
        print("Failed to load main configuration. Exiting.")
        return

    # --- Week 1: Geometric Correction ---
    print("\n--- Stage 1: Geometric Correction ---")
    
    # A. Get path to calibration .npz file and load calibration data
    config_dir_path = os.path.dirname(main_config_filepath) # Directory where main_config.json resides
    calibration_npz_path = config_loader.get_camera_calibration_filepath(main_config, config_dir_path)
    
    if not calibration_npz_path or not os.path.exists(calibration_npz_path): # Simplified check
        print(f"ERROR: Calibration NPZ missing: {calibration_npz_path}")
        print("Check 'calibration_file_name' in your JSON config and file existence.")
        return

    calib_data_tuple = calibration.load_calibration_data(calibration_npz_path)
    if calib_data_tuple is None:
        print("Failed to load calibration data. Exiting.")
        return
    _mtx, _dist, _new_mtx, roi_from_calib, map1, map2, _img_size_calib = calib_data_tuple

    # B. Load Test Image
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        print(f"Error loading image: {image_path}") # Updated message
        return
    print(f"Loaded raw image: {os.path.basename(image_path)}, Shape: {raw_image.shape}")
    cv2.imwrite(os.path.join(output_dir, "00_raw_input.png"), raw_image)

    # C. Undistort
    undistorted_image = calibration.undistort_frame(raw_image, map1, map2, roi=roi_from_calib, crop_to_roi=False)
    if undistorted_image is None: return
    print(f"Undistorted image shape: {undistorted_image.shape}")
    cv2.imwrite(os.path.join(output_dir, "01_undistorted.png"), undistorted_image)

    # D. Perspective Warp using parameters from config
    src_pts_cfg, out_w_cfg, out_h_cfg = config_loader.get_perspective_warp_params(main_config)
    if src_pts_cfg is None:
        print("Failed to load perspective warp parameters from config. Exiting.")
        return
    
    print(f"Using warp parameters from config: src_pts={src_pts_cfg.tolist()}, output_size=({out_w_cfg}x{out_h_cfg})")
    perspective_matrix, (out_w, out_h) = geometric_transforms.get_perspective_transform_matrix(
        src_points=src_pts_cfg, output_width=out_w_cfg, output_height=out_h_cfg
    )
    if perspective_matrix is None: return

    rectified_image = geometric_transforms.warp_frame_perspective(undistorted_image, perspective_matrix, (out_w, out_h))
    if rectified_image is None: return
    print(f"Rectified image shape: {rectified_image.shape}")
    cv2.imwrite(os.path.join(output_dir, "02_rectified_roi.png"), rectified_image)
    
    current_processed_image = rectified_image
    print("Geometric correction complete.")
    
    # --- Stage 2: Photometric Preprocessing (Week 2) ---
    print("\n--- Stage 2: Photometric Preprocessing ---") # Updated stage name
    preprocessing_cfg = main_config.get("preprocessing_params", {}) # Get the whole sub-config
    
    # A. Resizing (Downsampling/Upsampling)
    downsample_factor_cfg = preprocessing_cfg.get("downsample_factor", 1) 
    
    if downsample_factor_cfg > 1:
        print(f"Applying downsampling by factor: {downsample_factor_cfg}")
        resized_image = geometric_transforms.downsample_image(current_processed_image, downsample_factor_cfg)
        if resized_image is not None:
            cv2.imwrite(os.path.join(output_dir, f"03_downsampled_x{downsample_factor_cfg}.png"), resized_image)
            current_processed_image = resized_image
            print(f"Downsampled image shape: {current_processed_image.shape}")
        else:
            print("Downsampling failed.")
    else:
        print("Downsampling skipped (factor <= 1 or not configured).")

    # B. Luminance Equalization
    luminance_eq_config = preprocessing_cfg.get("luminance_equalization", {})
    apply_lum_eq = luminance_eq_config.get("apply", False)
    image_before_lum_eq = current_processed_image.copy() # Define for potential display

    if apply_lum_eq:
        lum_eq_color_space_cfg = luminance_eq_config.get("color_space", "LAB") # Default to LAB
        print(f"Applying luminance equalization using {lum_eq_color_space_cfg} space...")
        eq_image, lum_orig, lum_eq = color_enhancement.equalize_luminance_channel(
            current_processed_image, lum_eq_color_space_cfg
        )
        if eq_image is not None:
            cv2.imwrite(os.path.join(output_dir, f"04a_luminance_original_{lum_eq_color_space_cfg}.png"), lum_orig)
            cv2.imwrite(os.path.join(output_dir, f"04b_luminance_equalized_{lum_eq_color_space_cfg}.png"), lum_eq)
            cv2.imwrite(os.path.join(output_dir, f"04_luminance_equalized.png"), eq_image) # Simplified filename
            current_processed_image = eq_image
            print(f"Luminance equalization complete. Output shape: {current_processed_image.shape}")
        else:
            print("Luminance equalization failed.")
    else:
        print("Luminance equalization skipped (not configured to apply).")

    # --- Stage 3: Denoising & Deblurring (Week 3) ---
    print("\n--- Stage 3: Denoising & Deblurring ---")
    
    # A. Denoising Filter
    denoising_config = preprocessing_cfg.get("denoising_filter", {})
    apply_denoising = denoising_config.get("apply", False)
    if apply_denoising:
        denoise_type = denoising_config.get("type", "none")
        denoise_params = denoising_config.get("params", {})
        print(f"Applying denoising filter: {denoise_type} with params: {denoise_params}")
        denoised_image = filtering.apply_denoising_filter(current_processed_image, denoise_type, denoise_params)
        if denoised_image is not None:
            cv2.imwrite(os.path.join(output_dir, f"05_denoised_{denoise_type}.png"), denoised_image)
            current_processed_image = denoised_image
        else:
            print(f"Denoising filter '{denoise_type}' failed or returned None.")
            
    # B. Deconvolution (Deblurring)
    deconv_config = preprocessing_cfg.get("deconvolution", {})
    apply_deconv = deconv_config.get("apply", False)
    if apply_deconv:
        deconv_type = deconv_config.get("type", "wiener")
        blur_kernel_type = deconv_config.get("blur_kernel_type", "gaussian")
        kernel_cfg_params = deconv_config.get("kernel_params", {})
        
        print(f"Applying deconvolution: {deconv_type} assuming {blur_kernel_type} blur...")

        # Generate assumed blur kernel based on config
        assumed_blur_kernel = None
        if blur_kernel_type.lower() == "gaussian":
            k_size = kernel_cfg_params.get("size", 15)
            k_sigma = kernel_cfg_params.get("sigma", 2.0)
            assumed_blur_kernel = deconvolution.generate_gaussian_blur_kernel(k_size, k_sigma)
        # Add other kernel types here if needed (e.g., motion)
        
        if assumed_blur_kernel is not None:
            # Deconvolution functions expect grayscale float [0,1] image
            if current_processed_image.ndim == 3:
                img_for_deconv_gray = cv2.cvtColor(current_processed_image, cv2.COLOR_BGR2GRAY)
            else:
                img_for_deconv_gray = current_processed_image
            
            img_for_deconv_float = img_for_deconv_gray.astype(np.float32) / 255.0
            
            deblurred_image_uint8 = None
            if deconv_type.lower() == "inverse":
                epsilon = deconv_config.get("inverse_epsilon", 1e-3)
                deblurred_image_uint8 = deconvolution.deconvolve_inverse_filter(
                    img_for_deconv_float, assumed_blur_kernel, epsilon=epsilon
                )
            elif deconv_type.lower() == "wiener":
                k_wiener = deconv_config.get("wiener_K", 0.005)
                deblurred_image_uint8 = deconvolution.deconvolve_wiener_filter(
                    img_for_deconv_float, assumed_blur_kernel, K_wiener=k_wiener
                )
            
            if deblurred_image_uint8 is not None:
                # If original was color, we might want to apply deblur to L and recombine,
                # or just show the deblurred grayscale for now.
                cv2.imwrite(os.path.join(output_dir, f"06_deblurred_{deconv_type}.png"), deblurred_image_uint8)
                # If you want to continue pipeline with deblurred image, and it was color:
                # This is a simplification. Proper color deblurring is more complex.
                if current_processed_image.ndim == 3:
                    # Example: convert deblurred grayscale back to BGR (will be grayscale color)
                    # Or apply deblurring channel-wise or on L channel of LAB/HSV
                    print("Note: Deblurring applied to grayscale. For color, more advanced techniques are needed.")
                    current_processed_image = cv2.cvtColor(deblurred_image_uint8, cv2.COLOR_GRAY2BGR) 
                else:
                    current_processed_image = deblurred_image_uint8
            else:
                print(f"Deconvolution type '{deconv_type}' failed.")
        else:
            print(f"Could not generate assumed blur kernel of type '{blur_kernel_type}'. Skipping deconvolution.")

    # --- Future Stages (Placeholders for Week 4 filters etc.) ---
    # print("\n--- Stage X: Advanced Filters (Week 4) ---")
    # filter_config = main_config.get("preprocessing_params", {}).get("advanced_filter", {})
    # filter_type = filter_config.get("type", "none")
    # filter_params = filter_config.get("params", {})
    # if filter_type != "none":
    #    from object_sorter_model.preprocessing import advanced_filtering # Assuming this exists
    #    current_processed_image = advanced_filtering.apply_advanced_filter(current_processed_image, filter_type, filter_params)
    #    cv2.imwrite(os.path.join(output_dir, f"07_advanced_filtered_{filter_type}.png"), current_processed_image)


    print(f"\n--- Pipeline Test Finished. Outputs saved in '{output_dir}' ---")
    try:
        cv2.imshow("Raw Input", raw_image)
        if apply_lum_eq:
            cv2.imshow("Before Luminance Eq", image_before_lum_eq)
        cv2.imshow("Final Processed", current_processed_image) # Show the latest processed image
        print("Displaying images. Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"Could not display images: {e}") # Updated message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for the object sorter CV pipeline.")
    parser.add_argument("--image", type=str, default=DEFAULT_TEST_IMAGE,
                        help="Path to input test image.") # help text aligned
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE, 
                        help="Path to main JSON config file.") # help text aligned
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save output images.") # help text aligned
    
    args = parser.parse_args()

    if not os.path.exists(args.image): exit(f"ERROR: Test image '{args.image}' not found.") # Updated exit
    if not os.path.exists(args.config): exit(f"ERROR: Main config '{args.config}' not found.") # Updated exit
            
    run_test_pipeline(args.image, args.config, args.output)