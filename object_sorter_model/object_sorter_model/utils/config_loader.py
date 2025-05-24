# object_sorter_model/object_sorter_model/utils/config_loader.py
import json
import os
import numpy as np

# --- Default Configuration Paths (relative to this file's location) ---
# Assuming 'utils' is a child of 'object_sorter_model' package, 
# and 'config' is a sibling of 'utils' within the package.
# Or, more robustly, assume config files are relative to a project root or passed as absolute.

# For this example, let's assume the main config file path is passed to functions,
# or we can define a default relative to the project root if this util is part of a larger app.

def load_json_config(config_filepath):
    """Loads a JSON configuration file."""
    if not os.path.exists(config_filepath):
        print(f"Error: Configuration file not found at '{config_filepath}'")
        return None
    try:
        with open(config_filepath, 'r') as f:
            config_data = json.load(f)
        # print(f"Configuration loaded successfully from '{config_filepath}'")
        return config_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{config_filepath}': {e}")
        return None
    except Exception as e:
        print(f"Error reading config file '{config_filepath}': {e}")
        return None

def get_camera_calibration_filepath(main_config_data, config_dir=None):
    """
    Gets the full path to the camera calibration .npz file.
    The .npz filename is expected to be in main_config_data.
    If config_dir is provided, the .npz path is resolved relative to it.
    """
    if not main_config_data or 'calibration_file_name' not in main_config_data:
        print("Error: 'calibration_file_name' not found in main configuration.")
        return None
    
    npz_filename = main_config_data['calibration_file_name']
    
    if config_dir: # If config_dir is provided, npz_filename is relative to it
        return os.path.join(config_dir, npz_filename)
    else: # Assume npz_filename might be an absolute path or relative to current working dir
          # This case might need more robust handling depending on project structure
        return npz_filename


def get_perspective_warp_params(main_config_data):
    """
    Extracts perspective warp parameters from the main configuration data.
    
    Returns:
        tuple: (src_points_np, output_width, output_height) or (None, 0, 0) if error.
    """
    if not main_config_data or 'perspective_warp' not in main_config_data:
        print("Error: 'perspective_warp' section not found in main configuration.")
        return None, 0, 0
        
    warp_config = main_config_data['perspective_warp']
    
    src_points_list = warp_config.get('undistorted_src_points')
    output_width = warp_config.get('output_width')
    output_height = warp_config.get('output_height')

    if not all([src_points_list, isinstance(output_width, int), isinstance(output_height, int)]):
        print("Error: Missing or invalid 'undistorted_src_points', 'output_width', or 'output_height' in perspective_warp config.")
        return None, 0, 0
        
    try:
        src_points_np = np.array(src_points_list, dtype=np.float32)
        if src_points_np.shape != (4, 2):
            raise ValueError("src_points must be a list of 4 [x,y] coordinates.")
    except Exception as e:
        print(f"Error converting 'undistorted_src_points' to NumPy array: {e}")
        return None, 0, 0
        
    return src_points_np, output_width, output_height

# Example main config structure this loader expects:
# {
#   "calibration_file_name": "camera_calibration_data.npz",
#   "perspective_warp": {
#     "undistorted_src_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
#     "output_width": 300,
#     "output_height": 250
#   },
#   ... other future settings ...
# }