# object_sorter_model/object_sorter_model/preprocessing/temporal_blending.py
import numpy as np
import cv2 # Only if absolutely necessary for a utility, try to keep core logic NumPy based

class IlluminationNormalizer:
    """
    A class to perform temporal blending for illumination normalization.
    It maintains a reference frame and blends incoming frames with it.
    """
    def __init__(self, alpha=0.95, initial_reference_frame=None, frame_shape=None, dtype=np.float32):
        """
        Initializes the IlluminationNormalizer.

        Args:
            alpha (float): Weight for the current frame in the blend. 
                           (1 - alpha) will be the weight for the reference frame.
                           A higher alpha means faster adaptation to new frames.
                           Value should be between 0 and 1.
            initial_reference_frame (np.ndarray, optional): An initial reference frame.
                                                             If None, the first processed frame will become
                                                             the initial reference. Must be float32.
            frame_shape (tuple, optional): Expected shape (h, w, c) of incoming frames.
                                           Required if initial_reference_frame is None to initialize it.
            dtype (np.dtype, optional): Data type for the reference frame (e.g., np.float32 for blending).
        """
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be between 0 (exclusive) and 1 (inclusive).")
            
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.reference_frame_float = None
        self.dtype = dtype

        if initial_reference_frame is not None:
            if initial_reference_frame.dtype != self.dtype:
                self.reference_frame_float = initial_reference_frame.astype(self.dtype)
            else:
                self.reference_frame_float = initial_reference_frame.copy()
        elif frame_shape is not None:
            # Will be initialized on the first call to normalize_frame
            self.frame_shape_for_init = frame_shape
            pass 
        else:
            # Will require the first frame to initialize
            pass
        
    def normalize_frame(self, current_frame_uint8):
        """
        Blends the current frame with the reference frame to normalize illumination.
        Updates the reference frame.

        Args:
            current_frame_uint8 (np.ndarray): The new incoming frame (uint8, BGR).

        Returns:
            np.ndarray: The blended/normalized frame (uint8, BGR).
            np.ndarray: The current reference frame used for blending (uint8, BGR).
        """
        if current_frame_uint8 is None:
            print("Error: current_frame_uint8 is None.")
            return None, None

        current_frame_float = current_frame_uint8.astype(self.dtype)

        if self.reference_frame_float is None:
            # First frame, initialize reference frame
            self.reference_frame_float = current_frame_float.copy()
            # The first "normalized" frame is just the current frame itself
            blended_frame_float = current_frame_float.copy()
        else:
            # Ensure reference frame and current frame have compatible shapes
            if self.reference_frame_float.shape != current_frame_float.shape:
                print(f"Warning: Shape mismatch. Reference: {self.reference_frame_float.shape}, Current: {current_frame_float.shape}. Re-initializing reference.")
                self.reference_frame_float = current_frame_float.copy()
                blended_frame_float = current_frame_float.copy()
            else:
                # Blend: reference_new = alpha * current + (1-alpha) * reference_old
                # The output of this step is also the new reference frame
                blended_frame_float = cv2.addWeighted(
                    current_frame_float, self.alpha,
                    self.reference_frame_float, self.beta,
                    0.0 # gamma (offset)
                )
                # Update the reference frame for the next iteration
                self.reference_frame_float = blended_frame_float.copy()
        
        # Clip and convert back to uint8 for output
        normalized_output_uint8 = np.clip(blended_frame_float, 0, 255).astype(np.uint8)
        current_reference_uint8 = np.clip(self.reference_frame_float, 0, 255).astype(np.uint8)
        
        return normalized_output_uint8, current_reference_uint8

    def get_reference_frame(self):
        """Returns the current reference frame as uint8."""
        if self.reference_frame_float is not None:
            return np.clip(self.reference_frame_float, 0, 255).astype(np.uint8)
        return None

    def reset_reference_frame(self, new_reference_frame_uint8=None):
        """Resets or sets a new reference frame."""
        if new_reference_frame_uint8 is not None:
            self.reference_frame_float = new_reference_frame_uint8.astype(self.dtype)
        else:
            self.reference_frame_float = None # Will re-initialize on next frame
        print("Reference frame reset.")