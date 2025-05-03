from flask import Flask, request
from datetime import datetime
import os
from PIL import Image
import numpy as np
import io # Keep io import

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def rgb565_to_rgb888(data_565, width, height):
    """Converts raw RGB565 byte data to an RGB888 NumPy array."""
    expected_bytes = width * height * 2
    if len(data_565) != expected_bytes:
        raise ValueError(f"Incorrect data size: expected {expected_bytes} bytes, got {len(data_565)}")

    # Read as little-endian unsigned 16-bit integers
    pixels_565 = np.frombuffer(data_565, dtype=np.uint16).astype(np.int32) # Read as uint16, cast to int32 for shifting safety

    # --- !!! ADD BYTE SWAP HERE !!! ---
    # ESP32 camera drivers often store RGB565 with bytes swapped within the 16-bit word.
    # Perform an explicit byte swap on each 16-bit element.
    pixels_565 = ((pixels_565 >> 8) & 0xff) | ((pixels_565 & 0xff) << 8)
    # --- End Byte Swap ---

    # Reshape *after* byte swapping
    # Note: Reshaping isn't strictly needed for this extraction method, but keep if preferred
    # pixels_565 = pixels_565.reshape((height, width)) # Optional reshape

    # Extract R, G, B components from the potentially byte-swapped values
    # R: Top 5 bits (after swap: bits 15-11)
    r = (pixels_565 >> 11) & 0x1F
    # G: Middle 6 bits (after swap: bits 10-5)
    g = (pixels_565 >> 5)  & 0x3F
    # B: Bottom 5 bits (after swap: bits 4-0)
    b = pixels_565         & 0x1F

    # Scale to 8-bit (0-255)
    # Use floating point for accuracy before converting back to uint8
    r = (r * 255.0 / 31.0).astype(np.uint8)
    g = (g * 255.0 / 63.0).astype(np.uint8)
    b = (b * 255.0 / 31.0).astype(np.uint8)

    # Stack R, G, B channels together
    # Create the final (height, width, 3) RGB array
    # We need height*width here because we didn't reshape earlier
    rgb_888 = np.dstack((r.reshape(height,width), g.reshape(height,width), b.reshape(height,width)))

    return rgb_888

# --- Rest of your Flask code remains the same ---

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return 'No image file field in request', 400
    file = request.files['image']

    if file.filename == '':
        return 'No image file selected', 400

    # --- Read file content ONCE ---
    try:
        image_data = file.read()
        if not image_data:
             return 'Empty image file uploaded', 400
        print(f"Received {len(image_data)} bytes.")
    except Exception as e:
        return f'Error reading file stream: {str(e)}', 500
    # --- End read file content ---

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # --- Save the raw RGB565 data ---
    raw_filename = f'{UPLOAD_FOLDER}/raw_image_{timestamp}.bin'
    try:
        with open(raw_filename, 'wb') as f_raw:
            f_raw.write(image_data)
        print(f'Raw data saved to: {raw_filename}')
    except Exception as e:
        print(f'Error saving raw file: {str(e)}')

    # --- Convert raw RGB565 data to JPEG ---
    try:
        width = 640  # VGA width (matches FRAMESIZE_VGA)
        height = 480 # VGA height (matches FRAMESIZE_VGA)

        # Convert using the dedicated function
        rgb888_array = rgb565_to_rgb888(image_data, width, height)

        # Create PIL image from the RGB888 array
        img = Image.fromarray(rgb888_array, 'RGB')

        # Save as JPEG
        jpeg_filename = f'{UPLOAD_FOLDER}/image_{timestamp}.jpg'
        img.save(jpeg_filename, 'JPEG', quality=85) # Add quality setting

        print(f'Raw data converted to JPEG: {jpeg_filename}')
        return 'Image saved and converted successfully', 200

    except ValueError as ve: # Catch specific data size error
         print(f'Data conversion error: {str(ve)}')
         # Include received size in error message for easier debugging
         return f'Error: {str(ve)} - Received {len(image_data)} bytes. Check image dimensions/format.', 400
    except Exception as e:
        print(f'JPEG conversion/saving error: {str(e)}')
        # Return 500, but mention the raw file might be saved
        return f'Error converting to JPEG: {str(e)} (Raw file might be saved as {raw_filename})', 500

@app.before_request
def log_request_info():
    if request.path != '/favicon.ico':
        print(f'Method: {request.method}\tPath: {request.path}\tFrom: {request.remote_addr}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)