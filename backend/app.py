import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def embed_watermark_text(image_path, watermark_text, save_path):
    """Embed text watermark into image using DCT"""
    try:
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError("Could not load image")

        # Convert to YCrCb color space
        img_float = np.float32(img) / 255.0
        ycrcb = cv2.cvtColor(img_float, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        
        # Apply DCT
        dct = cv2.dct(y_channel)
        h, w = dct.shape
        
        # Create watermark from text
        # Convert text to binary and embed in DCT coefficients
        text_binary = ''.join(format(ord(char), '08b') for char in watermark_text)
        
        # Embed watermark in mid-frequency coefficients
        alpha = 0.1  # Increased strength for text
        watermark_matrix = np.zeros((h, w), dtype=np.float32)
        
        # Embed binary data in a pattern
        bit_index = 0
        for i in range(min(h//4, 100)):  # Limit embedding area
            for j in range(min(w//4, 100)):
                if bit_index < len(text_binary):
                    # Embed in mid-frequency area (avoid DC and high freq)
                    row = i + h//4
                    col = j + w//4
                    if row < h and col < w:
                        watermark_matrix[row, col] = float(text_binary[bit_index]) * alpha
                        bit_index += 1
                else:
                    break
            if bit_index >= len(text_binary):
                break
        
        # Add watermark to DCT coefficients
        dct_watermarked = dct + watermark_matrix
        
        # Inverse DCT
        y_watermarked = cv2.idct(dct_watermarked)
        y_watermarked = np.clip(y_watermarked, 0, 1)
        ycrcb[:, :, 0] = y_watermarked
        watermarked_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        watermarked_img_uint8 = np.uint8(watermarked_img * 255)

        cv2.imwrite(save_path, watermarked_img_uint8)
        return True
    except Exception as e:
        print(f"Error in embed_watermark_text: {e}")
        return False

def embed_watermark_image(image_path, watermark_path, save_path):
    """Embed image watermark into image using DCT"""
    try:
        img = cv2.imread(image_path)
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or watermark is None:
            raise ValueError("Could not load images")

        # Convert to YCrCb color space for better perceptual quality
        img_float = np.float32(img) / 255.0
        ycrcb = cv2.cvtColor(img_float, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        
        # Get dimensions and resize watermark to match Y channel
        h, w = y_channel.shape
        watermark_resized = cv2.resize(watermark, (w, h), interpolation=cv2.INTER_AREA)
        watermark_norm = np.float32(watermark_resized) / 255.0
        
        # Apply DCT to both Y channel and watermark
        dct_img = cv2.dct(y_channel)
        dct_watermark = cv2.dct(watermark_norm)
        
        # Embedding with improved alpha for better extraction
        alpha = 0.1  # Increased from 0.05 for better visibility in extraction
        dct_watermarked = dct_img + alpha * dct_watermark
        
        # Apply inverse DCT
        y_watermarked = cv2.idct(dct_watermarked)
        y_watermarked = np.clip(y_watermarked, 0, 1)
        
        # Reconstruct the image
        ycrcb[:, :, 0] = y_watermarked
        watermarked_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        watermarked_img_uint8 = np.uint8(watermarked_img * 255)

        cv2.imwrite(save_path, watermarked_img_uint8)
        return True
    except Exception as e:
        print(f"Error in embed_watermark_image: {e}")
        return False

def extract_watermark_text(original_image_path, watermarked_image_path):
    """Extract text watermark from watermarked image"""
    try:
        original_img = cv2.imread(original_image_path)
        watermarked_img = cv2.imread(watermarked_image_path)
        
        if original_img is None or watermarked_img is None:
            raise ValueError("Could not load images")

        original_float = np.float32(original_img) / 255.0
        watermarked_float = np.float32(watermarked_img) / 255.0

        original_ycrcb = cv2.cvtColor(original_float, cv2.COLOR_BGR2YCrCb)
        watermarked_ycrcb = cv2.cvtColor(watermarked_float, cv2.COLOR_BGR2YCrCb)

        y_original = original_ycrcb[:, :, 0]
        y_watermarked = watermarked_ycrcb[:, :, 0]

        dct_original = cv2.dct(y_original)
        dct_watermarked = cv2.dct(y_watermarked)

        # Extract difference
        alpha = 0.1
        diff_dct = dct_watermarked - dct_original
        
        # Extract binary data from mid-frequency area
        h, w = diff_dct.shape
        extracted_bits = []
        
        for i in range(min(h//4, 100)):
            for j in range(min(w//4, 100)):
                row = i + h//4
                col = j + w//4
                if row < h and col < w:
                    bit_value = diff_dct[row, col] / alpha
                    # Threshold to determine if bit is 0 or 1
                    if bit_value > 0.05:
                        extracted_bits.append('1')
                    else:
                        extracted_bits.append('0')
        
        # Convert binary to text
        extracted_text = ""
        for i in range(0, len(extracted_bits), 8):
            if i + 8 <= len(extracted_bits):
                byte_str = ''.join(extracted_bits[i:i+8])
                try:
                    char_code = int(byte_str, 2)
                    if 32 <= char_code <= 126:  # Printable ASCII
                        extracted_text += chr(char_code)
                    elif char_code == 0:  # End of string
                        break
                except ValueError:
                    continue
        
        return extracted_text.strip()
    except Exception as e:
        print(f"Error in extract_watermark_text: {e}")
        return None

def extract_watermark_image(original_image_path, watermarked_image_path, save_path):
    """Extract image watermark from watermarked image"""
    try:
        original_img = cv2.imread(original_image_path)
        watermarked_img = cv2.imread(watermarked_image_path)
        
        if original_img is None or watermarked_img is None:
            raise ValueError("Could not load images")

        # Convert to YCrCb color space
        original_float = np.float32(original_img) / 255.0
        watermarked_float = np.float32(watermarked_img) / 255.0

        original_ycrcb = cv2.cvtColor(original_float, cv2.COLOR_BGR2YCrCb)
        watermarked_ycrcb = cv2.cvtColor(watermarked_float, cv2.COLOR_BGR2YCrCb)

        y_original = original_ycrcb[:, :, 0]
        y_watermarked = watermarked_ycrcb[:, :, 0]

        # Apply DCT to both images
        dct_original = cv2.dct(y_original)
        dct_watermarked = cv2.dct(y_watermarked)

        # Extract the watermark with improved processing
        alpha = 0.1  # Must match embedding alpha
        diff_dct = (dct_watermarked - dct_original) / alpha
        
        # Apply inverse DCT to get the spatial domain watermark
        extracted_watermark_spatial = cv2.idct(diff_dct)
        
        # Enhance the extracted watermark for better visibility
        extracted_watermark_spatial = np.clip(extracted_watermark_spatial, 0, 1)
        
        # Apply contrast enhancement
        extracted_watermark_spatial = np.power(extracted_watermark_spatial, 0.5)  # Gamma correction
        
        # Normalize to full range
        min_val = np.min(extracted_watermark_spatial)
        max_val = np.max(extracted_watermark_spatial)
        if max_val > min_val:
            extracted_watermark_spatial = (extracted_watermark_spatial - min_val) / (max_val - min_val)
        
        # Apply threshold to make watermark clearer
        threshold = 0.3
        extracted_watermark_spatial = np.where(extracted_watermark_spatial > threshold, 1.0, 0.0)
        
        # Convert to uint8 and apply morphological operations for cleanup
        extracted_watermark_uint8 = np.uint8(extracted_watermark_spatial * 255)
        
        # Apply morphological closing to fill gaps
        kernel = np.ones((3,3), np.uint8)
        extracted_watermark_uint8 = cv2.morphologyEx(extracted_watermark_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur to smooth edges
        extracted_watermark_uint8 = cv2.GaussianBlur(extracted_watermark_uint8, (3,3), 0)

        cv2.imwrite(save_path, extracted_watermark_uint8)
        return True
    except Exception as e:
        print(f"Error in extract_watermark_image: {e}")
        return False

def embed_watermark_image_block_dct(image_path, watermark_path, save_path):
    """Embed image watermark using block-based DCT for better robustness"""
    try:
        img = cv2.imread(image_path)
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or watermark is None:
            raise ValueError("Could not load images")

        # Convert to YCrCb and work on Y channel
        img_float = np.float32(img) / 255.0
        ycrcb = cv2.cvtColor(img_float, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        
        h, w = y_channel.shape
        
        # Resize watermark to be smaller for block processing
        watermark_size = min(h//4, w//4, 64)  # Max 64x64 watermark
        watermark_resized = cv2.resize(watermark, (watermark_size, watermark_size))
        watermark_norm = np.float32(watermark_resized) / 255.0
        
        # Apply block-based DCT embedding
        block_size = 8
        alpha = 0.25  # Embedding strength (increased from 0.15 for better extraction)
        
        # Create a copy for modification
        y_watermarked = y_channel.copy()
        
        # Process blocks
        for i in range(0, min(h - block_size + 1, watermark_size * block_size), block_size):
            for j in range(0, min(w - block_size + 1, watermark_size * block_size), block_size):
                # Extract 8x8 block
                block = y_channel[i:i+block_size, j:j+block_size]
                
                # Apply DCT
                dct_block = cv2.dct(block)
                
                # Get corresponding watermark pixel
                wi = (i // block_size) % watermark_size
                wj = (j // block_size) % watermark_size
                watermark_bit = watermark_norm[wi, wj]
                
                # Embed watermark in mid-frequency coefficients (avoid DC and high freq)
                # Modify coefficients at positions (2,1), (1,2), (3,1), (1,3)
                dct_block[1, 2] += alpha * watermark_bit
                dct_block[2, 1] += alpha * watermark_bit
                
                # Apply inverse DCT
                modified_block = cv2.idct(dct_block)
                modified_block = np.clip(modified_block, 0, 1)
                
                # Put block back
                y_watermarked[i:i+block_size, j:j+block_size] = modified_block
        
        # Reconstruct image
        ycrcb[:, :, 0] = y_watermarked
        watermarked_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        watermarked_img_uint8 = np.uint8(watermarked_img * 255)

        cv2.imwrite(save_path, watermarked_img_uint8)
        return True
    except Exception as e:
        print(f"Error in embed_watermark_image_block_dct: {e}")
        return False

def extract_watermark_image_block_dct(original_image_path, watermarked_image_path, save_path):
    """Extract image watermark using block-based DCT"""
    try:
        original_img = cv2.imread(original_image_path)
        watermarked_img = cv2.imread(watermarked_image_path)
        
        if original_img is None or watermarked_img is None:
            raise ValueError("Could not load images")

        # Convert to YCrCb
        original_float = np.float32(original_img) / 255.0
        watermarked_float = np.float32(watermarked_img) / 255.0

        original_ycrcb = cv2.cvtColor(original_float, cv2.COLOR_BGR2YCrCb)
        watermarked_ycrcb = cv2.cvtColor(watermarked_float, cv2.COLOR_BGR2YCrCb)

        y_original = original_ycrcb[:, :, 0]
        y_watermarked = watermarked_ycrcb[:, :, 0]
        
        h, w = y_original.shape
        
        # Determine extraction size
        watermark_size = min(h//4, w//4, 64)
        extracted_watermark = np.zeros((watermark_size, watermark_size), dtype=np.float32)
        
        block_size = 8
        alpha = 0.25  # Must match embedding (increased from 0.15)
        
        # Extract from blocks
        for i in range(0, min(h - block_size + 1, watermark_size * block_size), block_size):
            for j in range(0, min(w - block_size + 1, watermark_size * block_size), block_size):
                # Extract blocks
                orig_block = y_original[i:i+block_size, j:j+block_size]
                water_block = y_watermarked[i:i+block_size, j:j+block_size]
                
                # Apply DCT
                dct_orig = cv2.dct(orig_block)
                dct_water = cv2.dct(water_block)
                
                # Extract watermark from mid-frequency coefficients
                diff1 = (dct_water[1, 2] - dct_orig[1, 2]) / alpha
                diff2 = (dct_water[2, 1] - dct_orig[2, 1]) / alpha
                
                # Average the differences for better extraction
                extracted_value = (diff1 + diff2) / 2.0
                
                # Map back to watermark position
                wi = (i // block_size) % watermark_size
                wj = (j // block_size) % watermark_size
                extracted_watermark[wi, wj] = extracted_value
        
        # Post-process extracted watermark
        extracted_watermark = np.clip(extracted_watermark, 0, 1)
        
        # Apply enhancement
        extracted_watermark = np.power(extracted_watermark, 0.4)  # Gamma correction
        
        # Normalize
        min_val = np.min(extracted_watermark)
        max_val = np.max(extracted_watermark)
        if max_val > min_val:
            extracted_watermark = (extracted_watermark - min_val) / (max_val - min_val)
        
        # Apply adaptive threshold
        threshold = np.mean(extracted_watermark) + 0.1
        extracted_watermark = np.where(extracted_watermark > threshold, 1.0, 0.0)
        
        # Resize to reasonable size for viewing
        output_size = 256
        extracted_watermark_resized = cv2.resize(extracted_watermark, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
        
        # Convert to uint8
        extracted_watermark_uint8 = np.uint8(extracted_watermark_resized * 255)
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        extracted_watermark_uint8 = cv2.morphologyEx(extracted_watermark_uint8, cv2.MORPH_CLOSE, kernel)
        extracted_watermark_uint8 = cv2.morphologyEx(extracted_watermark_uint8, cv2.MORPH_OPEN, kernel)

        cv2.imwrite(save_path, extracted_watermark_uint8)
        return True
    except Exception as e:
        print(f"Error in extract_watermark_image_block_dct: {e}")
        return False

def embed_watermark_image_block_dct_q(image_path, watermark_path, save_path, block_size=8, delta=0.04):
    """Embed image watermark using quantized block-DCT coefficient ordering.
    Each block encodes 1 bit by enforcing an order between two mid-frequency coefficients.
    delta controls separation strength (on normalized 0-1 range)."""
    try:
        img = cv2.imread(image_path)
        wm = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        if img is None or wm is None:
            raise ValueError("Could not load images")

        img_float = np.float32(img) / 255.0
        ycrcb = cv2.cvtColor(img_float, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]
        h, w = y.shape

        # Determine block grid size
        blocks_h = h // block_size
        blocks_w = w // block_size
        max_wm_size = min(blocks_h, blocks_w, 64)  # limit to 64x64 bits
        wm_resized = cv2.resize(wm, (max_wm_size, max_wm_size))
        wm_bin = (wm_resized > 127).astype(np.uint8)

        # Prepare output luminance
        y_mod = y.copy()

        for bi in range(max_wm_size):
            for bj in range(max_wm_size):
                bit = wm_bin[bi, bj]
                i = bi * block_size
                j = bj * block_size
                if i + block_size > h or j + block_size > w:
                    continue
                block = y[i:i+block_size, j:j+block_size]
                dct_block = cv2.dct(block)
                # Mid-frequency pair
                cA = dct_block[1,2]
                cB = dct_block[2,1]
                diff = cA - cB
                if bit == 1:
                    # enforce cA - cB >= delta
                    if diff < delta:
                        m = (cA + cB)/2.0
                        cA = m + delta/2.0
                        cB = m - delta/2.0
                else:
                    # enforce cB - cA >= delta
                    if -diff < delta:
                        m = (cA + cB)/2.0
                        cB = m + delta/2.0
                        cA = m - delta/2.0
                dct_block[1,2] = cA
                dct_block[2,1] = cB
                new_block = cv2.idct(dct_block)
                y_mod[i:i+block_size, j:j+block_size] = np.clip(new_block, 0, 1)

        ycrcb[:, :, 0] = y_mod
        out_img = (cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR) * 255).astype(np.uint8)
        cv2.imwrite(save_path, out_img)
        print(f"[EmbedQBlockDCT] bits={max_wm_size}x{max_wm_size} delta={delta}")
        return True
    except Exception as e:
        print(f"Error in embed_watermark_image_block_dct_q: {e}")
        return False

def extract_watermark_image_block_dct_q(original_image_path, watermarked_image_path, save_path, block_size=8):
    """Extract quantized block DCT watermark by comparing coefficient pairs.
    Returns a binary watermark image."""
    try:
        orig = cv2.imread(original_image_path)
        wmi = cv2.imread(watermarked_image_path)
        if orig is None or wmi is None:
            raise ValueError("Could not load images")

        o_float = np.float32(orig) / 255.0
        w_float = np.float32(wmi) / 255.0
        o_y = cv2.cvtColor(o_float, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        w_y = cv2.cvtColor(w_float, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        h, w = o_y.shape
        blocks_h = h // block_size
        blocks_w = w // block_size
        wm_size = min(blocks_h, blocks_w, 64)
        wm_bits = np.zeros((wm_size, wm_size), dtype=np.uint8)

        for bi in range(wm_size):
            for bj in range(wm_size):
                i = bi * block_size
                j = bj * block_size
                if i + block_size > h or j + block_size > w:
                    continue
                block_w = w_y[i:i+block_size, j:j+block_size]
                dct_block = cv2.dct(block_w)
                cA = dct_block[1,2]
                cB = dct_block[2,1]
                bit = 1 if cA > cB else 0
                wm_bits[bi, bj] = bit

        # Upscale for visibility
        upscale = 4 if wm_size * 4 <= 512 else max(1, 512 // wm_size)
        wm_up = cv2.resize(wm_bits*255, (wm_size*upscale, wm_size*upscale), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_path, wm_up)
        white_ratio = float(np.mean(wm_bits))
        print(f"[ExtractQBlockDCT] size={wm_size} white_ratio={white_ratio:.3f}")
        return True
    except Exception as e:
        print(f"Error in extract_watermark_image_block_dct_q: {e}")
        return False

FREQ_ALPHA = 0.08  # Increased embedding strength (was 0.02) for better extraction

def embed_watermark_frequency_domain(image_path, watermark_path, save_path):
    """Embed watermark using frequency domain with selective embedding.
    Uses a ring mask in mid-frequencies. Increase FREQ_ALPHA if extraction remains weak.
    """
    try:
        img = cv2.imread(image_path)
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        if img is None or watermark is None:
            raise ValueError("Could not load images")

        # Convert to YCrCb and take luminance
        img_float = np.float32(img) / 255.0
        ycrcb = cv2.cvtColor(img_float, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        h, w = y_channel.shape

        # Resize watermark to full image (simple approach); binarize
        watermark_resized = cv2.resize(watermark, (w, h))
        watermark_binary = (watermark_resized > 127).astype(np.float32)

        # FFTs
        f_shift_img = np.fft.fftshift(np.fft.fft2(y_channel))
        f_shift_watermark = np.fft.fftshift(np.fft.fft2(watermark_binary))

        # Mid-frequency ring mask
        center_h, center_w = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
        inner_radius = min(h, w) * 0.1
        outer_radius = min(h, w) * 0.4
        mask = ((distances >= inner_radius) & (distances <= outer_radius)).astype(np.float32)

        # Apply scaling proportional to watermark energy to keep visual quality
        wm_energy = np.mean(np.abs(f_shift_watermark) * mask + 1e-8)
        img_energy = np.mean(np.abs(f_shift_img) * mask + 1e-8)
        adaptive_scale = FREQ_ALPHA * (img_energy / (wm_energy + 1e-8)) * 0.3  # damp factor

        f_watermarked = f_shift_img + adaptive_scale * f_shift_watermark * mask

        # Inverse FFT
        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(f_watermarked)))
        img_back = np.clip(img_back, 0, 1)

        # Reconstruct final image
        ycrcb[:, :, 0] = img_back
        watermarked_img_uint8 = (cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR) * 255).astype(np.uint8)

        cv2.imwrite(save_path, watermarked_img_uint8)
        print(f"[EmbedFreq] alpha={FREQ_ALPHA:.3f} adaptive_scale={adaptive_scale:.6f} wm_energy={wm_energy:.4f} img_energy={img_energy:.4f}")
        return True
    except Exception as e:
        print(f"Error in embed_watermark_frequency_domain: {e}")
        return False

def extract_watermark_frequency_domain(original_image_path, watermarked_image_path, save_path):
    """Extract watermark using frequency domain analysis with enhanced processing"""
    try:
        original_img = cv2.imread(original_image_path)
        watermarked_img = cv2.imread(watermarked_image_path)
        
        if original_img is None or watermarked_img is None:
            raise ValueError("Could not load images")

        # Convert to YCrCb
        original_float = np.float32(original_img) / 255.0
        watermarked_float = np.float32(watermarked_img) / 255.0

        original_ycrcb = cv2.cvtColor(original_float, cv2.COLOR_BGR2YCrCb)
        watermarked_ycrcb = cv2.cvtColor(watermarked_float, cv2.COLOR_BGR2YCrCb)

        y_original = original_ycrcb[:, :, 0]
        y_watermarked = watermarked_ycrcb[:, :, 0]
        
        h, w = y_original.shape
        
        # Apply FFT
        f_original = np.fft.fftshift(np.fft.fft2(y_original))
        f_watermarked = np.fft.fftshift(np.fft.fft2(y_watermarked))
        
        # Extract difference in frequency domain
        f_diff = f_watermarked - f_original
        
        # Create extraction mask (same as embedding)
        mask = np.zeros((h, w), dtype=np.float32)
        center_h, center_w = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
        
        inner_radius = min(h, w) * 0.1
        outer_radius = min(h, w) * 0.4
        mask = ((distances >= inner_radius) & (distances <= outer_radius)).astype(np.float32)
        
        # Apply mask and extract
        f_extracted = f_diff * mask
        
        # Inverse transform
        extracted_watermark = np.fft.ifft2(np.fft.ifftshift(f_extracted))
        extracted_watermark = np.real(extracted_watermark)
        
        # Enhanced processing for better logo extraction
        extracted_watermark = np.abs(extracted_watermark)
        
        # Normalize to 0-1 range
        if np.max(extracted_watermark) > 0:
            extracted_watermark = extracted_watermark / np.max(extracted_watermark)
        
        # Apply adaptive thresholding instead of simple threshold
        # Use Otsu's method for automatic threshold selection
        extracted_8bit = (extracted_watermark * 255).astype(np.uint8)
        _, extracted_binary = cv2.threshold(extracted_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Enhanced morphological operations for logo extraction
        # Use different kernel sizes for better logo reconstruction
        kernel_small = np.ones((2,2), np.uint8)
        kernel_medium = np.ones((3,3), np.uint8)
        kernel_large = np.ones((5,5), np.uint8)
        
        # Close small gaps in the logo
        extracted_binary = cv2.morphologyEx(extracted_binary, cv2.MORPH_CLOSE, kernel_medium)
        
        # Remove small noise
        extracted_binary = cv2.morphologyEx(extracted_binary, cv2.MORPH_OPEN, kernel_small)
        
        # Dilate to make logo features more prominent
        extracted_binary = cv2.dilate(extracted_binary, kernel_medium, iterations=1)
        
        # Apply median filter to smooth
        extracted_binary = cv2.medianBlur(extracted_binary, 3)
        
        # Apply Gaussian blur to reduce sharp edges
        extracted_binary = cv2.GaussianBlur(extracted_binary, (3,3), 0.5)
        
        # Re-threshold after smoothing
        _, extracted_binary = cv2.threshold(extracted_binary, 127, 255, cv2.THRESH_BINARY)
        
        # Try to enhance contrast one more time
        extracted_binary = cv2.convertScaleAbs(extracted_binary, alpha=1.2, beta=10)

        cv2.imwrite(save_path, extracted_binary)
        print(f"Enhanced watermark extraction completed. Saved to: {save_path}")
        return True
    except Exception as e:
        print(f"Error in extract_watermark_frequency_domain: {e}")
        return False

def extract_watermark_frequency_domain_scaled(original_image_path, watermarked_image_path, save_path):
    """Frequency domain extraction that rescales using known embedding alpha for stronger recovery.
    Falls back gracefully if scaling saturates."""
    try:
        original_img = cv2.imread(original_image_path)
        watermarked_img = cv2.imread(watermarked_image_path)
        if original_img is None or watermarked_img is None:
            raise ValueError("Could not load images")

        # Y channel
        original_float = np.float32(original_img) / 255.0
        watermarked_float = np.float32(watermarked_img) / 255.0
        y_original = cv2.cvtColor(original_float, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        y_watermarked = cv2.cvtColor(watermarked_float, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        h, w = y_original.shape

        f_original = np.fft.fftshift(np.fft.fft2(y_original))
        f_watermarked = np.fft.fftshift(np.fft.fft2(y_watermarked))
        f_diff = f_watermarked - f_original

        center_h, center_w = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
        inner_radius = min(h, w) * 0.1
        outer_radius = min(h, w) * 0.4
        mask = ((distances >= inner_radius) & (distances <= outer_radius)).astype(np.float32)

        # Reconstruct watermark spectrum estimate
        # Avoid division by zero — we used adaptive scaling, approximate with inverse of adaptive_scale factor used.
        # Since adaptive_scale ~ FREQ_ALPHA * (img_energy / wm_energy) * 0.3, we can't invert precisely without wm_energy.
        # Empirically amplify by large factor relative to FREQ_ALPHA.
        amplify = 1.0 / max(FREQ_ALPHA, 1e-4) * 2.5  # tune factor
        f_est_wm = f_diff * mask * amplify

        spatial_est = np.real(np.fft.ifft2(np.fft.ifftshift(f_est_wm)))
        spatial_abs = np.abs(spatial_est)
        spatial_norm = spatial_abs - spatial_abs.min()
        if spatial_norm.max() > 0:
            spatial_norm /= spatial_norm.max()

        # Focus on strong components via percentile thresholding
        high_thresh = np.percentile(spatial_norm, 99)
        mid_thresh = np.percentile(spatial_norm, 95)
        binary = np.zeros_like(spatial_norm, dtype=np.uint8)
        binary[spatial_norm >= high_thresh] = 255
        binary[(spatial_norm >= mid_thresh) & (spatial_norm < high_thresh)] = 128

        # Resize to smaller representative watermark (improves visibility)
        target_size = min(256, h, w)
        binary_small = cv2.resize(binary, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # Clean
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(binary_small, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.medianBlur(cleaned, 3)

        mean_val = float(spatial_norm.mean())
        std_val = float(spatial_norm.std())
        cv2.imwrite(save_path, cleaned)
        print(f"[ExtractFreqScaled] amplify={amplify:.2f} mean={mean_val:.4f} std={std_val:.4f} high={high_thresh:.4f} mid={mid_thresh:.4f}")
        return True
    except Exception as e:
        print(f"Error in extract_watermark_frequency_domain_scaled: {e}")
        return False

def extract_watermark_logo_enhanced(original_image_path, watermarked_image_path, save_path):
    """Enhanced logo extraction using multiple methods"""
    try:
        original_img = cv2.imread(original_image_path)
        watermarked_img = cv2.imread(watermarked_image_path)
        
        if original_img is None or watermarked_img is None:
            raise ValueError("Could not load images")

        # Convert to grayscale for simpler processing
        original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        watermarked_gray = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2GRAY)
        
        # Ensure same size
        h, w = original_gray.shape
        watermarked_gray = cv2.resize(watermarked_gray, (w, h))
        
        # Convert to float
        original_float = np.float32(original_gray) / 255.0
        watermarked_float = np.float32(watermarked_gray) / 255.0
        
        # Method 1: Simple difference
        diff = watermarked_float - original_float
        
        # Method 2: Ratio-based extraction
        ratio = np.divide(watermarked_float, original_float + 0.001)  # Add small value to avoid division by zero
        ratio_norm = (ratio - 1.0) * 10  # Amplify small differences
        
        # Method 3: DCT-based extraction
        # Apply DCT to both images
        dct_original = cv2.dct(original_float)
        dct_watermarked = cv2.dct(watermarked_float)
        dct_diff = dct_watermarked - dct_original
        
        # Focus on mid-frequency components
        h_quarter = h // 4
        w_quarter = w // 4
        dct_watermark = dct_diff[h_quarter:h_quarter*3, w_quarter:w_quarter*3]
        
        # Inverse DCT of the extracted region
        if dct_watermark.size > 0:
            # Pad to make it square for better logo reconstruction
            size = max(dct_watermark.shape)
            dct_square = np.zeros((size, size))
            dct_square[:dct_watermark.shape[0], :dct_watermark.shape[1]] = dct_watermark
            
            # Apply inverse DCT
            logo_candidate = cv2.idct(dct_square)
        else:
            logo_candidate = diff
        
        # Combine results from different methods
        combined = (np.abs(diff) + np.abs(ratio_norm) + np.abs(logo_candidate)) / 3.0
        
        # Normalize
        combined = (combined - np.min(combined))
        if np.max(combined) > 0:
            combined = combined / np.max(combined)
        
        # Apply adaptive threshold
        combined_8bit = (combined * 255).astype(np.uint8)
        
        # Use adaptive threshold for better logo detection
        adaptive_thresh = cv2.adaptiveThreshold(combined_8bit, 255, 
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to enhance logo structure
        kernel_small = np.ones((2,2), np.uint8)
        kernel_medium = np.ones((4,4), np.uint8)
        
        # Clean up noise
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel_small)
        
        # Close gaps in logo
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        # Dilate to make logo more visible
        cleaned = cv2.dilate(cleaned, kernel_small, iterations=2)
        
        # Apply bilateral filter to preserve edges while reducing noise
        cleaned = cv2.bilateralFilter(cleaned, 9, 75, 75)
        
        # Final threshold
        _, final_logo = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite(save_path, final_logo)
        print(f"Enhanced logo extraction completed. Saved to: {save_path}")
        return True
        
    except Exception as e:
        print(f"Error in extract_watermark_logo_enhanced: {e}")
        return False

def extract_watermark_robust(original_image_path, watermarked_image_path, save_path):
    """Robust watermark extraction that matches the embedding algorithm exactly"""
    try:
        original_img = cv2.imread(original_image_path)
        watermarked_img = cv2.imread(watermarked_image_path)
        
        if original_img is None or watermarked_img is None:
            raise ValueError("Could not load images")

        # Use the same color space conversion as embedding
        original_float = np.float32(original_img) / 255.0
        watermarked_float = np.float32(watermarked_img) / 255.0

        original_ycrcb = cv2.cvtColor(original_float, cv2.COLOR_BGR2YCrCb)
        watermarked_ycrcb = cv2.cvtColor(watermarked_float, cv2.COLOR_BGR2YCrCb)

        y_original = original_ycrcb[:, :, 0]
        y_watermarked = watermarked_ycrcb[:, :, 0]
        
        h, w = y_original.shape
        
        # Apply FFT exactly as in embedding
        f_original = np.fft.fftshift(np.fft.fft2(y_original))
        f_watermarked = np.fft.fftshift(np.fft.fft2(y_watermarked))
        
        # Extract the difference - this should contain the watermark
        f_diff = f_watermarked - f_original
        
        # Create the same mask as used in embedding
        center_h, center_w = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
        
        inner_radius = min(h, w) * 0.1
        outer_radius = min(h, w) * 0.4
        ring_mask = ((distances >= inner_radius) & (distances <= outer_radius)).astype(np.float32)
        
        # Extract watermark from the masked region
        f_watermark = f_diff * ring_mask
        
        # Convert back to spatial domain
        extracted_spatial = np.fft.ifft2(np.fft.ifftshift(f_watermark))
        extracted_real = np.real(extracted_spatial)
        
        # Since we know the watermark was resized and positioned, try to extract it properly
        # Focus on the center region where the watermark was likely embedded
        center_region_size = min(h//3, w//3, 256)  # Reasonable watermark size
        start_h = (h - center_region_size) // 2
        start_w = (w - center_region_size) // 2
        
        # Extract center region
        center_extracted = extracted_real[start_h:start_h+center_region_size, 
                                        start_w:start_w+center_region_size]
        
        # Alternative: try extracting from the magnitude of the difference
        magnitude = np.abs(f_diff)
        magnitude_spatial = np.fft.ifft2(np.fft.ifftshift(magnitude * ring_mask))
        magnitude_real = np.real(magnitude_spatial)
        
        # Combine both approaches
        combined = (np.abs(center_extracted) + np.abs(magnitude_real)) / 2.0
        
        # Normalize to 0-255
        combined = (combined - np.min(combined))
        if np.max(combined) > 0:
            combined = combined / np.max(combined)
        
        combined_8bit = (combined * 255).astype(np.uint8)
        
        # Apply more sophisticated processing
        # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(combined_8bit)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 15, 80, 80)
        
        # Use adaptive threshold for better logo detection
        adaptive = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours and filter by area to remove small noise
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask for significant contours only
        min_area = combined.size * 0.001  # Minimum 0.1% of image area
        result = np.zeros_like(cleaned)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.fillPoly(result, [contour], 255)
        
        # If no significant contours found, use the cleaned version
        if np.sum(result) < np.sum(cleaned) * 0.1:  # Less than 10% of cleaned image
            result = cleaned
        
        cv2.imwrite(save_path, result)
        print(f"Robust watermark extraction completed. Saved to: {save_path}")
        return True
        
    except Exception as e:
        print(f"Error in extract_watermark_robust: {e}")
        return False
        
        # Ensure same size
        h, w = original_gray.shape
        watermarked_gray = cv2.resize(watermarked_gray, (w, h))
        
        # Convert to float
        original_float = np.float32(original_gray) / 255.0
        watermarked_float = np.float32(watermarked_gray) / 255.0
        
        # Method 1: Simple difference
        diff = watermarked_float - original_float
        
        # Method 2: Ratio-based extraction
        ratio = np.divide(watermarked_float, original_float + 0.001)  # Add small value to avoid division by zero
        ratio_norm = (ratio - 1.0) * 10  # Amplify small differences
        
        # Method 3: DCT-based extraction
        # Apply DCT to both images
        dct_original = cv2.dct(original_float)
        dct_watermarked = cv2.dct(watermarked_float)
        dct_diff = dct_watermarked - dct_original
        
        # Focus on mid-frequency components
        h_quarter = h // 4
        w_quarter = w // 4
        dct_watermark = dct_diff[h_quarter:h_quarter*3, w_quarter:w_quarter*3]
        
        # Inverse DCT of the extracted region
        if dct_watermark.size > 0:
            # Pad to make it square for better logo reconstruction
            size = max(dct_watermark.shape)
            dct_square = np.zeros((size, size))
            dct_square[:dct_watermark.shape[0], :dct_watermark.shape[1]] = dct_watermark
            
            # Apply inverse DCT
            logo_candidate = cv2.idct(dct_square)
        else:
            logo_candidate = diff
        
        # Combine results from different methods
        combined = (np.abs(diff) + np.abs(ratio_norm) + np.abs(logo_candidate)) / 3.0
        
        # Normalize
        combined = (combined - np.min(combined))
        if np.max(combined) > 0:
            combined = combined / np.max(combined)
        
        # Apply adaptive threshold
        combined_8bit = (combined * 255).astype(np.uint8)
        
        # Use adaptive threshold for better logo detection
        adaptive_thresh = cv2.adaptiveThreshold(combined_8bit, 255, 
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to enhance logo structure
        kernel_small = np.ones((2,2), np.uint8)
        kernel_medium = np.ones((4,4), np.uint8)
        
        # Clean up noise
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel_small)
        
        # Close gaps in logo
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        # Dilate to make logo more visible
        cleaned = cv2.dilate(cleaned, kernel_small, iterations=2)
        
        # Apply bilateral filter to preserve edges while reducing noise
        cleaned = cv2.bilateralFilter(cleaned, 9, 75, 75)
        
        # Final threshold
        _, final_logo = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite(save_path, final_logo)
def extract_watermark_robust(original_image_path, watermarked_image_path, save_path):
    """Robust watermark extraction that matches the embedding algorithm exactly"""
    try:
        original_img = cv2.imread(original_image_path)
        watermarked_img = cv2.imread(watermarked_image_path)
        
        if original_img is None or watermarked_img is None:
            raise ValueError("Could not load images")

        # Use the same color space conversion as embedding
        original_float = np.float32(original_img) / 255.0
        watermarked_float = np.float32(watermarked_img) / 255.0

        original_ycrcb = cv2.cvtColor(original_float, cv2.COLOR_BGR2YCrCb)
        watermarked_ycrcb = cv2.cvtColor(watermarked_float, cv2.COLOR_BGR2YCrCb)

        y_original = original_ycrcb[:, :, 0]
        y_watermarked = watermarked_ycrcb[:, :, 0]
        
        h, w = y_original.shape
        
        # Apply FFT exactly as in embedding
        f_original = np.fft.fftshift(np.fft.fft2(y_original))
        f_watermarked = np.fft.fftshift(np.fft.fft2(y_watermarked))
        
        # Extract the difference - this should contain the watermark
        f_diff = f_watermarked - f_original
        
        # Create the same mask as used in embedding
        center_h, center_w = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
        
        inner_radius = min(h, w) * 0.1
        outer_radius = min(h, w) * 0.4
        ring_mask = ((distances >= inner_radius) & (distances <= outer_radius)).astype(np.float32)
        
        # Extract watermark from the masked region
        f_watermark = f_diff * ring_mask
        
        # Convert back to spatial domain
        extracted_spatial = np.fft.ifft2(np.fft.ifftshift(f_watermark))
        extracted_real = np.real(extracted_spatial)
        
        # Since we know the watermark was resized and positioned, try to extract it properly
        # Focus on the center region where the watermark was likely embedded
        center_region_size = min(h//3, w//3, 256)  # Reasonable watermark size
        start_h = (h - center_region_size) // 2
        start_w = (w - center_region_size) // 2
        
        # Extract center region
        center_extracted = extracted_real[start_h:start_h+center_region_size, 
                                        start_w:start_w+center_region_size]
        
        # Alternative: try extracting from the magnitude of the difference
        magnitude = np.abs(f_diff)
        magnitude_spatial = np.fft.ifft2(np.fft.ifftshift(magnitude * ring_mask))
        magnitude_real = np.real(magnitude_spatial)
        
        # Combine both approaches
        combined = (np.abs(center_extracted) + np.abs(magnitude_real)) / 2.0
        
        # Normalize to 0-255
        combined = (combined - np.min(combined))
        if np.max(combined) > 0:
            combined = combined / np.max(combined)
        
        combined_8bit = (combined * 255).astype(np.uint8)
        
        # Apply more sophisticated processing
        # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(combined_8bit)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 15, 80, 80)
        
        # Use adaptive threshold for better logo detection
        adaptive = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours and filter by area to remove small noise
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask for significant contours only
        min_area = combined.size * 0.001  # Minimum 0.1% of image area
        result = np.zeros_like(cleaned)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.fillPoly(result, [contour], 255)
        
        # If no significant contours found, use the cleaned version
        if np.sum(result) < np.sum(cleaned) * 0.1:  # Less than 10% of cleaned image
            result = cleaned
        
        cv2.imwrite(save_path, result)
        print(f"Robust watermark extraction completed. Saved to: {save_path}")
        return True
        
    except Exception as e:
        print(f"Error in extract_watermark_robust: {e}")
        return False

def auto_detect_watermark_type(original_path, watermarked_path):
    """Automatically detect if watermark is text or image based on analysis"""
    try:
        print("Starting auto-detection...")
        
        # Method 1: Try text extraction first
        extracted_text = extract_watermark_text(original_path, watermarked_path)
        print(f"Text extraction result: '{extracted_text}'")
        
        # Check if extracted text is meaningful
        if extracted_text and len(extracted_text.strip()) > 2:
            # Check if it contains mostly printable ASCII characters
            printable_ratio = sum(c.isprintable() for c in extracted_text) / len(extracted_text)
            print(f"Printable character ratio: {printable_ratio}")
            
            if printable_ratio > 0.7:  # 70% printable characters
                print("Auto-detected as TEXT watermark")
                return 'text', extracted_text
        
        # Method 2: Analyze image differences to detect image watermarks
        original_img = cv2.imread(original_path)
        watermarked_img = cv2.imread(watermarked_path)
        
        if original_img is not None and watermarked_img is not None:
            # Resize to same dimensions
            h, w = original_img.shape[:2]
            watermarked_img = cv2.resize(watermarked_img, (w, h))
            
            # Calculate difference
            diff = cv2.absdiff(original_img, watermarked_img)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Check if there are significant differences (indicating image watermark)
            mean_diff = np.mean(diff_gray)
            std_diff = np.std(diff_gray)
            max_diff = np.max(diff_gray)
            
            print(f"Image difference stats - Mean: {mean_diff:.2f}, Std: {std_diff:.2f}, Max: {max_diff:.2f}")
            
            # If there are moderate differences, likely an image watermark
            if mean_diff > 2 and std_diff > 5 and max_diff > 30:
                print("Auto-detected as IMAGE watermark based on difference analysis")
                return 'image', None
        
        # Default to image if uncertain
        print("Auto-detected as IMAGE watermark (default)")
        return 'image', None
        
    except Exception as e:
        print(f"Auto-detection error: {e}")
        return 'image', None  # Default to image if detection fails

@app.route('/embed', methods=['POST'])
def embed_watermark_api():
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'Image file is required'}), 400
        
        image_file = request.files['image']
        watermark_type = request.form.get('type', 'text')  # Default to text
        
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
        
        # Generate unique filenames
        unique_id = str(uuid.uuid4())
        image_filename = f"image_{unique_id}.{image_file.filename.rsplit('.', 1)[1].lower()}"
        output_filename = f"watermarked_{unique_id}.png"
        
        # Save uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        image_file.save(image_path)
        
        success = False
        
        if watermark_type == 'text':
            # Handle text watermark
            watermark_text = request.form.get('watermark', '')
            if not watermark_text:
                return jsonify({'error': 'Watermark text is required'}), 400
            success = embed_watermark_text(image_path, watermark_text, output_path)
            
        elif watermark_type == 'image':
            # Handle image watermark
            if 'watermark' not in request.files:
                return jsonify({'error': 'Watermark image file is required'}), 400
            
            watermark_file = request.files['watermark']
            if watermark_file.filename == '':
                return jsonify({'error': 'No watermark file selected'}), 400
                
            if not allowed_file(watermark_file.filename):
                return jsonify({'error': 'Invalid watermark file type. Only PNG, JPG, JPEG allowed'}), 400
            
            # Save watermark image
            watermark_filename = f"watermark_{unique_id}.{watermark_file.filename.rsplit('.', 1)[1].lower()}"
            watermark_path = os.path.join(app.config['UPLOAD_FOLDER'], watermark_filename)
            watermark_file.save(watermark_path)
            
            # Choose algorithm
            algorithm = request.form.get('algorithm', 'block_dct_q')  # block_dct_q | block_dct | freq
            if algorithm not in ['block_dct_q', 'block_dct', 'freq']:
                algorithm = 'block_dct_q'
            print(f"[EmbedAPI] Using algorithm={algorithm}")

            if algorithm == 'block_dct_q':
                delta = request.form.get('delta', None)
                try:
                    delta = float(delta) if delta is not None else 0.04
                except:
                    delta = 0.04
                success = embed_watermark_image_block_dct_q(image_path, watermark_path, output_path, delta=delta)
            elif algorithm == 'block_dct':
                success = embed_watermark_image_block_dct(image_path, watermark_path, output_path)
            else:
                success = embed_watermark_frequency_domain(image_path, watermark_path, output_path)
            
            # Cleanup watermark file
            if os.path.exists(watermark_path):
                os.remove(watermark_path)
        else:
            return jsonify({'error': 'Invalid watermark type. Must be "text" or "image"'}), 400
        
        if success:
            # Return watermarked plus metadata headers
            resp = send_file(output_path, as_attachment=True, download_name='watermarked_image.png')
            if watermark_type == 'image':
                resp.headers['X-Watermark-Algorithm'] = request.form.get('algorithm', 'block_dct')
            return resp
        else:
            return jsonify({'error': 'Failed to embed watermark'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        # Cleanup temporary files
        if 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)

@app.route('/extract', methods=['POST'])
def extract_watermark_api():
    try:
        # Check if files are present
        if 'original' not in request.files or 'watermarked' not in request.files:
            return jsonify({'error': 'Both original and watermarked images are required'}), 400
        
        original_file = request.files['original']
        watermarked_file = request.files['watermarked']
        extraction_type = request.form.get('type', 'image')  # Default to image now
        
        print(f"Extraction request: type={extraction_type}")  # Debug logging
        
        if original_file.filename == '' or watermarked_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if not (allowed_file(original_file.filename) and allowed_file(watermarked_file.filename)):
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
        
        # Generate unique filenames
        unique_id = str(uuid.uuid4())
        original_filename = f"original_{unique_id}.{original_file.filename.rsplit('.', 1)[1].lower()}"
        watermarked_filename = f"watermarked_{unique_id}.{watermarked_file.filename.rsplit('.', 1)[1].lower()}"
        
        # Save uploaded files
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        watermarked_path = os.path.join(app.config['UPLOAD_FOLDER'], watermarked_filename)
        
        original_file.save(original_path)
        watermarked_file.save(watermarked_path)
        
        # Auto-detect watermark type if extraction fails
        if extraction_type == 'auto':
            detected_type, detected_text = auto_detect_watermark_type(original_path, watermarked_path)
            extraction_type = detected_type
            print(f"Auto-detected watermark type: {extraction_type}")
        
        if extraction_type == 'text':
            # Extract text watermark
            extracted_text = extract_watermark_text(original_path, watermarked_path)
            
            if extracted_text is not None and extracted_text.strip():
                return jsonify({'watermark': extracted_text})
            else:
                return jsonify({'error': 'No text watermark found. The image may contain an image watermark instead of text, or no watermark at all. Try selecting "Image" extraction type.'}), 400
                
        elif extraction_type == 'image':
            # Extract image watermark using multiple strategies
            output_filename = f"extracted_{unique_id}.png"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            # 1. Try quantized block DCT extraction first (if image was embedded with block_dct_q)
            success = extract_watermark_image_block_dct_q(original_path, watermarked_path, output_path)
            if success:
                print("Quantized block DCT extraction succeeded")
            
            # 2. Scaled frequency domain extraction
            if not success:
                success = extract_watermark_frequency_domain_scaled(original_path, watermarked_path, output_path)
                if success:
                    print("Scaled frequency extraction succeeded")
            
            # 3. Robust extraction
            if not success:
                print("Scaled frequency method failed, trying robust method...")
                success = extract_watermark_robust(original_path, watermarked_path, output_path)
            
            # If robust method fails, try enhanced logo extraction
            if not success:
                print("Robust method failed, trying enhanced logo extraction...")
                success = extract_watermark_logo_enhanced(original_path, watermarked_path, output_path)
            
            # If still failing, try legacy frequency domain method
            if not success:
                print("Enhanced method failed, trying frequency domain method...")
                success = extract_watermark_frequency_domain(original_path, watermarked_path, output_path)

            # Final fallback: classic block DCT amplitude embedding extraction
            if not success:
                print("Frequency domain methods failed, trying classic block DCT extraction fallback...")
                success = extract_watermark_image_block_dct(original_path, watermarked_path, output_path)
            
            if success:
                return send_file(output_path, as_attachment=True, download_name='extracted_watermark.png')
            else:
                return jsonify({'error': 'Failed to extract image watermark. Try selecting "Text" extraction type if the watermark is text-based.'}), 500
        else:
            return jsonify({'error': 'Invalid extraction type. Must be "text", "image", or "auto"'}), 400
            
    except Exception as e:
        print(f"Error in extract_watermark_api: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        # Cleanup temporary files
        if 'original_path' in locals() and os.path.exists(original_path):
            os.remove(original_path)
        if 'watermarked_path' in locals() and os.path.exists(watermarked_path):
            os.remove(watermarked_path)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Watermarking API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
