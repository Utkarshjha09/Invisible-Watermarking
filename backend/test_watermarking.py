#!/usr/bin/env python3
"""
Test script for watermarking algorithms
Use this to test different watermarking methods and compare results
"""

import cv2
import numpy as np
import os
from app import (
    embed_watermark_image, extract_watermark_image,
    embed_watermark_image_block_dct, extract_watermark_image_block_dct,
    embed_watermark_frequency_domain, extract_watermark_frequency_domain
)

def test_all_methods(original_path, watermark_path, output_dir="test_results"):
    """Test all watermarking methods and save results"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    methods = [
        ("basic_dct", embed_watermark_image, extract_watermark_image),
        ("block_dct", embed_watermark_image_block_dct, extract_watermark_image_block_dct),
        ("frequency_domain", embed_watermark_frequency_domain, extract_watermark_frequency_domain)
    ]
    
    results = {}
    
    for method_name, embed_func, extract_func in methods:
        print(f"\nTesting {method_name} method...")
        
        # Embed watermark
        watermarked_path = os.path.join(output_dir, f"watermarked_{method_name}.png")
        extracted_path = os.path.join(output_dir, f"extracted_{method_name}.png")
        
        try:
            # Embedding
            embed_success = embed_func(original_path, watermark_path, watermarked_path)
            if not embed_success:
                print(f"❌ {method_name}: Embedding failed")
                continue
            
            # Extraction
            extract_success = extract_func(original_path, watermarked_path, extracted_path)
            if not extract_success:
                print(f"❌ {method_name}: Extraction failed")
                continue
            
            # Calculate quality metrics
            quality = calculate_quality_metrics(original_path, watermarked_path, watermark_path, extracted_path)
            results[method_name] = quality
            
            print(f"✅ {method_name}: Success")
            print(f"   PSNR: {quality['psnr']:.2f} dB")
            print(f"   SSIM: {quality['ssim']:.4f}")
            print(f"   Extraction Quality: {quality['extraction_score']:.4f}")
            
        except Exception as e:
            print(f"❌ {method_name}: Error - {e}")
    
    # Print comparison
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    for method, metrics in results.items():
        print(f"{method:15} | PSNR: {metrics['psnr']:6.2f} | SSIM: {metrics['ssim']:.4f} | Extraction: {metrics['extraction_score']:.4f}")
    
    # Find best method
    if results:
        best_method = max(results.keys(), key=lambda x: results[x]['extraction_score'])
        print(f"\n🏆 Best method for extraction quality: {best_method}")
    
    return results

def calculate_quality_metrics(original_path, watermarked_path, watermark_path, extracted_path):
    """Calculate PSNR, SSIM and extraction quality"""
    
    # Load images
    original = cv2.imread(original_path)
    watermarked = cv2.imread(watermarked_path)
    watermark_orig = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    extracted = cv2.imread(extracted_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate PSNR between original and watermarked
    mse = np.mean((original.astype(float) - watermarked.astype(float)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Calculate SSIM (simplified version)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    watermarked_gray = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    
    # Simple SSIM calculation
    mu1 = np.mean(original_gray)
    mu2 = np.mean(watermarked_gray)
    sigma1 = np.var(original_gray)
    sigma2 = np.var(watermarked_gray)
    sigma12 = np.mean((original_gray - mu1) * (watermarked_gray - mu2))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    
    # Calculate extraction quality (correlation with original watermark)
    if extracted is not None and watermark_orig is not None:
        # Resize extracted to match watermark size
        h, w = watermark_orig.shape
        extracted_resized = cv2.resize(extracted, (w, h))
        
        # Normalize both
        watermark_norm = watermark_orig.astype(float) / 255.0
        extracted_norm = extracted_resized.astype(float) / 255.0
        
        # Calculate normalized cross-correlation
        correlation = np.corrcoef(watermark_norm.flatten(), extracted_norm.flatten())[0, 1]
        extraction_score = max(0, correlation)  # Ensure non-negative
    else:
        extraction_score = 0
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'extraction_score': extraction_score
    }

if __name__ == "__main__":
    # Example usage
    original_path = "test_original.jpg"  # Replace with your test image
    watermark_path = "test_watermark.png"  # Replace with your watermark
    
    if os.path.exists(original_path) and os.path.exists(watermark_path):
        results = test_all_methods(original_path, watermark_path)
    else:
        print("Please provide test_original.jpg and test_watermark.png files")
        print("Or modify the paths in this script")