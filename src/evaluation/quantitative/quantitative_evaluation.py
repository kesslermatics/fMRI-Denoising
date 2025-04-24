#!/usr/bin/env python3
"""
evaluate_slices_no_quant.py

This script loads a ground-truth volume and a denoised/SR volume from .npy files,
then computes PSNR and SSIM slice by slice in a batch-wise fashion to avoid
excessive RAM usage.

Note on saving images:
  - Exporting each slice as a 64×64 PNG (~12 KB) for 819×300 ≈ 245 700 files → ~2.8 GB.
  - It’s more efficient to process in-memory float arrays directly.

Note on memory:
  - Full volume shape: (64, 64, 819, 300) ⇒ 64·64·819·300 ≈ 1 006 387 200 floats.
  - At 8 bytes each ≈ 7.5 GiB RAM if fully loaded.
  - We avoid this by using `mmap_mode='r'` and processing one slice at a time.

Quantization:
  - **Removed**: metrics are computed on the original float data, preserving
    subtle noise differences that uint8 conversion would round away.
"""

import math
import numpy as np
import cv2

# ─── Metric Functions (float) ───────────────────────────────────────────────────

def calculate_psnr_float(img1, img2, peak):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between two float images.
    PSNR = 20 * log10( peak / sqrt(MSE) )
      - peak: maximum possible signal value (e.g. global max of GT volume)
      - MSE = mean((img1 - img2)**2)
    Range:
      - [0, ∞) dB, where ∞ indicates a perfect match (MSE=0).
      - Typical “good” SR yields 30–40 dB.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(peak / math.sqrt(mse))


def _ssim2d_float(ch1, ch2, peak):
    """
    Compute SSIM for a single-channel 2D float image using an 11×11 Gaussian window.
    SSIM formula references:
      - C1 = (0.01*peak)**2, C2 = (0.03*peak)**2
    Range:
      - [−1, 1] (though for positive images typically [0, 1]),
        where 1 = perfect structural similarity.
    """
    C1 = (0.01 * peak) ** 2
    C2 = (0.03 * peak) ** 2

    ch1 = ch1.astype(np.float64)
    ch2 = ch2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)

    mu1 = cv2.filter2D(ch1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(ch2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(ch1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(ch2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12   = cv2.filter2D(ch1*ch2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim_float(img1, img2, peak):
    """
    Compute SSIM between two float images.
      - Grayscale (2D) or multi-channel (3D: average per-channel).
    Range:
      - [0,1], where 1 = perfect match.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return _ssim2d_float(img1, img2, peak)
    elif img1.ndim == 3:
        # average SSIM over each channel
        vals = [_ssim2d_float(img1[:,:,c], img2[:,:,c], peak) for c in range(img1.shape[2])]
        return float(np.mean(vals))
    else:
        raise ValueError("Unsupported image dimensions for SSIM.")


# ─── Configuration & Loading ────────────────────────────────────────────────────

GT_PATH = '../input/gt_func_train_1.npy'       # ground-truth volume
SR_PATH = '../input/noisy_func_train_1.npy'    # noisy/denoised volume

# Memory-map the volumes to avoid a 7.5 GiB spike
gt = np.load(GT_PATH, mmap_mode='r')  # shape (64, 64, 819, 300)
sr = np.load(SR_PATH, mmap_mode='r')

# Determine the global maximum signal for PSNR/SSIM constants
global_peak = float(np.max(gt))


# ─── Batch-wise Evaluation ──────────────────────────────────────────────────────

d0, d1, d2, d3 = gt.shape
total_psnr = 0.0
total_ssim = 0.0
count = 0

for i in range(d2):
    for j in range(d3):
        # extract one 64×64 float slice
        slice_gt = gt[:, :, i, j].astype(np.float64)
        slice_sr = sr[:, :, i, j].astype(np.float64)

        # compute metrics on float data
        psnr = calculate_psnr_float(slice_sr, slice_gt, global_peak)
        ssim = calculate_ssim_float(slice_sr, slice_gt, global_peak)

        total_psnr += 0.0 if psnr == float('inf') else psnr
        total_ssim += ssim
        count += 1

        # print interim stats every 10 000 slices
        if count % 10000 == 0 or (i == d2-1 and j == d3-1):
            avg_psnr = total_psnr / count
            avg_ssim = total_ssim / count
            print(f"Processed {count}/{d2*d3} slices")
            print(f" Interim PSNR: {avg_psnr:.4f} dB | SSIM: {avg_ssim:.4f}")
            print("------------------------------------------------------")

# ─── Final Averages ─────────────────────────────────────────────────────────────

avg_psnr = total_psnr / count
avg_ssim = total_ssim / count
print(f"Final PSNR: {avg_psnr:.4f} dB")
print(f"Final SSIM: {avg_ssim:.4f}")
