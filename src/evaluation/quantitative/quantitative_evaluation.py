"""
evaluate_slices.py

This script loads a ground-truth volume and a denoised/ SR volume from .npy files,
then computes PSNR and SSIM slice by slice in a batch-wise fashion to avoid
excessive RAM usage.

Note on saving images:
  - If you were to export each slice as a 64×64 PNG (~12 KB per file),
    you’d end up with 819 × 300 ≈ 245 700 files.
  - Total disk space ≈ 245 700 × 12 KB ≈ 2.8 GB.
  - In most cases it’s more efficient to process in-memory NumPy arrays
    rather than round‐trip through image files.

Note on memory:
  - The full volume has shape (64, 64, 819, 300) ⇒ 64·64·819·300 ≈ 1 006 387 200 float64 values.
  - At 8 bytes each that’s ≈ 7.5 GiB of RAM if fully loaded.
  - We avoid this by using `mmap_mode='r'` and processing one slice at a time.
"""

import math
import numpy as np
import cv2

# ─── Metric Functions ───────────────────────────────────────────────────────────

def calculate_psnr(img1, img2):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between two uint8 images.
    PSNR = 20 * log10(MAX_I / sqrt(MSE)),
    where MAX_I=255 and MSE = mean((img1-img2)^2).
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def _ssim2d(ch1, ch2):
    """
    Compute SSIM for a single-channel 2D image (following the standard formula).
    Uses an 11×11 Gaussian window (σ=1.5) and valid convolution to avoid border effects.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ch1 = ch1.astype(np.float64)
    ch2 = ch2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)
    # compute local means
    mu1 = cv2.filter2D(ch1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(ch2, -1, window)[5:-5, 5:-5]
    # squares and products
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    # variances and covariance
    sigma1_sq = cv2.filter2D(ch1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(ch2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12   = cv2.filter2D(ch1 * ch2, -1, window)[5:-5, 5:-5] - mu1_mu2
    # SSIM map
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """
    Compute SSIM between two images.
    Supports:
      - Grayscale: 2D arrays
      - RGB: 3D arrays with 3 channels (averages per-channel SSIM).
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return _ssim2d(img1, img2)
    elif img1.ndim == 3 and img1.shape[2] == 3:
        # average SSIM over R, G, B channels
        return float(np.mean([_ssim2d(img1[:, :, c], img2[:, :, c]) for c in range(3)]))
    else:
        raise ValueError("Unsupported image dimensions for SSIM.")


def normalize_to_uint8(arr):
    """
    Scale a floating-point array to uint8 [0,255].
    - Subtracts min, divides by max to normalize to [0,1], then scales to [0,255].
    - Rounds before casting to preserve accuracy.
    """
    a = arr.astype(np.float64)
    a -= a.min()
    max_val = a.max()
    if max_val != 0:
        a /= max_val
    a = (a * 255.0).round()
    return a.astype(np.uint8)


# ─── Configuration ─────────────────────────────────────────────────────────────

GT_INPUT_PATH    = '../../../input/gt_func_train_1.npy'    # ground-truth volume
DENOISED_INPUT_PATH = '../../../input/noisy_func_train_1.npy'  # denoised / SR volume

GT_PATH = GT_INPUT_PATH
DENOISED_PATH = DENOISED_INPUT_PATH


# ─── Load volumes as memory-mapped arrays ────────────────────────────────────────
# mmap_mode='r' avoids loading the entire array into RAM at once.
gt = np.load(GT_PATH, mmap_mode='r')  # shape: (64, 64, 819, 300)
sr = np.load(DENOISED_PATH, mmap_mode='r')


# ─── Batch-wise Evaluation ──────────────────────────────────────────────────────
# We iterate over the last two dims (819 × 300 = 245700 slices) to avoid 7.5 GiB RAM spike.
d0, d1, d2, d3 = gt.shape
total_psnr = 0.0
total_ssim = 0.0
count = 0

for i in range(d2):
    for j in range(d3):
        # extract one 64×64 slice from GT and SR volumes
        slice_gt = normalize_to_uint8(gt[:, :, i, j])
        slice_sr = normalize_to_uint8(sr[:, :, i, j])

        # accumulate PSNR & SSIM
        total_psnr += calculate_psnr(slice_sr, slice_gt)
        total_ssim += calculate_ssim(slice_sr, slice_gt)
        count += 1

        # print interim averages every 10 000 slices to track progress
        if count % 10000 == 0 or (i == d2-1 and j == d3-1):
            avg_psnr = total_psnr / count
            avg_ssim = total_ssim / count
            print(f"Processed {count}/{d2*d3} slices")
            print(f" Interim Average PSNR: {avg_psnr:.4f} dB | SSIM: {avg_ssim:.4f}")
            print("------------------------------------------------------")

# ─── Final Averages ─────────────────────────────────────────────────────────────
avg_psnr = total_psnr / count
avg_ssim = total_ssim / count

print(f"Final Average PSNR: {avg_psnr:.4f} dB")
print(f"Final Average SSIM: {avg_ssim:.4f}")
