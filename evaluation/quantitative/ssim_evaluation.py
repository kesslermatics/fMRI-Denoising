# -----------------------------------
# Quantitative fMRI Evaluation: SSIM
# -----------------------------------

# === Libraries ===
import numpy as np
import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import nibabel as nib            
import os

# === Main SSIM Evaluation Function ===
def main():
    # Step 1: Load x0 and x_hat as 4D arrays (T, D, H, W) and convert to tensors
    # - Normalize both arrays to the same intensity scale (0–1)

    # Step 2: Prepare SSIM metric
    # - Use torchmetrics StructuralSimilarityIndexMeasure (data_range=1.0)
    # - Note: only supports 2D inputs, so we will compare slice-wise

    # Step 3: Loop through time frames and select a slice direction (e.g. axial)
    # - For each time frame and selected slice:
    #   - Extract 2D slices from x0 and x_hat
    #   - Compute SSIM per slice
    #   - Average across slices and/or time points

    # Step 4: Aggregate all SSIM scores (e.g. mean SSIM over time)

    # Step 5: Print average SSIM score
    # - Optionally: plot SSIM progression over time

    # Step 6: Optional – Save results
    
    print("SSIM Evaluation")

# === Entry Point ===
if __name__ == "__main__":
    main()
