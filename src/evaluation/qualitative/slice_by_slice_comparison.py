# -----------------------------
# Qualitative fMRI Evaluation
# Slice-by-Slice Viewer Script
# -----------------------------

# === Libraries ===
import numpy as np                
import matplotlib.pyplot as plt   
import random                     
import nibabel as nib             
import os                         

# === Main Evaluation Function ===
def main():
    # Step 1: Load fMRI data
    # - Load x0 (original/clean) and x_hat (denoised) 4D arrays
    # - Shape: (T, D, H, W) or (T, H, W, D)

    # Step 2: Check shape consistency and normalize intensities if needed

    # Step 3: Choose a number of random time frames to inspect
    # - Use random.sample to pick t1, t2, ..., tn from total T
    # Maybe make this automated for each iteration

    # Step 4: For each selected time frame:
    # - Extract the 3D volume from x0 and x_hat
    # - Choose a fixed slice direction (e.g. axial)
    # - Select one or multiple slice indices (e.g. middle slice)

    # Step 5: Plot original and denoised slice side-by-side
    # - Use consistent color scaling (vmin, vmax)
    # - Add titles: "Original" and "Denoised"
    # - Optional: Add difference map

    # Step 6: Optional – Save plots to output directory
    # - Use consistent filenames (e.g. t10_slice32.png)
    print("Slice-by-Slice Comparison")

# === Entry Point ===
if __name__ == "__main__":
    main()
