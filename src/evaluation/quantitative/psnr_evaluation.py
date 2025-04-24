# ----------------------------------
# Quantitative fMRI Evaluation: PSNR
# ----------------------------------

# === Libraries ===
import numpy as np                
import torch                    
import torch.nn.functional as F  
import nibabel as nib            
import os                        

# === Main PSNR Evaluation Function ===
def main():
    # Step 1: Load x0 (ground truth) and x_hat (denoised) as 4D arrays (T, D, H, W)
    # - Convert to torch tensors if necessary

    # Step 2: Normalize both arrays to the same intensity range (e.g., 0 to 1)

    # Step 3: Loop over time frames (t = 0 to T-1)
    # - For each time frame:
    #   - Compute MSE between x_hat[t] and x0[t]
    #   - Use PSNR formula: PSNR = 10 * log10(MAX^2 / MSE)

    # Step 4: Store and average PSNR values across all frames

    # Step 5: Print final average PSNR
    # - Optionally: print per-frame PSNR or visualize over time

    # Step 6: Optional – Save results to file or CSV
    
    print("PSNR Evaluation")

# === Entry Point ===
if __name__ == "__main__":
    main()
