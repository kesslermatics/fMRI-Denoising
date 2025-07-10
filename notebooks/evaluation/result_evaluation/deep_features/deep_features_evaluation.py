# --------------------------------------------------------
# Perceptual Evaluation using Deep Features (e.g., LPIPS)
# --------------------------------------------------------

# Source: https://arxiv.org/pdf/1801.03924

# === Libraries ===
import numpy as np                  
import torch                        
import torchvision.models as models  # Pretrained CNNs (e.g., VGG)
import torchvision.transforms as transforms  
import lpips                        # if using the LPIPS library
import os

# === Main Deep Feature Evaluation Function ===
def main():
    # Step 1: Load x0 (ground truth) and x_hat (denoised) as 4D tensors (T, H, W, D)
    # - Optional: convert slices to 2D images for CNN input
    # - Normalize values to match pretrained model expectations (e.g., range [0,1] or [-1,1])

    # Step 2: Define a pretrained feature extractor
    # Option A: Use LPIPS directly (e.g., lpips.LPIPS(net='vgg'))
    # Option B: Use torchvision.models.vgg16(pretrained=True) and extract features manually

    # Step 3: For each time frame and selected slice (e.g. axial slice 32):
    # - Extract corresponding slices from x0 and x_hat
    # - Resize to 224x224 if required (for pretrained nets)
    # - Normalize channels (e.g. replicate grayscale → RGB if needed)
    # - Convert to torch tensors with correct shape: (B, C, H, W)

    # Step 4: Pass both images through the feature network
    # - Compute L2 or cosine distance between feature activations
    # - If using LPIPS: get score directly via lpips_model(x0, x_hat)

    # Step 5: Repeat over time and/or slices
    # - Average deep feature distances across all comparisons

    # Step 6: Output mean perceptual distance
    # - Optionally: visualize example patches or distances over time
    print("Deep Feature Evaluation")

# === Entry Point ===
if __name__ == "__main__":
    main()
