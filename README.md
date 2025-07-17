# fMRI Denoising using Conditional Diffusion Models

## 1. Project Description

This project implements and evaluates a conditional diffusion model, inspired by Google's SR3, for the task of denoising functional Magnetic Resonance Imaging (fMRI) data. The repository provides a complete workflow from data preprocessing and model training to inference and detailed result evaluation.

## 2. Repository Note

**Please note:** The code and experiments for this project were originally developed in the Kaggle notebook environment. The various notebooks for preprocessing, training, and evaluation have been consolidated and organized into the directory structure of this repository for clarity, version control, and final submission.
Keep in mind, that the input paths should be adjusted accordingly.


## 3. Repository Structure

The project is organized into two main directories:

-   **/models/**: This directory contains pre-trained model checkpoints. The `baseline.md` file provides information about the baseline models used for comparison.
-   **/notebooks/**: This directory contains all Jupyter notebooks, organized by function into the following subdirectories:
    -   `preprocessing/`: Notebooks for data preparation, such as coregistering and normalization.
    -   `modeling/`: The primary notebooks for training the different denoising models.
    -   `inference/`: Notebooks to run a trained model on test data and generate denoised images.
    -   `evaluation/`: Scripts and notebooks for the final analysis, including quantitative metrics (PSNR, SSIM, DBPS, tSNR) and qualitative visualizations.

## 4. Setup and Installation

To run this project, please ensure you have a Python environment with PyTorch and GPU support.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/kesslermatics/fMRI-Denoising
    cd fMRI-Denoising
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment. All required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## 5. How to Run the Project (Workflow)

The project is designed to be run in a logical sequence. Pre-trained models are provided in the `/models` directory, allowing you to skip directly to the inference or evaluation steps.

### Step 1: Preprocessing (Optional)

If you wish to process your own raw data, use the notebooks in `/notebooks/preprocessing`.

### Step 2: Training (Optional)

To train the models from scratch:
1.  **Denoising Model:** Use the notebooks in `/notebooks/modeling`.
2.  **Custom Perceptual Metric (DBPS):** Use the `dbps.ipynb` notebook found in `/notebooks/evaluation/result_evaluation/deep_features/`.

### Step 3: Inference

To generate denoised images from a trained model:
1.  Navigate to `/notebooks/inference/`.
2.  Open one of the inference notebooks (e.g., `SR3_inference.ipynb`).
3.  In the configuration cell, set the `resume_state` path to point to your desired model checkpoint.
4.  Run all cells to generate and save the denoised `.npy` files.

### Step 4: Evaluation

This is the final step to analyze and visualize the performance of the generated results.
1.  Navigate to `/notebooks/evaluation/result_evaluation/`.
2.  Open the `showcase.ipynb` notebook.
3.  Ensure the paths in the configuration cells point to the `.npy` files generated in Step 3 and any required metric models (like the DBPS model).
4.  Run the cells to produce quantitative charts (PSNR, SSIM, etc.) and qualitative visual comparisons.