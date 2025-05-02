# NEUBIAS Computational Pathology Course

## Purpose
Teach students the full pipeline of computational pathology using PyTorch, PyTorch Lightning, Albumentations, and Cytomine — from dataset prep to model inference — via engaging, step-by-step notebooks.

## Target Audience
Grad students, researchers, and professionals in computational biology, medical imaging, or AI in healthcare.

## Introduction to Computational Pathology

Computational Pathology is an emerging field that utilizes digital pathology images (whole-slide images or WSIs) and applies computational methods, including machine learning and artificial intelligence, to analyze them. This field aims to enhance diagnostic accuracy, predict patient outcomes, discover new biomarkers, and ultimately improve patient care.

This course focuses on the practical application of deep learning techniques within this domain.

## Course Overview

This repository contains a series of interactive Jupyter notebooks designed to guide you through the complete workflow of a computational pathology project. We will cover:

1.  **Dataset Handling:** Creating PyTorch Datasets and DataLoaders specifically for pathology images (patches), and utilizing Albumentations for effective data augmentation.
2.  **Model Development:** Building and training deep learning models (CNNs, potentially using transfer learning with ResNet) using PyTorch Lightning for structured and efficient training.
3.  **Experiment Management:** Leveraging Lightning Callbacks, understanding training hooks, and logging experiments (optionally with Weights & Biases).
4.  **Inference and Deployment:** Exporting models using TorchScript and performing inference on large whole-slide images using the Cytomine platform.

Each notebook builds upon the previous one, providing hands-on experience with the core technologies and concepts.

## Dataset

This course uses a patch classification dataset where images are organized into subfolders based on their labels.

**Action Required:** Please place your dataset folder inside the `data/raw/patch_classification_dataset/` directory. The expected structure is:

```
data/
└── raw/
    └── patch_classification_dataset/
        ├── label_A/
        │   ├── image1.png
        │   └── image2.png
        │   └── ...
        ├── label_B/
        │   ├── image3.png
        │   └── ...
        └── ...
```

Once the data is in place, we can proceed with preprocessing and splitting (Tasks 2.2 and onwards).

## Setup

1.  **Create a Python environment:**
    It is recommended to use a virtual environment (like `venv` or `conda`) to manage dependencies.
    ```bash
    # Using venv
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    
    # Or using conda
    conda create -n neubias-course python=3.10 # Or your preferred Python version
    conda activate neubias-course
    ```

2.  **Install Dependencies:**
    Install all required packages using the provided requirements file:
    ```bash
    pip install -r requirements.txt
    ```

*(More details about structure, setup, and modules will be added here.)*
