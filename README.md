# FS2025_PR_SVM_MLP_CNN

This repository contains implementations of three different classification methods applied to the MNIST dataset: Support Vector Machines (SVM), Multilayer Perceptron (MLP), and Convolutional Neural Networks (CNN). This project is part of the Pattern Recognition course at the University of Fribourg, Spring Semester 2025.

## Project Structure

```
FS2025_PR_SVM_MLP_CNN/
├── mlp/
│   └── plots/                # Learning curves and performance visualizations
│       └── *.png
├── MLP.ipynb                 # Jupyter notebook with MLP implementation and experiments
├── cnn/                      # CNN implementation directory
│   ├── logs.txt              # Training logs and experiment results
│   ├── models/               # Saved CNN models
│   │   └── best_model.pth    # Best performing CNN model
│   └── plots/                # Learning curves and performance visualizations
│       └── trial_*_curves.png
├── cnn.ipynb                 # Jupyter notebook with CNN implementation and experiments
├── pyproject.toml            # Project dependencies and metadata
├── svm/                      # SVM implementation directory
│   ├── model/                # Saved SVM models
│   │   └── svm_model.joblib  # Best performing SVM model
│   ├── svm.ipynb             # Jupyter notebook with SVM experiments
│   └── svm.py                # SVM implementation code
└── utils.py                  # Common utility functions for data loading and preprocessing
```

## Dataset

We use the full MNIST dataset consisting of:
- 60,000 training images
- 10,000 test images

Each image is a 28x28 grayscale representation of a handwritten digit (0-9).

## Classification Methods

### Support Vector Machine (SVM)

The SVM implementation explores different kernels (linear and RBF) and optimizes hyperparameters through cross-validation.

- **Location**: `svm/` directory and `svm.ipynb`
- **Key Features**:
  - Kernel comparison
  - Hyperparameter optimization
  - Cross-validation results

### Multilayer Perceptron (MLP)

The MLP is a feedforward neural network with at least one hidden layer, implemented to recognize the handwritten digits.

- **Location**: `MLP.ipynb`
- **Key Features**:
  - Hidden layer size optimization
  - Learning rate experiments
  - Training and validation curves

### Convolutional Neural Network (CNN)

The CNN implementation uses convolutional layers for feature extraction followed by fully connected layers for classification.

- **Location**: `cnn/` directory and `cnn.ipynb`
- **Key Features**:
  - Multiple convolution layers
  - Various kernel sizes
  - Learning rate optimization
  - Feature aggregation experiments


## Usage

### Prerequisites

This project requires Python 3.8+ and the following libraries:
- PyTorch
- scikit-learn
- NumPy
- Matplotlib
- Jupyter

You can install all dependencies using:

```bash
pip install -e .
```

### Running the Models

1. **SVM**:
   Open `svm.ipynb` in Jupyter.

2. **MLP**:
   Open `MLP.ipynb` in Jupyter and run all cells.

3. **CNN**:
   Open `cnn.ipynb` in Jupyter and run all cells.

## Contributors

- Richmond Tetteh Djwerter
- Raunak Pillai
- Kaushik Raghupathruni
- Nathan Wegmann
- Yi-Shuin Alan Wu

## Acknowledgements

This project was completed as part of the Pattern Recognition course supervised by Dr. Andreas Fischer and teaching assistant Michael Jungo at the University of Fribourg.
