# Gradient Leakage Simulations

This repository contains adaptations of the Deep Leakage from Gradients (DLG) method to reproduce the simulations mentioned in my bachelor thesis. The experiments explore the feasibility of reconstructing input images from gradients in different scenarios using different models.
The code is based on the original DLG paper [find here the repository https://github.com/mit-han-lab/dlg], and it has been modified to work with different underlying models.

## Files and Experiments

### 1. **DLG on (single image) linear regression**
   - **Description**: We adapt the DLG method for a linear regression model with a single input image. The goal is to reconstruct the original image from the gradients of the model.
   - **Characteristics**:
     - Uses a single image from the CIFAR-100 dataset or a custom image provided by the user.
     - Implements a simple linear regression model.
     - Attempts to reconstruct the input image using gradient matching.

### 2. **Alpha reconstruction on LR**
   - **Description**: We build on the linear regression model with a single image but reconstructs the original image by multiplying the input by a specifically computed constant (`1/alpha`).
   - **Characteristics**:
     - Demonstrates a simpler reconstruction approach compared to DLG.
     - Highlights the fact that the gradient in this case encloses the structure of the input image, allowing for a straightforward reconstruction.

### 3. **DLG on linear regression (multiple)**
   - **Description**: We extend the DLG method to a linear regression model with multiple input images. However, the reconstruction fails in this scenario.
   - **Characteristics**:
     - Uses multiple images as input to the linear regression model.
     - Explores the limitations of the DLG method in reconstructing inputs when gradients are derived from multiple images.

### 4. **multiple images - LeNet**
   - **Description**: We implement a neural network model (LeNet) to explore the DLG and iDLG attacks.
   - **Characteristics**:
     - Uses a LeNet architecture for image classification.
     - Explores the potential for gradient leakage in a more complex model compared to linear regression.
     - The images can be chosen from either the CIFAR-100 dataset or the MINIST dataset. 

### 5. **different batch sizes comparison**
   - **Description**: This script compares the performance of the DLG method with different batch sizes, for datasets MNIST and CIFAR-100.
   - **Characteristics**:
     - Plots the average loss for n experiments for different batch sizes.


## Repository Structure


📂 **Linear Regression GL Simulations**
- **`DLG on (single image) linear regression.py`**: Code for the first experiment with a single image.
- **`alpha reconstruction on LR.py`**: Code for the second experiment using the alpha reconstruction method.
- **`DLG on linear regression (multiple).py`**: Code for the third experiment with multiple images.

📂 **NeuralNetwork simulations**
- **`multiple images - LeNet.py`**: Code for a neural network simulation with multiple images.
- **`different batch sizes comparison.py`**: Code for comparing the performance of the DLG method with different batch sizes.
- 📂 **models**: Necessary to set up the underliying model (LeNet)



