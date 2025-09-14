#  L-Layer Neural Network from Scratch (NumPy)

This project is a complete implementation of an **L-layer neural network** from scratch using only **NumPy**.  
The goal was to learn and demonstrate the **fundamentals of deep learning** by coding everything manually, without using libraries like TensorFlow or PyTorch.



##  What I Built
From scratch, I implemented all the key components of a deep neural network:

- **Parameter Initialization**  
- **Forward Propagation**  
  - Linear step  
  - Activation functions: ReLU (hidden layers), Sigmoid (output layer)  
- **Cost Function** (Binary Cross-Entropy)  
- **Backward Propagation**  
  - Gradients for each layer  
- **Parameter Updates** (Gradient Descent)  
- **Prediction Function**  
- **Accuracy Calculation**  
- **Training Loop** (`L_layer_model` function with cost tracking)



##  Architecture


The network structure is flexible: you can define any number of layers and neurons with a Python list.

Example:
layers_dims = [2, 10, 5, 1]

Input layer → 2 features

Hidden layer 1 → 10 neurons (ReLU)

Hidden layer 2 → 5 neurons (ReLU)

Output layer → 1 neuron (Sigmoid for binary classification)
