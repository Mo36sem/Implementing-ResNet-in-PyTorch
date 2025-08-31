# Implementing ResNet in PyTorch

This project implements a simplified version of the ResNet (Residual Network) architecture in PyTorch and applies it to the CIFAR-10 image classification dataset.

## 🎯 Objective

- Build custom residual blocks with skip connections.
- Construct the ResNet-18 architecture from scratch using PyTorch.
- Train the model using CIFAR-10 dataset and evaluate its performance.

## 🧱 Key Components

### 🔹 Residual Block
- Two convolutional layers with Batch Normalization.
- Identity (skip) connections to allow gradient flow and mitigate vanishing gradients.

### 🔹 ResNet-18 Architecture
- Initial Conv + BN + MaxPool.
- Four stages of residual blocks (increasing depth).
- Fully connected (Linear) layer for classification.

### 🔹 Training Strategy
- Dataset: CIFAR-10 (60,000 images, 10 classes)
- Split: 70% Training / 30% Validation
- Data Augmentation: RandomCrop, HorizontalFlip, Normalization
- Optimizer: SGD / Adam (with tuning)
- Evaluation: Accuracy and Loss plots over epochs

## 📊 Results

The notebook includes plots for:
- Training vs. Validation Accuracy
- Training vs. Validation Loss

These visualizations help understand model convergence and generalization.

## 📁 Files

- `Resnet_with_pytorch.ipynb`: Jupyter Notebook containing full model code and training procedure.

## 📚 References

- [CIFAR-10 Tutorial – PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [UDLBook: Residual Networks](https://github.com/udlbook/udlbook/blob/main/Notebooks/Chap11/11_2_Residual_Networks.ipynb)
- [UDLBook: Batch Normalization](https://github.com/udlbook/udlbook/blob/main/Notebooks/Chap11/11_3_Batch_Normalization.ipynb)

## 🚀 Future Improvements
- Try deeper ResNet versions (e.g., ResNet-34, ResNet-50)
- Use learning rate scheduling and early stopping
- Experiment with regularization techniques

