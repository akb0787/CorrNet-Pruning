markdown

Collapse

Wrap

Copy
# CorrNet: Pearson Correlation Based Pruning for Efficient CNNs

This repository implements the "CorrNet" methodology from the research article "CorrNet: Pearson Correlation Based Pruning for Efficient Convolutional Neural Networks" published in the *International Journal of Machine Learning and Cybernetics*. It provides a framework for pruning convolutional neural networks (CNNs) using Pearson correlation to reduce model size and computational cost while maintaining accuracy.

## Features
- Preprocessing and augmentation of CIFAR-10 dataset.
- VGG16 model construction and training.
- Correlation-based filter pruning (CFS).
- Fine-tuning of pruned models.
- Evaluation and visualization of results.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CorrNet-Pruning.git
   cd CorrNet-Pruning
