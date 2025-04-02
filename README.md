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
   git clone https://github.com/akb0787/CorrNet-Pruning.git
   cd CorrNet-Pruning
   
## Install dependencies:
pip install -r requirements.txt

## Usage
### Run the main script to train, prune, and evaluate a VGG16 model on CIFAR-10:
python main.py

## Project Structure
- corrnet/: Core package with modular components.
- - data_preprocessing.py: Data loading and augmentation.
- - model_builder.py: CNN model construction.
- - pruning.py: Correlation-based pruning logic.
- - training.py: Model training and fine-tuning.
- - evaluation.py: Model evaluation.
- - utils.py: Utility functions (e.g., plotting).
- main.py: Entry point for running the pipeline.

## Citation
Kumar, A., Yin, B., Shaikh, A.M. et al. CorrNet: pearson correlation based pruning for efficient convolutional neural networks. Int J Mach Learn Cyber (2022). https://doi.org/10.1007/s13042-022-01624-5
