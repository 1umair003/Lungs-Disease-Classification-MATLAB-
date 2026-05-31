# Fine-Tuned Automated Lung Disease Diagnosis on Chest Radiographs Using Deep Neural Networks in MATLAB

### Overview

This repository contains the MATLAB implementation of a deep learning pipeline for automated lung disease classification from chest radiographs. The study benchmarks three state-of-the-art CNN architectures - ResNet-18, Xception, and DenseNet-201 - fine-tuned with transfer learning on a curated dataset of ~10,000 chest X-ray images spanning five disease categories.

The work addresses a critical bottleneck in clinical diagnostics: the shortage of experienced radiologists, particularly in underserved healthcare settings, which leads to delayed or inconsistent diagnoses. By automating radiograph interpretation, this system aims to support faster and more reliable clinical decisions.


### Disease Categories

|    Class    |    Description   |
|-------------|------------------|
| Bacterial Pneumonia | Lung infection caused by bacteria |
| Viral Pneumonia | Lung infection caused by viruses |
| COVID-19 | SARS-CoV-2 pulmonary manifestation |
| Tuberculosis | Mycobacterium tuberculosis infection |
| Normal | Healthy lung radiograph |

## Models

### ResNet-18

An 18-layer Residual Network with skip connections that allow gradients to propagate effectively. Pretrained on ImageNet and fine-tuned with a new fully connected classification head. Input size: 224×224 px.

### Xception

An "Extreme Inception" architecture using depthwise-separable convolutions to drastically reduce parameters while maintaining representational power. Input size: 299×299 px.

### DenseNet-201

A 201-layer Densely Connected Network where each layer receives feature maps from all preceding layers, encouraging feature reuse and improving gradient flow. Input size: 224×224 px.

All models use a softmax + cross-entropy classification head replacing the original ImageNet output layers.

## Results 

|Model|Training Accuracy|Testing Accuracy|Training Time|
|-----|-----------------|----------------|-------------|
|ResNet-18|87.60%|86.30%|~755 min|
|Xception|86.41%|87.91%|~616 min|
|DenseNet-201|89.60%|90.63%|~1334 min|

DenseNet-201 achieved the highest accuracy (90.63%), benefiting from dense connectivity that captures subtle diagnostic features in chest X-rays.

## Per-Class Accuracy (DenseNet-201)

|Class|Correctly Classified|Accuracy|
|-----|--------------------|--------|
|COVID-19|401 / 410|98.8%|
|Tuberculosis398 | 401|98.0%|
|Normal|392 / 421|97.5%|
|Bacterial Pneumonia|350 / 355|87.3%|
|Viral Pneumonia|266 / 324|66.3%|

## Methodology

### Dataset & Preprocessing

~10,000 chest X-ray images from publicly available repositories

Resized to 224×224 px (ResNet-18, DenseNet-201) or 299×299 px (Xception)

Grayscale images converted to 3-channel RGB by channel duplication

Data augmentation: random horizontal flipping + random translation (±30 px)


### Training Configuration

Optimizer: Stochastic Gradient Descent with Momentum (SGDM)

Learning Rate: 1×10⁻⁴

Mini-batch Size: 10

Epochs: 10 (ResNet-18, DenseNet-201), 6 (Xception)

Validation: after every 3 iterations

Environment: MATLAB

### Transfer Learning Strategy

All models are initialized with ImageNet-pretrained weights. The original classification layers are removed and replaced with a task-specific fully connected layer with C=5 outputs, followed by softmax activation and cross-entropy loss minimization.

  
