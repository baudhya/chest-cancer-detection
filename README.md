# 🩺 CT Scan Classification Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-orange.svg)](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)

A comprehensive deep learning project for automated classification of chest CT scan images using PyTorch. This repository implements multiple model architectures including a custom CNN, DenseNet121, ResNet50, and VGG19 with advanced data augmentation and comprehensive evaluation pipelines.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
- [Model Architectures](#-model-architectures)
- [Results](#-results)
- [Configuration](#-configuration)
- [System Requirements](#-system-requirements)
- [License](#-license)

## 🎯 Overview

This project focuses on developing robust deep learning models for chest CT scan image classification. The system can automatically categorize CT scan images into four different chest conditions, providing valuable assistance for medical diagnosis and research.

### Key Capabilities:
- **Multi-class Classification**: Distinguish between 4 chest conditions
- **Multiple Model Architectures**: Custom CNN, DenseNet121, ResNet50, and VGG19
- **Advanced Data Augmentation**: Comprehensive image transformations for improved generalization
- **Automated Evaluation**: Complete pipeline with metrics and visualizations
- **Transfer Learning**: Leverage pre-trained models for better performance

## ✨ Features

- 🔧 **Four Model Architectures**: Custom CNN, DenseNet121, ResNet50, and VGG19
- 🎨 **Advanced Data Augmentation**: Random flips, affine transformations, color jittering, Gaussian blur, and random erasing
- 📊 **Comprehensive Evaluation**: Loss curves, accuracy plots, confusion matrices, and classification reports
- 🚀 **Easy Training Pipeline**: One-command training with configurable parameters
- 📈 **Real-time Monitoring**: Progress tracking and performance visualization
- 🔄 **Automated Dataset Management**: Direct download from Kaggle
- 🎯 **Class Mapping**: Simplified class names for better understanding

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)
- 8GB+ RAM recommended
- 2GB+ free disk space

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ct-scan-model.git
   cd ct-scan-model
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dependencies** (for dataset download)
   ```bash
   pip install requests tqdm
   ```
   Note: These are already included in requirements.txt

## 📊 Dataset

The project uses the [Chest CT-Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) from Kaggle, containing CT scan images categorized into four chest conditions.

### Class Categories
- **Adenocarcinoma**: Left lower lobe adenocarcinoma (T2 N0 M0 Ib)
- **Large Cell Carcinoma**: Left hilum large cell carcinoma (T2 N2 M0 IIIa)
- **Normal**: Normal chest CT scans
- **Squamous Cell Carcinoma**: Left hilum squamous cell carcinoma (T1 N2 M0 IIIa)

### Download Dataset
```bash
python dataset_download.py
```

This script will:
- ✅ Download the dataset from a public source
- ✅ Extract it to the `dataset/` directory
- ✅ Show download progress with a progress bar
- ✅ Clean up temporary files
- ✅ Provide fallback instructions if download fails

## 🚀 Usage

### Basic Training

Run the model with default parameters (DenseNet121):
```bash
python main.py
```

### Advanced Training with Custom Parameters

```bash
python main.py \
    --batch_size 32 \
    --lr 0.0001 \
    --num_epoch 50 \
    --model_name densenet \
    --pretrained True
```

### Train All Models

Use the provided script to train all models:
```bash
chmod +x run.sh
./run.sh
```

### Command-line Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 32 | Training batch size |
| `--lr` | 0.0001 | Learning rate |
| `--num_epoch` | 10 | Number of training epochs |
| `--model_name` | densenet | Model architecture (custom/densenet/resnet/vgg) |
| `--pretrained` | True | Use pre-trained weights |

## 🧠 Model Architectures

### CustomCNN
- **Architecture**: Custom convolutional neural network
- **Input Size**: 224x224x3
- **Features**: 6 convolutional blocks with batch normalization
- **Classifier**: 3-layer fully connected network (512×14×14 → 1024 → 128 → num_classes)
- **Regularization**: Dropout (0.3, 0.1)

### ImprovedDenseNetCTScan
- **Backbone**: DenseNet121 (pre-trained)
- **Classifier**: Custom 3-layer architecture
- **Features**: 1024 → 128 → num_classes
- **Regularization**: Dropout (0.3, 0.1)

### ImprovedResNetCTScan
- **Backbone**: ResNet50 (pre-trained)
- **Classifier**: Custom 3-layer architecture
- **Features**: 2048 → 1024 → 128 → num_classes
- **Regularization**: Dropout (0.3, 0.1)

### ImprovedVGG19CTScan
- **Backbone**: VGG19 (pre-trained)
- **Classifier**: Custom 3-layer architecture
- **Features**: 4096 → 1024 → 128 → num_classes
- **Regularization**: Dropout (0.3, 0.1)

## 📈 Results

The training process automatically generates comprehensive visualizations and metrics for each model:

### Training Progress Visualization

#### Loss Curves
- `results/Loss_custom_graph.jpg`
- `results/Loss_densenet_graph.jpg`
- `results/Loss_resnet_graph.jpg`
- `results/Loss_vgg_graph.jpg`

#### Accuracy Curves
- `results/Accuracy_custom_graph.jpg`
- `results/Accuracy_densenet_graph.jpg`
- `results/Accuracy_resnet_graph.jpg`
- `results/Accuracy_vgg_graph.jpg`

#### Confusion Matrices
- `results/confusion_custom_matrix.jpg`
- `results/confusion_densenet_matrix.jpg`
- `results/confusion_resnet_matrix.jpg`
- `results/confusion_vgg_matrix.jpg`

#### Class Distribution
- `results/class_distribution.jpg`

### Generated Metrics
- **Training/Validation Loss**: Track model convergence
- **Training/Validation Accuracy**: Monitor performance improvement
- **Confusion Matrix**: Class-wise performance analysis
- **Classification Report**: Precision, recall, F1-score for each class
- **Class Distribution**: Dataset balance visualization

## ⚙️ Configuration

### Data Augmentation Pipeline

```python
# Training Transforms
- Resize: (IMG_SIZE, IMG_SIZE)
- Random Horizontal Flip: p=0.5
- Random Affine: degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
- Color Jitter: brightness=0.2, contrast=0.2
- Gaussian Blur: kernel_size=(5, 9), sigma=(0.1, 5.0)
- Normalization: ImageNet statistics
- Random Erasing: p=0.5, scale=(0.02, 0.2)
```

### Training Configuration

- **Optimizer**: Adam with weight decay (0.0001)
- **Loss Function**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau (factor=0.1, patience=5)
- **Device**: Auto-detects CUDA/CPU
- **Validation**: Per-epoch evaluation

### Image Sizes
- **Custom CNN**: 224x224
- **Pre-trained Models**: 200x200 (optimized for GPU memory)

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8 | 3.9+ |
| **RAM** | 4GB | 8GB+ |
| **Storage** | 1GB | 2GB+ |
| **GPU** | CPU only | CUDA-compatible |
| **OS** | Linux/macOS/Windows | Linux |

## 📁 Project Structure

```
ct-scan-model/
├── main.py                 # Main training script
├── dataset_download.py     # Dataset download utility
├── run.sh                  # Batch training script
├── requirements.txt        # Python dependencies
├── model/
│   └── model.py           # Model architectures
├── dataloader/
│   └── dataloader.py      # Data loading and augmentation
├── trainer/
│   └── trainer.py         # Training loop and evaluation
├── utils/
│   ├── argument_parser.py # Command-line arguments
│   ├── config.py          # Configuration management
│   └── plot.py           # Visualization utilities
├── dataset/               # Dataset directory (created after download)
└── results/              # Generated plots and metrics
```

## 📝 License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Mohamed Hany](https://www.kaggle.com/mohamedhanyyy) for the Chest CT-Scan Images Dataset
- **Framework**: PyTorch and torchvision for deep learning capabilities
- **Community**: Open-source contributors and researchers in medical imaging