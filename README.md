# CT Scan Classification Model

A deep learning project for classifying chest CT scan images using PyTorch. This project implements improved versions of DenseNet and VGG19 architectures for medical image classification.

## 🏥 Project Overview

This project focuses on developing a robust deep learning model for chest CT scan image classification. The model can distinguish between different chest conditions from CT scan images, making it useful for medical diagnosis assistance.

## 🚀 Features

- **Multiple Model Architectures**: Supports both DenseNet121 and VGG19 with custom improvements
- **Data Augmentation**: Comprehensive image transformations for better model generalization
- **Training Pipeline**: Complete training and evaluation pipeline with progress tracking
- **Visualization Tools**: Automatic generation of training curves and confusion matrices
- **Easy Dataset Management**: Automated dataset download from Kaggle

## 📁 Project Structure

```
ct-scan-model/
├── main.py                 # Main training script
├── dataset_download.py     # Dataset download utility
├── requirements.txt        # Python dependencies
├── model/
│   ├── __init__.py
│   └── model.py           # Model architectures (DenseNet, VGG19)
├── dataloader/
│   └── dataloader.py      # Data loading and preprocessing
├── trainer/
│   └── trainer.py         # Training and evaluation logic
├── utils/
│   └── plot.py           # Visualization utilities
├── dataset/              # Dataset directory (created after download)
└── results/              # Generated plots and metrics
    ├── Accuracy_graph.jpg
    ├── Loss_graph.jpg
    └── confusion_matrix.jpg
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ct-scan-model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle API** (for dataset download)
   - Install Kaggle CLI: `pip install kaggle`
   - Get your Kaggle API credentials from [Kaggle Settings](https://www.kaggle.com/settings/account)
   - Place `kaggle.json` in `~/.kaggle/` directory

## 📊 Dataset

The project uses the [Chest CT-Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) from Kaggle, which contains CT scan images categorized into different chest conditions.

### Download Dataset

```bash
python dataset_download.py
```

This will automatically:
- Download the dataset from Kaggle
- Extract it to the `dataset/` directory
- Organize it into train/test/validation splits

## 🎯 Usage

### Training the Model

1. **Run the main training script**:
   ```bash
   python main.py
   ```

2. **Customize hyperparameters** in `main.py`:
   ```python
   IMG_SIZE = 200          # Image size for training
   BATCH_SIZE = 32         # Batch size
   lr = 0.0001            # Learning rate
   NUM_EPOCH = 30         # Number of training epochs
   model_name = 'densenet' # Model architecture ('densenet' or 'vgg')
   ```

### Model Architectures

The project supports two improved model architectures:

1. **ImprovedDenseNetCTScan**: Enhanced DenseNet121 with custom classifier
2. **ImprovedVGG19CTScan**: Enhanced VGG19 with custom classifier

Both models include:
- Pre-trained backbones
- Custom classifier layers with dropout
- Optimized for medical image classification

## 📈 Results

The training process automatically generates:

- **Training/Validation Loss Curves**: Track model convergence
- **Training/Validation Accuracy Curves**: Monitor performance improvement
- **Confusion Matrix**: Visualize classification performance
- **Classification Report**: Detailed metrics (precision, recall, F1-score)

All results are saved in the `results/` directory.

### Training Progress Visualization

#### Loss Curves
![Training and Validation Loss](results/Loss_graph.jpg)

*The loss curves show the training and validation loss over epochs, helping to identify overfitting and convergence patterns.*

#### Accuracy Curves
![Training and Validation Accuracy](results/Accuracy_graph.jpg)

*The accuracy curves demonstrate the model's learning progress and generalization performance.*

#### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.jpg)

*The confusion matrix provides a detailed view of the model's classification performance across all classes, showing true positives, false positives, true negatives, and false negatives.*

## 🔧 Configuration

### Data Augmentation

The training pipeline includes comprehensive data augmentation:

```python
# Training transforms
- Resize to specified image size
- Random horizontal flip
- Random rotation (±10 degrees)
- Color jittering (brightness, contrast)
- Normalization with ImageNet statistics
```

### Model Configuration

Both models use a custom classifier with:
- 1024 → 128 → num_classes architecture
- ReLU activation functions
- Dropout layers (0.3, 0.1) for regularization

## 🖥️ System Requirements

- **Python**: 3.8+
- **CUDA**: Compatible GPU for faster training (optional)
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ free space for dataset

## 📋 Dependencies

Key dependencies include:
- `torch` & `torchvision`: Deep learning framework
- `matplotlib` & `seaborn`: Visualization
- `scikit-learn`: Metrics and evaluation
- `kaggle`: Dataset download
- `PIL`: Image processing


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [Mohamed Hany](https://www.kaggle.com/mohamedhanyyy) on Kaggle
- Built with PyTorch and torchvision
- Inspired by medical image classification research