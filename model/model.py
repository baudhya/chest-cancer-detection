import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights, ResNet50_Weights, DenseNet121_Weights

class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            # Input: 3 x 224 x 224
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # Output: 64 x 56 x 56

            # Block 2: Standard convolutional layers to process features
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 3: Strided convolution for learnable downsampling
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), # Output: 256 x 28 x 28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: More standard convolutional layers
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 5: Increase channel depth before final downsampling
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 6: Final strided convolution to reduce to final feature map size
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1), # Output: 512 x 14 x 14
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # --- Fully Connected Classifier (FNN) ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImprovedDenseNetCTScan(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ImprovedDenseNetCTScan, self).__init__()
        self.densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)
    

class ImprovedVGG19CTScan(nn.Module):
    def __init__(self, out_classes, pretrained=True):
        super(ImprovedVGG19CTScan, self).__init__()
        self.vgg = models.vgg19(weights=VGG19_Weights.DEFAULT if pretrained else None)

        in_features = self.vgg.classifier[0].in_features
        self.vgg.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_classes)
        )

    def forward(self, x):
        return self.vgg(x)


class ImprovedResNetCTScan(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ImprovedResNetCTScan, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


def get_model(num_classes, model_name = 'densenet', pretrained = True):
    if model_name == 'densenet':
        return ImprovedDenseNetCTScan(num_classes, pretrained=pretrained)
    elif model_name == "resnet":
        return ImprovedResNetCTScan(num_classes, pretrained=pretrained)
    elif model_name == "vgg":
        return ImprovedVGG19CTScan(num_classes, pretrained=pretrained)
    elif model_name == "custom":
        return CustomCNN(num_classes)
    else:
        raise ValueError(f"Model {model_name} not found")