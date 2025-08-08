import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights


class ImprovedDenseNetCTScan(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ImprovedDenseNetCTScan, self).__init__()
        self.densenet = models.densenet121(pretrained=pretrained)
        
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
        self.vgg = models.vgg19(weights=VGG19_Weights.DEFAULT, pretrained=pretrained)

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
        self.resnet = models.resnet50(pretrained=pretrained)
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
    return ImprovedVGG19CTScan(num_classes, pretrained=pretrained)