import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# set TORCH_HOME to /tmp
import os
os.environ["TORCH_HOME"] = "/tmp"
class Net(nn.Module):
    """Fine-tuned ResNet-50 model for medical image classification"""

    def __init__(self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
        super(Net, self).__init__()
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer with our custom classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Optionally freeze the backbone for transfer learning
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Only train the final classifier
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
    
    def get_backbone_parameters(self):
        """Get parameters of the backbone (excluding final classifier)"""
        return [p for name, p in self.backbone.named_parameters() if 'fc' not in name]
    
    def get_classifier_parameters(self):
        """Get parameters of the final classifier"""
        return [p for name, p in self.backbone.named_parameters() if 'fc' in name]