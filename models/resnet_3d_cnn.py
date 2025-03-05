import torch.nn as nn
from torchvision import models
import torch
from base_model import BaseModel


class ResNet3DClassifier(BaseModel):
    def __init__(self, dropout: float, freeze : bool = False):
        super(ResNet3DClassifier, self).__init__()
        self.cnn3d = models.video.r3d_18(pretrained=True)
        
        if freeze:
            for param in self.cnn3d.parameters():
                param.requires_grad = False
                
        in_features = self.cnn3d.fc.in_features
        
        self.cnn3d.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, 1),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.permute(0, 2, 1, 3, 4)
        return self.cnn3d(x) # (batch_size, 1)