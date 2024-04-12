'''
A simple implementation of CNN for classification
'''
import torch.nn as nn
from .base_classifier import BasePLClassifier


class SimpleCNN(BasePLClassifier):
    def __init__(self, **kwargs
                 ):
        super().__init__(**kwargs)
        self.model_name = 'simplecnn'


        # network structure
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=self.dropout_rate),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=self.dropout_rate),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * ((self.img_size//4)**2), 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(128, self.num_classes)
        )


    def forward(self, x,t):
        x = self.features(x)
        x = self.classifier(x)
        return x
