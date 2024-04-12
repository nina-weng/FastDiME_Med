
from torchvision import models
import torch.nn as nn
from .base_classifier import BasePLClassifier


'''
adopted from bias causes project
for current use, only need the situation where num_classes=1
'''


class ResNet(BasePLClassifier):
    def __init__(self, 
                 pretrained,
                 model_scale='18',
                 add_dropout_layer=False,
                 dropout_rate=0.5,
                 in_channels=1,
                **kwargs ):
        super().__init__(**kwargs)
        self.model_name = 'resnet'
        self.pretrained=pretrained
        self.model_scale = model_scale
        self.add_dropout_layer= add_dropout_layer
        self.dropout_rate = dropout_rate

        print(self.model_scale)
        
        if self.model_scale == '18':
            self.model = models.resnet18(pretrained=self.pretrained)
        elif self.model_scale == '34':
            self.model = models.resnet34(pretrained=self.pretrained)
        elif self.model_scale == '50':
            self.model = models.resnet50(pretrained=self.pretrained)
        else:
            raise Exception('not implemented model scale: '+self.model_scale)
        
        # change the input channel to 1
        if in_channels == 1:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

        
        


        if self.add_dropout_layer:
            self.append_dropout(self.model)
            print(self.model)


    def forward(self, x, t):
        return self.model.forward(x)
    
    def append_dropout(self, model):
        '''
        add dropout layers
        '''
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.append_dropout(module)
            if isinstance(module, nn.ReLU):
                new = nn.Sequential(module, nn.Dropout2d(p=self.dropout_rate, inplace=False))
                setattr(model, name, new)

   
