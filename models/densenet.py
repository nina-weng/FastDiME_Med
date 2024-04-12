
from torchvision import models
import torch.nn as nn
from .base_classifier import BasePLClassifier


'''
adopted from bias causes project
for current use, only need the situation where num_classes=1
'''


class DenseNet(BasePLClassifier):
    def __init__(self, 
                 pretrained,
                 model_scale='121',
                 add_dropout_layer=False,
                 dropout_rate=0.5,
                **kwargs ):
        super().__init__(**kwargs)
        self.model_name = 'densenet'
        self.pretrained=pretrained
        self.model_scale = model_scale
        self.add_dropout_layer= add_dropout_layer
        self.dropout_rate = dropout_rate

        print(self.model_scale)
        
        if self.model_scale == '121':
            self.model = models.densenet121(pretrained=self.pretrained)
        else:
            raise Exception('not implemented model scale: '+self.model_scale)
        

        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)

        
        


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

   
