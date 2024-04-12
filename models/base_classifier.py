'''
A simple implementation of CNN for classification
'''
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics import Accuracy,AUROC
from torchmetrics.classification import MultilabelAUROC

class BasePLClassifier(pl.LightningModule):
    def __init__(self, num_classes,
                 lr, 
                 img_size = 32, 
                 loss_func_type='BCE',
                 weights = None,
                 dropout_rate=0.5,
                 weight_decay = 0.01):
        super().__init__()
        self.model_name = None
        self.num_classes = num_classes
        self.loss_func_type= loss_func_type
        self.weights = weights
        self.lr=lr
        self.img_size = img_size
        self.weight_decay = weight_decay

        if self.loss_func_type == 'WeightedBCE':
            assert (self.weights is not None), 'weights can not be None when using WeightedBCE'
        assert self.img_size%4 == 0, 'imgsize can not be devided by 4'
        self.dropout_rate = dropout_rate

        if self.loss_func_type == 'BCE':
            # self.loss_func = F.binary_cross_entropy
            self.loss_func = nn.BCELoss()
        elif self.loss_func_type == 'WeightedBCE':
            pos_weight = torch.tensor(self.weights)
            # Define the loss function with weighted binary cross-entropy
            self.loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            raise Exception('Not implemented loss function type : {}'.format(self.loss_func_type))

        if self.num_classes == 1:
            self.accu_func = Accuracy(task="binary", num_labels=num_classes)
            self.auroc_func = AUROC(task='binary',num_labels=num_classes, average='macro', thresholds=None)
        elif self.num_classes >1:
            self.accu_func= Accuracy(task="multilabel", num_labels=num_classes)
            self.auroc_func = MultilabelAUROC(num_labels=num_classes,average='macro', thresholds=None)


    def forward(self, x,t):
        return

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=self.lr,weight_decay = self.weight_decay)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['t'], batch['label']

    def process_batch(self, batch):
        img, t, lab = self.unpack_batch(batch)
        out = self.forward(img,t)
        prob = torch.sigmoid(out)
        prob = prob.squeeze()
        loss = self.loss_func(prob, lab)

        multi_accu = self.accu_func(prob, lab)
        multi_auroc = self.auroc_func(prob,lab.long())
        return loss,multi_accu,multi_auroc

    def training_step(self, batch,batch_idx):
        loss,multi_accu,multi_auroc = self.process_batch(batch)
        self.log('train_loss', loss)
        self.log('train_accu', multi_accu)
        self.log('train_auroc', multi_auroc)
        return loss

    def validation_step(self, batch,batch_idx):
        loss, multi_accu, multi_auroc = self.process_batch(batch)
        self.log('val_loss', loss)
        self.log('val_accu', multi_accu)
        self.log('val_auroc', multi_auroc)

    def test_step(self, batch,batch_idx):
        loss,multi_accu,multi_auroc = self.process_batch(batch)
        self.log('test_loss', loss)
        self.log('test_accu', multi_accu)
        self.log('test_auroc', multi_auroc)