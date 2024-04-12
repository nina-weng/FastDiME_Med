'''
train biassed classifier from biassed dataset
'''
import os
import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import Accuracy,AUROC
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from shortcut_detection.config import CLASSIFIERCONFIG

from models.resnet import ResNet
from models.cnn import SimpleCNN
from models.encoder import EncoderUNetModel
from train.config import TrainingConfig
from global_config import REPO_HOME_DIR
from train.dataloader_classifier import NoisyBaseDataModule
from train.utils import create_dir_if_nonexist, get_version_num


import pprint
import copy
import pickle
from tqdm import tqdm
import pandas as pd





def test_func(model, data_loader, n_classes,device):
    model.eval()
    logits = []
    preds = []
    targets = []
    img_name_rec = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, t, lab = batch['image'].to(device), batch['t'].to(device), batch['label'].to(device)
            img_name = batch['img_name']
            _ = 0 # placeholder for t which is useless here

            out = model(img,t)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)
            img_name_rec.extend(img_name)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)


        logits = torch.squeeze(logits)
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)


    return preds,targets,logits,img_name_rec







def main_train(
    seed: int,
    classifier_config: dict,
):
    classifier_config_str = pprint.pformat(classifier_config,sort_dicts=False)
    classifier_config_backup = copy.deepcopy(classifier_config)
    work_dir = REPO_HOME_DIR

    # save and return
    model_pth_list = {}

    # pop before the iteration over percentages
    model_name = classifier_config.pop('model_name')
    ds_name_prefix = classifier_config.pop('ds_name')
    percentages = classifier_config.pop('percentages')
    device = torch.device(classifier_config.pop('device_name'))
    img_size = classifier_config.pop('img_size')
    labels = classifier_config.pop('labels')
    batch_size = classifier_config.pop('batch_size')
    num_workers= classifier_config.pop('num_workers')
    img_manipulation = classifier_config.pop('img_manipulation')
    sd = classifier_config.pop('sd')
    timestep_range = classifier_config.pop('timestep_range')

    lr = classifier_config.pop('lr')

    model_scale = classifier_config.pop('model_scale')
    loss_func_type=classifier_config.pop('loss_func')
    weights = classifier_config.pop('weights')
    weight_decay = classifier_config.pop('weight_decay')
    add_dropout_layer = classifier_config.pop('add_dropout_layer')
    dropout_rate = classifier_config.pop('dropout_rate')
    epochs = classifier_config.pop('epochs')

    rest_config = classifier_config


    assert model_name in ['resnet','simplecnn','unet_encoder'], f'!Model not implemented: {model_name}'

    pl.seed_everything(seed, workers=True)

    for p in percentages:
        ds_name  = ds_name_prefix + '_{}'.format(str(p))
        print('#'*50)
        print(f'{ds_name=}')



    
        assert ds_name in ['CHEXPERT_CardioPM_100',
                        'CHEXPERT_CardioPM_75',
                        'CHEXPERT_CardioPM_50',
                        'NIH_Drain_100',
                        'NIH_Drain_75',
                        'NIH_Drain_50',
                        'ISIC_Ruler_100',
                        'ISIC_Ruler_75',
                        'ISIC_Ruler_50',
                        ]

    
        data_dir = os.path.join(work_dir, 'datasets',ds_name)
        models_dir = os.path.join(work_dir, 'records', 'biassed_classifier',ds_name,f'{model_name}-{seed}')

        create_dir_if_nonexist(models_dir)

        # print and record the config
        print(classifier_config_str)
        version_num = get_version_num(models_dir+'/lightning_logs/')
        with open(models_dir+f'/version_{version_num}_classifier_training_config.txt', "w") as f:
            f.write(classifier_config_str)
    
        
    


        data_module = NoisyBaseDataModule(
            data_dir = data_dir,
            img_size = img_size,
            labels = labels,
            batch_size = batch_size,
            num_workers = num_workers,
            img_manipulation = img_manipulation,
            sd = sd,
            timestep_range= timestep_range,
            unet = None,
            )

        n_classes = data_module.num_classes
   

        if model_name == 'resnet':
            model = ResNet(num_classes=n_classes,
                        lr=lr,
                        img_size=img_size[2],
                        pretrained=True,
                        model_scale=model_scale,
                        loss_func_type=loss_func_type,
                        weights = weights,
                        weight_decay =weight_decay,
                        add_dropout_layer = add_dropout_layer,
                        dropout_rate = dropout_rate,
                        in_channels = img_size[0],
                        ).to(device)
        elif model_name == 'simplecnn':
            model = SimpleCNN(num_classes=n_classes,
                            lr=lr,
                            img_size=img_size[2],
                            loss_func_type=loss_func_type,
                            weights = weights,
                            dropout_rate=dropout_rate,
                            weight_decay = weight_decay,
                            ).to(device)
        elif model_name == 'unet_encoder':
            image_size = img_size[2]
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            elif image_size == 224:
                channel_mult = (1, 2, 4, 4)
            else:
                raise ValueError(f"unsupported image size: {image_size}")

            attention_ds = []

            # defaulted parameters
            classifier_attention_resolutions="32,16,8"
            classifier_width=128
            classifier_depth=2
            classifier_use_fp16=False
            classifier_use_scale_shift_norm=True  # False
            classifier_resblock_updown=True  # False
            classifier_pool="attention"


            for res in classifier_attention_resolutions.split(","):
                attention_ds.append(image_size // int(res))

            model= EncoderUNetModel(
                in_channels=img_size[0],
                model_channels=classifier_width,
                out_channels=n_classes,
                num_res_blocks=classifier_depth,
                attention_resolutions=tuple(attention_ds),
                channel_mult=channel_mult,
                use_fp16=classifier_use_fp16,
                num_head_channels=64,
                use_scale_shift_norm=classifier_use_scale_shift_norm,
                resblock_updown=classifier_resblock_updown,
                pool=classifier_pool,
                # ori parameters
                num_classes=n_classes,
                lr=lr,
                img_size=TrainingConfig.IMG_SHAPE[2],
                loss_func_type=loss_func_type,
                weights = weights,
                dropout_rate=dropout_rate,
                weight_decay =weight_decay,
                ).to(device)

        loss_precesion = 3
        loss_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=3, filename='loss-{epoch}-{val_loss:.%sf}' % loss_precesion)
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")

        trainer = pl.Trainer(
            default_root_dir=models_dir,
            callbacks=[early_stop_callback],
            max_epochs=epochs,
            logger=CSVLogger(models_dir, flush_logs_every_n_steps=100),
            log_every_n_steps=10,
            **rest_config
        )
        
        trainer.fit(
            model,
            data_module
        )

        # record the config in .pkl
        with open(models_dir+f'/lightning_logs/version_{version_num}/config.pickle', "wb") as handle:
            pickle.dump(classifier_config_backup, handle)
            print('Successfully record the config in .pkl file.')


        # pytorch lightning
        trainer.test(ckpt_path="best",datamodule=data_module)

        # inference
        cols_names_classes = ['class_' + str(i) for i in range(0,n_classes)]
        cols_names_logits = ['logit_' + str(i) for i in range(0, n_classes)]
        cols_names_targets = ['target_' + str(i) for i in range(0, n_classes)]
        cols_names_imgnames = ['img_name']
        print('TESTING')
        preds_test, targets_test, logits_test, img_name_rec = test_func(model, 
                                                        data_module.test_dataloader(), 
                                                        n_classes,
                                                        device='cpu')
        # compute the metrics
        accu_func = Accuracy(task="binary", num_labels=n_classes)
        auroc_func = AUROC(task='binary',num_labels=n_classes, average='macro', thresholds=None)
        multi_accu = accu_func(preds_test, targets_test)
        multi_auroc = auroc_func(preds_test,targets_test.long())

        # save the predicts
        preds_test = preds_test.numpy()
        targets_test = targets_test.numpy()
        logits_test = logits_test.numpy()

        df_img_names = pd.DataFrame(data=img_name_rec, columns=cols_names_imgnames)
        df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
        df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
        df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
        df = pd.concat([df_img_names, df, df_logits, df_targets], axis=1)
        df.to_csv(models_dir+f'/lightning_logs/version_{version_num}/'+'predictions.test.csv',
                                index=False)
        
        
        # record the result in .txt
        with open(models_dir+f'/lightning_logs/version_{version_num}/result_test.txt', "w") as f:
            f.write(f'test set\naccu: {multi_accu=: .4f}\n{multi_auroc=: .4f}')
            print('Successfully record the text result in .txt file.')

        model_pth_list[str(p)] = models_dir+f'/lightning_logs/version_{version_num}'
    
    return model_pth_list

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    

    seed = 42

    
    main_train(
        seed,
        classifier_config=CLASSIFIERCONFIG
    )

