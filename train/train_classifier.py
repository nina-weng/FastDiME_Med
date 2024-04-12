'''
adopted from niels'
'''
import os
import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import sys
sys.path.append('../FastDiME_Med/')
print(sys.path)

from models.resnet import ResNet
from models.cnn import SimpleCNN
from models.diffusion import GuidedDiffusion
from models.encoder import EncoderUNetModel
from models.densenet import DenseNet
from train.config import TrainingConfig
from global_config import REPO_HOME_DIR
from train.dataloader_classifier import NoisyBaseDataModule
from train.utils import create_dir_if_nonexist, get_version_num
import pprint
import copy
import pickle
from generate_cf import load_diffusion_model
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sd = GuidedDiffusion(
            num_diffusion_timesteps = 1000,
            img_shape               = TrainingConfig.IMG_SHAPE,
            device                  = 'cpu',
        )


CLASSIFIERCONFIG=dict(
            # model
            model_name='unet_encoder',  # simplecnn, resnet, unet_encoder
            model_scale='50', # only for resnet, densenet
            loss_func = 'WeightedBCE', # WeightedBCE, BCE
            weights = [4.8], #[0.85/0.15] - ISIC_Ruler + CHE_PM, for Drain, it is balanced
            weight_decay = 0.05,
            # schedule noisy or denoised img, or original img
            img_manipulation = 'None' ,  # ['add_noise','denoise',None]
            sd = sd,
            timestep_range= 400,   
            # if needed unet (for denoised img)
            diff_train_from_ds = 'ISIC_Ruler',
            diff_version_num = 50,
            # dataset
            ds_name= 'ISIC_Ruler',
            img_size = (3,224,224),
            labels =['non-Ruler','Ruler'],
            # training
            epochs=50,
            batch_size=16,
            lr=1e-6,
            dropout_rate = 0.5,
            # hardware
            accelerator='gpu',
            strategy='ddp',
            devices=torch.cuda.device_count(),
            num_nodes=1,
            num_workers=min(16, os.cpu_count()),
            device_name='cuda',
                   
        )




def main(
    seed: int,
    classifier_config: dict,
):
    classifier_config_str = pprint.pformat(classifier_config,sort_dicts=False)
    classifier_config_backup = copy.deepcopy(classifier_config)


    model_name = classifier_config.pop('model_name')
    assert model_name in ['resnet','simplecnn','unet_encoder', 'densenet'], f'!Model not implemented: {model_name}'
    ds_name = classifier_config.pop('ds_name')
    assert ds_name in ['MNIST_combined_random_clean_gaussbox_labelled',
                       'MNIST_combined_random_clean_whitebox_labelled',
                       'CHEXPERT_CardioPM',
                       'NIH_Drain',
                       'ISIC_Ruler',
                       'ISIC_Ruler_onlyNV']

    work_dir = REPO_HOME_DIR
    data_dir = os.path.join(work_dir, 'datasets',ds_name)
    models_dir = os.path.join(work_dir, 'records', 'classifier',ds_name,f'{model_name}-{seed}')

    create_dir_if_nonexist(models_dir)

    # print and record the config
    print(classifier_config_str)
    version_num = get_version_num(models_dir+'/lightning_logs/')
    with open(models_dir+f'/version_{version_num}_classifier_training_config.txt', "w") as f:
        f.write(classifier_config_str)
    
    device = torch.device(classifier_config.pop('device_name'))
    
    pl.seed_everything(seed, workers=True)

    try:
        unet, _ = load_diffusion_model(diff_train_from_ds=classifier_config.pop('diff_train_from_ds'),
                                    version_num=classifier_config.pop('diff_version_num'))
        unet.to('cpu') # only for loading data, so we dont use cuda to avoid multi cuda process
        # https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
        # Don't call cuda() inside Dataset's __getitem__() method
    except:
        unet=None


    data_module = NoisyBaseDataModule(
        data_dir = data_dir,
        img_size = classifier_config.pop('img_size'),
        labels = classifier_config.pop('labels'),
        batch_size = classifier_config.pop('batch_size'),
        num_workers = classifier_config.pop('num_workers'),
        img_manipulation = classifier_config.pop('img_manipulation'),
        sd = classifier_config.pop('sd'),
        timestep_range= classifier_config.pop('timestep_range'),
        unet = unet,
    )

    n_classes = data_module.num_classes
    lr = classifier_config.pop('lr')
    model_scale= classifier_config.pop('model_scale')

    if model_name == 'resnet':
        model = ResNet(num_classes=n_classes,
                       lr=lr,
                       img_size=data_module.img_size[2],
                       pretrained=True,
                       model_scale=model_scale,
                       loss_func_type=classifier_config.pop('loss_func'),
                       weights = classifier_config.pop('weights'),
                       dropout_rate=classifier_config.pop('dropout_rate'),
                       weight_decay = classifier_config.pop('weight_decay'),
                       in_channels = data_module.img_size[0],
                       ).to(device)
    elif model_name == 'simplecnn':
        model = SimpleCNN(num_classes=n_classes,
                          lr=lr,
                          img_size=data_module.img_size[2],
                          loss_func_type=classifier_config.pop('loss_func'),
                          weights = classifier_config.pop('weights'),
                          dropout_rate=classifier_config.pop('dropout_rate'),
                          weight_decay = classifier_config.pop('weight_decay'),
                          ).to(device)
    elif model_name == 'densenet':
        model = DenseNet(num_classes=n_classes,
                          lr=lr,
                          pretrained=True,
                          model_scale=model_scale,
                          img_size=data_module.img_size[2],
                          loss_func_type=classifier_config.pop('loss_func'),
                          weights = classifier_config.pop('weights'),
                          dropout_rate=classifier_config.pop('dropout_rate'),
                          weight_decay = classifier_config.pop('weight_decay'),
                          ).to(device)
    elif model_name == 'unet_encoder':
        image_size = data_module.img_size[2]
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
            in_channels=data_module.img_size[0],
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
            img_size=data_module.img_size[2],
            loss_func_type=classifier_config.pop('loss_func'),
            weights = classifier_config.pop('weights'),
            dropout_rate=classifier_config.pop('dropout_rate'),
            weight_decay = classifier_config.pop('weight_decay'),
            ).to(device)

    loss_precesion = 3
    loss_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=3, filename='loss-{epoch}-{val_loss:.%sf}' % loss_precesion)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")
    
    trainer = pl.Trainer(
        default_root_dir=models_dir,
        callbacks=[early_stop_callback],
        max_epochs=classifier_config.pop('epochs'),
        logger=CSVLogger(models_dir, flush_logs_every_n_steps=100),
        log_every_n_steps=10,
        **classifier_config
    )
    
    trainer.fit(
        model,
        data_module
    )

    # record the config in .pkl
    with open(models_dir+f'/lightning_logs/version_{version_num}/config.pickle', "wb") as handle:
        pickle.dump(classifier_config_backup, handle)
        print('Successfully record the config in .pkl file.')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    

    seed = 42

    
    main(
        seed,
        classifier_config=CLASSIFIERCONFIG
    )

