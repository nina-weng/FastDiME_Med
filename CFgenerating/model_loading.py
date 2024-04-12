'''
functions for loading the pre-trained models
'''
import pickle
import torch 
import os

from models.cnn import SimpleCNN
from models.densenet import DenseNet
from models.encoder import EncoderUNetModel
from models.unet import UNet
from models.resnet import ResNet
from train.dataloader import get_dataloader,get_dataset,inverse_transform
from CFgenerating.utils import *


def load_diffusion_model(checkpoint_dir:str):
    '''
    load the diffusion model trained on combined dataset
    Parameters:
    - checkpoint_dir: the dir path of the pre-trained diffusion model

    Return:
    - model: the diffusion model (Unet)
    - config_set: configs of the diffusion model
    '''

    # get the config
    with open(checkpoint_dir+'/config.pickle', 'rb') as handle:
        config_set = pickle.load(handle)

    
    print('^'*30+f'\nLoading trained DF model from {checkpoint_dir}')
    model = UNet(
            input_channels          = config_set['TrainingConfig']['IMG_SHAPE'][0],
            output_channels         = config_set['TrainingConfig']['IMG_SHAPE'][0],
            base_channels           = config_set['ModelConfig']['BASE_CH'],
            base_channels_multiples = config_set['ModelConfig']['BASE_CH_MULT'],
            apply_attention         = config_set['ModelConfig']['APPLY_ATTENTION'],
            attention_type          = config_set['ModelConfig']['ATTENTION_TYPE'],
            dropout_rate            = config_set['ModelConfig']['DROPOUT_RATE'],
            time_multiple           = config_set['ModelConfig']['TIME_EMB_MULT'],
        )
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "ckpt.tar"), map_location='cpu')['model'])

    model.to(config_set['BaseConfig']['DEVICE'])
    
    model.eval()

    print('^'*30+f'\nSuccessful loaded model {checkpoint_dir}.')
    return model, config_set


def load_test_set(ds_name:str,
                  device:str,):
    '''
    load the test set of the chosen dataset for generating counterfactuals
    Parameters:
    - ds_name: the dataset that used for generating counterfactuals 
    - device: device
    '''
    # load the test set
    print('^'*30+ '\nLoad data from test set')

    loader = get_dataloader(
            dataset_name=ds_name+'-test',
            batch_size=128,
            device=device,
            shuffle = False
            )
    print('^'*30+'\nSuccessful loaded the test set.')
    return loader


def load_classifier(model_path: str,
                    model_type: str,
                    device: str ):
    config_set = get_classifier_config(model_path)
    print(f'{config_set=}')

    if model_type == 'resnet':
        try:
            model_scale = config_set['model_scale']
        except:
            model_scale = '18'

        try:
            in_channels = config_set['img_size'][0]
        except:
            in_channels = 1

        classifier = ResNet.load_from_checkpoint(model_path,
                                        num_classes=1,
                                        lr=config_set['lr'],
                                        pretrained=True,
                                        model_scale=model_scale,
                                        loss_func_type = config_set['loss_func'], # WeightedBCE, BCE
                                        weights = config_set['weights'],
                                        in_channels = in_channels,
                                        ).eval().to(device)
    elif model_type == 'densenet':
        try:
            model_scale = config_set['model_scale']
        except:
            model_scale = '121'
        classifier = DenseNet.load_from_checkpoint(model_path,
                                        num_classes=1,
                                        lr=config_set['lr'],
                                        pretrained=True,
                                        model_scale=model_scale,
                                        loss_func_type = config_set['loss_func'], # WeightedBCE, BCE
                                        weights = config_set['weights'],
                                        ).eval().to(device)
    elif model_type == 'simplecnn':
        classifier = SimpleCNN.load_from_checkpoint(model_path,
                                                    num_classes=1,
                                                    lr=config_set['lr'],
                               ).eval().to(device)
    elif model_type == 'unet_encoder':
        classifier_attention_resolutions="32,16,8"
        classifier_width=128
        classifier_depth=2
        classifier_use_fp16=False
        classifier_use_scale_shift_norm=True  # False
        classifier_resblock_updown=True  # False
        classifier_pool="attention"

        attention_ds = []
        for res in classifier_attention_resolutions.split(","):
            attention_ds.append(config_set['img_size'][2] // int(res))
        channel_mult = (1, 2, 4, 4)

        classifier = EncoderUNetModel.load_from_checkpoint(model_path,
                                                    in_channels=config_set['img_size'][0],
                                                    model_channels=classifier_width,
                                                    out_channels=1,
                                                    num_res_blocks=classifier_depth,
                                                    attention_resolutions=tuple(attention_ds),
                                                    channel_mult=channel_mult,
                                                    use_fp16=classifier_use_fp16,
                                                    num_head_channels=64,
                                                    use_scale_shift_norm=classifier_use_scale_shift_norm,
                                                    resblock_updown=classifier_resblock_updown,
                                                    pool=classifier_pool,
                                                    # ori parameters
                                                    num_classes=1,
                                                    lr=config_set['lr'],
                                                    img_size=config_set['img_size'][2],
                                                    loss_func_type= config_set['loss_func'],
                                                    weights = config_set['weights'],
                                                    dropout_rate=config_set['dropout_rate'],
                                                    weight_decay = config_set['weight_decay'],
                               ).eval().to(device)
    return classifier


