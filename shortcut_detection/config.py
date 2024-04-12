from dataclasses import dataclass,asdict,field
import os
import torch
from global_config import *



# ----------- parameters that need to set up -------
which_ds = 'isic'
isTrain = True # train the classifier, if False, use the trained classifier by adding the model path in model_path_dict
isExam = True # examine the performance
# --------------------------------------------------



ClassifierBasePATH = REPO_HOME_DIR + '/records/biassed_classifier/'


ds_name = 'CHEXPERT_CardioPM' if which_ds == 'chexpert' else\
        'NIH_Drain' if which_ds == 'nih' else \
        'ISIC_Ruler' if which_ds == 'isic' else 'none'
model_type = 'resnet'  # ['resnet','simplecnn','unet_encoder']


percentages = [100,75,50]
if 'CHE' in ds_name:
    d_labels = ['non-Cardiomegaly','Cardiomegaly']
    img_size = (1,224,224)
    model_scale = '18'
    num_epochs = 20


elif 'NIH' in ds_name:
    d_labels = ['non-Pneumothorax','Pneumothorax']
    img_size = (1,224,224)
    model_scale = '50'
    num_epochs = 50

elif 'ISIC' in ds_name:
    d_labels = ['non-Mal','Mal']
    img_size = (3,224,224)
    model_scale = '50'
    num_epochs = 50


CLASSIFIERCONFIG=dict(
            # model
            model_name=model_type,  # simplecnn, resnet, unet_encoder
                # for shortcut parts, the ratio is always 1:1 for the disease: non-disease, therefore we only use BCE
            loss_func = 'BCE', # WeightedBCE, BCE
            weights = 4.63, #[0.85/0.15],
            weight_decay = 0.01,
            model_scale = model_scale,
            add_dropout_layer = False,
            dropout_rate = 0.1,

            # schedule noisy or denoised img, or original img
                # not needed for this part of experiments - shortcuts
            img_manipulation = None ,  # ['add_noise','denoise',None]
            sd = None,
            timestep_range= None,   

            # dataset
            ds_name= ds_name,
            percentages = percentages,
            img_size = img_size,
            labels = d_labels, # ['clean','noisy']

            # training
            epochs=num_epochs,
            batch_size=8,
            lr=1e-6,

            # hardware
            accelerator='gpu',
            strategy='ddp',
            devices=torch.cuda.device_count(),
            num_nodes=1,
            num_workers=min(16, os.cpu_count()),
            device_name='cuda',  
        )



@dataclass
class scdConfig:
    # classifier
    ds_name:str
    img_size:tuple[int]
    
    model_type:str

    # load test set
    cf_store_path:str 

    # disease and sc info
    img_index_col_name:str
    disease_col_name:str
    sc_col_name:str

    isTrain:bool
    isExam:bool


    percentages:list[int] = field(default_factory=list)

    model_path_dict:dict = field(default_factory= lambda: {
        '100': None,
        '75': None,
        '50': None,
    })




model_type = model_type
if ds_name == 'CHEXPERT_CardioPM':
    # your cf generation results path, e.g. /records/cf_generating/DATASET_common_testset_MODELVAR/version_x/
    cf_store_path = '' # your path
    
    img_index_col_name = 'img_name'
    disease_col_name = 'Cardiomegaly'
    sc_col_name = 'PM'

    if model_type == 'resnet': 
        model_path_dict ={'100': '', # your path
                            '75': '', # your path
                            '50': '',# your path
                            }



elif ds_name == 'NIH_Drain':
    cf_store_path = '' # your path

    
    img_index_col_name = 'img_name'
    disease_col_name = 'Pneumothorax'
    sc_col_name = 'Drain'

    if model_type == 'resnet':
        model_path_dict ={'100': '', # your path
                            '75': '', # your path
                            '50': '',# your path
                            }
    else:
        model_path_dict = None

elif ds_name == 'ISIC_Ruler':
    cf_store_path = '' # your path


    img_index_col_name = 'img_name'
    disease_col_name = 'Mal'
    sc_col_name = 'Ruler'

    if model_type == 'resnet': 
        model_path_dict ={'100': '', # your path
                            '75': '', # your path
                            '50': '',# your path
                            }

    else:
        model_path_dict = None



assert model_type in ['resnet','simplecnn','unet_encoder']






scdconfig = scdConfig(ds_name = ds_name,
                      img_size=img_size,
                      model_path_dict = model_path_dict,
                      model_type=model_type,
                      cf_store_path = cf_store_path,
                      img_index_col_name=img_index_col_name,
                      disease_col_name=disease_col_name,
                      sc_col_name=sc_col_name,
                      percentages=percentages,
                      isTrain=isTrain,
                      isExam=isExam,
                    )


