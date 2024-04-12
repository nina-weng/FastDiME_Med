import sys
from train.utils import get_default_device

from dataclasses import dataclass,asdict
import os
import pprint
import pickle

from global_config import *




@dataclass
class BaseConfig:
    repo_home_dir: str = REPO_HOME_DIR
    DEVICE : str = get_default_device()
    ds_name : str = 'OUT_NIH_All' 
    
    # -------- ds_name could be chose from ---------
    # CHEXPERT_CardioPM
    # NIH_Drain
    # ISIC_Ruler

    # ---- you can also train the diffusion model with larger datasets outside the REPO_HOME_DIR
    # OUT_xxx
    # OUT_yyy

    split: str ='train' # 'train', 'test'
    record_dir: str = './records/diffusion/'
    DATASET: str
    if (ds_name.startswith('OUT')):
        DATASET: str = ds_name
    else: DATASET: str = ds_name+'-'+split  #  "MNIST"
    

    # For logging inferece images and saving checkpoints.
    root_log_dir: str = os.path.join(record_dir,DATASET,"Logs_Checkpoints", "Inference")
    root_checkpoint_dir:str = os.path.join(record_dir,DATASET,"Logs_Checkpoints", "checkpoints")
    root_prevs_dir:str = os.path.join(record_dir,DATASET,'PreVs')


    # Current log and checkpoint directory.
    log_dir:str = "version_0"
    checkpoint_dir:str = "version_0"



@dataclass
class TrainingConfig:
    TIMESTEPS: int = 1000 # Define number of diffusion timesteps
    IMG_SHAPE: tuple[int]
    BATCH_SIZE: int
    SAMPLE_NUM: int
    NUM_EPOCHS: int

    if ('CHEXPERT' in BaseConfig.DATASET) or ('NIH' in BaseConfig.DATASET) :
        IMG_SHAPE = (1,224,224)
        BATCH_SIZE = 16
        SAMPLE_NUM = min(BATCH_SIZE*4,32)
        NUM_EPOCHS = 500

    elif 'ISIC' in BaseConfig.DATASET:
        IMG_SHAPE = (3,224,224)
        BATCH_SIZE = 16
        SAMPLE_NUM = min(BATCH_SIZE*4,32)
        
        if IMG_SHAPE[2] == 224:
            NUM_EPOCHS = 500
        elif IMG_SHAPE[2] == 64:
            NUM_EPOCHS = 200
        else:
            print(f'{NUM_EPOCHS=}')
    else:
        raise Exception(f'Not implemented {BaseConfig.DATASET=}')
    LR : float = 2e-4
    NUM_WORKERS : int = 2



@dataclass
class ModelConfig:
    BASE_CH: int = 64  # 64, 128, 256, 256
    BASE_CH_MULT: tuple[int] = (1, 2, 4, 4) # 32, 16, 8, 8 
    APPLY_ATTENTION: tuple[bool] = (False, False, False, False)
    ATTENTION_TYPE: str= 'ori_att' # ['ori_att','simple_att',] # not using anymore
    DROPOUT_RATE : float = 0.1
    TIME_EMB_MULT : int = 4 # 128



def save_config_train_diffusion(tc: TrainingConfig,
                                mc: ModelConfig,
                                bc: BaseConfig,
                                save_dir: str):
    '''
    save the config of diffusion model trainingS
    '''
    # save as dict
    config_dict = {}
    config_dict['BaseConfig'] = asdict(bc)
    config_dict['ModelConfig'] = asdict(mc)
    config_dict['TrainingConfig'] = asdict(tc)

    with open(save_dir+'/config.pickle', 'wb') as handle:
        pickle.dump(config_dict, handle)

    # save to txt
    config_str = pprint.pformat(config_dict,sort_dicts=False)
    with open(save_dir+'/config.txt', "w") as f:
        f.write(config_str)

    return

