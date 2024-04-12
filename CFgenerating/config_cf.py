from dataclasses import dataclass,asdict
import configparser
import pprint
from global_config import *



@dataclass
class CfConfig:
    
    
    total_timesteps: int
    cf_from_timestep: int
    ts_guided: bool # whether the classifier takes in the timestep
    rescale_t: bool # scale total time step
    rescale_to : int #

    # use_inpaint: bool 
    model_var:str  # variant of models, ['DiME','FastDiME','FastDiME-2','FastDiME-2+']
    start_tau: int
    dilation:int
    inpaint_th:float

    start_scale: int

    grad_loss_types: list[str]
    grad_loss_weights: list[float]
    
    ds_name: str

    checkpoint_dir: str

    best_model_path: str
    model_type:str

    plot_cf_process: bool # whether ploting the process of cf generation
    saveshow_gap: int

    confounder_filter:str


#-------------- hyperparameters ------------
model_var = 'FastDiME' # choose from ['DiME','FastDiME','FastDiME-2','FastDiME-2+','FastDiME-woM']
dataset = 'isic_common_testset' # choose from ['chexpert', 'nih', 'isic','chexpert_common_testset', 'nih_common_testset', 'isic_common_testset' ]


#-------------------------------------------



# ------------- pre defined parameters -----
# for basic use, do not need to change the default value of those parameters
# could be changed for ablation study

# pre-trained shortcut classifiers
classifier_ckpt_path = {'chexpert': REPO_HOME_DIR+ \
                        '/pretrained_models/classifier/CHEXPERT_CardioPM/unet_encoder.ckpt',
                        
                        'nih': REPO_HOME_DIR + \
                        '/pretrained_models/classifier/NIH_Drain/unet_encoder.ckpt',
                         
                        'isic': REPO_HOME_DIR + \
                            '/pretrained_models/classifier/ISIC_Ruler/unet_encoder.ckpt',
                         }


ds_name = 'CHEXPERT_CardioPM' if dataset == 'chexpert' else \
    'NIH_Drain' if dataset == 'nih' else \
    'ISIC_Ruler' if dataset == 'isic' else \
    'CHEXPERT_CardioPM_common_testset' if dataset == 'chexpert_common_testset' else \
    'NIH_Drain_common_testset' if dataset == 'nih_common_testset' else \
    'ISIC_Ruler_common_testset' 


grad_loss_types = ['Lc','Denoised_L1','L1']
if model_var == 'DiME':
    grad_loss_weights = [1.0,0.0,50.0]
else: grad_loss_weights = [1.0, 200.0, 0.0]


# diffusion model parameter (GuidedDiffusion)
start_scale = 100
total_timesteps = 1000
cf_from_timestep = 400

ts_guided = True

rescale_t = False
rescale_to = 400

# inpainting parameter
start_tau = 200 
dilation = 21 
inpaint_th = 0.15


confounder_filter = '0' # ['1','0','all'], where '1' stands for having shortcuts, and '0' stands for not having shortcuts


if ds_name.endswith('common_testset'):
    # for this part of experiment it's always with all samples
    confounder_filter = 'all'
else:
    confounder_filter = '1'



if rescale_t:
    start_scale = int(rescale_to*start_scale/total_timesteps)
    cf_from_timestep = int(cf_from_timestep*rescale_to/total_timesteps)
    start_tau = int(start_tau*rescale_to/total_timesteps)
    # need to change at the last time step
    total_timesteps = rescale_to
    
    



# ddpm model - UNet
if 'CHEXPERT' in ds_name:
    checkpoint_dir = REPO_HOME_DIR +'/pretrained_models/diffusion/OUT_CHEXPERT_CardioSplit/'
elif 'NIH' in ds_name:
    checkpoint_dir = REPO_HOME_DIR + '/pretrained_models/diffusion/OUT_NIH_All/'
elif 'ISIC' in ds_name:
    checkpoint_dir = REPO_HOME_DIR + '/pretrained_models/diffusion/ISIC_Ruler-train/'

ds_base = dataset.split('_')[0]
best_model_path = classifier_ckpt_path[ds_base]
model_type = best_model_path.split('/')[-1].split('.ckpt')[0] 



# ----VALIDATION-----
assert model_var in ['DiME','FastDiME','FastDiME-2','FastDiME-2+','FastDiME-woM'], NotImplementedError
assert dataset in ['chexpert', 'nih', 'isic','chexpert_common_testset', 'nih_common_testset', 'isic_common_testset'], NotImplementedError
assert model_type in ['resnet','simplecnn','unet_encoder','densenet'], f'{model_type=}'
# -------------------

# ---- config instance -----
cfconfig = CfConfig(# diff model hp
                    total_timesteps=total_timesteps,
                    cf_from_timestep=cf_from_timestep,
                    ts_guided = ts_guided,
                    rescale_t= rescale_t,
                    rescale_to=rescale_to,
                    # inpaint hp
                    model_var = model_var,
                    start_tau = start_tau,
                    dilation= dilation,
                    inpaint_th= inpaint_th,
                    # gradient related: 
                    # gradient decend
                    start_scale=start_scale,
                    grad_loss_types = grad_loss_types,
                    grad_loss_weights = grad_loss_weights,
                    # test data load from
                    ds_name = ds_name,
                    # load diff model (Unet)
                    checkpoint_dir = checkpoint_dir,
                    # load classifier (ResNet)
                    best_model_path = best_model_path,
                    model_type = model_type,
                    # plotting
                    plot_cf_process = False,
                    saveshow_gap=20,
                    # confounder_filter
                    confounder_filter = confounder_filter,
                    )


# -------------------