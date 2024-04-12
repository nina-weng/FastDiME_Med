import cv2
import torch
import os

import pprint
from dataclasses import asdict
from torchvision.utils import make_grid

from train.dataloader import inverse_transform
from train.transform_config import CHE_transforms_che,ISIC_transforms_isic
from global_config import REPO_HOME_DIR
from CFgenerating.config_cf import CfConfig
import pickle
from PIL import Image
import torch.nn.functional as F 
import numpy as np

def save_tensor2img(img: torch.Tensor,
                    savepath: str):
    '''
    save from torch tensor to images
    '''
    img4save = inverse_transform(img).cpu().permute(1,2,0).numpy()
    cv2.imwrite(savepath, img4save)
    return

def get_save_path_with_version(path_dir:str,
                               with_version:bool = True):
    '''
    prepare for the save dir with version numbers
    '''
    if not with_version:
        this_save_path = path_dir + '/'
    else:
        if os.path.isdir(path_dir): # if exists
            # Get all folders numbers in the root_log_dir
            # print(os.listdir(path_dir))
            folder_numbers = [int(folder.replace("version_", "")) for folder in os.listdir(path_dir)]
            if len(folder_numbers) == 0:
                version_name = 'version_0'
            else:
                # Find the latest version number present in the log_dir
                last_version_number = max(folder_numbers)
                # New version name
                version_name = f"version_{last_version_number + 1}"

        else:
            version_name = 'version_0'


        this_save_path = os.path.join(path_dir,version_name)+'/'
    os.makedirs(this_save_path, exist_ok=False)
    return this_save_path


def save_config(config: CfConfig,
                save_path: str):
    config_str = pprint.pformat(asdict(config),sort_dicts=False)
    with open(save_path, "w") as f:
        f.write(config_str)
    return



def print_config(config: CfConfig):
    pprint.pprint(asdict(config), sort_dicts=False)
    return

def added_config(save_path: str,
                add_info: str):
    with open(save_path, "a") as f:
        f.write('\n'+add_info+'\n')

    return


def save_imgs_tensor_batch(x: torch.Tensor,
                           save_path: str,
                           nrow:int = 8):
    '''
    Save a batch of tensor images to one plot
    Parmeter:
    x: shape: [BS, C, H, W]
    save_path: the save path, should be ended with the plot name rather than folder
    '''

    assert len(x.shape) == 4, f'Incorrect Shape: {x.shape=}'

    grid = make_grid(inverse_transform(x),nrow=nrow, pad_value=255.0).to("cpu")
    ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
    cv2.imwrite(save_path, ndarr)

    return


def get_classifier_config(model_path:str,
                          )-> dict:
    '''
    get the config of the classifier from pickle

    Return:
    - config_dict
    '''
    file_name = model_path.split('/')[-1].split('.ckpt')[0]
    upper_folder = '/'.join(model_path.split('/')[:-1])

    try:
        with open(upper_folder+f'/{file_name}_config.pickle', 'rb') as handle:
            config_dict = pickle.load(handle)
    except:
        try:
            version_dir = '/'.join(model_path.split('/')[:-2])
            with open(version_dir+'/config.pickle', 'rb') as handle:
                config_dict = pickle.load(handle)
        except:
            print('Unkown pickle file position')

    return config_dict


def get_checkpoint_dir(diff_train_from_ds:str,
                       version_num:int):
    dir_name = diff_train_from_ds if diff_train_from_ds.startswith('OUT') else diff_train_from_ds+'-train'
    return REPO_HOME_DIR+'/records/diffusion/'+dir_name+'/Logs_Checkpoints/checkpoints/version_{}/'.format(version_num)


def get_dict(names,
             weights,):
    assert len(names) == len(weights)
    dict_ = {}
    for idx in range(len(names)):
        dict_[names[idx]] = weights[idx]
    return dict_

def save_lab_target(sample_lab:torch.Tensor,
                    sample_traget:torch.Tensor,
                save_path: str):
    str_ = 'lab:{}\ntarget:{}\n'.format(sample_lab.cpu().numpy(),sample_traget.cpu().numpy())
    with open(save_path, "w") as f:
        f.write(str_)
    return


def get_one_cf_img(img_path:str,
                   device:str):
    '''
    read one image to tensor with the shape of [1,C,W,H]
    need to have the same transform as the dataloader in //dataloader.py
    '''
    image = Image.open(img_path)
    ds_name = img_path.split('/')[-4]
    if ('CHE' in ds_name) or ('NIH' in ds_name):
        transform = CHE_transforms_che
    elif 'ISIC' in ds_name:
        transform = ISIC_transforms_isic
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    assert len(image.shape) == 4
    return image


def get_fixed_mask(x1,x2,
                   dilation,
                   inpaint_th,
                   ):
    '''
    get the bool mask from x1 and x2
    x1, and x2 shape: [BS, C, W, H]
    '''
    assert (dilation %2) ==1, 'dilation must be an odd number'

    x1 = (x1+1)/2
    x2 = (x2+1)/2

    mask = (x1 - x2).abs().sum(dim=1, keepdim=True)
    mask = mask / mask.view(mask.size(0), -1).max(dim=1)[0].view(-1,1,1,1)
    dil_mask = F.max_pool2d(mask,
                            dilation, stride=1,
                            padding = (dilation-1)//2
    )
    boolmask = (dil_mask < inpaint_th).float()
    return boolmask

def save_fixed_mask(mask,
                    save_pth):
    # save it as npy
    mask_np = mask.cpu().detach().numpy()
    np.save(save_pth+'mask.npy', mask_np)

    # save it as image
    if len(mask.shape) == 4:
        mask = mask[0]

    img4save = inverse_transform(mask).cpu().permute(1,2,0).numpy()

    cv2.imwrite(save_pth+'mask.png', img4save)
    return


def randomized_fix_mask(img_size,mask_size):
    '''
    Parameters: 
    - img_size: (img_height, img_width)
    - mask_size: (mask_height, mask_width)
    #TODO the reproducible experiments (from one seed to a series of seed? sth like that)

    Return:
    - output_mask: (1, img_height, img_width), contains only 0 and 1
    '''


    mask_height,mask_width = mask_size[0],mask_size[1]
    image_height,image_width = img_size[0],img_size[1]
 
    assert mask_height <= image_height and mask_width <= image_width, "Mask size must be smaller than image size."

    # Calculate maximum starting points for the mask
    max_x_start = image_width - mask_width
    max_y_start = image_height - mask_height

    # Generate random starting points
    x_start = np.random.randint(0, max_x_start + 1)
    y_start = np.random.randint(0, max_y_start + 1)

    # Initialize a zero matrix representing the image
    output_mask = np.ones((image_height, image_width))

    # Create a binary mask with ones within the specified dimensions
    mask_ones = np.zeros((mask_height, mask_width))

    # Place the binary mask in the image at the random starting position
    output_mask[y_start:y_start+mask_height, x_start:x_start+mask_width] = mask_ones

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_mask = torch.from_numpy(output_mask).float().to(device)

    output_mask = output_mask.unsqueeze(0)

    return output_mask


def get_prediction_from_classifier(classifier,img,device):
    classifier.eval()
    t_tensor = torch.ones(1, dtype=torch.long, device=device) * 0
    out_logits = classifier.forward(img,t_tensor)
    prob = torch.sigmoid(out_logits)
    return prob


def save_probs(txt_file:str,
               prob_x_0:torch.tensor,
               prob_x_cf:torch.tensor):
    str_ = 'Prob of having shortcut:\nprob_x_0:{}\nprob_x_cf:{}\n'.format(prob_x_0.cpu().detach().numpy(),prob_x_cf.cpu().detach().numpy())
    with open(txt_file, "a") as f:
        f.write(str_)
    return