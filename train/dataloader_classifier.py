import pytorch_lightning as pl
import os

import os

from numpy import random

import torch
from torch import nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset,random_split
import torchvision.transforms as TF 
from PIL import Image
import numpy as np
from torchvision.utils import save_image


from models.diffusion import GuidedDiffusion
from train.dataloader import inverse_transform
from models.unet import UNet
from train.utils import get
from global_config import *




# add noise to images
def add_noise_diff(image: torch.Tensor,
                   t:int,
                   sd: GuidedDiffusion) -> torch.Tensor:
    '''
    Add  noise to the image using diffusion models wrt t=t
    '''
    timestep = t
    timestep_tensor = torch.as_tensor(timestep, dtype=torch.long)
    x_t,_ = sd.forward_diffusion(image,timestep_tensor)
    return x_t[0],timestep


def get_denoised_img(image:torch.Tensor,
                     t:int,
                     sd:GuidedDiffusion,
                     unet:UNet,
                     ):
    '''
    get denoised img from time= t
    '''
    timestep = t
    timestep_tensor = torch.as_tensor(timestep, dtype=torch.long)
    x_t,_ = sd.forward_diffusion(image,timestep_tensor)
    err = unet(x_t,timestep_tensor)
    err = err.detach() #detach from the grad graph, it is only a tensor img now
    x_0_blur = (x_t-err * get(sd.sqrt_one_minus_alpha_cumulative,timestep_tensor))/get(sd.sqrt_alpha_cumulative,timestep_tensor) # x_0_blur.shape = [BS, C, H, W]

    return x_0_blur[0], timestep


class NoisyBaseDataset(Dataset):
    def __init__(self, 
                 img_dir ,
                 labels= ['clean','noisy'], 
                 transform=None,
                 img_manipulation=None, # ['add_noise','denoise',None]
                 timestep_range= None,
                 sd = None,
                 unet = None, # unet model - trained

                 ):
        self.img_dir = img_dir
        self.transform = transform
        self.labels = labels
        self.num_classes = len(self.labels)
        self.img_manipulation = img_manipulation
        self.timestep_range = timestep_range
        self.sd = sd
        self.unet = unet

        self.img_file_names = []
        self.img_labels = []

        assert not((self.img_manipulation is not None) and (self.sd == None)),f'when {self.img_manipulation=} is not None, sd is needed.'
        assert not((self.img_manipulation == 'denoise') and (self.unet == None) ), f'when {self.img_manipulation=}, unet is needed.'

        for idx,each in enumerate(self.labels):
            sub_dir = os.path.join(self.img_dir,each)
            img_files = os.listdir(sub_dir)
            img_files.sort()

            for each_f in img_files:
                self.img_file_names.append(os.path.join(self.img_dir,each,each_f))
                self.img_labels.append(np.float32(idx))

        assert len(self.img_file_names) == len(self.img_labels)
        


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_file_names[idx]
        image = Image.open(img_path)

        img_name = img_path.split('/')[-1]
        
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        
        if self.img_manipulation == 'add_noise':
            # uniformly sample
            t = random.randint(1,self.timestep_range,size = 1) 
            image,_ = add_noise_diff(image,t,self.sd)
        elif self.img_manipulation == 'denoise':
            t = random.randint(1,self.timestep_range,size = 1) 
            denoised_image,_ = get_denoised_img(image,t,self.sd,self.unet)
            image = denoised_image
        elif self.img_manipulation is None:
            t = 0
        else:
            print(f'not implemented {self.img_manipulation=}')

        return  {'image': image, 't':t, 'label': label, 'img_name':img_name}



class NoisyBaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        img_size: tuple[int] = None,
        labels : list[str] = None,
        batch_size: int= None,
        num_workers: int =None,
        normalize: bool =True,
        train_val_split: list[int] =[0.8,0.2],
        data_split_seed: int =42,
        img_manipulation: str = None,# ['add_noise','denoise',None]
        sd: GuidedDiffusion = None,
        timestep_range: int = None,
        unet: UNet = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = 2 if num_workers is None else num_workers
        self.normalize = normalize
        self.train_val_split = train_val_split
        self.data_split_seed = data_split_seed

        self.sd = sd
        self.timestep_range= timestep_range
        self.img_manipulation = None if img_manipulation == 'None' else img_manipulation
        self.unet = unet

        if (self.img_manipulation is not None) and (self.timestep_range==None or self.sd == None):
            raise Exception(f'Diffusion model and timestep_range can not be None when applying adding noise.')
        
        self.normelization = TF.Normalize((0.,), (1.,)) if normalize else nn.Identity() # transforms.Normalize((0.1307,), (0.3081,)),
        
        self.num_classes = 1
        self.compose_list = self.get_compose_list()
        self.transform = TF.Compose(self.compose_list)
    
        self.trainval_set = NoisyBaseDataset(img_dir=self.data_dir+'/train/',labels = self.labels,transform=self.transform,
                                             img_manipulation=self.img_manipulation,
                                             timestep_range=self.timestep_range,
                                             sd=self.sd,
                                             unet = self.unet,
                                                )
        print('#train+val:',len(self.trainval_set))
        train_set, val_set = random_split(self.trainval_set, self.train_val_split, generator=torch.Generator().manual_seed(self.data_split_seed))
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = NoisyBaseDataset(img_dir=self.data_dir+'/test/',labels= self.labels,transform=self.transform,
                                         img_manipulation=self.img_manipulation,
                                             timestep_range=self.timestep_range,
                                             sd=self.sd,
                                             unet=self.unet,)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))
            
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
    
    def exam(self, 
                          idx_list: list[int],
                          save_dir: str)-> None:
        '''
        exam (the augmentaion) by saving some images from train set
        '''
        for idx in idx_list:
            img, t, label = self.train_set[idx]['image'],  self.train_set[idx]['t'][0],self.train_set[idx]['label']
            save_image(img, save_dir+f'/train_idx{idx}_lab{label}_t{t}.png')
            img = inverse_transform(img)
            img = img/255.0
            save_image(img, save_dir+f'/train_idx{idx}_lab{label}_t{t}_inv.png')
            

    def get_compose_list(self):
        '''
        get the compose function list from the parameters
        '''
        compose_list=[]

        if self.img_size[0] == 1:
            compose_list.append(TF.Grayscale(num_output_channels=self.img_size[0]))
        compose_list.extend(
            # every setting has the followings
            [TF.ToTensor(),
            TF.Resize((self.img_size[1],self.img_size[2]), 
                    interpolation=TF.InterpolationMode.BICUBIC, 
                    antialias=True),
            TF.Lambda(lambda t: (t * 2) - 1), # Scale between [-1, 1] 
            ]
        )
        return compose_list




    