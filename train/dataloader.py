import torchvision.transforms as TF
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
from torchvision.datasets import ImageFolder
import os
from PIL import Image

from train.config import TrainingConfig,BaseConfig
from train.utils import DeviceDataLoader
from global_config import REPO_HOME_DIR,DATASET_DIR
from train.transform_config import CHE_transforms_che,ISIC_transforms_isic


img_size = (TrainingConfig.IMG_SHAPE[1],TrainingConfig.IMG_SHAPE[2])




class CustomImageFolder(ImageFolder):
    '''
    custommed dataloader, to make sure that the classes index are assigned correctly aligned with NoisyBaseDataModule
    '''
    def __init__(self, root, transform, ds_name=None):
        self.ds_name = ds_name
        super(CustomImageFolder, self).__init__(root, transform)
        print(f'{self.ds_name=}')

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Override this method to load from setting file instead of scanning directory
        """
        classes = list(self.classes_to_idx_func().keys())
        classes_to_idx = self.classes_to_idx_func()
        return classes, classes_to_idx
    
    def classes_to_idx_func(self):
        if (self.ds_name == 'CHEXPERT_CardioPM') or \
            (self.ds_name == 'CHEXPERT_CardioPM_all') or \
                (self.ds_name == 'CHEXPERT_CardioPM_control') or\
                (self.ds_name == 'CHEXPERT_CardioPM_common_testset') :
            dict_ = {'non-PM':0,'PM':1}
        elif (self.ds_name == 'CHEXPERT_CardioPM_100') or (self.ds_name == 'CHEXPERT_CardioPM_75') or (self.ds_name == 'CHEXPERT_CardioPM_50'):
            dict_ = {'non-Cardiomegaly':0,'Cardiomegaly':1}
        elif (self.ds_name == 'NIH_Drain') or (self.ds_name == 'NIH_Drain_common_testset') :
            dict_ = {'non-Drain':0,'Drain':1}
        elif (self.ds_name == 'ISIC_Ruler') or (self.ds_name == 'ISIC_Ruler_common_testset'):
            dict_ = {'non-Ruler':0,'Ruler':1}
        elif 'MNIST' in self.ds_name:
            dict_ = {'clean':0, 'noisy':1}
        else:
            raise Exception(f'Not implemented {self.ds_name=}')
        return dict_




class NonlabelImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(image_path)
        placeholder = 0 # placeholder to makersure the formnat is the same as the other way of dataloading

        if self.transform:
            image = self.transform(image)

        return image, placeholder
    

def get_dataset(dataset_name='MNIST'):
    transforms_che = CHE_transforms_che
    transforms_isic = ISIC_transforms_isic


    if ('CHEXPERT' in dataset_name) or ('NIH' in dataset_name):
        if len(dataset_name.split('-')) == 1:
            ds_name = dataset_name
            split = 'train'
        else:
            assert len(dataset_name.split('-'))==2
            ds_name, split = dataset_name.split('-')

        if dataset_name.startswith('OUT'):
            pure_dataset_name = dataset_name.split('OUT_')[-1]
            root = os.path.join(DATASET_DIR,pure_dataset_name)


            dataset = datasets.ImageFolder(root=root,
                                        transform=transforms_che)

        else: 
            root = f'{BaseConfig.repo_home_dir}/datasets/{ds_name}/{split}/'
            dataset = CustomImageFolder(root=root,
                                        transform=transforms_che,
                                        ds_name=ds_name)



    elif ('ISIC' in dataset_name):
        if len(dataset_name.split('-')) == 1:
            ds_name = dataset_name
            split = 'train'
        else:
            assert len(dataset_name.split('-'))==2
            ds_name, split = dataset_name.split('-')

        dataset = CustomImageFolder(root=f'{BaseConfig.repo_home_dir}/datasets/{ds_name}/{split}/',
                                       transform=transforms_isic,
                                       ds_name=ds_name)

        
    
    return dataset

def get_dataloader(dataset_name='CHEXPERT', 
                   batch_size=32, 
                   pin_memory=False, 
                   shuffle=True, 
                   num_workers=0, 
                   device="cpu"
                  ):
    dataset    = get_dataset(dataset_name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            pin_memory=pin_memory, 
                            num_workers=num_workers, 
                            shuffle=shuffle
                           )

        
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader

def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0 


