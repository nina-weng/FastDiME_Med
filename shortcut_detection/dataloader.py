from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from PIL import Image
import os
import torchvision.transforms as TF 
from torch import nn
from functools import cmp_to_key

def compare(s1, s2):
    loader_idx_1, idx_1 = int(s1.split('_')[-2]),int(s1.split('_')[-1])
    loader_idx_2, idx_2 = int(s2.split('_')[-2]),int(s2.split('_')[-1])
    if loader_idx_1 < loader_idx_2:
        return -1
    elif loader_idx_1 > loader_idx_2:
        return 1
    else: # loader_idx_1 == loader_idx_2
        if idx_1 < idx_2:
            return -1
        elif idx_1 > idx_2:
            return 1
        return 0



class CFDataset(Dataset):
    def __init__(self, 
                 img_dir,
                 transform=None,
                 img_type = 'ori',
                 ):
        self.img_dir = img_dir
        self.transform = transform
        self.img_type = img_type

        assert self.img_type in ['ori','cf']

        self.selected_file_name = 'x_0.png' if self.img_type == 'ori' else 'x_1_0.png'

        self.img_file_names = []

        self.img_sc_labels = []

        sample_list = os.listdir(self.img_dir)
        sample_list = [each for each in sample_list if not each.startswith('config')]
        sample_list = sorted(sample_list, key=cmp_to_key(compare))
        self.sample_list = sample_list

        for idx,each in enumerate(self.sample_list):
            sample_dir = os.path.join(self.img_dir,each)
            self.img_file_names.append(os.path.join(sample_dir,self.selected_file_name))

            with open(sample_dir+'/lab_target.txt') as f:
                contents = f.read()
                this_sc_lab = contents.split('\n')[0].split('lab:')[-1]
                this_sc_lab = int(this_sc_lab[1:-1])
                self.img_sc_labels.append(this_sc_lab)

        


    def __len__(self):
        return len(self.img_sc_labels)

    def __getitem__(self, idx):
        img_path = self.img_file_names[idx]
        image = Image.open(img_path)
        sc_lab = self.img_sc_labels[idx]

        t = 0

        if self.transform:
            image = self.transform(image)


        return  {'image': image,'t':t, 'sc_lab':sc_lab}



class CFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        img_size: tuple[int] = None,
        batch_size: int= None,
        num_workers: int =None,
        normalize: bool =True,
        data_split_seed: int =42,
        img_type:str = 'ori',
    ):
        super().__init__()
        # labels are not important here
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = 2 if num_workers is None else num_workers
        self.normalize = normalize
        self.data_split_seed = data_split_seed
        self.img_type = img_type


        self.normelization = TF.Normalize((0.,), (1.,)) if normalize else nn.Identity() 
        
        self.num_classes = 1
        self.compose_list = self.get_compose_list()
        self.transform = TF.Compose(self.compose_list)
    
        self.trainval_set = []
        self.train_set = []
        self.val_set = []
        self.test_set = CFDataset(img_dir=self.data_dir,
                                  transform=self.transform,
                                  img_type = self.img_type)

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
    
            

    def get_compose_list(self):
        '''
        get the compose function list from the parameters
        '''
        compose_list=[]

        if self.img_size[0] == 1:
            compose_list.append(TF.Grayscale(num_output_channels=self.img_size[0]))
        compose_list.extend(
            [TF.ToTensor(),
            TF.Resize((self.img_size[1],self.img_size[2]), 
                    interpolation=TF.InterpolationMode.BICUBIC, 
                    antialias=True),
            TF.Lambda(lambda t: (t * 2) - 1), 
            ]
        )
        return compose_list