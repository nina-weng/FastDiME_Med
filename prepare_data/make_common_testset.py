'''
commen test set:
balanced with disease and shortcut
'''
import pandas as pd
import os
import shutil
from tqdm import tqdm
from global_config import *

def mkdir_train_test(dataset_path,shortcuts_name):
    os.makedirs(dataset_path)
    train_test = ['train','test']
    shortcuts = ['non-'+shortcuts_name,shortcuts_name]
    for each_t in train_test:
        for each_sc in shortcuts:
            os.makedirs(os.path.join(dataset_path,each_t,each_sc))

    return

def make_common_testset(ds_name,from_dataset_path,shortcuts_name):
    ori_dataset_dir = REPO_HOME_DIR + f'/datasets/{ds_name}_100/' # it does not matter which one to choose from
    dataset_name = f'{ds_name}_common_testset'

    dataset_dir = './datasets/' # under this repo path

    meta_data_df = pd.read_csv(ori_dataset_dir+'metadata.csv')

    mkdir_train_test(dataset_path=os.path.join(dataset_dir,dataset_name),
                     shortcuts_name=shortcuts_name)

    for index, row in tqdm(meta_data_df.iterrows()):
        # scr path can get directly from the csv file
        path = from_dataset_path + row['img_name']

        sc_int = int(row[shortcuts_name])
        shortcuts_dir_name = ['non-'+shortcuts_name,shortcuts_name]
        sc = shortcuts_dir_name[sc_int]

        split = row['split']

        if split == 'train': # skip the train set as we dont need them
            continue

        dst_path = os.path.join(dataset_dir,dataset_name,split,sc)

        shutil.copy(path, dst_path)

    meta_data_df.to_csv(os.path.join(dataset_dir,dataset_name,'metadata.csv'),index=False)
