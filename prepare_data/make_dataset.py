'''
prapare the datasets
'''
import sys
sys.path.append("../FastDiME_Med/")
print(sys.path)

import pandas as pd
import os
from global_config import REPO_HOME_DIR, DATASET_DIR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

def mkdir_train_test(dataset_path,shortcuts_name):
    os.makedirs(dataset_path)
    train_test = ['train','test']
    shortcuts = ['non-'+shortcuts_name,shortcuts_name]
    for each_t in train_test:
        for each_sc in shortcuts:
            os.makedirs(os.path.join(dataset_path,each_t,each_sc))

    return

def make_dataset_with_labels(which_ds):
    assert which_ds in ['chexpert','nih','isic']
    print('-'*30)
    dataset_dir = './datasets/' # under this repo path
    shortcut_lab_dir = REPO_HOME_DIR +'/shortcut_annotations/'

    if which_ds == 'chexpert':
        dataset_name = 'CHEXPERT_CardioPM'
        ori_dataset_dir = DATASET_DIR + '/chexpert/preproc_224x224/' # change to yours
        shortcuts_name = 'PM'
        metadata_filename = 'che_pm_shortcut_labels.csv'
    elif which_ds == 'nih':
        dataset_name = 'NIH_Drain'
        ori_dataset_dir = DATASET_DIR + '/NIH/preproc_224x224/' # change to yours
        shortcuts_name = 'Drain'
        metadata_filename = 'nih_drain_shortcut_labels.csv'
    else: # isic
        dataset_name = 'ISIC_Ruler'
        ori_dataset_dir =  DATASET_DIR + '/isic/ISIC2018_preproc_224x224/' # change to yours
        shortcuts_name = 'Ruler'
        metadata_filename = 'isic_ruler_shortcut_labels.csv'

    meta_data_df = pd.read_csv(shortcut_lab_dir+metadata_filename)

    if 'split' in meta_data_df.columns:
        pass
    else:
        # split the dataset
        shortcut_df = meta_data_df[meta_data_df[shortcuts_name] == 1]
        non_shortcut_df = meta_data_df[meta_data_df[shortcuts_name] == 0]

        dfs = [shortcut_df,non_shortcut_df]

        result_df = None
        for each_df in dfs:
            train_df, test_df = train_test_split(each_df, test_size=0.1, random_state=42)
            train_df['split'] = 'train'
            test_df['split'] = 'test'
            if result_df is None:
                result_df = pd.concat([train_df, test_df])
            else:
                tmp = pd.concat([train_df, test_df])
                result_df = pd.concat([result_df,tmp])

        result_df.to_csv(shortcut_lab_dir+metadata_filename)
        meta_data_df = pd.read_csv(shortcut_lab_dir+metadata_filename)

    mkdir_train_test(dataset_path=os.path.join(dataset_dir,dataset_name),
                     shortcuts_name=shortcuts_name)

    for index, row in tqdm(meta_data_df.iterrows()):
        img_name = row['img_name']

        sc_int = int(row[shortcuts_name])
        shortcuts_dir_name = ['non-'+shortcuts_name,shortcuts_name]
        sc = shortcuts_dir_name[sc_int]

        split = row['split']

        src_path = ori_dataset_dir+img_name
        dst_path = os.path.join(dataset_dir,dataset_name,split,sc)

        shutil.copy(src_path, dst_path)

    meta_data_df.to_csv(os.path.join(dataset_dir,dataset_name,'metadata.csv'),index=False)

    print(f'{len(meta_data_df)=}')
    print(f'{meta_data_df.groupby(shortcuts_name).count()=}')

    print('-'*30)
    return










if __name__ == '__main__':
    # prepare the dataset
    make_dataset_with_labels(which_ds='chexpert')

    