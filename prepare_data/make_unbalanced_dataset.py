'''
Prepare 3 unbalanced dataset,
with the 100/75/50% 'correlation' between diesase and confounders 
100: all disease samples have shortcuts
75: 75% of the disease samples have shortcuts
50: 50% of the disease samples have shortcuts (balanced dataset)

BaseSize: the basic size for disease/non-disease samples, or confouder/non-confouder samples
---------------
|.      | D | non-D|
|     C | a | b    |
| non-C | c | d.   |

100: a = d = BaseSize
     c = b = 0

75:  a = d = 0.75 * BaseSize
     c = b = 0.25 * BaseSize

50:  a = b = c = d = 0.5 BaseSize

'''





import pandas as pd 
from tqdm import tqdm
import os
import shutil
import sys
sys.path.append('../FastDiME_Med/')
from global_config import *
from prepare_data.make_common_testset import make_common_testset

def mkdir_train_test(dataset_path,shortcuts_name):
    os.makedirs(dataset_path)
    train_test = ['train','test']
    shortcuts = ['non-'+shortcuts_name,shortcuts_name]
    for each_t in train_test:
        for each_sc in shortcuts:
            os.makedirs(os.path.join(dataset_path,each_t,each_sc))

    return

def get_basesize(df_collection,
                 disease_name,
                 confounder_name,
                 split = 'train'):
    '''
    basesize is the smallest size of [D_C, D_nonC, nonD_C, nonD_nonC] in train set 
    '''
    all_sizes = []
    for each_d in [disease_name,'non-'+disease_name]:
        for each_c in [confounder_name, 'non-'+ confounder_name]:
            this_df = df_collection[each_d][each_c]
            this_df_train = this_df[this_df['split'] == split]
            this_size = len(this_df_train)
            ### need test
            print('WARNING ! need test')
            if each_d.startswith('non-') ^ each_c.startswith('non-'):
                this_size *=2
                print(f'{each_d=},{each_c=}')
            all_sizes.append(this_size)

    min_ = min(all_sizes)

    if split == 'train':
        while min_ % 4 !=0:
            min_ -=1

    print(f'{min_=}')
    return int(min_)




def make_unbalanced_ds(ds_name,
                       from_dataset_path, # copy the images from this path
                       disease_name,
                       confounder_name,
                       metadata_filename,
                       ):
    
    seed = 42
    print(f'{seed=}')
    based_ds_name = ds_name #'CHEXPERT_CardioPM'
    dataset_dir = './datasets/'
    meta_df = pd.read_csv(REPO_HOME_DIR +'prepare_data/metadata_files/' +metadata_filename)


    
    
    df_collection={}
    for each_d in [disease_name,'non-'+disease_name]:
        indicator_d = 0 if each_d.startswith('non-') else 1
        df_collection[each_d] = {}
        df_ = meta_df[meta_df[disease_name] == indicator_d]

        for each_c in [confounder_name, 'non-'+ confounder_name]:
            indicator_cf = 0 if each_c.startswith('non-') else 1
            df_collection[each_d][each_c] = df_[df_[confounder_name] == indicator_cf]

    
    


    print(f'{df_collection=}')

    percentages = [100, 75, 50]
    baseSize = get_basesize(df_collection,disease_name,confounder_name)
    baseSize_test = get_basesize(df_collection,disease_name,confounder_name,split='test')

    # get the test set's meta csv
    # randomly sample using seed 
    sampled_test = None
    for each_d in [disease_name,'non-'+disease_name]:
        for each_c in [confounder_name, 'non-'+ confounder_name]:
            this_df = df_collection[each_d][each_c]
            test_df = this_df[this_df['split'] == 'test']

            sampled = test_df.sample(n=baseSize_test,
                                     replace=False,
                                     random_state=seed)
            
            if sampled_test is None:
                sampled_test = sampled
            else: sampled_test = pd.concat([sampled_test,sampled], ignore_index=True)

    print(f'{sampled_test=}')

    # sample the train according to percentages
    meta_df_collection={}
    for each_p in percentages:
        sampled_train = None
        for each_d in [disease_name,'non-'+disease_name]:
            for each_c in [confounder_name, 'non-'+ confounder_name]:
                this_df = df_collection[each_d][each_c]
                test_df = this_df[this_df['split'] == 'train']

                if (each_c.startswith('non-')) != (each_d.startswith('non-')) :
                    n = int(baseSize*(100-each_p)/100)
                else: 
                    n = int(baseSize*each_p/100)
                print(f'{each_d=},{each_c=},{n=}')

                sampled = test_df.sample(n=n,
                                        replace=False,
                                        random_state=seed)
                
                if sampled_train is None:
                    sampled_train = sampled
                else: sampled_train = pd.concat([sampled_train,sampled], ignore_index=True)
        
        sampled_all = pd.concat([sampled_train,sampled_test],ignore_index=True)
        meta_df_collection[each_p] = sampled_all
        print(f'{each_p=}')
        print(f'{sampled_all=}')

        this_p_ds_name = based_ds_name+'_'+str(each_p)
        shortcuts_name = disease_name

        mkdir_train_test(dataset_path=os.path.join(dataset_dir,this_p_ds_name),
                     shortcuts_name=shortcuts_name)

        for index, row in tqdm(sampled_all.iterrows()):
            # scr path can get directly from the csv file
            path = from_dataset_path + row['img_name']

            sc_int = int(row[shortcuts_name])
            shortcuts_dir_name = ['non-'+shortcuts_name,shortcuts_name]
            sc = shortcuts_dir_name[sc_int]

            split = row['split']

            dst_path = os.path.join(dataset_dir,this_p_ds_name,split,sc)

            shutil.copy(path, dst_path)

        sampled_all.to_csv(os.path.join(dataset_dir,this_p_ds_name,'metadata.csv'),index=False)
    

        

if __name__ == '__main__':
    
    which_ds = 'isic'# ['chexpert','nih','isic']
    assert which_ds in ['chexpert','nih','isic']
    print(f'prepare unbalanced dataset for {which_ds}')

    if which_ds == 'chexpert':
        dataset_name = 'CHEXPERT_CardioPM'
        ori_dataset_dir = CHE_DATASET_DIR  # change to yours
        metadata_filename = 'che_pm_meta.csv'
        disease_name = 'Cardiomegaly'
        confounder_name = 'PM'
    elif which_ds == 'nih':
        dataset_name = 'NIH_Drain'
        ori_dataset_dir = NIH_DATASET_DIR # change to yours
        metadata_filename = 'nih_drain_meta.csv'
        disease_name = 'Pneumothorax'
        confounder_name = 'Drain'
    else: # isic
        dataset_name = 'ISIC_Ruler'
        ori_dataset_dir =  ISIC_DATASET_DIR  # change to yours
        metadata_filename = 'isic_ruler_meta.csv'
        disease_name = 'Mal'
        confounder_name = 'Ruler'


    make_unbalanced_ds(ds_name=dataset_name,
                       from_dataset_path=ori_dataset_dir,
                       disease_name = disease_name,
                       confounder_name = confounder_name,
                       metadata_filename = metadata_filename)
    make_common_testset(ds_name=dataset_name,
                        from_dataset_path=ori_dataset_dir,
                        shortcuts_name=confounder_name,)