'''
get the test sample sequence with img_name , sc_lab, anb cardio_lab
'''

import pandas as pd
import numpy as np
from global_config import *


def main_sort_test(ds_name = 'CHEXPERT_CardioPM_common_testset',
         img_index_col_name = 'img_name',
         disease_col_name = 'Cardiomegaly',
         sc_col_name = 'PM',
         ):
    print(f'start getting sorted meta data csv for {ds_name}')
    CardioPM_common_testset_metadata_csv_pth = REPO_HOME_DIR + f'/datasets/{ds_name}/metadata.csv'

    df = pd.read_csv(CardioPM_common_testset_metadata_csv_pth)


    df_test = df[df['split'] == 'test']
    df_test = df_test[[img_index_col_name,disease_col_name,sc_col_name,'split']]
    df_test_sc = df_test[df_test[sc_col_name] == 1]
    df_test_non_sc = df_test[df_test[sc_col_name] == 0]
    df_test_sc = df_test_sc.sort_values(by=img_index_col_name, ascending=True)
    df_test_non_sc = df_test_non_sc.sort_values(by=img_index_col_name, ascending=True)
    df_sorted_test = pd.concat([df_test_sc,df_test_non_sc],ignore_index=True)
    df_sorted_test[sc_col_name] = df_sorted_test[sc_col_name].astype(np.int64)
    df_sorted_test.to_csv(REPO_HOME_DIR +f'/datasets/{ds_name}/metadata_test_sorted.csv')

    return


if __name__ == '__main__':
    main_sort_test()