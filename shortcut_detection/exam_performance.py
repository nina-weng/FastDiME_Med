'''
shortcuts detection
Method 1: compare the similarity between prediction prob of original test set and the CF ones
'''

from global_config import REPO_HOME_DIR
from models.resnet import ResNet
from models.cnn import SimpleCNN
from models.encoder import EncoderUNetModel
from train.utils import *
from CFgenerating.utils import *
from shortcut_detection.dataloader import CFDataModule
from shortcut_detection.get_test_img_sequence import main_sort_test
from shortcut_detection.utils import cos_similarity
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
import pandas as pd
import numpy as np

def load_classifier(model_path: str,
                    model_type: str,
                    device: str ):

    config_set = get_classifier_config(model_path)
    print(f'{config_set=}')

    if model_type == 'resnet':
        classifier = ResNet.load_from_checkpoint(model_path,
                                        num_classes=1,
                                        lr=config_set['lr'],
                                        pretrained=True,
                                        model_scale=config_set['model_scale'],
                                        loss_func_type = config_set['loss_func'], # WeightedBCE, BCE
                                        weights = config_set['weights'],
                                        in_channels = config_set['img_size'][0],
                                        ).eval().to(device)
    elif model_type == 'simplecnn':
        classifier = SimpleCNN.load_from_checkpoint(model_path,
                                                    num_classes=1,
                                                    lr=config_set['lr'],
                               ).eval().to(device)
    elif model_type == 'unet_encoder':
        classifier_attention_resolutions="32,16,8"
        classifier_width=128
        classifier_depth=2
        classifier_use_fp16=False
        classifier_use_scale_shift_norm=True  # False
        classifier_resblock_updown=True  # False
        classifier_pool="attention"

        attention_ds = []
        for res in classifier_attention_resolutions.split(","):
            attention_ds.append(config_set['img_size'][2] // int(res))
        channel_mult = (1, 2, 4, 4)

        classifier = EncoderUNetModel.load_from_checkpoint(model_path,
                                                    in_channels=config_set['img_size'][0],
                                                    model_channels=classifier_width,
                                                    out_channels=1,
                                                    num_res_blocks=classifier_depth,
                                                    attention_resolutions=tuple(attention_ds),
                                                    channel_mult=channel_mult,
                                                    use_fp16=classifier_use_fp16,
                                                    num_head_channels=64,
                                                    use_scale_shift_norm=classifier_use_scale_shift_norm,
                                                    resblock_updown=classifier_resblock_updown,
                                                    pool=classifier_pool,
                                                    # ori parameters
                                                    num_classes=1,
                                                    lr=config_set['lr'],
                                                    img_size=config_set['img_size'][2],
                                                    loss_func_type= config_set['loss_func'],
                                                    weights = config_set['weights'],
                                                    dropout_rate=config_set['dropout_rate'],
                                                    weight_decay = config_set['weight_decay'],
                               ).eval().to(device)
    print('classifier loaded.')
    return classifier


def test_func(model, data_loader, n_classes,device):
    model.eval()
    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, t, sc_lab = batch['image'].to(device), batch['t'].to(device), batch['sc_lab'].to(device)
            out = model(img,t)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(sc_lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        logits = torch.squeeze(logits)
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)


    return preds,targets,logits


def main_sc_exam(scdconfig,
                 model_pth_list=None):

    if model_pth_list is not None:
        model_path = model_pth_list 
    else:
        model_path = scdconfig.model_path_dict

    print(f'{model_path=}')


    if torch.cuda.is_available():
        device='cuda'
    else: device = 'cpu'

    
    # get sorted meta data
    if os.path.exists(REPO_HOME_DIR + f'/datasets/{scdconfig.ds_name}_common_testset/metadata_test_sorted.csv'):
        pass
    else:
        main_sort_test(ds_name = scdconfig.ds_name+'_common_testset',
            img_index_col_name = scdconfig.img_index_col_name,
            disease_col_name = scdconfig.disease_col_name,
            sc_col_name = scdconfig.sc_col_name,
            )

    for p in scdconfig.percentages:
        print('percentage:{}'.format(p))
        p = str(p)
        # load the classifier
        classifier = load_classifier(model_path=model_path[p],
                                    model_type=scdconfig.model_type,
                                    device=device,)

        # load the dataset
        dataset_ori = CFDataModule(data_dir = scdconfig.cf_store_path,
                            img_size =scdconfig.img_size,
                            batch_size = 32,
                            num_workers = 16,
                            img_type = 'ori'

        )
        print('original image loaded.')

        dataset_cf = CFDataModule(data_dir = scdconfig.cf_store_path,
                            img_size =scdconfig.img_size,
                            batch_size = 32,
                            num_workers = 16,
                            img_type = 'cf'

        )
        print('cf image loaded.')


        n_classes = dataset_ori.num_classes

        # get the inference 
        print('inferencing the original images')
        preds_ori , targets_test, _  = test_func(classifier, 
                                                        dataset_ori.test_dataloader(), 
                                                        n_classes,
                                                        device=device)
        
        print('inferencing the cf images')
        preds_cf , _, _  = test_func(classifier, 
                                                        dataset_cf.test_dataloader(), 
                                                        n_classes,
                                                        device=device)
        

        if device == 'cuda':
            preds_ori = preds_ori.cpu()
            targets_test = targets_test.cpu()
            preds_cf = preds_cf.cpu()

        preds_ori = preds_ori.numpy()
        targets_test = targets_test.numpy()
        preds_cf = preds_cf.numpy()

        cols_names_targets = ['sc_lab']
        cols_names_preds_ori = ['preds_disease_ori']
        cols_names_preds_cf = ['preds_disease_cf']
        
        df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
        df_preds_ori = pd.DataFrame(data=preds_ori, columns=cols_names_preds_ori)
        df_preds_cf = pd.DataFrame(data=preds_cf, columns=cols_names_preds_cf)
        df = pd.concat([df_targets, df_preds_ori, df_preds_cf], axis=1)

        # concat it with the img_names, cardiomagely labels and shortcuts labels
        sorted_test_meta_csv = REPO_HOME_DIR + f'/datasets/{scdconfig.ds_name}_common_testset/metadata_test_sorted.csv'
        meta_test_sorted = pd.read_csv(sorted_test_meta_csv)

        assert len(df) == len(meta_test_sorted), f'length does not match: {len(df) = }, {len(meta_test_sorted)=}'

        df = pd.concat([meta_test_sorted,df],axis=1)

        assert df['sc_lab'].equals(df[scdconfig.sc_col_name]),'{},{}'.format(df['sc_lab'][:10],df[scdconfig.sc_col_name][:10])

        # place to save
        save_dir = REPO_HOME_DIR+'/records/shortcuts/'+scdconfig.ds_name+f'_{p}'+'/'+scdconfig.model_type+'/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_csv(save_dir+'preds_ori_cf.csv',
                                index=False)
        
        # save config
        save_config(scdconfig,save_dir+'config_record.txt')
        

        # compute the metrics
        # 1. mean(abs(difference))
        mean_abs_diff = np.mean(np.abs(preds_ori-preds_cf))


        # 2. the changes of flipped from PM to non-PM
        # wrt. confounder label

        confounder_cases_index = np.argwhere(targets_test==1)
        non_confounder_cases_index = np.argwhere(targets_test==0)


        mean_diff_p_ori_p_cf_confounder = np.mean(preds_ori[confounder_cases_index]-preds_cf[confounder_cases_index])
        mean_diff_p_ori_p_cf_non_confounder = np.mean(preds_ori[non_confounder_cases_index]-preds_cf[non_confounder_cases_index])

        mean_abs_diff_p_ori_p_cf_confounder = np.mean(np.abs(preds_ori[confounder_cases_index]-preds_cf[confounder_cases_index]))
        mean_abs_diff_p_ori_p_cf_non_confounder = np.mean(np.abs(preds_ori[non_confounder_cases_index]-preds_cf[non_confounder_cases_index]))


        # 3. similarity score
        cos_sim_0 = cos_similarity(preds_ori,preds_cf)
        cos_sim = cosine_similarity(preds_ori.reshape(1,-1),preds_cf.reshape(1,-1))[0][0]
        print(f'{cos_sim_0=},{cos_sim=}')


        res_dict = {
            'mean_abs_diff': mean_abs_diff,
            'mean(p(ori) - p(cf)),confounder':mean_diff_p_ori_p_cf_confounder,
            'mean(p(ori) - p(cf)),non_confounder':mean_diff_p_ori_p_cf_non_confounder,
            'mean(abs(p(ori) - p(cf))),confounder': mean_abs_diff_p_ori_p_cf_confounder,
            'mean(abs(p(ori) - p(cf))),non-counfounder':mean_abs_diff_p_ori_p_cf_non_confounder,
            'cos_sim': cos_sim,
        }

        # record the result in .txt
        with open(save_dir+f'result_test.txt', "w") as f:
            str_ = 'result:\n'
            for each in res_dict.keys():
                str_+=f'{each}:{res_dict[each]: .4f}\n'
            print(str_)
            f.write(str_)
            print('Successfully record the text result in .txt file.')



    return



if __name__ == '__main__':


    main_sc_exam()