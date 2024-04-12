'''
Generating the counterfactuals.
'''

import torch
import datetime

from global_config import REPO_HOME_DIR
from models.resnet import ResNet

from models.unet import UNet
from models.diffusion import GuidedDiffusion, GuidedDiffusionTS, GuidedDiffusionTSRescaleT
from CFgenerating.utils import *
from CFgenerating.config_cf import cfconfig
from CFgenerating.model_loading import *
from train.utils import *
from train.dataloader import inverse_transform

import os

from torchvision.utils import make_grid
import matplotlib.pyplot as plt 


import seaborn as sns
sns.set_theme(style='white')

import gc


def get_sd(cfconfig,
           iteration=1,
           fixed_mask = None):
    '''
    Get the different guided diffusion model based on the chosen model variants
    
    - FastDiME: 1st iteration with dynamic mask and inpainting, no 2nd iteration
    - FastDiME-2: 1st iteration without mask and inpainting, 2nd iterarion with a fixed mask and inpainting
    - FastDiME-2+: 1st iteration with dynamic mask and inpainting, 2nd iterarion with a fixed mask and inpainting
    - FastDiME-woM: (without mask) 1st iteration without mask and inpainting, no 2nd iteration
    - DiME: 1st iteration without mask and inpainting, no 2nd iteration
    '''

    assert iteration in [1,2]
    assert not(iteration ==2 and (not cfconfig.model_var in ['FastDiME-2','FastDiME-2+' ]))

    if cfconfig.model_var in ['FastDiME-2','FastDiME-2+']:
        iter_num = '{}/2'.format(iteration)
        if cfconfig.model_var == 'FastDiME-2':
            # FastDiME-2: 1st iteration without mask and inpainting, 2nd iterarion with a fixed mask and inpainting
            if iteration == 1:
                use_inpaint = False
                inpaint_type = None
                fixed_mask = None
            else: #iteration == 2
                use_inpaint = True
                inpaint_type = 'fix'
                fixed_mask = fixed_mask
        elif cfconfig.model_var == 'FastDiME-2+':
            # FastDiME-2+: 1st iteration with dynamic mask and inpainting, 2nd iterarion with a fixed mask and inpainting
            use_inpaint = True
            if iteration == 1:
                inpaint_type = 'dynamic'
                fixed_mask = None
            else: #iteration == 2
                inpaint_type = 'fix'
                fixed_mask = fixed_mask

    
    elif cfconfig.model_var in ['FastDiME-woM','DiME'] :
        use_inpaint = False
        inpaint_type = None
        fixed_mask = None
        iter_num = '1/1'
    elif cfconfig.model_var == 'FastDiME':
        use_inpaint = True
        inpaint_type = 'dynamic'
        fixed_mask = None
        iter_num = '1/1'

    if 'FastDiME' in cfconfig.model_var:
        grad_obt_from = 'denoised_img'
    else: # DiME
        grad_obt_from = 'generated_img'
    

    print(f'getting sd,{use_inpaint=},{inpaint_type=},{fixed_mask=},{iter_num=}')
    assert inpaint_type in ['fix','dynamic',None]


    if cfconfig.rescale_t == True and cfconfig.ts_guided == True:
        sd = GuidedDiffusionTSRescaleT(
            num_diffusion_timesteps = cfconfig.total_timesteps,
            img_shape               = diff_config_set['TrainingConfig']['IMG_SHAPE'],
            device                  = device,
            grad_obt_from           = grad_obt_from,
            grad_loss_types         = cfconfig.grad_loss_types,
            grad_loss_weights       = cfconfig.grad_loss_weights,
            rescale                 = cfconfig.rescale_t,
            rescaled_totalt         = cfconfig.rescale_to,
            use_inpaint             = use_inpaint,
            start_tau               = cfconfig.start_tau,
            dilation                = cfconfig.dilation,
            inpaint_th              = cfconfig.inpaint_th,
            inpaint_type            = inpaint_type,
            fixed_mask              = fixed_mask,
            iter_num                = iter_num,
            )

    elif cfconfig.ts_guided == True:
        sd = GuidedDiffusionTS(
            num_diffusion_timesteps = cfconfig.total_timesteps,
            img_shape               = diff_config_set['TrainingConfig']['IMG_SHAPE'],
            device                  = device,
            grad_obt_from           = grad_obt_from,
            grad_loss_types         = cfconfig.grad_loss_types,
            grad_loss_weights       = cfconfig.grad_loss_weights,
            use_inpaint             = use_inpaint,
            start_tau               = cfconfig.start_tau,
            dilation                = cfconfig.dilation,
            inpaint_th              = cfconfig.inpaint_th,
            inpaint_type            = inpaint_type,
            fixed_mask              = fixed_mask,
            iter_num                = iter_num,
        )
    else:
        sd = GuidedDiffusion(
                num_diffusion_timesteps = cfconfig.total_timesteps,
                img_shape               = diff_config_set['TrainingConfig']['IMG_SHAPE'],
                device                  = device,
                grad_obt_from           = grad_obt_from,
                grad_loss_types         = cfconfig.grad_loss_types,
                grad_loss_weights       = cfconfig.grad_loss_weights,
                use_inpaint             = use_inpaint,
                start_tau               = cfconfig.start_tau,
                dilation                = cfconfig.dilation,
                inpaint_th              = cfconfig.inpaint_th,
                inpaint_type            = inpaint_type,
                fixed_mask              = fixed_mask,
                iter_num                = iter_num,
                )
    return sd     




def plot_cf_process(records: tuple[list],
                    this_save_path: str,
                    total_timesteps: int,
                    cf_from_timestep: int,
                    x_0:torch.Tensor,
                    ):
    '''
    plot the process of generating the counterfactuals
    Parameter:
    - records
    - this_save_path: save dir path
    - 

    Return:
    1. 'x_cf_t_0_{}-ttal_{}.png': 
    2. ..
    3. ..
    '''
    # 
    x_t_0_list,x_t_minus_1_list,max_grad_list,prob_list, losses_dict= records

    img_size = x_t_0_list[0].shape
    assert len(img_size) == 4, f'{img_size}'
    if img_size[3] < 64:
        imgplot_interval = 5
        nrow = 20
    else: 
        imgplot_interval = 20
        nrow = 5

    # 1. x_t_0_list
    grid = make_grid([inverse_transform(each[0]).type(torch.uint8) for each in x_t_0_list[0::imgplot_interval]],nrow=nrow, pad_value=255.0).to("cpu")
    ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
    # display(Image.fromarray(ndarr))
    cv2.imwrite(this_save_path+f'x_cf_t_0_{cf_from_timestep}-ttal_{total_timesteps}.png', ndarr)    

    # 2. x_t_minus_1_list
    grid = make_grid([inverse_transform(each[0]).type(torch.uint8) for each in x_t_minus_1_list[0::imgplot_interval]],nrow=nrow, pad_value=255.0).to("cpu")
    ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
    cv2.imwrite(this_save_path+'x_cf_t_minus_1.png', ndarr)

    # 3. prob and max_grad in the same plot shares the same x-axis
    max_grad_list = [each.detach().cpu().numpy() for each in max_grad_list]
    prob_list= [each.detach().cpu().numpy() for each in prob_list]
    
    fig, ax1 = plt.subplots(figsize=(10, 3))
    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  

    ax1.plot(range(cf_from_timestep,1,-1),max_grad_list,label='max_grad',color='olive')
    ax2.plot(range(cf_from_timestep,1,-1),prob_list,label='prob',color='gold', linestyle = 'dashed');
    ax1.set_xlabel('t')
    # plt.xticks(range(timestep,1,-1),labels=None)
    ax1.set_xlim(int(cf_from_timestep*1.05),int(-cf_from_timestep*0.05))
    ax1.set_yscale('log')

    ax1.set_ylabel('max_grad')
    ax2.set_ylabel('prob')

    fig.legend()
    # fig.show()
    fig.savefig(this_save_path+'prob_maxgrad.png')

    # 4. subplots prob, maxgrad and losses
    num_type_loss =len(losses_dict.keys())
    fig, axs = plt.subplots(1, 2+num_type_loss,figsize=(3*(2+num_type_loss), 3))
    # max_grad
    axs[0].plot(range(cf_from_timestep,1,-1),max_grad_list,color='olive')
    axs[0].set_title('max_grad')
    # prob
    axs[1].plot(range(cf_from_timestep,1,-1),prob_list,color='gold')
    axs[1].set_title('prob of having confounders')
    # losses
    for i,key in enumerate(losses_dict.keys()):
        loss_cpu = [each.squeeze().detach().cpu().numpy() for each in losses_dict[key]]
        axs[i+2].plot(range(cf_from_timestep,1,-1),loss_cpu)
        axs[i+2].set_title(f'{key}')

    for i in range(2+num_type_loss):
        axs[i].set_xlabel('t')
        axs[i].set_xlim(int(cf_from_timestep*1.05),int(-cf_from_timestep*0.05))
    
    fig.tight_layout()
    fig.savefig(this_save_path+'all_records_in_subplots.png')


    # 5. Mask between generated img and x_0
    for idx in range(img_size[0]):
        mask_list = [inverse_transform(each[idx]).type(torch.uint8)-inverse_transform(x_0[idx]).type(torch.uint8) for each in x_t_0_list[0::imgplot_interval]]
        grid = make_grid(mask_list,
                        nrow=nrow, pad_value=255.0).to("cpu")
        ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
        # display(Image.fromarray(ndarr))
        cv2.imwrite(this_save_path+f'mask_x_1_0minusx_0_idx{idx}.png', ndarr)    



    return

def one_CF_procedure(sample_img: torch.Tensor,
                     sample_lab:torch.Tensor,
                     sample_traget:torch.Tensor,
                     sd: GuidedDiffusion,
                     diffusion_model: UNet,
                     classifier: ResNet,
                     showplot:bool=False,
                     saveplot:bool=True,
                     this_save_path:str=None):
    '''
    One precedure for CF
    '''
    # save the label and target
    save_lab_target(sample_lab,sample_traget[0],this_save_path+'lab_target.txt')

    # get the prediction from the classifer on original image
    prob_x_0 = get_prediction_from_classifier(classifier, sample_img,device)

    # save the plots  x_0 
    save_imgs_tensor_batch(sample_img,this_save_path+'x_0.png')

    # get x_t and save it
    timestep_tensor = torch.as_tensor(cfconfig.cf_from_timestep, dtype=torch.long).to(device)
    x_t,noise = sd.forward_diffusion(sample_img,timestep_tensor)
    save_imgs_tensor_batch(x_t,this_save_path+'x_t_{}-ttal_{}.png'.format(cfconfig.cf_from_timestep,cfconfig.total_timesteps))
    x_0 = sample_img


    assert len(x_0.shape) == 4 and len(x_t.shape) == 4 and len(sample_traget.shape) == 2, f'Incorrect Shape: {x_0.shape=}, {x_t.shape=}, {sample_traget.shape}'

    

    # generating cf
    records = sd.p_sample_loop(diffusion_model,
                        classifier,
                        timestep=cfconfig.cf_from_timestep, # the timestep we start to generate cf from
                        x_t = x_t, # shape: [BS,C,H,W]
                        x_0 = x_0,
                        target=sample_traget, # shape: [BS,1]
                        showplot=showplot,
                        saveplot=saveplot,
                        this_save_path = this_save_path,
                        scale = cfconfig.start_scale,
                        saveshow_gap=cfconfig.saveshow_gap,
                        )
    
    # pred the pobability of the cf
    x_cf = records[0][-1] # the last one of x_t_0_list
    prob_x_cf = get_prediction_from_classifier(classifier, x_cf,device)

    # save the prob(x_0) and prob(x_cf) to the same file where we store label and target
    save_probs(this_save_path+'lab_target.txt',prob_x_0=prob_x_0,prob_x_cf=prob_x_cf)
    
    # plotting
    if cfconfig.plot_cf_process:
        # all samples skip this
        plot_cf_process(records=records,
                    this_save_path=this_save_path,
                    total_timesteps=cfconfig.total_timesteps,
                    cf_from_timestep=cfconfig.cf_from_timestep,
                    x_0=x_0)
    
    del records
    torch.cuda.empty_cache()
    # then collect the garbage
    gc.collect()
    





if __name__ == '__main__':
    

    ##################################
    # print out config
    print_config(cfconfig)

    ##################################
    

    # clear the cache
    with torch.no_grad():
        torch.cuda.empty_cache()

    


    # path where the results will be saved
    code_repo_path = REPO_HOME_DIR
    save_res_path = code_repo_path+f'records/cf_generating/{cfconfig.ds_name}_{cfconfig.model_var}/'
    create_dir_if_nonexist(save_res_path)

    version_num = get_version_num(save_res_path)
    save_res_path = save_res_path + 'version_{}'.format(version_num)
    create_dir_if_nonexist(save_res_path)


    # diffusion model
    if torch.cuda.is_available():
        device='cuda'
    else: device = 'cpu'
    
    

    # load dataset used for inference, notice that we only do the test set part
    ds_test_loader = load_test_set(ds_name=cfconfig.ds_name,
                                device= device)
    
    
    # --------- run counterfactuals -----------

    # Get the current date and time for a unique string for record purposes
    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.strftime("%Y%m%d%H%M%S")
    save_config(cfconfig,save_res_path+f'/config_record_{timestamp}.txt')


    for loader_idx,(img, lab) in enumerate(ds_test_loader):
        # get each sample
        for i in range(len(lab)):
            print(f'{loader_idx=},{i}/{len(lab)}')

            sample_img = img[i]
            sample_lab = lab[i]

            # filter samples if needed
            if cfconfig.confounder_filter != 'all':
                if int(cfconfig.confounder_filter) != sample_lab.cpu().numpy():
                    continue

            print('First iteration:')

            # get the save dir pth
            # create the dir for saving the results
            config_name = 'ts_{}_{}-sidx_{}_{}'.format(cfconfig.cf_from_timestep,
                                                            cfconfig.total_timesteps,
                                                            loader_idx,
                                                            i)
            
            this_save_path = get_save_path_with_version(os.path.join(save_res_path,config_name),with_version = False)

            # load the trained diffusion model (UNet)
            diffusion_model, diff_config_set = load_diffusion_model(checkpoint_dir=cfconfig.checkpoint_dir)

            # get sd
            sd = get_sd(cfconfig)
            
            # load the classifier for guidance
            # load classifier to avoid svaing grad graph
            classifier = load_classifier(cfconfig.best_model_path,cfconfig.model_type,device)

            

            sample_img = sample_img.to(device).unsqueeze(0)
            sample_lab = sample_lab.to(device).unsqueeze(0)

            sample_target = torch.tensor(1.0) - sample_lab
            sample_target = sample_target.to(device)
            sample_target = sample_target.unsqueeze(0)

                
            # CF
            one_CF_procedure(sample_img,sample_lab,sample_target,
                            # Note that it takes a large space to save the plot for each step
                            saveplot=cfconfig.plot_cf_process,
                            sd=sd,
                            diffusion_model=diffusion_model,
                            classifier=classifier,
                            this_save_path = this_save_path,)
            
            

            del classifier
            del sample_img,sample_lab, sample_target
            del sd
            del diffusion_model
            
            torch.cuda.empty_cache()
            gc.collect()

            if cfconfig.model_var in ['FastDiME-2','FastDiME-2+']:
                # the second iteration
                print('Second iteration: ')
                # load the trained diffusion model (UNet)
                diffusion_model, diff_config_set = load_diffusion_model(checkpoint_dir=cfconfig.checkpoint_dir)

                
                # load the classifier for guidance
                # load classifier to avoid svaing grad graph
                classifier = load_classifier(cfconfig.best_model_path,cfconfig.model_type,device)

                sample_img = img[i]
                sample_lab = lab[i]

                if cfconfig.confounder_filter != 'all':
                    if int(cfconfig.confounder_filter) != sample_lab.cpu().numpy():
                        continue

                sample_img = sample_img.to(device).unsqueeze(0)
                sample_lab = sample_lab.to(device).unsqueeze(0)

                sample_target = torch.tensor(1.0) - sample_lab
                sample_target = sample_target.to(device)
                sample_target = sample_target.unsqueeze(0)

                
                # get the cf img for 1st iteration and obtain the fixed mask from cf and original image
                cf_1st_iter_img = get_one_cf_img(img_path = this_save_path+'x_1_0_iter_1.png',device=device)
                fixed_mask = get_fixed_mask(sample_img,cf_1st_iter_img,dilation=cfconfig.dilation,inpaint_th=cfconfig.inpaint_th)
                # save the fixed mask
                save_fixed_mask(fixed_mask,this_save_path) 

                # get sd
                sd = get_sd(cfconfig,iteration=2,fixed_mask=fixed_mask)
                    
                # CF
                one_CF_procedure(sample_img,sample_lab,sample_target,
                                 # Note that it takes a large space to save the plot for each step
                                saveplot=cfconfig.plot_cf_process,
                                sd=sd,
                                diffusion_model=diffusion_model,
                                classifier=classifier,
                                this_save_path= this_save_path)
                
                

                del classifier
                del sample_img,sample_lab, sample_target
                del sd
                del diffusion_model
                
                torch.cuda.empty_cache()
                gc.collect()



        




    


    
    
