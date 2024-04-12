import torch
from tqdm import tqdm
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np


from train.utils import *
from models.unet import UNet
from models.resnet import ResNet
from train.dataloader import get_dataloader,get_dataset,inverse_transform
from CFgenerating.utils import *


@torch.no_grad()
def generate_mask(x1,x2,dilation):
    assert (dilation %2) ==1, 'dilation must be an odd number'

    x1 = (x1+1)/2
    x2 = (x2+1)/2

    mask = (x1 - x2).abs().sum(dim=1, keepdim=True)
    mask = mask / mask.view(mask.size(0), -1).max(dim=1)[0].view(-1,1,1,1)
    dil_mask = F.max_pool2d(mask,
                            dilation, stride=1,
                            padding = (dilation-1)//2
    )
    return mask, dil_mask


class SimpleDiffusion:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        img_shape=(3, 64, 64),
        device="cpu",
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device

        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta  = self.get_betas()
        self.alpha = 1 - self.beta
        
        self.sqrt_beta                       = torch.sqrt(self.beta)
        self.alpha_cumulative                = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative           = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha               = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)
         
    def get_betas(self):
        """linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
        
    def forward_diffusion(self, x0: torch.Tensor, timesteps: torch.Tensor):
        eps = torch.randn_like(x0)  # Noise
        mean    = get(self.sqrt_alpha_cumulative, t=timesteps) * x0  # Image scaled
        std_dev = get(self.sqrt_one_minus_alpha_cumulative, t=timesteps) # Noise scaled
        sample  = mean + std_dev * eps # scaled inputs * scaled noise

        return sample, eps  # return ... , gt noise --> model predicts this)




class GuidedDiffusion:
    def __init__(
        self,
        num_diffusion_timesteps: int =1000,
        img_shape: tuple[int] =(3, 64, 64),
        device: str ="cpu",
        grad_obt_from: str ='generated_img',
        grad_loss_types: list[str] = ['Lc'],
        grad_loss_weights :list[float] = [1.0],
        use_inpaint:bool = False,
        inpaint_type:str='dynamic', #['dynamic','fix']
        fixed_mask:torch.Tensor = None,
        iter_num:str='1/1',
        start_tau:int=100,
        dilation:int=5,
        inpaint_th:float=0.15):
        '''
        Initialize Parameters:
        - num_diffusion_timesteps
        - img_shape
        - device
        - grad_obt_from: Gradient obtained from ['generated img', 'noisy img', 'denoised img']
        - grad_loss_type: The type(s) of loss to get gradient from. Subset of list: ['Lc','L1',]
            - Lc: loss from classifier 
            - L1: L1 loss, bewteen reconstrated img x_t_0 and the original img x_0
            - ...
        - grad_loss_weights: corresponding weight for each loss terms
        - use_inpaint: use inpaint or not
        - start_tau: if using inpaint, the tau to start from
        - dilation: dilation used for inpaint
        - inpaint_th: threshold for inpaint
        '''

        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device
        self.grad_obt_from = grad_obt_from
        self.grad_loss_types = grad_loss_types
        self.grad_loss_weights = grad_loss_weights
        self.use_inpaint = use_inpaint
        self.inpaint_type = inpaint_type
        self.start_tau = start_tau
        self.dilation = dilation
        self.inpaint_th = inpaint_th
        self.iter_num = iter_num
        self.fixed_mask = fixed_mask

        assert not (('L1' in self.grad_loss_types) and (self.grad_obt_from == 'noisy_img')), 'Noisy img can not used for L1 loss.'
        assert len(self.grad_loss_types) == len(self.grad_loss_weights), 'Length of the loss types and the loss weights does not match.'
        assert not ((self.use_inpaint == False) and (self.start_tau == None or self.dilation == None or self.inpaint_th == None)), \
            'when using inpaint, parameters: start_tau, dilation and inpaint_th can not be None.'
        assert not ((self.use_inpaint == True) and (self.inpaint_type == 'fix') and (self.fixed_mask == None)), \
            'when using inpaint and inpaint type == fix, parameters: fixed_mask can not be None.'


        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta  = self.get_betas()
        self.alpha = 1 - self.beta
        
        self.sqrt_beta                       = torch.sqrt(self.beta)
        self.alpha_cumulative                = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative           = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha               = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)
         
    def get_betas(self):
        """linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
        
    def forward_diffusion(self, x0: torch.Tensor, timesteps: torch.Tensor):
        eps = torch.randn_like(x0)  # Noise
        mean    = get(self.sqrt_alpha_cumulative, t=timesteps) * x0  # Image scaled
        std_dev = get(self.sqrt_one_minus_alpha_cumulative, t=timesteps) # Noise scaled
        sample  = mean + std_dev * eps # scaled inputs * scaled noise

        return sample, eps  # return ... , gt noise --> model predicts this)
    

    def model_err_pred_to_mean(self,
                               err: torch.Tensor, 
                               x_t: torch.Tensor, 
                               t: torch.Tensor) -> torch.Tensor:
        """
        Used to calculate the estimate for the mean in the reverse process using the predicted noise

        # formula 13 in improved DDPM paper
        mu_theta(x_t,t) = (1/sqrt(alpha_t) )* (x_t - (beta_t/sqrt(1-alpha_bar_t)) * err_theta(x_t,t))
        
        # formula 9 in improved DDPM paper
        x_0 = (x_t - sqrt(1-alpha_bar_t) * err) / sqrt(alpha_bar_t)

        Return:
        - mu_theta_xt_t:  mu_theta(x_t,t) in formula 13, the estismated mean of x_t given t and predicted err
        - x_0_blur: x_0 in formula 9, as it is reconstrated from the reversed noise implementation, 
                    the image will be a bit blur
        """
        c1 = get(self.one_by_sqrt_alpha, t)
        noise_coef = self.beta / self.sqrt_one_minus_alpha_cumulative
        c2 = get(noise_coef, t)

        mu_theta_xt_t = c1 * (x_t - c2 * err)  # mu_theta_xt_t.shape = [BS, C, H, W]
        x_0_blur = (x_t-err*get(self.sqrt_one_minus_alpha_cumulative,t))/get(self.sqrt_alpha_cumulative,t) # x_0_blur.shape = [BS, C, H, W]

        return mu_theta_xt_t, x_0_blur
    

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alpha_cumulative, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alpha_cumulative, t, x_start.shape)
            * noise
        )
    
    def p_mean_std(self,
                   model: UNet, 
                   x_t: torch.Tensor, 
                   t: torch.Tensor, 
                   y: torch.Tensor=None) -> dict[str, torch.Tensor]:
        """
        Calculate mean and std of p(x_{t-1} | x_t) using the reverse process and model
        Parameter:
        - x_t: shape = [BS, C, H, W]
        - t 
        

        Return:
        - out
            - out['mean']: shape = [BS, C, H, W]
            - out['std']: shape = [BS, 1,1,1] #for simple scheduled std
            - out['denoised']: shape = [BS, C, H, W]
        """
        out = dict()

        model_out = model(x_t,t) # get the noise ftom unet
        # model_out.shape = [BS, C, H, W]

        err = model_out

        # scheduled std
        assert len(x_t.shape) == 4
        out['std'] = get(self.sqrt_beta, t).repeat(x_t.shape[0],1,1,1)
        out['mean'], out['denoised'] = self.model_err_pred_to_mean(err, x_t, t)
        return out
    
    @torch.no_grad()
    def reserve_process_with_start_x_one_img(self,
                                    model: UNet, 
                                    x_t: torch.Tensor,
                                    t_start: int, #  e.g. 100
                                    t_end: int  = 0, # normally 
                                    img_shape: tuple =(3, 64, 64), 
                                    **kwargs):
        '''
        Reverse process (without any guidance) from x_t_start to x_t_end for only one image
        Parameter:
        - model: the UNet model
        - x_t: the image at t
        - t_start: the starting timestep
        - t_end: the end timestep
        - img_shape: 
        - device:

        Return:
        - out_imgs: A list of images 
        - x_end: the reversed img at t_end
        '''

        x = x_t
        model.eval()

        out_imgs = []

        for time_step in (reversed(range(t_end, t_start))):
            ts = torch.ones(1, dtype=torch.long, device=self.device) * time_step
            z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

            predicted_noise = model(x, ts)

            beta_t                            = get(self.beta, ts)
            one_by_sqrt_alpha_t               = get(self.one_by_sqrt_alpha, ts)
            sqrt_one_minus_alpha_cumulative_t = get(self.sqrt_one_minus_alpha_cumulative, ts) 

            x = (
                one_by_sqrt_alpha_t
                * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
                + torch.sqrt(beta_t) * z
            )
            
            
            x_inv = inverse_transform(x).type(torch.uint8)
            out_imgs.append(x_inv)
        
        x_end = x
        return out_imgs, x_end
    
    def get_classifier_loss(self,
                            classifier: ResNet,
                            x_t_0: torch.Tensor,
                            y_target: torch.Tensor,):
        '''
        get the loss of classification
        '''
        out_logits = classifier.forward(x_t_0)
        prob = torch.sigmoid(out_logits)
        class_loss = F.binary_cross_entropy(prob, y_target, reduction='none')
        return class_loss, prob.detach()
    

    def get_l1_loss(self,
                    x_t_0: torch.Tensor,
                    x_0: torch.Tensor):
        '''
        Get L1 loss
        '''
        # reduction have to set to none to compute the gradient seperately 
        loss_func = nn.L1Loss(reduction = 'none')
        
        l1_loss = loss_func(x_t_0,x_0)
        # get the mean of loss for each instance
        l1_loss = l1_loss.view(l1_loss.size(0), -1).mean(1).view(-1,1)
        return l1_loss
    


    def get_guided_gradient(self,
            classifier: ResNet,
            x_t_0: torch.Tensor,
            y_target: torch.Tensor,
            x_0: torch.Tensor,
            ):
        '''
        get the guided grad
        Parameter:
        - classifier: resnet classifier
        - x_t_0: reconstructed img from timestep t, should be a clear one
        - y_target: the target y class

        Return:
        - gradient: the guided gradient, shape =  [BS,C,H,W]
        - prob: the prob of being the target class y (could be moved afterwards)
        - losses: a dict that stores the type of loss and the loss value
        '''
        # cloning reconstructed image to get gradient
        x_t_0_clone = torch.clone(x_t_0)
        x_t_0_clone.requires_grad_(True)
        loss = 0
        prob = None
        losses = {}


        for i,each_type in enumerate(self.grad_loss_types):
            if each_type == 'Lc':
                 # for class loss
                class_loss, prob = self.get_classifier_loss(classifier,x_t_0_clone,y_target)
                loss+= self.grad_loss_weights[i]*class_loss
                losses[each_type] = torch.clone(class_loss).detach()
            elif each_type == 'L1':
                l1_loss = self.get_l1_loss(x_t_0_clone,x_0)
                loss+= self.grad_loss_weights[i]*l1_loss
                losses[each_type] =  torch.clone(l1_loss).detach()
            else:
                raise Exception('Loss type did not implemented: {}'.format(each_type))



        loss.mean().backward()
        gradient = x_t_0_clone.grad.detach()
        return gradient, prob, losses
    

    def p_sample_once(self,
                        model: UNet,
                        classifier: ResNet,
                        timestep: int, 
                        x_t: torch.Tensor, 
                        x_0: torch.Tensor,
                        target: torch.Tensor, 
                        scale: int = 100, 
                        ):

        '''
        sample once, get P{x_{t-1}} from P{x_t}

        Parameter:
        - model: diffusion model
        - classifier: 
        - timestep: the starting timestep to generate cf from
        - x_t: noisy imgs, shape: [BS,C,H,W]
        - x_0: oroiginal imgs, shape: [BS,C,H,W]
        - target: shape: [BS,1]
        - scale: the scale for the gradient decending 

        Return:
        - x_t_0: the recontructed img from x_t
        - x_t_minus_1: x_{t-1}
        - max_grad: max of the grad, for recording and plotting
        - prob: prob of being the target class, for recording and plotting
        '''

        # tensor t
        t = torch.ones(1, dtype=torch.long, device=self.device) * timestep

        # adjust the shape of x_t
        if len(x_t.shape) == 3:
            x_t_unsqueeze = x_t.unsqueeze(0)
        else: x_t_unsqueeze = x_t
        assert len(x_t_unsqueeze.shape) == 4

        # get the mean and std of the x_t
        out = self.p_mean_std(model=model,x_t=x_t_unsqueeze,t=t)

        if self.grad_obt_from == 'generated_img':
            # get x_t_0, the generated img from x_t
            _,x_t_0 = self.reserve_process_with_start_x_one_img(model,
                                                    x_t_unsqueeze,
                                                    t_start = timestep,
                                                    t_end=0,
                                                    img_shape=self.img_shape,)
        elif self.grad_obt_from == 'denoised_img':
            x_t_0 = out['denoised'].detach()
        else:
            raise Exception(f'Not implemented {self.grad_obt_from=}')

        # get the grad from x_t_0 and the classifer
        grad, prob, losses = self.get_guided_gradient(classifier=classifier,
                                x_t_0 = x_t_0,
                                y_target = target,
                                x_0 = x_0)
        
        max_grad = torch.amax(grad, dim=(1,2,3))

        # update the mean and std
        timestep_tensor = torch.as_tensor(timestep, dtype=torch.long).to(self.device)
        cov = get(self.beta, timestep_tensor) 
        one_over_sqrt_alpha_t = 1/torch.sqrt(get(self.alpha, timestep_tensor))
        

        # update out
        out['mean'] = out['mean'] - cov*grad*scale*one_over_sqrt_alpha_t
        out['std'] = out['std'] 

        # get x_t_minus_1
        z = torch.randn_like(x_t) if timestep > 1 else torch.zeros_like(x_t) # the noise
        # z.shape = [BS,C,H,W]
        x_t_minus_1 = out['mean']+ out['std']*z

        if self.use_inpaint:
            if self.inpaint_type == 'dynamic':
                if timestep <= self.start_tau:
                    print(f'{timestep <=self.start_tau=}')
                    dilation = self.dilation
                    inpaint = self.inpaint_th

                    mask, dil_mask = generate_mask(x_0,x_t_0,dilation)
                    boolmask = (dil_mask < inpaint).float()

                    x_t_0 = (x_t_0 * (1-boolmask) + boolmask * x_0)
                    x_t_minus_1 = (x_t_minus_1 * (1-boolmask) + boolmask * self.q_sample(x_0,t))
            elif self.inpaint_type == 'fix':
                boolmask = self.fixed_mask

                x_t_0 = (x_t_0 * (1-boolmask) + boolmask * x_0)
                x_t_minus_1 = (x_t_minus_1 * (1-boolmask) + boolmask * self.q_sample(x_0,t))

        return x_t_0, x_t_minus_1.detach(), max_grad, prob, losses
    
    def show_save(self,
                  x:torch.Tensor,
                  showplot: bool=False,
                  saveplot: bool=False,
                  this_save_path: str= None,
                  save_name: str =None):
        '''
        show or save the img (only one img)

        '''
        assert len(x.shape) == 4
        # assert x.shape[0] == 1

        if showplot and self.device == 'cuda':
            print('WARNING!'*30)
            print('Not being able to show with plt in GPU runs.')

        if showplot and self.device != 'cuda':
            assert x.shape[0] == 1
            plt.figure()
            plt.imshow(x[0].cpu().permute(1,2,0).detach().numpy(),cmap='gray', vmin = -1, vmax = 1)
            plt.axis('off')
            plt.show()

        if saveplot:
            save_imgs_tensor_batch(x=x,
                                   save_path=this_save_path+'{}.png'.format(save_name),
                                   )
        return

    def p_sample_loop(self,
                        model: UNet,
                        classifier: ResNet,
                        timestep: int, # the timestep we start to generate cf from
                        x_t: torch.Tensor, # shape: [BS,C,H,W]
                        x_0: torch.Tensor, # shape: [BS,C,H,W]
                        target: torch.Tensor, # shape: [BS,1]
                        showplot: bool = False,
                        saveplot: bool = False,
                        this_save_path: str = None,
                        scale: int = 100,
                        saveshow_gap: int = 10,
                        ):
        '''
        the loop of guided cf generation
        
        Parameters:
        - model
        - classifier
        - timestep: the starting timestep used to generate cf from
        - ...
        - saveshow_gap: the gap number between two save/shop operations

        Return:
        - x_t_0_list: the list of reconstracted x_t_0
        - x_t_minus_1_list: the list of x_{t-1}
        - max_grad_list: the list of max grad
        - prob_list: the list of prob
        '''

        assert not (saveplot==True and this_save_path==None), 'No save path for saving results!'

        # intialize
        this_x_t = x_t
        batch_size = x_0.shape[0]
        this_scale = scale * batch_size 

        # records
        x_t_0_list = []
        x_t_minus_1_list = []
        max_grad_list = []
        prob_list=[]
        losses_dict={}

        
        

        for each_tstep in tqdm(iterable=reversed(range(1, timestep)), 
                            total=timestep-1, dynamic_ncols=False, 
                            desc="cf generating :: ", position=0):
            
                
            x_t_0, x_t_minus_1,max_grad,prob,losses = self.p_sample_once(model,
                                            classifier,
                                            timestep=each_tstep,
                                            x_t=this_x_t,
                                            x_0 = x_0,
                                            target=target,
                                            scale = this_scale
                                    )
            

            # record
            max_grad_list.append(max_grad)
            prob_list.append(prob.squeeze())

            if each_tstep == timestep-1:
                for each_k in losses:
                    losses_dict[each_k] = [losses[each_k]]
            else:
                for each_k in losses:
                    losses_dict[each_k].append(losses[each_k])

            x_t_0_list.append(x_t_0)
            x_t_minus_1_list.append(x_t_minus_1)

            this_x_t = x_t_minus_1

            if (showplot or saveplot) and ((each_tstep+1) % saveshow_gap==0 or each_tstep==1):
                self.show_save(x_t_0,showplot,saveplot,this_save_path,'x_{}_0'.format(each_tstep))
                if self.iter_num.split('/')[-1] == '2':
                    iter_this = self.iter_num.split('/')[0]
                    self.show_save(x_t_0,showplot,True,this_save_path,'x_{}_0_iter_{}'.format(each_tstep,iter_this))
            elif each_tstep == 1:
                self.show_save(x_t_0,showplot,True,this_save_path,'x_{}_0'.format(each_tstep))
                if self.iter_num.split('/')[-1] == '2':
                    iter_this = self.iter_num.split('/')[0]
                    self.show_save(x_t_0,showplot,True,this_save_path,'x_{}_0_iter_{}'.format(each_tstep,iter_this))
            

        return x_t_0_list, x_t_minus_1_list, max_grad_list, prob_list, losses_dict
    

class GuidedDiffusionTS(GuidedDiffusion):
    def __init__(self, 
                 **kwargs
                 ):
        super().__init__(**kwargs)
    

    def get_classifier_loss(self,
                            classifier: ResNet,
                            x_t_0: torch.Tensor,
                            t:torch.Tensor,
                            y_target: torch.Tensor,):
        '''
        get the loss of classification
        '''
        out_logits = classifier.forward(x_t_0,t)
        prob = torch.sigmoid(out_logits)
        class_loss = F.binary_cross_entropy(prob, y_target, reduction='none')
        return class_loss, prob.detach()

    def get_guided_gradient(self,
            classifier: ResNet,
            x_t_0: torch.Tensor,
            t:torch.Tensor,
            x_t_0_denoise:torch.Tensor,
            y_target: torch.Tensor,
            x_0: torch.Tensor,
            x_t: torch.Tensor,
            model: UNet,
            ):
        '''
        get the guided grad
        Parameter:
        - classifier: resnet classifier
        - x_t_0: reconstructed img from timestep t, should be a clear one
        - x_t_0_denoise: denosied x_0
        - x_0: original image
        - x_t: x_t, noisy image at timestep t (in CF process)
        - y_target: the target y class

        Return:
        - gradient: the guided gradient, shape =  [BS,C,H,W]
        - prob: the prob of being the target class y (could be moved afterwards)
        - losses: a dict that stores the type of loss and the loss value
        '''
        # cloning reconstructed image to get gradient
        x_t_0_clone = torch.clone(x_t_0)
        x_t_0_clone.requires_grad_(True)

        x_t_0_denoise_clone = torch.clone(x_t_0_denoise)
        x_t_0_denoise_clone.requires_grad_(True)




        loss = 0
        prob = None
        losses = {}


        for i,each_type in enumerate(self.grad_loss_types):
            if each_type == 'Lc':
                 # for class loss
                class_loss, prob = self.get_classifier_loss(classifier,x_t_0_denoise_clone,t,y_target)
                loss+= self.grad_loss_weights[i]*class_loss
                losses[each_type] = torch.clone(class_loss).detach()
            elif each_type == 'Denoised_L1':
                denoised_l1_loss = self.get_l1_loss(x_t_0_denoise_clone,x_0)
                loss+= self.grad_loss_weights[i]*denoised_l1_loss
                losses[each_type] =  torch.clone(denoised_l1_loss).detach()
            elif each_type == 'L1':
                l1_loss = self.get_l1_loss(x_t_0_clone,x_0)
                loss+= self.grad_loss_weights[i]*l1_loss
                losses[each_type] =  torch.clone(l1_loss).detach()
            else:
                raise Exception('Loss type did not implemented: {}'.format(each_type))
           

        loss.mean().backward()
        gradient = x_t_0_clone.grad.detach()

        gradient_denoised = x_t_0_denoise_clone.grad.detach()
        
        return gradient+gradient_denoised, prob, losses

    def p_sample_once(self,
                        model: UNet,
                        classifier: ResNet,
                        timestep: int, 
                        x_t: torch.Tensor, 
                        x_0: torch.Tensor,
                        target: torch.Tensor, 
                        scale: int = 100, 
                        ):

        '''
        sample once, get P{x_{t-1}} from P{x_t}

        Parameter:
        - model: diffusion model
        - classifier: 
        - timestep: the starting timestep to generate cf from
        - x_t: noisy imgs, shape: [BS,C,H,W]
        - x_0: oroiginal imgs, shape: [BS,C,H,W]
        - target: shape: [BS,1]
        - scale: the scale for the gradient decending 

        Return:
        - x_t_0: the recontructed img from x_t
        - x_t_minus_1: x_{t-1}
        - max_grad: max of the grad, for recording and plotting
        - prob: prob of being the target class, for recording and plotting
        '''

        # tensor t
        t = torch.ones(1, dtype=torch.long, device=self.device) * timestep

        # adjust the shape of x_t
        if len(x_t.shape) == 3:
            x_t_unsqueeze = x_t.unsqueeze(0)
        else: x_t_unsqueeze = x_t
        assert len(x_t_unsqueeze.shape) == 4

        # get the mean and std of the x_t
        out = self.p_mean_std(model=model,x_t=x_t_unsqueeze,t=t)

        if self.grad_obt_from == 'generated_img':
            # get x_t_0, the generated img from x_t
            _,x_t_0 = self.reserve_process_with_start_x_one_img(model,
                                                    x_t_unsqueeze,
                                                    t_start = timestep,
                                                    t_end=0,
                                                    img_shape=self.img_shape,)
        elif (self.grad_obt_from == 'noisy_img') or (self.grad_obt_from == 'TS_noisy_img'):
            # the classifier is supposed to take the noisy image
            x_t_0 = torch.clone(x_t_unsqueeze).detach()
        elif (self.grad_obt_from == 'denoised_img') or (self.grad_obt_from == 'TS_denoised_img') or (self.grad_obt_from == 'TS0_denoised_img'):
            x_t_0 = out['denoised'].detach()
        else:
            raise Exception(f'Not implemented {self.grad_obt_from=}')
        
        if self.grad_obt_from == 'TS0_denoised_img':
            t_tensor = torch.ones(1, dtype=torch.long, device=self.device) * 0
        else:
            t_tensor = t
        

        x_t_0_denoise = out['denoised'].detach()

        # get the grad from x_t_0 and the classifer
        grad, prob, losses = self.get_guided_gradient(classifier=classifier,
                                x_t_0 = x_t_0,
                                t =t_tensor,
                                x_t_0_denoise = x_t_0_denoise,
                                y_target = target,
                                x_0 = x_0,
                                x_t= x_t_unsqueeze,
                                model = model,)
        
        max_grad = torch.amax(grad, dim=(1,2,3))

        # update the mean and std
        timestep_tensor = torch.as_tensor(timestep, dtype=torch.long).to(self.device)
        cov = get(self.beta, timestep_tensor) 
        one_over_sqrt_alpha_t = 1/torch.sqrt(get(self.alpha, timestep_tensor))
        

        # update out
        out['mean'] = out['mean'] - cov*grad*scale*one_over_sqrt_alpha_t
        out['std'] = out['std'] 

        # get x_t_minus_1
        z = torch.randn_like(x_t) if timestep > 1 else torch.zeros_like(x_t) # the noise
        # z.shape = [BS,C,H,W]
        x_t_minus_1 = out['mean']+ out['std']*z

        if self.use_inpaint:
            if self.inpaint_type == 'dynamic':
                if timestep <= self.start_tau:
                    print(f'{timestep <=self.start_tau=}')
                    dilation = self.dilation
                    inpaint = self.inpaint_th

                    mask, dil_mask = generate_mask(x_0,x_t_0,dilation)
                    boolmask = (dil_mask < inpaint).float()

                    x_t_0 = (x_t_0 * (1-boolmask) + boolmask * x_0)
                    x_t_minus_1 = (x_t_minus_1 * (1-boolmask) + boolmask * self.q_sample(x_0,t))
            elif self.inpaint_type == 'fix':
                boolmask = self.fixed_mask

                x_t_0 = (x_t_0 * (1-boolmask) + boolmask * x_0)
                x_t_minus_1 = (x_t_minus_1 * (1-boolmask) + boolmask * self.q_sample(x_0,t))
            


        if (self.grad_obt_from == 'noisy_img') or (self.grad_obt_from == 'TS_noisy_img'):
            return x_t_0, x_t_minus_1.detach(), max_grad, prob, losses
        else:
            return x_t_0, x_t_minus_1.detach(), max_grad, prob, losses


class GuidedDiffusionTSRescaleT(GuidedDiffusionTS):
    def __init__(self, 
                 rescale:bool,
                 rescaled_totalt:int,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.rescale = rescale
        self.rescaled_totalt = rescaled_totalt
    

    def p_mean_std(self,
                   model: UNet, 
                   *args, **kwargs):
        return super().p_mean_std(self._wrap_model(model),*args, **kwargs )
    
    def reserve_process_with_start_x_one_img(self,
                                    model: UNet, 
                                    *args, **kwargs):
        return super().reserve_process_with_start_x_one_img(self._wrap_model(model),
                                                            *args, **kwargs)

    def get_classifier_loss(self,
                            classifier: ResNet,
                            x_t_0: torch.Tensor,
                            t:torch.Tensor,
                            y_target: torch.Tensor,):
        return super().get_classifier_loss(self._wrap_model(classifier),x_t_0=x_t_0,
                                           t=t,
                                           y_target=y_target)
    
    def get_guided_gradient(self,
            classifier: ResNet,
            x_t_0: torch.Tensor,
            t:torch.Tensor,
            x_t_0_denoise:torch.Tensor,
            y_target: torch.Tensor,
            x_0: torch.Tensor,
            x_t: torch.Tensor,
            model:UNet,
            ):
        return super().get_guided_gradient(self._wrap_model(classifier),
                                           x_t_0=x_t_0,
                                           t=t,
                                           x_t_0_denoise=x_t_0_denoise,
                                           y_target=y_target,
                                           x_0=x_0,
                                           x_t=x_t,
                                           model =model,
                                           )
    
    def p_sample_once(self,
                        model: UNet,
                        classifier: ResNet,
                        *args, **kwargs
                        ):
        return super().p_sample_once(self._wrap_model(model),self._wrap_model(classifier),*args, **kwargs)


    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.rescale, self.rescaled_totalt
        )


class _WrappedModel:
    def __init__(self, model, rescale, rescaled_totalt):
        self.model = model
        self.rescale = rescale
        self.rescaled_totalt = rescaled_totalt

    def __call__(self, x, ts, **kwargs):
        # map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        # new_ts = map_tensor[ts]
        new_ts = ts
        if self.rescale:
            new_ts = new_ts.float() * (1000.0 / self.rescaled_totalt)
            new_ts = new_ts.to(torch.long)
        return self.model(x, new_ts, **kwargs)
    
    def forward(self, x, ts, **kwargs):
        return self.__call__(x,ts,**kwargs)
    
    def eval(self,):
        return self.model.eval()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if isinstance(arr,np.ndarray):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    else:
        res = get(arr,timesteps)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)