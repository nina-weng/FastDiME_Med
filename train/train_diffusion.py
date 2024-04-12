import sys
sys.path.append('../FastDiME_Med/')
print(sys.path)

from train.config import BaseConfig,TrainingConfig,ModelConfig,save_config_train_diffusion
from train.utils import *
from train.dataloader import get_dataloader,inverse_transform

from models.unet import UNet
from models.diffusion import SimpleDiffusion



import gc
import os
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.cuda import amp
import torchvision.transforms as TF
from torchvision.utils import make_grid



from torchmetrics import MeanMetric

from IPython.display import display




# Algorithm 1: Training
def train_one_epoch(model, sd, loader, optimizer, scaler, loss_fn, epoch=800, 
                   base_config=BaseConfig(), training_config=TrainingConfig()):
    
    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")
         
        for x0s, _ in loader:
            tq.update(1)
            
            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=base_config.DEVICE)
            xts, gt_noise = sd.forward_diffusion( x0s, ts)

            with amp.autocast():
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.6f}")

        mean_loss = loss_record.compute().item()
    
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.6f}")
    
    return mean_loss 


# Algorithm 2: Sampling
@torch.no_grad()
def reverse_diffusion(model, sd, timesteps=1000, img_shape=(3, 64, 64), 
                      num_images=5, nrow=8, device="cpu", **kwargs):

    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()

    if kwargs.get("generate_video", False):
        outs = []

    for time_step in tqdm(iterable=reversed(range(1, timesteps)), 
                          total=timesteps-1, dynamic_ncols=False, 
                          desc="Sampling :: ", position=0):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)

        beta_t                            = get(sd.beta, ts)
        one_by_sqrt_alpha_t               = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts) 

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z # noise term
        )

        if kwargs.get("generate_video", False):
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            outs.append(ndarr)


    if kwargs.get("generate_video", False): # Generate and save video of the entire reverse process. 
        frames2vid(outs, kwargs['save_path'])
        display(Image.fromarray(outs[-1][:, :, ::-1])) # Display the image at the final timestep of the reverse process.
        return None

    else: # Display and save the image at the final timestep of the reverse process. 
        x = inverse_transform(x).type(torch.uint8)
        grid = make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        pil_image = TF.functional.to_pil_image(grid)
        pil_image.save(kwargs['save_path'], format=save_path[-3:].upper())
        display(pil_image)
        return None



if __name__ == '__main__':
    
    print(f'Device: {BaseConfig.DEVICE}')

    log_dir, checkpoint_dir, prevs_dir, version_name = setup_log_directory(config=BaseConfig())

    # save the config
    save_config_train_diffusion(TrainingConfig(),ModelConfig(),BaseConfig(),save_dir = checkpoint_dir)

    # visualize the dataset
    print('^'*30+ 'Visualize the dataset')
    loader = get_dataloader(
        dataset_name=BaseConfig.DATASET,
        batch_size=128,
        device='cpu',
        )

    plt.figure(figsize=(24, 12), facecolor='white',)

    for b_image, _ in loader:
        b_image = inverse_transform(b_image).cpu()
        grid_img = make_grid(b_image / 255.0, nrow=16, padding=True, pad_value=1, normalize=True)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis("off")
        break

    plt.savefig(os.path.join(prevs_dir,'visualize_ds.png'),dpi=300)

    # check the forward process
    print('^'*30+ 'Check the forward process')
    sd = SimpleDiffusion(num_diffusion_timesteps=TrainingConfig.TIMESTEPS, device="cpu")

    loader = iter(  # converting dataloader into an iterator for now.
        get_dataloader(
            dataset_name=BaseConfig.DATASET,
            batch_size=6,
            device="cpu",
        )
    )

    x0s, _ = next(loader)

    noisy_images = []
    if TrainingConfig.TIMESTEPS == 1000:
        specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]
    elif TrainingConfig.TIMESTEPS == 200:
        specific_timesteps = [0, 5, 10, 25, 50, 100, 150, 199]
    else:
        raise Exception('Not implemented timesteps')

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long)

        xts, _ = sd.forward_diffusion(x0s, timestep)
        xts = inverse_transform(xts) / 255.0
        xts = make_grid(xts, nrow=1, padding=1)

        noisy_images.append(xts)

    # Plot and see samples at different timesteps

    _, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor="white")

    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.9)
    plt.axis("off")
    plt.savefig(os.path.join(prevs_dir,'forward_diffusion_process_example.png'))

    # training
    print('^'*30+'Training')

    model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
    model.to(BaseConfig.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)

    dataloader = get_dataloader(
        dataset_name  = BaseConfig.DATASET,
        batch_size    = TrainingConfig.BATCH_SIZE,
        device        = BaseConfig.DEVICE,
        pin_memory    = True,
        num_workers   = TrainingConfig.NUM_WORKERS,
    )

    loss_fn = nn.MSELoss()

    sd = SimpleDiffusion(
        num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
        img_shape               = TrainingConfig.IMG_SHAPE,
        device                  = BaseConfig.DEVICE,
    )

    scaler = amp.GradScaler()

    total_epochs = TrainingConfig.NUM_EPOCHS + 1

    generate_video = False
    ext = ".mp4" if generate_video else ".png"

    for epoch in range(1, total_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        
        # Algorithm 1: Training
        train_one_epoch(model, sd, dataloader, optimizer, scaler, loss_fn, epoch=epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(f'Sampling at {epoch=}')
            save_path = os.path.join(log_dir, f"{epoch}{ext}")
            
            # Algorithm 2: Sampling
            reverse_diffusion(model, sd, timesteps=TrainingConfig.TIMESTEPS, num_images=TrainingConfig.SAMPLE_NUM, generate_video=generate_video,
                save_path=save_path, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE,
            )

            # clear_output()
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model": model.state_dict()
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
            del checkpoint_dict


    # inference
    print('^'*30+'Inference')
    model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )

    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "ckpt.tar"), map_location='cpu')['model'])

    model.to(BaseConfig.DEVICE)

    sd = SimpleDiffusion(
        num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
        img_shape               = TrainingConfig.IMG_SHAPE,
        device                  = BaseConfig.DEVICE,
    )

        
    generate_video = False

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)

    reverse_diffusion(
        model,
        sd,
        num_images=TrainingConfig.SAMPLE_NUM,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=32,
    )
    print(save_path)

    