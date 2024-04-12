from train.config import TrainingConfig,BaseConfig
import torchvision.transforms as TF

img_size = (TrainingConfig.IMG_SHAPE[1],TrainingConfig.IMG_SHAPE[2])

ORI_transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize(img_size, 
                      interpolation=TF.InterpolationMode.BICUBIC, 
                      antialias=True),
            TF.Lambda(lambda t: (t * 2) - 1) 
        ]
    )

    
CHE_transforms_che = TF.Compose(
        [
            TF.Grayscale(num_output_channels=1),
            TF.ToTensor(),
            TF.Resize(img_size,
                      interpolation=TF.InterpolationMode.BICUBIC, 
                      antialias=True
                ),
            TF.Lambda(lambda t: (t * 2) - 1)
        ]
    )

ISIC_transforms_isic = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize(img_size,
                      interpolation=TF.InterpolationMode.BICUBIC, 
                      antialias=True
                ),
            TF.Lambda(lambda t: (t * 2) - 1)
        ]
    )