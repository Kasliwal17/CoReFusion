import wandb
import segmentation_models_pytorch as smp
from .train_utils import TrainEpoch, ValidEpoch
from .loss import custom_loss
from .dataloader import Dataset
from .transformations import get_training_augmentation, get_validation_augmentation, get_preprocessing
from .model import Unet
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import torch
from torch.utils.data import DataLoader
def train(epochs, batch_size, hr_dir, tar_dir, th_dir, hr_val_dir, tar_val_dir, th_val_dir,encoder='resnet34', encoder_weights='imagenet', device='cuda', lr=1e-4, pretrain =False, checkpoint='' ):

    activation = 'tanh' 
    # create segmentation model with pretrained encoder
    model = Unet(
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
        encoder_depth = 5,
        classes=1, 
        activation=activation,
        contrastive=True,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    train_dataset = Dataset(
        hr_dir,
        th_dir,
        tar_dir,
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    valid_dataset = Dataset(
        hr_val_dir,
        th_val_dir,
        tar_val_dir,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)#, drop_last=True)

    loss = custom_loss(batch_size, pretrain=pretrain)
    lossV = custom_loss(batch_size, pretrain=True)

    Z = StructuralSimilarityIndexMeasure()
    P = PeakSignalNoiseRatio()
    P.__name__ = 'psnr'
    Z.__name__ = 'ssim'
    metrics = [
        Z,
        P,
    ]

    
    if pretrain:
        model.encoder.load_state_dict(torch.load('./encoder.pth', map_location=torch.device(device)))
        model.encoder2.load_state_dict(torch.load('./encoder2.pth', map_location=torch.device(device)))
        print('Encoder weights loaded!')
        ##define optimizer excluding model.encoder and model.encoder2
        optimizer = torch.optim.Adam([
            dict(params=model.decoder.parameters(), lr=lr),
            dict(params=model.segmentation_head.parameters(), lr=lr),
        ])
    else:
        optimizer = torch.optim.Adam([ 
            dict(params=model.parameters(), lr=lr),
        ])
        if checkpoint != '':
            model.load_state_dict(torch.load(checkpoint, map_location=torch.device(device)))
            print('Model weights loaded!')

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,250)
    train_epoch = TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
        contrastive=True
    )
    valid_epoch = ValidEpoch(
        model, 
        loss=lossV, 
        metrics=metrics, 
        device=device,
        verbose=True,
        contrastive=True
    )

    max_ssim = 0
    max_psnr = 0
    counter = 0
    for i in range(0, epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        # scheduler.step()
        print(train_logs)
        wandb.log({'epoch':i+1,'t_loss':train_logs['custom_loss'],'t_ssim':train_logs['ssim'],'v_loss':valid_logs['custom_loss'],'v_ssim':valid_logs['ssim']})
        # do something (save model, change lr, etc.)
        if max_ssim <= valid_logs['ssim']:
            max_ssim = valid_logs['ssim']
            max_psnr = valid_logs['psnr']
            wandb.config.update({'max_ssim':max_ssim, 'max_psnr':max_psnr}, allow_val_change=True)
            torch.save(model.state_dict(), './best_model.pth')
            print('Model saved!')
            counter = 0
        counter = counter+1
        if counter>50:
            break
    print(f'max ssim: {max_ssim} max psnr: {max_psnr}')
    del model
    torch.cuda.empty_cache()

def train_model(configs):
    train(configs['epochs'], configs['batch_size'], configs['hr_dir'],
         configs['tar_dir'], configs['th_dir'], configs['hr_val_dir'],
         configs['tar_val_dir'], configs['th_val_dir'], configs['encoder'],
         configs['encoder_weights'], configs['device'], configs['lr'], configs['pretrain'])
         
