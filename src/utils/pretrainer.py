import wandb
import segmentation_models_pytorch as smp
from .pretrain_utils import TrainEpoch, ValidEpoch
from .loss import ContrastiveLoss as custom_loss
from .dataloader import Pretrain_Dataset as Dataset
from .transformations import get_pretraining_augmentation, get_validation_augmentation, get_preprocessing
from .model import Unet
import torch
from torch.utils.data import DataLoader
def pretrain(epochs, batch_size, hr_dir, tar_dir, th_dir, hr_val_dir, tar_val_dir, th_val_dir,encoder='resnet34', encoder_weights='imagenet', device='cuda', lr=1e-4 ):

    activation = 'tanh' 
    # create segmentation model with pretrained encoder
    model = Unet(
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
        encoder_depth = 5,
        classes=1, 
        activation=activation,
        contrastive=True,
        pretrain=True
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    train_dataset = Dataset(
        hr_dir,
        th_dir,
        tar_dir,
        augmentation=get_pretraining_augmentation(), 
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
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    loss = custom_loss(batch_size)
    loss.__name__='custom_loss'
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=lr),
    ])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,250)
    train_epoch = TrainEpoch(
        model, 
        loss=loss, 
        optimizer=optimizer,
        device=device,
        verbose=True,
        contrastive=True
    )
    valid_epoch = ValidEpoch(
        model, 
        loss=loss, 
        device=device,
        verbose=True,
        contrastive=True
    )

    min_loss = 100000
    counter = 0
    for i in range(0, epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        print(train_logs)
        wandb.log({'pr_epoch':i+1,'pre_t_loss':train_logs['custom_loss'],'pre_v_loss':valid_logs['custom_loss']})
        # do something (save model, change lr, etc.)
        if min_loss >= valid_logs['custom_loss']:
            min_loss = valid_logs['custom_loss']
            wandb.config.update({'min_loss':min_loss}, allow_val_change=True)
            torch.save(model.encoder.state_dict(), './encoder.pth')
            torch.save(model.encoder2.state_dict(), './encoder2.pth')
            print('encoders saved!')
            counter = 0
        counter = counter+1
        if counter>10:
            break
    del model
    torch.cuda.empty_cache()

def pre_train_model(configs):
    pretrain(configs['epochs'], configs['batch_size'], configs['hr_dir'],
         configs['tar_dir'], configs['th_dir'], configs['hr_val_dir'],
         configs['tar_val_dir'], configs['th_val_dir'], configs['encoder'],
         configs['encoder_weights'], configs['device'], configs['lr'])
         