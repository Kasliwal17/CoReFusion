import argparse
from .utils.trainer import train_model
from .utils.pretrainer import pre_train_model
import wandb
import torch

def main(args):
    config = {
        'hr_dir': args.hr_dir,
        'th_dir': args.th_dir,
        'tar_dir': args.tar_dir,
        'hr_val_dir': args.hr_val_dir,
        'th_val_dir': args.th_val_dir,
        'tar_val_dir': args.tar_val_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'device':args.device,
        'encoder': args.encoder,
        'encoder_weights': args.encoder_weights,
        'lr': args.lr,
        'pretrain': True if args.pretrain =='True' else False,
        'checkpoint': args.checkpoint
    }
    wandb.init(project="ThermalSuperResolutionN", entity="kasliwal17",
               config={'model':'resnet34 d5','fusion_technique':'img 2 encoders concat decoder*2 avg tanh x+p/10+z/100+y/10 saving:ssim',
                'lr':args.lr, 'max_ssim':0, 'max_psnr':0, 'min_loss':1000}, allow_val_change=True)
    if args.pretrain:
        pre_train_model(config)
        print('Pretraining done')
        print('Now training model with frozen encoders')
        train_model(config)
        torch.cuda.synchronize()
    config['pretrain'] = False
    print('Training model with unfrozen encoders')
    train_model(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=False, default='./Dataset/train_val/training/train_VIS_HR')
    parser.add_argument('--th_dir', type=str, required=False, default='./Dataset/train_val/training/train_input_THER_LR_bicubic/X8')
    parser.add_argument('--tar_dir', type=str, required=False, default='./Dataset/train_val/training/train_output_gt_THER_HR')
    parser.add_argument('--hr_val_dir', type=str, required=False, default='./Dataset/train_val/validation/valid_VIS_HR')
    parser.add_argument('--th_val_dir', type=str, required=False, default='./Dataset/train_val/validation/valid_input_THER_LR_bicubic/X8')
    parser.add_argument('--tar_val_dir', type=str, required=False, default='./Dataset/train_val/validation/valid_output_gt_THER_HR')
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--epochs', type=int, required=False, default=250)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--encoder', type=str, required=False, default='resnet34')
    parser.add_argument('--encoder_weights', type=str, required=False, default='imagenet')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--pretrain', type=str, required=False, default='False')
    parser.add_argument('--checkpoint', type=str, required=False, default='./best_model.pth')
    arguments = parser.parse_args()
    main(arguments)
