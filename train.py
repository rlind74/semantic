import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

# this is important to add in as it imports all the other parts of Unet that are not changed and so you can just update UNet project and all projects will update with latest version of the model.
import sys
sys.path.append('../pytorch-unet')

#from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
#from utils.dataset import BasicDataset # need to update path to below
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

#dir_img = '/db/pszaj/whitefly-eggplant/scaled/'
#dir_img = '/home/rob/Pictures/Mites/'
#dir_img = '/home/rob/Pictures/whitefly_medium/'
#dir_img = '/home/rob/Pictures/whitefly_coomplete/'
#dir_img = '/home/rob/Pictures/maize_counts/'
#dir_img = '/home/rob/Pictures/thrips_leafcradle/patches/'
#dir_img = '/home/rob/Pictures/whitefly_final_trainingset/Adults_small_large/'
dir_img = '/home/rob/Pictures/whitefly_final_trainingset/Adults_scales/' # all these are 256 patches
dir_mask = dir_img
#dir_checkpoint = 'checkpoints_whitefly_tomato_complete_sigma1/'
#dir_checkpoint = 'checkpoints_maize_finaldataset/'
dir_checkpoint = 'checkpoints_WF_adult_scales_with_aphids_warmRestarts/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.0001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=0.00001, last_epoch=-1) # T_0 how many epochs before reset learning rate. T_mult how to change learning rate. e.g. set to 1 to reset back to orginal. Setting to 10 would divide by 10.
    # eta_min lowest possible value the lr can go to to stop it going too low. last_epoch is the epoch you want to start from. -1 is the default to start from scratch. highest lr is the one you set, the lowest is eta_min. need say 10 - 100 difference
    criterion = nn.BCELoss() #nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        framenumber = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for (i,batch) in enumerate (train_loader): # tupole to define i. need this to vary learning rate within and between epochs
                imgs = batch['image']
                true_masks = batch['mask']

                # use this method to output guassian masks only to check.
                # make_mask_output_images(true_masks[0], true_masks, framenumber, 'filename_')

                # use this method to output images + masks + guassian masks to check.
                # make_mask_output_images(imgs[0], true_masks, framenumber, 'ALS_')

                # makes an image to save out to check the guassian masks fit well over objects
                #tmp = imgs[0] # use this to make masks over the original RGB image
                # tmp = true_masks[0] # use this to just make masks to check dimensions

                ##tmp[:,:,:] = 0 # RJL for a single channel 1 class image

                # RJL for a RGB image with 3 classes. If more classes have to choose which to put into RGB channels
                #tmp[0,:,:] += true_masks[0,0,:,:] # channel 1
                #tmp[1,:,:] += true_masks[0,1,:,:] # channel 2
                #tmp[2,:,:] += true_masks[0,2,:,:] # channel 3
                #framenumberstr = str(framenumber)
                #save_image(tmp, 'sigma0point5_64dist.png')  # RJL save out the image tmp into the project so you can check them.
                #save_image(tmp, 'sigma0point5_64dist'+framenumberstr+'.png') # RJL save out the image tmp into the project so you can check them.
                #framenumber = framenumber + 1
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 #if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
#                nn.utils.clip_grad_value_(net.parameters(), 0.1) # comment out
                optimizer.step()
                scheduler.step(epoch + i / n_train)
                #print(scheduler.get_last_lr()) # the lr changes within the epoch not at the epoch!

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                        # put back validation to improve accuracy RJL
#                    val_score = eval_net(net, val_loader, device) # comment out
#                    scheduler.step(val_score) # comment out
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

#                    if net.n_classes > 1: # comment out
#                        logging.info('Validation cross entropy: {}'.format(val_score)) # comment out
#                        writer.add_scalar('Loss/test', val_score, global_step) # comment out
#                    else: # comment out
#                        logging.info('Validation Dice Coeff: {}'.format(val_score)) # comment out
#                        writer.add_scalar('Dice/test', val_score, global_step) # comment out

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


def make_mask_output_images(tmp, true_masks, frameNumber, filename):
    # RJL for a RGB image with 3 classes. If more classes have to choose which to put into RGB channels
    tmp[0, :, :] += true_masks[0, 0, :, :]  # channel 1
    tmp[1, :, :] += true_masks[0, 1, :, :]  # channel 2
    tmp[2, :, :] += true_masks[0, 2, :, :]  # channel 3
    framenumberstr = str(frameNumber)
    save_image(tmp,
               filename + framenumberstr + '.png')  # RJL save out the image tmp into the project so you can check them.


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=3, bilinear=True) # RJL change to number of classes
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
