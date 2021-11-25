import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# this is important to add in as it imports all the other parts of Unet that are not changed and so you can just update UNet project and all projects will update with latest version of the model.
import sys
sys.path.append('../pytorch-unet')

from unet import UNet
#from utils.data_vis import plot_img_and_mask
from data_vis import plot_img_and_mask
#from utils.dataset import BasicDataset
from dataset import BasicDataset

def get_mask(net_output, resize) :
    probs = net_output  # we removed the extra sigmoid

    probs = probs.squeeze(0)


    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(resize),
            transforms.ToTensor()
        ]
    )

    probs = tf(probs.cpu())
    full_mask = probs.squeeze().cpu().numpy()

    return full_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    img_size = img.size()
    print(img_size)
    crop_number = 10
    img_size = (img_size[2], img_size[3]) # height and width as tuple
    if img_size[0]>2048 and img_size[1]>2048 :
        full_mask = torch.zeros(img.size())
        crop_x = img_size[0]//crop_number # a double // converts to an int from the double
        print(img_size)
        crop_y = img_size[1]//crop_number
        with torch.no_grad():
            for x in range(crop_number) :
                for y in range(crop_number) :
                    #print(str(crop_x))
                    full_mask[:,:,x*crop_x : x*crop_x+crop_x, y*crop_y : y*crop_y+crop_y] = net(img[:,:,x*crop_x : x*crop_x+crop_x, y*crop_y : y*crop_y+crop_y] )
    else :
        full_mask = net(img)

    full_mask= get_mask(full_mask, full_img.size[1])

    print(full_mask.max())
    return full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=3)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files): # can input multiple files
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]

            # get mask dimensions for rgb output image
            arrayWidth = mask.shape[1]
            arrayHeight = mask.shape[2]

            # make output rgb to be filled with mask channels
            result_rgb = Image.new("RGB", [arrayWidth, arrayHeight], 255)

            result_r = mask_to_image(mask[0,:,:])
            result_g = mask_to_image(mask[1,:,:])
            result_b = mask_to_image(mask[2,:,:])
            # save out individual channels if needed
            result_r.save(out_files[i] + '0.png')
            result_g.save(out_files[i] + '1.png')
            result_b.save(out_files[i] + '2.png')


            # use the split and merge commands to unpack and pack rgb images
            result_rgb = Image.merge('RGB', (result_r, result_g, result_b))

            result_rgb.save(out_files[i] + 'ALL.png') # save out rgb mask

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
