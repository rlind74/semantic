from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

import random
import math
from skimage.transform import AffineTransform, warp
import copy

# this is important to add in as it imports all the other parts of Unet that are not changed and so you can just update UNet project and all projects will update with latest version of the model.
import sys
sys.path.append('../Pytorch-UNet')


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.') and file.endswith('.jpg')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod # use this if all objects have the same size heatmap
    def draw_gaussian(self, heatmap, xloc, yloc, sigma=0.2, mu=0, channel=0):
        x, y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
        dst = np.sqrt(x * x + y * y)
        gauss = np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2)))
        gauss = (gauss - np.min(gauss)) / np.ptp(gauss)  # ptp returns max - min
        heatmap[channel, (xloc - 16):(xloc + 16), (yloc - 16):(yloc + 16)] += gauss

    @classmethod # use this if you want different size heatmaps for the different object classes, e.g. whitefly adults vs scales
    def draw_gaussian_channel(self, heatmap, xloc, yloc, channel, sigma=0.2, mu=0): # default mu is 0, default sigma is 0.2
        #x, y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32)) #  defines the extend of the mask, may need to increase this.
        #x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64)) #  defines the extend of the mask, may need to increase this.
        #dst = np.sqrt(x * x + y * y)
        if channel == 0:
            x, y = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(-1, 1, 40))  # defines the extend of the mask, may need to increase this.whitefly
            dst = np.sqrt(x * x + y * y)
            #gauss = np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2)))  # original. mu is the mean
            gauss = np.exp(-((dst - mu) ** 2 / (2.0 * (sigma+0.2) ** 2))) # whitefly
            #gauss = np.exp(-((dst - mu) ** 2 / (2.0 * (sigma + 0.6) ** 2)))  # BIG whitefly
            #gauss = np.exp(-((dst - mu) ** 2 / (2.0 * (sigma + 1.8) ** 2)))  # SUPER BIG whitefly
            #gauss = np.exp(-((dst - mu) ** 2 / (2.0 * (sigma) ** 2))) # maize plants large
            gauss = (gauss - np.min(gauss)) / np.ptp(gauss)  # ptp returns max - min
            # heatmap[channel, (xloc - 16):(xloc + 16), (yloc - 16):(yloc + 16)] += gauss # original size
            heatmap[channel, (xloc - 20):(xloc + 20), (yloc - 20):(yloc + 20)] += gauss # whitefly

        if channel == 1:
            x, y = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(-1, 1, 40))
            dst = np.sqrt(x * x + y * y)
            gauss = np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2)))  # original. mu is the mean
            gauss = (gauss - np.min(gauss)) / np.ptp(gauss)  # ptp returns max - min
            # heatmap[channel, (xloc - 16):(xloc + 16), (yloc - 16):(yloc + 16)] += gauss # original size
            heatmap[channel, (xloc - 20):(xloc + 20), (yloc - 20):(yloc + 20)] += gauss
        if channel ==2:
            x, y = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(-1, 1, 40))
            dst = np.sqrt(x * x + y * y)
            gauss = np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2))) # original
            #gauss = np.exp(-((dst - mu) ** 2 / (2.0 * (sigma+0.2) ** 2))) # BIG whitefly
            #gauss = np.exp(-((dst - mu) ** 2 / (2.0 * (sigma + 0.8) ** 2)))  # SUPER BIG whitefly
            gauss = (gauss - np.min(gauss)) / np.ptp(gauss)  # ptp returns max - min
            # heatmap[channel, (xloc - 16):(xloc + 16), (yloc - 16):(yloc + 16)] += gauss # original size
            heatmap[channel, (xloc - 20):(xloc + 20), (yloc - 20):(yloc + 20)] += gauss
        #gauss = (gauss - np.min(gauss)) / np.ptp(gauss)  # ptp returns max - min
        # heatmap[channel, (xloc - 16):(xloc + 16), (yloc - 16):(yloc + 16)] += gauss # original size
        #heatmap[channel, (xloc - 32):(xloc + 32), (yloc - 32):(yloc + 32)] += gauss

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def augment(self, img, pts):
        txy = (random.randint(-40, 40), random.randint(-40, 40))
        rot = random.random() * math.pi * 2  # 0 to 2*pi
        #scale = 1 - (random.random() * 0.4 - 0.2)  # 0.8 to 1.2 # RJL go tighter for consistent imagery like leaf cradle?
        scale = 1 - (random.random() * 0.1 - 0.05)  # 0.95 to 1.05 # RJL go tighter for consistent imagery like leaf cradle?
        #scale = 1 # no scaling!

        m_shift = np.array([[1, 0, -img.shape[1] / 2],
                            [0, 1, -img.shape[2] / 2],
                            [0, 0, 1]])

        m_rot = np.array([[math.cos(rot), math.sin(rot), 0],
                          [-math.sin(rot), math.cos(rot), 0],
                          [0, 0, 1]])
        m_scale = np.eye(3) * scale
        m_scale[2, 2] = 1
        m_txy = np.array([[1, 0, txy[0]],
                          [0, 1, txy[1]],
                          [0, 0, 1]])

        # inv(m_shift) * m_scale * m_rot * m_rxy * m_shift
        m_T = np.matmul(np.linalg.inv(m_shift),
                        np.matmul(m_scale, np.matmul(m_rot, np.matmul(m_txy, m_shift))))

        # classes = pts[:,2] # save this for later. RJL this was the problem why classes 1 and 2 were combined!
        classes = copy.copy(pts[:, 2])

        pts[:, 2] = 1

        pts = np.matmul(m_T, pts.transpose()).transpose()  # -> Nx3 transformed

        pts[:, 2] = classes  # restore classes after transform

        affine = AffineTransform(matrix=np.linalg.inv(m_T))

        for c in range(3):
            img[c, :, :] = warp(img[c, :, :], affine, clip=False, preserve_range=True)

        pts = np.round(pts).astype(int)

        # Apply some colour channel scaling
        for i in range(3):
            #channel_mul = 1 + (random.random() * 1.0 - 0.5) # 0.5 to 1.5
            #channel_mul = 1 + (random.random() * 1.0 - 0.5) # 0.5 to 1.5 # default wide
            #channel_mul =  1 + (random.random() * 1.0 - 0.8) # 0.8 to 1.2
            channel_mul = 1 - (random.random() * 0.4 - 0.2) # 0.8 to 1.2 # RJL
            #channel_mul = 1 # no colour augmentation
            img[i, :, :] *= channel_mul # each channel multiplied
            img[i, :, :] = img[i, :, :].clip(0, 1) # clip so all values between 0 and 1

        return img, pts

    def __getitem__(self, i):
        idx = self.ids[i]
        pts_file = glob(self.masks_dir + idx + '.txt')
        img_file = glob(self.imgs_dir + idx + '.jpg')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file[0])
        pts = np.loadtxt(open(pts_file[0], "rb"), delimiter=",").astype(int)
        if pts.shape[0] == 3 and len(pts.shape) < 2:
            pts2 = np.zeros((1, 3))
            pts2[0, :] = pts
            pts = pts2

        if pts.shape[0] == 0:
            pts = np.zeros((0, 3))

        img = self.preprocess(img, 1)

        img, pts = self.augment(img, pts)

        pad = 32
        heatmap = np.zeros((3, (pad * 2) + img.shape[1], (pad * 2) + img.shape[2]))

        # draw gauss function indicies python style, so x refers to row and y column. use y, x
        for p in pts:
            if p[0] > 0 and p[1] > 0 and p[0] < img.shape[2] and p[1] < img.shape[1]:
                #self.draw_gaussian(heatmap, pad + p[1], pad + p[0], channel=p[2])
                self.draw_gaussian_channel(heatmap, pad + p[1], pad + p[0], channel=p[2])

        # for p in pts:
        #    if p[0] > 0 and p[1] > 0 and p[0] < img.shape[2] and p[1] < img.shape[1]:
        #        self.draw_gaussian(heatmap, pad + p[1], pad + p[0], channel=p[2])

        heatmap = heatmap[:, pad:-pad, pad:-pad]
        #        heatmap = np.expand_dims(heatmap, axis=2)
        #        heatmap = heatmap.transpose((2, 0, 1)) # numpy squashes first dimension

        # resize img and heatmap by taking off another 64 pixels. This takes the 384x384 patch down to 256x256 and copes with pts at edges and removes much black areas due to rotation. Might miss very edges of image.
        heatmap = heatmap[:, pad:-pad, pad:-pad]
        img = img[:, pad:-pad, pad:-pad]

        #print(heatmap)

        # need to normalise RGB images!


        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(heatmap).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
