from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import datasets, models, transforms

import random
import math
from skimage.transform import AffineTransform, warp
import copy
from torchvision.utils import save_image

import sys
import numpy
from PIL import Image

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not (file.startswith('.') or mask_suffix in file)
                    and file.endswith('.png')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW (converts from Height/Width/Channel to Channel/Height/Width
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def preprocessMask(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)



        #if len(img_nd.shape) == 2:
        #    img_nd = np.expand_dims(img_nd, axis=2)

        #print(img_nd.shape)

        # HWC to CHW (converts from Height/Width/Channel to Channel/Height/Width
        #img_trans = img_nd.transpose((2, 0, 1))
        #if img_trans.max() > 1:
        #    img_trans = img_trans / 255

        #print(img_trans.shape)

        return img_nd

    def augment(self, img, mask):
        txy = (random.randint(-40, 40), random.randint(-40, 40))
        rot = random.random() * math.pi * 2  # 0 to 2*pi
        scale = 1 - (random.random() * 0.4 - 0.2)  # 0.8 to 1.2

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

        affine = AffineTransform(matrix=np.linalg.inv(m_T))

        for c in range(3):
            img[c, :, :] = warp(img[c, :, :], affine, clip=False, preserve_range=True)

        # Apply some colour channel scaling
        #for i in range(3):
        #    channel_mul = 1 + (random.random() * 1.0 - 0.5)
        #    img[i, :, :] *= channel_mul
        #    img[i, :, :] = img[i, :, :].clip(0, 1)

        # adjust mask for same transform already been applied to the img

        mask = warp(mask, affine, clip=False, preserve_range=True)

            # dont apply colour transform for mask as need the classes (!)


        return img, mask


    def augmentColour(self, img):


        # Apply some colour channel scaling
        for i in range(3):
            #channel_mul = 1 + (random.random() * 1.0 - 0.5) # 0.5 to 1.5
            #channel_mul = 1 + (random.random() * 1.0 - 0.8) # 0.2 to 1.2 for darker plants on edge and corners
            #channel_mul = 1 + (random.random() * 1.0 - 0.1) # 0.9 to 1.1 for darker plants on edge and corners
            channel_mul = 1 - (random.random() * 0.4 - 0.2)  # 0.8 to 1.2 # RJL
            img[i, :, :] *= channel_mul
            img[i, :, :] = img[i, :, :].clip(0, 1)

        # adjust mask for same transform already been applied to the img

        #mask = warp(mask, affine, clip=False, preserve_range=True)

            # dont apply colour transform for mask as need the classes (!)


        return img

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.png')
        img_file = glob(self.imgs_dir + idx + '.png')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        rotation_degree = random.random()*360
        img = transforms.functional.rotate(img, rotation_degree, interpolation=transforms.InterpolationMode.BICUBIC) # bicubic to give smooth RGB image rotation
        mask = transforms.functional.rotate(mask, rotation_degree, interpolation=transforms.InterpolationMode.NEAREST) # nearest neighbour to avoid interpolation errors!

        #scale_percent = 1 - (random.random() * 0.4 - 0.2)
        #scale_percent = 200
        #img = transforms.Resize(img, scale_percent, interpolation=transforms.InterpolationMode.NEAREST)
        #mask = transforms.Resize(mask, scale_percent, interpolation=transforms.InterpolationMode.NEAREST)

        img = self.preprocess(img, self.scale)
        # mask = self.preprocess(mask, self.scale) > (1/255) # for stanton images plants were greater than 1 # needs changing RJL
        mask = self.preprocessMask(mask, self.scale) # mask should be in the 3 classes of 0,1,2

        # augment colour before tranforming image
        img = self.augmentColour(img)  # RJL might need two different methods. Import to not allow mask to take in interpolated values

        #img, mask = self.augment(img, mask) # RJL might need two different methods. Import to not allow mask to take in interpolated values

        #Image image = mask

        # need to put mask back into 0,1,2 values as converted to floats due to the transformation!
        # round numbers

        #image = Image.open(mask_file[0])

        #print(mask.format)
        #print(mask.size)
        #print(mask.mode)

        # RJL Oct 14th update mask
        # need to generate an array that has 5 channels (1 per class) and has class detection as 255.
        # need to repack the png mask values into the different channels




        # img, mask = self.augment(img, mask) # RJL might need two different methods. Import to not allow mask to take in interpolated values

        # set up heatmap array to import the png mask into. default of zero values
        heatmap = np.zeros((3, 270, 270))

        # when building masks the values run from 0 to 1! not 0-255!

        for y in range (269):
            for x in range (269) :
                value = mask[x,y]
                if value==2 :
                    heatmap[2,x,y] = 1 # bleach
                if value==1:
                    heatmap[1,x,y] = 1 # healthy leaves
                if value==0 :
                    heatmap[0,x,y] = 1 # background




        #print(heatmap.shape)
        #heatmap = np.squeeze(heatmap)


        #print(heatmap.shape)

        #for i in range (100):
        #imArrayInt = imgArray.astype(int) # converts a float into ints

        #print(len(imArrayInt))

        #array = numpy.arange(0, picArraySize, 1, numpy.uint8)

        # repack array for mask with ints
        # after a rotation and scaling can get a range of values.
        # have to sample image before and after transformation to put back only values in the before image!
        #for i in range (picArraySize) :
            #array[i] = imArrayInt[i]
            #array[i] = numpy.around(imgArray[i]) # rounds the data rather than just convert 32bit to int
            #array[i] = numpy.floor(imgArray[i]) # floor
        # check that only ints are in the new image
        #count0 = 0
        #count1 = 0
        #count2 = 0
        #count3 = 0
        #count4 = 0
        #for i in range (picArraySize) :
        #    if (array[i]==0) :
        #        count0 = count0 + 1
        #    if (array[i] == 1):
        #        count1 = count1 + 1
        #    if (array[i] == 2):
        #        count2 = count2 + 1
        #    if (array[i] == 3):
        #        count3 = count3 + 1
        #    if (array[i] == 4):
        #        count4 = count4 + 1

        # print(count0, count1, count2, count3, count4, count0+count1+count2+count3+count4)

        # now update array into right shape
        #mask = numpy.reshape(array, (size, size))

        #mask = Image.fromarray(array)

        # reshape back into the mask image
        #im = numpy.reshape(im, (512,512))
        #mask = Image.fromarray(im)

        #for i in range(100):
        #    print(im[i])
        #result = Image.fromarray(im.astype(numpy.uint16))

        #result = Image.fromarray(round(im)).astype(numpy.uint16)
        #print(im.max(), im.min(), im.mean())
        #fft_mag = numpy.abs(numpy.fft.fftshift(numpy.fft.fft2(im)))

        #visual = numpy.log(fft_mag)
        #visual = (visual - visual.min()) / (visual.max() - visual.min())

        #result = Image.fromarray((visual * 255).astype(numpy.uint8))
        #image.save('out.bmp')

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(heatmap).type(torch.FloatTensor) # return the packed heatmap with 3 channels
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

class StantonDataset(BasicDataset):
    def __init__(self, imgs_dir):
        super().__init__(imgs_dir, imgs_dir, 1, mask_suffix='_mask') # images are 1024 x 1024 to allow for good pooliing/down sizing
