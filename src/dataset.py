import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils

def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:   output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        color_img, depth_img, edge_img,edge_c, normal_img =  sample['color'], sample['depth'],sample['edge'],sample['edge_pix'], sample['normal']
        h, w = color_img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        color_im = transform.resize(color_img, (new_h, new_w), preserve_range=True)
        depth_im = transform.resize(depth_img, (new_h, new_w), preserve_range=True)
        normal_im = transform.resize(normal_img, (new_h, new_w), preserve_range=True)
        edge_im = transform.resize(edge_img, (new_h, new_w), preserve_range=True)
        #  print edge_im.max(), edge_im.min(), np.count_nonzero(edge_im), np.sum(edge_img[edge_img == 1])
        return {'color':color_im, 'depth':depth_im, 'edge':edge_im,'edge_pix': edge_c, 'normal':normal_im}


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:   output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = (output_size, output_size)

    def __call__(self, sample):
        color_img, depth_img, edge_img,edge_c, normal_img =  sample['color'], sample['depth'],sample['edge'],sample['edge_pix'], sample['normal']
        key = 1
        try:
            mask_img = sample['mask']
        except KeyError:
            key = 0

        h, w = color_img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        color_im = color_img[top: top + new_h, left: left + new_w]
        depth_im = depth_img[top: top + new_h, left: left + new_w]
        edge_im = edge_img[top: top + new_h, left: left + new_w]
        normal_im = normal_img[top: top + new_h, left: left + new_w]

        if key == 0:
            return {'color':color_im, 'depth':depth_im, 'edge':edge_im,'edge_pix': edge_c, 'normal':normal_im}
        elif key==1:
            mask_im = mask_img[top: top + new_h, left: left + new_w]
            return {'color':color_im, 'depth':depth_im, 'edge':edge_im,'edge_pix': edge_c, 'normal':normal_im,
                    'mask':mask_im}

class CenterCrop(object):
    """Crop centor the image in a sample.
    Args:   output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        color_img, depth_img, edge_img,edge_c, normal_img =  sample['color'], sample['depth'],sample['edge'],sample['edge_pix'], sample['normal']
        key = 1
        try:
            mask_img = sample['mask']
        except KeyError:
            key = 0

        h, w = color_img.shape[:2]
        new_h, new_w = self.output_size

        top = int((h - new_h) / 2)
        left = int((w - new_w) / 2)

        color_im = color_img[top: top + new_h, left: left + new_w]
        depth_im = depth_img[top: top + new_h, left: left + new_w]
        edge_im = edge_img[top: top + new_h, left: left + new_w]
        normal_im = normal_img[top: top + new_h, left: left + new_w]
        if key == 0:
            return {'color':color_im, 'depth':depth_im, 'edge':edge_im,'edge_pix': edge_c, 'normal':normal_im}
        elif key==1:
            mask_im = mask_img[top: top + new_h, left: left + new_w]
            return {'color':color_im, 'depth':depth_im, 'edge':edge_im,'edge_pix': edge_c, 'normal':normal_im,
                    'mask':mask_im}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        color_img, depth_img, edge_img,edge_c, normal_img =  sample['color'], sample['depth'],sample['edge'],sample['edge_pix'], sample['normal']
        key = 1
        try:
            mask_img = sample['mask']
        except KeyError:
            key = 0

        # swap color axis because
        # numpy image: H x W x C  --> torch image: C x H x W (C=1,3)
        color_img = color_img.transpose((2, 0, 1))
        normal_img = normal_img.transpose((2, 0, 1))
        h,w = depth_img.shape
        depth_img = depth_img.reshape((1, h, w))
        edge_img = edge_img.reshape((1, h, w))
        if key == 0:
            return {'color':torch.from_numpy(color_img.astype('double')),
                    'depth':torch.from_numpy(depth_img.astype('double')),
                    'edge':torch.from_numpy(edge_img.astype('double')),
                    'normal':torch.from_numpy(normal_img.astype('double')),
                    'edge_pix': edge_c}
        elif key == 1:
            mask_img = mask_img.reshape((1, h, w))
            return {'color':torch.from_numpy(color_img.astype('double')),
                    'depth':torch.from_numpy(depth_img.astype('double')),
                    'edge':torch.from_numpy(edge_img.astype('double')),
                    'normal':torch.from_numpy(normal_img.astype('double')),
                    'mask':torch.from_numpy(mask_img.astype('double')),
                    'edge_pix': edge_c}



class GameDataset(Dataset):
    """
        Game dataset: SUNCG, SceneNet
    """

    def __init__(self, txt_file, suncg_dir=None, scenenet_dir=None, transform=None, norm=None):
        """
        Args:
            txt_file (string): Path to the txt file with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.indexlist = [line.rstrip('\n') for line in open(txt_file,'r')]
        self.scenenet_dir = scenenet_dir
        self.suncg_dir = suncg_dir
        self.transform = transform
        self.norm = norm

    def __len__(self):
        return len(self.indexlist)

    def __getitem__(self, idx):
        while True:
            info = self.indexlist[idx].split()
            if info[0] == 'suncg':
                color_img = io.imread(self.suncg_dir + '/mlt/' + info[1] + '_256.png')
                depth_img = io.imread(self.suncg_dir + '/depth/' + info[1] + '_256.png')
                edge_img = io.imread(self.suncg_dir + '/edge/' + info[1] + '_256.png')
                normal_img = io.imread(self.suncg_dir + '/normal/' + info[1] + '_256.png')
            elif info[0] == 'scenenet':
                color_img = io.imread(self.scenenet_dir + info[1] + '/photo/' + info[2] + '.jpg')
                depth_img = io.imread(self.scenenet_dir + info[1] + '/depth/'+ info[2] + '.png')
                edge_img = io.imread(self.scenenet_dir + info[1] + '/edge/' + info[2] + '.png')
                normal_img = io.imread(self.scenenet_dir + info[1] + '/normal/' + info[2] + '.png')
            else:
                print('wrong dataset!')
            #  print info[0], color_img.shape, depth_img.shape, edge_img.shape, normal_img.shape
            #  print depth_img.max(), depth_img.min(), edge_img.max()

            edge_img[edge_img>0.7] = 1
            edge_c = np.count_nonzero(edge_img)
            if edge_c < 350:
                idx  = np.random.randint(len(self.indexlist))
            else:
                break

        # processing data
        #  _ch = np.random.randint(3)
        #  color_img[:,:,0] = color_img[:, :, _ch]
        #  color_img[:,:,1] = color_img[:, :, 0]
        #  color_img[:,:,2] = color_img[:, :, 0]

        depth_img[depth_img<0] = 0
        depth_img = np.log(depth_img + 1.1)

        edge_img[edge_img>0.7] = 1
        edge_c = np.count_nonzero(edge_img) + 1

        sample = {'color':color_img, 'depth':depth_img, 'edge':edge_img,'edge_pix': edge_c, 'normal':normal_img}
        if self.transform:
            sample = self.transform(sample)
        if self.norm:
            sample['color'] = sample['color'].float().div(255)
            sample['color'] = self.norm(sample['color'])

        return sample


#  n = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])
#  g_data = GameDataset(txt_file='./dataset/train_1m_shuffle.txt',
            #  scenenet_dir='/home/SSD5/jason-data/scenenet5m/',
            #  suncg_dir='/home/SSD5/jason-data/SUNCG/suncg_image256/',
            #  transform=transforms.Compose([CenterCrop(227), ToTensor()]), norm=n)

#  g_data = NYUDataset(txt_file='debug.txt', root_dir='/home/SSD5/jason-data/SUNCG/nyud_256/',
#              transform=transforms.Compose([RandomCrop(224), ToTensor()]), norm=n)

#  g_data = NYUTestDataset(txt_file='test.txt', root_dir='/home/SSD2/jason-data/NYUD/NYU-D/data/raw_size/',
            #  transform=transforms.Compose([ToTensor()]), norm=n)

#  g_data = NNDataset(txt_file='/home/SSD5/jason-data/VOCdevkit/VOC2012/ImageSets/Main/trainval_new.txt',
            #  root_dir='/home/SSD5/jason-data/VOCdevkit/VOC2012/JPEGImages/')

#  print g_data[10].size()


#  dataloader = DataLoader(g_data, batch_size=8, shuffle=False, num_workers=1)
#  for i, spl in enumerate(dataloader):
    #  print '\n'
#      print spl['mask'].size(), spl['mask'].max(), spl['mask'].min()
#      print(len(g_data), spl['color'].size(), spl['depth'].size(), spl['edge'].size(), spl['normal'].size())
#      print spl['color'].max(), spl['normal'].max(), spl['depth'].max(), spl['edge'].max()
#      print spl['color'].min(), spl['normal'].min(), spl['depth'].min(), spl['edge'].min()
    #  print spl['edge'].max(), spl['edge_pix'].numpy().std(), spl['edge_pix'].min()


