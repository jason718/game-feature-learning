import os
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import io, transform

###################################
# Dataset tranformation code
# Most of them are borrowed from PyTorch repo
# Modify them to deal with several different modality pics together
##################################

class Rescale(object):
    """ Rescale the image in a sample to a given size. """

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
        return {'color':color_im, 'depth':depth_im, 'edge':edge_im,'edge_pix': edge_c, 'normal':normal_im}


class RandomCrop(object):
    """ Crop randomly the image in a sample. """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = (output_size, output_size)

    def __call__(self, sample):
        color_img, depth_img, edge_img,edge_c, normal_img =  sample['color'], sample['depth'],sample['edge'],sample['edge_pix'], sample['normal']

        h, w = color_img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        color_im = color_img[top: top + new_h, left: left + new_w]
        depth_im = depth_img[top: top + new_h, left: left + new_w]
        edge_im = edge_img[top: top + new_h, left: left + new_w]
        normal_im = normal_img[top: top + new_h, left: left + new_w]

        return {'color':color_im, 'depth':depth_im, 'edge':edge_im,'edge_pix': edge_c, 'normal':normal_im}


class CenterCrop(object):
    """ Crop centor the image in a sample. """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        color_img, depth_img, edge_img,edge_c, normal_img =  sample['color'], sample['depth'],sample['edge'],sample['edge_pix'], sample['normal']

        h, w = color_img.shape[:2]
        new_h, new_w = self.output_size

        top = int((h - new_h) / 2)
        left = int((w - new_w) / 2)

        color_im = color_img[top: top + new_h, left: left + new_w]
        depth_im = depth_img[top: top + new_h, left: left + new_w]
        edge_im = edge_img[top: top + new_h, left: left + new_w]
        normal_im = normal_img[top: top + new_h, left: left + new_w]

        return {'color':color_im, 'depth':depth_im, 'edge':edge_im,'edge_pix': edge_c, 'normal':normal_im}

class ToTensor(object):
    """ Convert ndarrays in sample to Tensors. """

    def __call__(self, sample):
        color_img, depth_img, edge_img,edge_c, normal_img =  sample['color'], sample['depth'],sample['edge'],sample['edge_pix'], sample['normal']

        # swap color axis because
        # numpy image: H x W x C  --> torch image: C x H x W (C=1,3)
        color_img = color_img.transpose((2, 0, 1))
        normal_img = normal_img.transpose((2, 0, 1))
        h,w = depth_img.shape
        depth_img = depth_img.reshape((1, h, w))
        edge_img = edge_img.reshape((1, h, w))

        return {'color':torch.from_numpy(color_img.astype('double')),
                'depth':torch.from_numpy(depth_img.astype('double')),
                'edge':torch.from_numpy(edge_img.astype('double')),
                'normal':torch.from_numpy(normal_img.astype('double')),
                'edge_pix': edge_c}


#########################################
# Dataset
########################################

class GameDataset(Dataset):
    """ Game dataset: SUNCG, SceneNet """

    def __init__(self, txt_file, suncg_dir=None, scenenet_dir=None, transform=None, norm=None):
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
                color_img = io.imread(os.path.join(self.suncg_dir, 'mlt', info[1]+'_256.png'))
                depth_img = io.imread(os.path.join(self.suncg_dir, 'depth', info[1]+'_256.png'))
                edge_img = io.imread(os.path.join(self.suncg_dir, 'edge', info[1]+'_256.png'))
                normal_img = io.imread(os.path.join(self.suncg_dir, 'normal', info[1]+'_256.png'))
            elif info[0] == 'scenenet':
                color_img = io.imread(os.path.join(self.scenenet_dir, info[1], 'photo', info[2]+'.jpg'))
                depth_img = io.imread(os.path.join(self.scenenet_dir, info[1], 'depth', info[2]+'.png'))
                edge_img = io.imread(os.path.join(self.scenenet_dir, info[1], 'edge', info[2]+'.png'))
                normal_img = io.imread(os.path.join(self.scenenet_dir, info[1], 'normal', info[2]+'.png'))
            else:
                raise ValueError('wrong dataset!')

            # choose the pics with decent edge map
            edge_img[edge_img>0.] = 1
            edge_c = np.count_nonzero(edge_img)
            if edge_c < 350:
                idx  = np.random.randint(len(self.indexlist))
            else:
                break

        # TODO: to fix
        depth_img[depth_img<0] = 0
        depth_img = np.log(depth_img / 1000. + 1e-8)

        sample = {'color':color_img, 'depth':depth_img, 'edge':edge_img,'edge_pix': edge_c, 'normal':normal_img}
        if self.transform:
            sample = self.transform(sample)
        if self.norm:
            sample['color'] = sample['color'].float().div(255)
            sample['color'] = self.norm(sample['color'])

        return sample


##################################
#   Test Code
##################################

#  from torchvision import transforms, utils
#  n = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])
#  g_data = GameDataset(txt_file='./data/train_1m_shuffle.txt',
            #  scenenet_dir='/home/jason/saturn-1TB/scenenet5m',
            #  suncg_dir='/home/jason/saturn-1TB/SUNCG/suncg_image256/',
            #  transform=transforms.Compose([CenterCrop(227), ToTensor()]), norm=n)

#  #  g_data = NYUDataset(txt_file='debug.txt', root_dir='/home/SSD5/jason-data/SUNCG/nyud_256/',
            #  #  transform=transforms.Compose([RandomCrop(224), ToTensor()]), norm=n)

#  #  g_data = NYUTestDataset(txt_file='test.txt', root_dir='/home/SSD2/jason-data/NYUD/NYU-D/data/raw_size/',
            #  #  transform=transforms.Compose([ToTensor()]), norm=n)

#  #  g_data = NNDataset(txt_file='/home/SSD5/jason-data/VOCdevkit/VOC2012/ImageSets/Main/trainval_new.txt',
#  #              root_dir='/home/SSD5/jason-data/VOCdevkit/VOC2012/JPEGImages/')

#  dataloader = torch.utils.data.DataLoader(g_data, batch_size=8, shuffle=False, num_workers=1)
#  for i, spl in enumerate(dataloader):
    #  print '\n'
    #  print(len(g_data), spl['color'].size(), spl['depth'].size(), spl['edge'].size(), spl['normal'].size())
    #  print spl['color'].max(), spl['normal'].max(), spl['depth'].max(), spl['edge'].max()
    #  print spl['color'].min(), spl['normal'].min(), spl['depth'].min(), spl['edge'].min()
    #  print spl['edge'].max(), spl['edge_pix'].numpy().std(), spl['edge_pix'].min()


