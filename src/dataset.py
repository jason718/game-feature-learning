import os
import pdb
import torch
import torchvision
import numpy as np
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
        color_img, depth_img, edge_img, normal_img = sample['color'], sample['depth'],sample['edge'], sample['normal']

        h, w = color_img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        color_im = transform.resize(color_img, (new_h, new_w), preserve_range=True, mode='reflect', anti_aliasing=True)
        depth_im = transform.resize(depth_img, (new_h, new_w), preserve_range=True, mode='reflect',anti_aliasing=True)
        normal_im = transform.resize(normal_img, (new_h, new_w), preserve_range=True, mode='reflect', anti_aliasing=True)
        edge_im = transform.resize(edge_img, (new_h, new_w), preserve_range=True, mode='reflect', anti_aliasing=True,)
        return {'color':color_im, 'depth':depth_im, 'edge':edge_im, 'edge_pix':sample['edge_pix'], 'normal':normal_im}


class RandomCrop(object):
    """ Crop randomly the image in a sample. """
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = (output_size, output_size)

    def __call__(self, sample):
        color_img, depth_img, edge_img, normal_img =  sample['color'], sample['depth'], sample['edge'], sample['normal']

        h, w = color_img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        color_im = color_img[top: top + new_h, left: left + new_w]
        depth_im = depth_img[top: top + new_h, left: left + new_w]
        edge_im = edge_img[top: top + new_h, left: left + new_w]
        normal_im = normal_img[top: top + new_h, left: left + new_w]

        return {'color':color_im, 'depth':depth_im, 'edge':edge_im, 'edge_pix':sample['edge_pix'], 'normal':normal_im}


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors. """
    def __call__(self, sample):
        color_img, depth_img, edge_img, normal_img =  sample['color'], sample['depth'], sample['edge'], sample['normal']

        # swap color axis because
        # numpy image: H x W x C  --> torch image: C x H x W (C=1,3)
        color_img = color_img.transpose((2, 0, 1))
        normal_img = normal_img.transpose((2, 0, 1))
        h,w = depth_img.shape
        depth_img = depth_img.reshape((1, h, w))
        edge_img = edge_img.reshape((1, h, w))

        return {'color': torch.from_numpy(color_img.astype('float32')),
                'depth': torch.from_numpy(depth_img.astype('float32')),
                'edge': torch.from_numpy(edge_img.astype('float32')),
                'normal': torch.from_numpy(normal_img.astype('float32')),
                'edge_pix': sample['edge_pix']}

#########################################
# Dataset
########################################

class GameDataset(torch.utils.data.Dataset):
    """ Game dataset: SUNCG, SceneNet """
    def __init__(self, cfg, norm=None):
        self.indexlist = [line.rstrip('\n') for line in open(cfg['TRAIN_FILE'], 'r')]
        self.cfg = cfg
        self.norm = norm

    def __len__(self):
        return len(self.indexlist)

    def __getitem__(self, idx):
        while True:
            info = self.indexlist[idx].split()
            if info[0] == 'suncg':
                color_img = io.imread(os.path.join(self.cfg['SUNCG_DIR'], 'mlt', info[1]+'_256.png'))
                depth_img = io.imread(os.path.join(self.cfg['SUNCG_DIR'], 'depth', info[1]+'_256.png'))
                edge_img = io.imread(os.path.join(self.cfg['SUNCG_DIR'], 'edge', info[1]+'_256.png'))
                normal_img = io.imread(os.path.join(self.cfg['SUNCG_DIR'], 'normal', info[1]+'_256.png'))
            elif info[0] == 'scenenet':
                color_img = io.imread(os.path.join(self.cfg['SCENENET_DIR'], info[1], 'photo', info[2]+'.jpg'))
                depth_img = io.imread(os.path.join(self.cfg['SCENENET_DIR'], info[1], 'depth', info[2]+'.png'))
                edge_img = io.imread(os.path.join(self.cfg['SCENENET_DIR'], info[1], 'edge', info[2]+'.png'))
                normal_img = io.imread(os.path.join(self.cfg['SCENENET_DIR'], info[1], 'normal', info[2]+'.png'))
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

        # image transformation
        if self.cfg['TRAIN']:
            _transforms = torchvision.transforms.Compose([
                            Rescale((256,256)),
                            RandomCrop(self.cfg['CROP_SIZE']),
                            ToTensor()])
        else:
            _transforms = torchvision.transforms.Compose([
                            Rescale(self.cfg['CROP_SIZE'], self.cfg['CROP_SIZE']),
                            ToTensor()])
        sample = _transforms(sample)

        ## only normalize RGB image
        if self.norm:
            sample['color'] = sample['color'].float().div(255)
            sample['color'] = self.norm(sample['color'])

        return sample


##################################
#   Test Code
##################################
#  import yaml
#  import pdb
#  from torchvision import transforms, utils

#  with open('./configs/alexnet.yaml', 'r') as fin:
    #  cfg=yaml.load(fin)
#  print(cfg)

#  n = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])
#  g_data = GameDataset(cfg,norm=n)
#  dataloader = torch.utils.data.DataLoader(g_data, batch_size=1, shuffle=False, num_workers=0)
#  iterator = iter(dataloader)
#  count=0
#  while True:
    #  try:
        #  #  print(count)
        #  count += 1
        #  spl = next(iterator)
    #  except StopIteration:
        #  iterator = iter(dataloader)
        #  spl = next(iterator)

    #  print(len(g_data), spl['color'].size(), spl['depth'].size(), spl['edge'].size(), spl['normal'].size())
    #  print spl['color'].max(), spl['normal'].max(), spl['depth'].max(), spl['edge'].max()
    #  print spl['color'].min(), spl['normal'].min(), spl['depth'].min(), spl['edge'].min()
    #  print spl['edge'].max(), spl['edge_pix'].numpy().std(), spl['edge_pix'].min()

