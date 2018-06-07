import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torchvision
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight, gain=np.sqrt(2.0))
        init.constant(m.bias, 0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

##############################################################################
# Base net as feature learner
# Options: Alex-net
##############################################################################
class alex_base(nn.Module):
    def __init__(self, ngpu):
        super(alex_base, self).__init__()
        self.ngpu = ngpu
        self.alexnet = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
          #    nn.ReLU(inplace=True),
            #  nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            #  #  fc layers
            #  nn.Conv2d(256, 4096, kernel_size=6, padding=5, dilation=2),
            #  nn.BatchNorm2d(4096),
            #  nn.ReLU(inplace=True),

            #  nn.Conv2d(4096, 4096, kernel_size=1, padding=0, dilation=1),
            #  nn.BatchNorm2d(4096),
          #    nn.ReLU(inplace=True),
        )
        self.alexnet.apply(weights_init)

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.alexnet, x, range(self.ngpu))
        else:
            out = self.alexnet(x)
        return out

# output at diff. layer
class alex_base_side_out(nn.Module):
    def __init__(self, ngpu, layer):
        super(alex_base_side_out, self).__init__()
        self.layer = layer
        self.ngpu = ngpu
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96))

        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256))

        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384))

        self.conv4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384))

        self.conv5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256))

        self.fc6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 4096, kernel_size=6, padding=5, dilation=2),
            nn.BatchNorm2d(4096))

        self.fc7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True))

        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv4.apply(weights_init)
        self.conv5.apply(weights_init)
        self.fc6.apply(weights_init)
        self.fc7.apply(weights_init)

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            out_1 = nn.parallel.data_parallel(self.conv1, x, range(self.ngpu))
            out_2 = nn.parallel.data_parallel(self.conv2, out_1, range(self.ngpu))
            out_3 = nn.parallel.data_parallel(self.conv3, out_2, range(self.ngpu))
            out_4 = nn.parallel.data_parallel(self.conv4, out_3, range(self.ngpu))
            out_5 = nn.parallel.data_parallel(self.conv5, out_4, range(self.ngpu))
            out_fc6 = nn.parallel.data_parallel(self.fc6, out_5, range(self.ngpu))
            out_fc  = nn.parallel.data_parallel(self.fc7, out_fc6, range(self.ngpu))
        else:
            out_1 = self.conv1(x)
            out_2 = self.conv2(out_1)
            out_3 = self.conv3(out_2)
            out_4 = self.conv4(out_3)
            out_5 = self.conv5(out_4)
            out_fc6 = self.fc6(out_5)
            out_fc = self.fc7(out_fc6)

        if self.layer == 'conv1':
            return out_fc, out_1
        elif self.layer == 'conv2':
            return out_fc, out_2
        elif self.layer == 'conv3':
            return out_fc, out_3
        elif self.layer == 'conv4':
            return out_fc, out_4
        elif self.layer == 'conv5':
            return out_fc, out_5
        elif self.layer == 'fc':
            return out_fc, out_fc6
        elif self.layer == 'all':
            return out_fc, out_fc6, out_5, out_4, out_3, out_2, out_1

##############################################################################
# Base net w/o batchnorm
# Options: Alex-net
# TODO: VGG-16
##############################################################################
class alex_wo_bn(nn.Module):
    def __init__(self, ngpu, pretrained=None):
        super(alex_wo_bn, self).__init__()
        self.ngpu = ngpu
        self.pretrained = pretrained
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )
        if self.pretrained:
            print 'wrong implementation[TBD]'
            raise NotImplementedError
            # What I am using is actually caffenet not alexnent
            model = torchvision.models.alexnet(pretrained=True)
            self.features[0].weight.data = model.features[0].weight.data
            self.features[0].bias.data   = model.features[0].bias.data
            self.features[3].weight.data = model.features[3].weight.data
            self.features[3].bias.data   = model.features[3].bias.data
            self.features[6].weight.data = model.features[6].weight.data
            self.features[6].bias.data   = model.features[6].bias.data
            self.features[8].weight.data = model.features[8].weight.data
            self.features[8].bias.data   = model.features[8].bias.data
            self.features[10].weight.data = model.features[10].weight.data
            self.features[10].bias.data   = model.features[10].bias.data
        else:
            self.features.apply(weights_init)

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            feat = nn.parallel.data_parallel(self.features, x, range(self.ngpu))
            feat = feat.view(x.size(0), 256 * 6 * 6)
            out = nn.parallel.data_parallel(self.classifier, feat, range(self.ngpu))
        else:
            x = self.features(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            out = self.classifier(x)
        return out

class alex_c5(nn.Module):
    def __init__(self, ngpu, pretrained=None):
        super(alex_c5, self).__init__()
        self.ngpu = ngpu
        self.pretrained = pretrained
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
        )
        if self.pretrained:
            model = torchvision.models.alexnet(pretrained=True)
            self.features[0].weight.data = model.features[0].weight.data
            self.features[0].bias.data   = model.features[0].bias.data
            self.features[3].weight.data = model.features[3].weight.data
            self.features[3].bias.data   = model.features[3].bias.data
            self.features[6].weight.data = model.features[6].weight.data
            self.features[6].bias.data   = model.features[6].bias.data
            self.features[8].weight.data = model.features[8].weight.data
            self.features[8].bias.data   = model.features[8].bias.data
            self.features[10].weight.data = model.features[10].weight.data
            self.features[10].bias.data   = model.features[10].bias.data
        else:
            self.features.apply(weights_init)

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.features, x, range(self.ngpu))
        else:
            out = self.features(x)
        return out


##############################################################################
# Head model for sub tasks
# Current task: Surface Normal, Depth, Edge
##############################################################################
class tasks_head(nn.Module):
    def __init__(self, ngpu):
        super(tasks_head, self).__init__()
        self.ngpu = ngpu
        self.normal = nn.Sequential(
            nn.ConvTranspose2d(4096, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2))

        self.depth = nn.Sequential(
            nn.ConvTranspose2d(4096, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2))

        self.edge = nn.Sequential(
            nn.ConvTranspose2d(4096, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2))

        self.normal.apply(weights_init)
        self.depth.apply(weights_init)
        self.edge.apply(weights_init)

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            n = nn.parallel.data_parallel(self.normal, x, range(self.ngpu))
            d = nn.parallel.data_parallel(self.depth, x, range(self.ngpu))
            e = nn.parallel.data_parallel(self.edge, x, range(self.ngpu))
        else:
            n = self.normal(x)
            d = self.depth(x)
            e = self.edge(x)
        return d,e,n

# Can also persom colorization
class tasks_head_color(nn.Module):
    def __init__(self, ngpu):
        super(tasks_head_color, self).__init__()
        self.ngpu = ngpu
        self.normal = nn.Sequential(
            nn.ConvTranspose2d(4096, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, padding=1, stride=2))

        self.color = nn.Sequential(
            nn.ConvTranspose2d(4096, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 313, kernel_size=4, stride=2))

        self.depth = nn.Sequential(
            nn.ConvTranspose2d(4096, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, padding=1, stride=2))

        self.edge = nn.Sequential(
            nn.ConvTranspose2d(4096, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, padding=1, stride=2))

        self.normal.apply(weights_init)
        self.color.apply(weights_init)
        self.depth.apply(weights_init)
        self.edge.apply(weights_init)

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            n = nn.parallel.data_parallel(self.normal, x, range(self.ngpu))
            d = nn.parallel.data_parallel(self.depth, x, range(self.ngpu))
            e = nn.parallel.data_parallel(self.edge, x, range(self.ngpu))
            c = nn.parallel.data_parallel(self.color, x, range(self.ngpu))
        else:
            n = self.normal(x)
            d = self.depth(x)
            e = self.edge(x)
            c = self.color(x)
        return d,e,n,c

##############################################################################
# Discriminator for Domain Adaptation
# Similar to DCGAN
##############################################################################
class alex_D(nn.Module):
    def __init__(self, ngpu):
        super(alex_D, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is 4096 x 13 x 13
            nn.Conv2d(4096, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 13 x 13
            nn.Conv2d(64 , 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 6 x 6
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 3 x 3
            nn.Conv2d(256,   1, kernel_size=1, stride=3, padding=0),
            nn.Sigmoid()
        )
        self.main.apply(weights_init)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

#  # patch D for alexnet, similar idea from pix2pix
#  class alex_patch_D(nn.Module):
     #  def __init__(self, ngpu, layer):
        #  super(alex_patch_D, self).__init__()
        #  self.ngpu = ngpu
        #  self.layer = layer
        #  if layer == 'fc':
            #  self.main = nn.Sequential(
                #  nn.Conv2d(4096, 512, kernel_size=3, stride=2, padding=0),
                #  nn.BatchNorm2d(512),
                #  nn.LeakyReLU(0.2, inplace=True),
                #  # state size. 1 x 6 x 6
                #  nn.Conv2d(512,   1, kernel_size=1, stride=1, padding=0),
                #  nn.Sigmoid())

        #  elif layer == 'conv1':
            #  self.main = nn.Sequential(
                #  # input is 96 x 27 x 27
                #  nn.Conv2d(96,  192, kernel_size=3, stride=2, padding=0),
                #  nn.BatchNorm2d(192),
                #  nn.LeakyReLU(0.2, inplace=True),
                #  # state size. 192 x 13 x 13
                #  nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=0),
                #  nn.BatchNorm2d(384),
                #  nn.LeakyReLU(0.2, inplace=True),
                #  #  # state size. 256 x 6 x 6
                #  #  nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0),
                #  #  nn.BatchNorm2d(384),
                #  #  nn.LeakyReLU(0.2, inplace=True),
                #  #  # state size. 1 x 6 x 6
                #  nn.Conv2d(384,   1, kernel_size=1, stride=1, padding=0),
                #  nn.Sigmoid())

        #  elif layer =='conv2' or layer == 'conv5':
            #  self.main = nn.Sequential(
                #  # input is 256 x 13 x 13
                #  nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
                #  nn.BatchNorm2d(512),
                #  nn.LeakyReLU(0.2, inplace=True),
                #  # state size. 512 x 6 x 6
                #  nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                #  nn.BatchNorm2d(512),
                #  nn.LeakyReLU(0.2, inplace=True),
                #  # state size. 1 x 6 x 6
                #  nn.Conv2d(512,   1, kernel_size=1, stride=1, padding=0),
                #  nn.Sigmoid())

        #  elif layer =='conv3' or layer == 'conv4':
            #  self.main = nn.Sequential(
                #  # input is 384 x 13 x 13
                #  nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=0),
                #  nn.BatchNorm2d(512),
                #  nn.LeakyReLU(0.2, inplace=True),
                #  # state size. 512 x 6 x 6
                #  nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                #  nn.BatchNorm2d(512),
                #  nn.LeakyReLU(0.2, inplace=True),
                #  # state size. 1 x 6 x 6
                #  nn.Conv2d(512,   1, kernel_size=1, stride=1, padding=0),
                #  nn.Sigmoid())

        #  self.main.apply(weights_init)

     #  def forward(self, input):
        #  if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            #  output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        #  else:
            #  output = self.main(input)
        #  return output.squeeze(1)

# patch D for alexnet, similar idea from pix2pix
class alex_patch_D(nn.Module):
     def __init__(self, ngpu, layer):
        super(alex_patch_D, self).__init__()
        self.ngpu = ngpu
        self.layer = layer
        if layer == 'fc':
            self.main = nn.Sequential(
                nn.Conv2d(4096, 512, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512,256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 1 x 6 x 6
                nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid())

        elif layer == 'conv1':
            self.main = nn.Sequential(
                # input is 96 x 55 x 55
                nn.Conv2d(96,  192, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(192),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 192 x 27 x 27
                nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(384),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 384 x 13 x 13
                nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 512 x 6 x 6
                nn.Conv2d(512,   1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid())

        elif layer =='conv2':
            self.main = nn.Sequential(
                # input is 256 x 27 x 27
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 512 x 13 x 13
                nn.Conv2d(512, 640, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(640),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 1 x 6 x 6
                nn.Conv2d(640,   1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid())

        elif layer =='conv3' or layer == 'conv4':
            self.main = nn.Sequential(
                # input is 384 x 13 x 13
                nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 512 x 6 x 6
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 1 x 6 x 6
                nn.Conv2d(512,   1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid())

        elif layer == 'conv5':
            self.main = nn.Sequential(
                # input is 256 x 13 x 13
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 512 x 6 x 6
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 1 x 6 x 6
                nn.Conv2d(512,   1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid())
        else:
            print 'you have to choose opt.feat_layer'
            return

        self.main.apply(weights_init)

     def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.squeeze(1)


# D pyramid: take inputs from all layers
class alex_pyramid_D(nn.Module):
     def __init__(self, ngpu):
        super(alex_pyramid_D, self).__init__()
        self.ngpu = ngpu
        self.c1 = nn.Sequential(
            # input is 96 x 55 x 55
            nn.Conv2d(96,  384, kernel_size=7, stride=4, padding=0),
            nn.BatchNorm2d(384))

        self.c2 = nn.Sequential(
            # input is 256 x 27 x 27
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True))

        self.c3 = nn.Sequential(
            # input is 384 x 13 x 13
            nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True))

        self.c4 = nn.Sequential(
            # input is 384 x 13 x 13
            nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True))

        self.c5 = nn.Sequential(
            # input is 256 x 13 x 13
            nn.Conv2d(256, 384, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True))

        self.f = nn.Sequential(
            # input is 4096 x 13 x 13
            nn.Conv2d(4096, 384, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True))

        self.D = nn.Sequential(
            # input is 384 x 13 x 13
            nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 512 x 6 x 6
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 1 x 6 x 6
            nn.Conv2d(512,   1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

        self.c1.apply(weights_init)
        self.c2.apply(weights_init)
        self.c3.apply(weights_init)
        self.c4.apply(weights_init)
        self.c5.apply(weights_init)
        self.f.apply(weights_init)
        self.D.apply(weights_init)


     def forward(self, in1, in2, in3, in4, in5, inf):
        if isinstance(in1.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            o1 = nn.parallel.data_parallel(self.c1, in1, range(self.ngpu))
            o2 = nn.parallel.data_parallel(self.c2, in2, range(self.ngpu))
            o3 = nn.parallel.data_parallel(self.c3, in3, range(self.ngpu))
            o4 = nn.parallel.data_parallel(self.c4, in4, range(self.ngpu))
            o5 = nn.parallel.data_parallel(self.c5, in5, range(self.ngpu))
            of = nn.parallel.data_parallel(self.f, inf, range(self.ngpu))
            o = o1 + o2 + o3 + o4 + o5 + of
            output = nn.parallel.data_parallel(self.D, o, range(self.ngpu))
        else:
            o1 = self.c1(in1)
            o2 = self.c2(in2)
            o3 = self.c3(in3)
            o4 = self.c4(in4)
            o5 = self.c5(in5)
            of = self.f(inf)
            o  = o1 + o2 + o3 + o4 + o5 + of
            output = self.D(o)

        return output.squeeze(1)


##########################################################################
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
##########################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


##############################################################################
# Test code
##############################################################################
#  lay = ['conv1','conv2','conv3', 'conv4', 'conv5', 'fc']
#  for l in lay:
    #  print '\n' + l
#  l ='all'
#  base_n = alex_base_side_out(1, l)
#  #  #  print base_n.alexnet[1].running_mean
#  D = alex_pyramid_D(1)
#  #  #  head = tasks_head(1)

#  inp = Variable(torch.randn(10, 3, 227, 227))
#  fc, c5, c4, c3, c2, c1 = base_n(inp)
#  print fc.size(), c5.size(), c4.size(), c3.size(), c2.size(), c1.size()

#  #  #  s,d,e = head(base_out)
#  #  #  print d.size(), e.size(), s.size()

#  prob_d = D(c1,c2,c3,c4,c5,fc)
#  print prob_d.size(), prob_d.data.max(), prob_d.data.min()
