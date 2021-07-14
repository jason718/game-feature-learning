import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision.models.resnet import resnet50

#############################################
#  Functions definition
#  Most of the code here is borrowed from:
#  pytorch-CycleGAN-and-pix2pix
#############################################
def init_net(net, gpu_ids=None, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    if gpu_ids:
        net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    return net

def get_scheduler(optimizer, cfg):
    if cfg['LR_POLICY'] == 'lambda':
        # TODO
        raise NotImplementedError('need to rewrite code here')
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + cfg['N_EPOCH'] - cfg.niter) / float(cfg.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif cfg['LR_POLICY'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg['LR_DECAY_EP'], gamma=0.1)
    elif cfg['LR_POLICY'] == 'multi-step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg['LR_DECAY_EP'], gamma=0.1)
    elif cfg['LR_POLICY'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt['LR_POLICY'])
    return scheduler

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

#############################
# alexnet-model definition
#############################
class netB_alexnet(nn.Module):
    def __init__(self):
        super(netB_alexnet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpooling5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # this is a conv layer but we still call it fc6, please read the paper for details
        self.fc6 = nn.Conv2d(256, 4096, kernel_size=6, padding=5, dilation=2)
        self.bn6 = nn.BatchNorm2d(4096)
        self.relu6 = nn.ReLU(inplace=True)

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1, padding=0, dilation=1)
        self.bn7 = nn.BatchNorm2d(4096)
        self.relu7 = nn.ReLU(inplace=True)

    def forward(self, x):
        """Compute feature before relu/pooling following github.com/soumith/ganhacks"""
        feat1 = self.bn1(self.conv1(x))
        feat2 = self.bn2(self.conv2(self.maxpooling1(self.relu1(feat1))))
        feat3 = self.bn3(self.conv3(self.maxpooling2(self.relu2(feat2))))
        feat4 = self.bn4(self.conv4(self.relu3(feat3)))
        feat5 = self.bn5(self.conv5(self.relu4(feat4)))
        feat6 = self.bn6(self.fc6(self.maxpooling5(self.relu5(feat5))))
        feat7 = self.relu7(self.bn7(self.fc7(self.relu6(feat6))))

        return {'conv1':feat1, 'conv2':feat2, 'conv3':feat3, 'conv4':feat4,
                'conv5':feat5, 'fc6':feat6, 'fc7':feat7, 'out':feat7}

class netH_alexnet(nn.Module):
    def __init__(self):
        super(netH_alexnet, self).__init__()
        self.normal = nn.Sequential(
            nn.ConvTranspose2d(4096, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2))

        self.depth = nn.Sequential(
            nn.ConvTranspose2d(4096, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2))

        self.edge = nn.Sequential(
            nn.ConvTranspose2d(4096, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2))

    def forward(self, x):
        return {'depth':self.depth(x), 'edge':self.edge(x), 'norm':self.normal(x)}

class netD_alexnet(nn.Module):
    def __init__(self, layer='conv5'):
        super(netD_alexnet, self).__init__()
        if layer == 'fc6':
            self.main = nn.Sequential(
                nn.Conv2d(4096, 512, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512,256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
                # state size. 1 x 6 x 6
                nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid())
        elif layer == 'conv1':
            self.main = nn.Sequential(
                # input is 96 x 55 x 55
                nn.Conv2d(96,  192, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(192), nn.LeakyReLU(0.2, inplace=True),
                # state size. 192 x 27 x 27
                nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(384), nn.LeakyReLU(0.2, inplace=True),
                # state size. 384 x 13 x 13
                nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
                # state size. 512 x 6 x 6
                nn.Conv2d(512,   1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid())
        elif layer =='conv2':
            self.main = nn.Sequential(
                # input is 256 x 27 x 27
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
                # state size. 512 x 13 x 13
                nn.Conv2d(512, 640, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(640), nn.LeakyReLU(0.2, inplace=True),
                # state size. 1 x 6 x 6
                nn.Conv2d(640,   1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid())
        elif layer =='conv3' or layer == 'conv4':
            self.main = nn.Sequential(
                # input is 384 x 13 x 13
                nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
                # state size. 512 x 6 x 6
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
                # state size. 1 x 6 x 6
                nn.Conv2d(512,   1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid())
        elif layer == 'conv5':
            self.main = nn.Sequential(
                # input is 256 x 13 x 13
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
                # state size. 512 x 6 x 6
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
                # state size. 1 x 6 x 6
                nn.Conv2d(512,   1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid())
        else:
            raise ValueError('wrong layer name')

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)

#############################
# vgg16-model definition
# TODO: test
#############################

class netB_vgg16(nn.Module):
    def __init__(self):
        super(netB_vgg16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))

        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128))

        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256))

        self.conv4 = nn.Sequential(
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512))

        self.conv5 = nn.Sequential(
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True))

    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)
        feat5 = self.conv5(feat4)

        return {'conv1':feat1, 'conv2':feat2, 'conv3':feat3, 'conv4':feat4, 'conv5':feat5}

class netH_vgg16(nn.Module):
    def __init__(self):
        super(netH_vgg16, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.n_2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.n_3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.n_4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1))

        self.d_2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.d_3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.d_4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1))

        self.e_2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.e_3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.e_4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, f2, f3, f4, f5):
        feat1 = self.main(f5)
        d_feat2 = self.d_2(torch.cat((feat1,f4), dim=1))
        d_feat3 = self.d_3(torch.cat((d_feat2,f3), dim=1))
        d       = self.d_4(torch.cat((d_feat3,f2), dim=1))

        n_feat2 = self.n_2(torch.cat((feat1,f4), dim=1))
        n_feat3 = self.n_3(torch.cat((n_feat2,f3), dim=1))
        n       = self.n_4(torch.cat((n_feat3,f2), dim=1))

        e_feat2 = self.e_2(torch.cat((feat1,f4), dim=1))
        e_feat3 = self.e_3(torch.cat((e_feat2,f3), dim=1))
        e       = self.e_4(torch.cat((e_feat3,f2), dim=1))

        return d,e,n

class netD_vgg16(nn.Module):
     def __init__(self, layer='conv5'):
        super(netD_vgg16, self).__init__()
        self.layer = layer
        if self.layer == 'conv5':
            self.main = nn.Sequential(
                # input is 512 x 14 x 14
                nn.Conv2d(512, 1024, kernel_size=4, stride=2),
                nn.BatchNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
                # 1024 x 6 x 6
                nn.Conv2d(1024, 1024, kernel_size=1),
                nn.BatchNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
                # state size. 1024 x 6 x 6
                nn.Conv2d(1024,   1, kernel_size=1),
                nn.Sigmoid())
        elif self.layer == 'conv4':
            self.main = nn.Sequential(
                # input is 512 x 28 x 28
                nn.Conv2d(512, 1024, kernel_size=4, stride=2),
                nn.BatchNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
                # 1024 x 14 x 14
                nn.Conv2d(1024, 1024, kernel_size=2,stride=2),
                nn.BatchNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
                # state size. 1024 x 6 x 6
                nn.Conv2d(1024,   1, kernel_size=1),
                nn.Sigmoid())

     def forward(self, x):
        output = self.main(x)
        return output.squeeze(1)


#############################
# resnet50-model definition
# TODO: test
#############################


class netB_resnet(nn.Module):
    def __init__(self, use_pretrained = False):
        super(netB_resnet, self).__init__()
        self.network = resnet50(pretrained = use_pretrained)
        self.network = torch.nn.Sequential(*list(self.network.children())[:-2])
        
        
    def forward(self, images):
        output = self.network(images)
        return output


class netH_resnet(nn.Module):
    def __init__(self):
        super(netH_resnet, self).__init__()
        self.depth = nn.Sequential(
            nn.ConvTranspose2d(2048, 64, kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 64, kernel_size = 1, stride = 2, output_padding = 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 1, kernel_size = 1, stride = 2))


        self.normal = nn.Sequential(
            nn.ConvTranspose2d(2048, 64, kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 64, kernel_size = 1, stride = 2, output_padding = 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 3, kernel_size = 1, stride = 2))

        self.edge = nn.Sequential(
            nn.ConvTranspose2d(2048, 64, kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 64, kernel_size = 1, stride = 2, output_padding = 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 1, kernel_size = 1, stride = 2))

    def forward(self, x):
        return {'depth':self.depth(x), 'edge':self.edge(x), 'norm':self.normal(x)}


class netD_resnet(nn.Module):
    def __init__(self):
        super(netD_resnet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size = 3, stride = 2, padding = 0),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace = True),
            # state size. 1 x 6 x 6
            nn.Conv2d(256, 1, kernel_size = 1, stride = 1, padding = 0),
            nn.Sigmoid())
        
    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


