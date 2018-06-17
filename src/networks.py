import torch
from .basic_model import BasicModel

############################
#  Functions definition
############################
def init_weights(net, init_type='normal', gain=0.02):
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

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


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
        feat2 = self.bn2(self.conv2(self.maxpooling1(self.relu1(x))))
        feat3 = self.bn3(self.conv3(self.maxpooling2(self.relu2(x))))
        feat4 = self.bn4(self.conv4(self.relu3(x)))
        feat5 = self.bn5(self.conv5(self.relu4(x)))
        feat6 = self.bn6(self.fc6(self.maxpooling5(self.relu5(x))))
        feat7 = self.relu7(self.bn7(self.fc7(self.relu6(x))))

        return feat1, feat2, feat3, feat4, feat5, feat6, feat7

class netH_alexnet(nn.Module):
    def __init__(self, ngpu):
        super(netH_alexnet, self).__init__()
        self.ngpu = ngpu
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
        return d, e, n


class netD_alexnet(nn.Module):
    def __init__(self, ngpu):
        super(netD_alexnet, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is 4096 x 13 x 13
            nn.Conv2d(4096, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 13 x 13
            nn.Conv2d(64 , 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 6 x 6
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
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

#############################
# vgg16-model definition
#############################

class netB_vgg16(nn.Module):
    def __init__(self):
        super(netB_vgg16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
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
        pass


#############################
# TODO:resnet-model definition
#############################

class netB_resnet(nn.Module):
    def __init__(self, name):
        super(netB_resnet, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        pass

