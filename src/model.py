import os
import numpy as np
import torch
import torchvision
import pdb
from datetime import datetime

import networks
from utils import ImagePool, Logger

#########################################################################
#  Network definition
#  Options:
#      alexnet(caffenet-definition), vgg16, resnet50
#  Note:
#      github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet
#      use this to follow previous paper's practice.
##########################################################################
class Model():
    def initialize(self, cfg):
        self.cfg = cfg

        ## set devices
        if cfg['GPU_IDS']:
            assert(torch.cuda.is_available())
            self.device = torch.device('cuda:{}'.format(cfg['GPU_IDS'][0]))
            torch.backends.cudnn.benchmark = True
            print('Using %d GPUs'% len(cfg['GPU_IDS']))
        else:
            self.device = torch.device('cpu')

        # define network
        if cfg['ARCHI'] == 'alexnet':
            self.netB = networks.netB_alexnet()
            self.netH = networks.netH_alexnet()
            if self.cfg['USE_DA'] and self.cfg['TRAIN']:
                self.netD = networks.netD_alexnet(self.cfg['DA_LAYER'])
        elif cfg['ARCHI'] == 'vgg16':
            raise NotImplementedError
            self.netB = networks.netB_vgg16()
            self.netH = networks.netH_vgg16()
            if self.cfg['USE_DA'] and self.cfg['TRAIN']:
                self.netD = netD_vgg16(self.cfg['DA_LAYER'])
        elif 'resnet' in cfg['ARCHI']:
            self.netB = networks.netB_resnet()
            self.netH = networks.netH_resnet()
            if self.cfg['USE_DA'] and self.cfg['TRAIN']:
                self.netD = networks.netD_resnet()
        else:
            raise ValueError('Un-supported network')

        ## initialize network param.
        self.netB = networks.init_net(self.netB, cfg['GPU_IDS'], 'xavier')
        self.netH = networks.init_net(self.netH, cfg['GPU_IDS'], 'xavier')
        if self.cfg['USE_DA'] and self.cfg['TRAIN']:
            self.netD = networks.init_net(self.netD, cfg['GPU_IDS'], 'xavier')
        print(self.netB, self.netH, self.netD)

        # loss, optimizer, and scherduler
        if cfg['TRAIN']:
            self.total_steps = 0
            ## Output path
            self.save_dir = os.path.join(cfg['OUTPUT_PATH'], cfg['ARCHI'],
                    datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            self.logger = Logger(self.save_dir)

            ## model names
            self.model_names = ['netB', 'netH']
            ## loss
            self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionDepth1 = torch.nn.MSELoss().to(self.device)
            self.criterionNorm = torch.nn.CosineEmbeddingLoss().to(self.device)
            # define during running, rely on data weight
            self.criterionEdge = None

            ## optimizers
            self.lr = cfg['LR']
            self.optimizers = []
            self.optimizer_B = torch.optim.Adam(self.netB.parameters(),
                                    lr=cfg['LR'], betas=(cfg['BETA1'], cfg['BETA2']))
            self.optimizer_H = torch.optim.Adam(self.netH.parameters(),
                                    lr=cfg['LR'], betas=(cfg['BETA1'], cfg['BETA2']))
            self.optimizers.append(self.optimizer_B)
            self.optimizers.append(self.optimizer_H)
            if cfg['USE_DA']:
                self.real_pool = ImagePool(cfg['POOL_SIZE'])
                self.syn_pool = ImagePool(cfg['POOL_SIZE'])
                self.model_names.append('netD')
                ## use SGD for discriminator
                self.optimizer_D = torch.optim.SGD(self.netD.parameters(),
                                    lr=cfg['LR'], momentum=cfg['MOMENTUM'], weight_decay=cfg['WEIGHT_DECAY'])
                self.optimizers.append(self.optimizer_D)
            ## LR scheduler
            self.schedulers = [networks.get_scheduler(optimizer, cfg) for optimizer in self.optimizers]

        if cfg['TEST'] or cfg['RESUME']:
            self.load_networks(cfg['CKPT_PATH'])

    def set_input(self, inputs):
        if self.cfg['GRAY']:
            _ch = np.random.randint(3)
            _syn = inputs['syn']['color'][:, _ch, :, :]
            self.input_syn_color = torch.stack((_syn, _syn, _syn), dim=1).to(self.device)
        else:
            self.input_syn_color = inputs['syn']['color'].to(self.device)
        self.input_syn_dep = inputs['syn']['depth'].to(self.device)
        self.input_syn_edge = inputs['syn']['edge'].to(self.device)
        self.input_syn_edge_count = inputs['syn']['edge_pix'].to(self.device)
        self.input_syn_norm = inputs['syn']['normal'].to(self.device)
        if self.cfg['USE_DA']:
            if self.cfg['GRAY']:
                _ch = np.random.randint(3)
                _real = inputs['real'][0][:, _ch, :, :]
                self.input_real_color = torch.stack((_real, _real, _real), dim=1).to(self.device)
            else:
                self.input_real_color = inputs['real'][0].to(self.device)

    def forward(self):
        self.feat_syn = self.netB(self.input_syn_color)
        self.head_pred = self.netH(self.feat_syn['out'])
        if self.cfg['USE_DA'] and self.cfg['TRAIN']:
            self.feat_real = self.netB(self.input_real_color)
            self.pred_D_real = self.netD(self.feat_real[self.cfg['DA_LAYER']])
            self.pred_D_syn  = self.netD(self.feat_syn[self.cfg['DA_LAYER']])

    def backward_BH(self):
        ## forward to compute prediction
        self.task_pred = self.netH(self.feat_syn['out'])

        # depth
        depth_diff = self.task_pred['depth'] - self.input_syn_dep
        _n = self.task_pred['depth'].size(0) * self.task_pred['depth'].size(2) * self.task_pred['depth'].size(3)
        loss_depth2 = depth_diff.sum().div_(_n).pow(2).mul_(0.5)
        loss_depth1 = self.criterionDepth1(self.task_pred['depth'], self.input_syn_dep)
        self.loss_dep  = self.cfg['DEP_WEIGHT'] * (loss_depth1 + loss_depth2) * 0.5

        # surface normal
        ch = self.task_pred['norm'].size(1)
        _pred = self.task_pred['norm'].permute(0, 2, 3, 1).contiguous().view(-1,ch)
        _gt = self.input_syn_norm.permute(0, 2, 3, 1).contiguous().view(-1,ch)
        _gt = (_gt / 127.5) - 1
        _pred = torch.nn.functional.normalize(_pred, dim=1)
        self.task_pred['norm'] = _pred.view(self.task_pred['norm'].size(0), self.task_pred['norm'].size(2), self.task_pred['norm'].size(3),3).permute(0, 3, 1, 2)
        self.task_pred['norm'] = (self.task_pred['norm'] + 1) * 127.5
        cos_label = torch.ones(_gt.size(0)).to(self.device)
        self.loss_norm = self.cfg['NORM_WEIGHT'] * self.criterionNorm(_pred, _gt, cos_label)

        # edge
        weight_e = (self.task_pred['edge'].size(2) * self.task_pred['edge'].size(3) - self.input_syn_edge_count ) / self.input_syn_edge_count
        self.criterionEdge = torch.nn.BCEWithLogitsLoss(weight=weight_e.float().view(-1,1,1,1)).to(self.device)
        self.loss_edge = self.cfg['EDGE_WEIGHT'] * self.criterionEdge(self.task_pred['edge'], self.input_syn_edge)

        ## combined loss
        loss = self.loss_edge + self.loss_norm + self.loss_dep

        if self.cfg['USE_DA']:
            pred_syn = self.netD(self.feat_syn[self.cfg['DA_LAYER']])
            self.loss_DA = self.criterionGAN(pred_syn, True)
            loss += self.loss_DA * self.cfg['DA_WEIGHT']

        loss.backward()

    def backward_D(self):
        ## Synthetic
        # stop backprop to netB by detaching
        _feat_s = self.syn_pool.query(self.feat_syn[self.cfg['DA_LAYER']].detach().cpu())
        pred_syn = self.netD(_feat_s.to(self.device))
        self.loss_D_syn = self.criterionGAN(pred_syn, False)

        ## Real
        _feat_r = self.real_pool.query(self.feat_real[self.cfg['DA_LAYER']].detach().cpu())
        pred_real = self.netD(_feat_r.to(self.device))
        self.loss_D_real = self.criterionGAN(pred_real, True)

        ## Combined
        self.loss_D = (self.loss_D_syn + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def optimize(self):
        self.total_steps += 1
        self.forward()
        # if DA, update on real data
        if self.cfg['USE_DA']:
            self.set_requires_grad(self.netD, True)
            self.set_requires_grad([self.netB, self.netH], False)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # update on synthetic data
        self.set_requires_grad([self.netB, self.netH], True)
        self.set_requires_grad(self.netD, False)
        self.optimizer_B.zero_grad()
        self.optimizer_H.zero_grad()
        self.backward_BH()
        self.optimizer_B.step()
        self.optimizer_H.step()

    # make models eval mode during test time
    def eval(self):
        self.netB.eval()
        self.netH.eval()
        self.netD.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        self.lr = self.cfgimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % self.lr)

    #  return visualization images. train.py will save the images.
    def visualize_pred(self, ep=0):
        vis_dir = os.path.join(self.save_dir, 'vis')
        if not os.path.isdir(vis_dir):
            os.makedirs(vis_dir)
        if self.total_steps % self.cfg['VIS_FREQ'] == 0:
            num_pic = min(8, self.task_pred['norm'].size(0))
            torchvision.utils.save_image(self.input_syn_color[0:num_pic].cpu(),
                                        '%s/ep_%d_iter_%d_color.jpg' % (vis_dir,ep,self.total_steps),
                                        nrow=num_pic, normalize=True)
            vis_norm = torch.cat((self.input_syn_norm[0:num_pic], self.task_pred['norm'][0:num_pic]), dim=0)
            torchvision.utils.save_image(vis_norm.detach(),
                                        '%s/ep_%d_iter_%d_norm.jpg' % (vis_dir,ep,self.total_steps),
                                        nrow=num_pic, normalize=True)
            vis_depth = torch.cat((self.input_syn_dep[0:num_pic], self.task_pred['depth'][0:num_pic]), dim=0)
            torchvision.utils.save_image(vis_depth.detach(),
                                        '%s/ep_%d_iter_%d_depth.jpg' % (vis_dir,ep,self.total_steps),
                                        nrow=num_pic, normalize=True)
            # TODO: visualization
            edge_vis = torch.nn.functional.sigmoid(self.task_pred['edge'])
            vis_edge = torch.cat((self.input_syn_edge[0:num_pic], edge_vis[0:num_pic]), dim=0)
            torchvision.utils.save_image(vis_edge.detach(),
                                        '%s/ep_%d_iter_%d_edge.jpg' % (vis_dir,ep,self.total_steps),
                                        nrow=num_pic, normalize=False)
            if self.cfg['USE_DA']:
                torchvision.utils.save_image(self.input_real_color[0:num_pic].cpu(),
                                            '%s/ep_%d_iter_%d_real.jpg' % (vis_dir,ep,self.total_steps),
                                            nrow=num_pic, normalize=True)
            print('==> Saved epoch %d total step %d visualization to %s' % (ep, self.total_steps, vis_dir))

    # print on screen, log into tensorboard
    def print_n_log_losses(self, ep=0):
        if self.total_steps % self.cfg['PRINT_FREQ'] == 0:
            print('\nEpoch: %d  Total_step: %d  LR: %f' % (ep, self.total_steps, self.lr))
            print('Train on tasks: Loss_dep: %.4f   | Loss_edge: %.4f   | Loss_norm: %.4f'
                  % (self.loss_dep, self.loss_edge, self.loss_norm))
            info = {
                'loss_dep': self.loss_dep,
                'loss_norm': self.loss_norm,
                'loss_edge': self.loss_edge
                }
            if self.cfg['USE_DA']:
                print('Train for DA:   Loss_D_syn: %.4f | Loss_D_real: %.4f | Loss_DA: %.4f'
                      % (self.loss_D_syn, self.loss_D_real, self.loss_DA))
                info['loss_D_syn'] = self.loss_D_syn
                info['loss_D_real'] = self.loss_D_real
                info['loss_DA'] = self.loss_DA

            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, self.total_steps)

    # save models to the disk
    def save_networks(self, which_epoch):
        for name in self.model_names:
            save_filename = '%s_ep%s.pth' % (name, which_epoch)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, name)
            if isinstance(net, torch.nn.DataParallel):
                torch.save(net.module.cpu().state_dict(), save_path)
            else:
                torch.save(net.cpu().state_dict(), save_path)
            print('==> Saved to %s' % save_path)
            if torch.cuda.is_available:
                net.cuda(self.device)

    # load models from the disk
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                    print('loading the model from %s' % load_path)
                    # if you are using PyTorch newer than 0.4 (e.g., built from
                    # GitHub source), you can remove str() on self.device
                    state_dict = torch.load(load_path, map_location=str(self.device))
                net.load_state_dict(state_dict)

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


######################################
#  Test code
######################################
#  import yaml
#  config_file = 'configs/alexnet.yaml'
#  with open(config_file, 'r') as f_in:
    #  cfg = yaml.load(f_in)
#  print(cfg)
#  model = Model()
#  model.initialize(cfg)
#  print(model.netB, model.netH, model.netD)
