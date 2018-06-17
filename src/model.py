import os
import torch
from collections import OrderedDict

import networks

#########################################################################
#  Network definition
#  Options:
#      alexnet(caffenet-definition), vgg16, resnet(TODO)
#  Note:
#      github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet
#      use this to follow previous paper's practice.
##########################################################################
class Model():
    def initialize(self, cfg):
        self.cfg = cfg
        #  self.save_dir = os.path.join(cfg['checkpoints_dir'], cfg['archi'])

        # if using GPUs
        if not cfg['cpu_mode']:
            #  self.gpu_ids = cfg.gpu_ids
            #  self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
            torch.backends.cudnn.benchmark = True

        #  # specify losses
        #  self.loss_name = ['l_norm', 'l_dep', 'l_edge']
        #  if self.cfg['use_DA']:
        #      self.loss_name += ['loss_D', 'loss_G']

        # define network
        if cfg['archi'] == 'alexnet':
            self.netB = networks.netB_alexnet()
            self.netH = networks.netH_alexnet()
            if self.cfg['use_DA'] and self.cfg['isTrain']:
                self.netD = networks.netD_alexnet(cfg)
        elif cfg['archi'] == 'vgg16':
            self.netB = networks.netB_vgg16(cfg)
            self.netH = networks.netH_vgg16(cfg)
            if self.cfg['use_DA'] and self.cfg['isTrain']:
                self.netD = netD_vgg16(cfg)
        elif 'resnet' in cfg['archi']:
            raise NotImplementedError
            self.netB = networks.netB_resnet(cfg)
            self.netH = networks.netH_resnet(cfg)
            if self.cfg['use_DA'] and self.cfg['isTrain']:
                self.netD = networks.netD_resnet(cfg)
        else:
            raise ValueError('Un-supported network')

        if cfg['isTrain']:
            self.schedulers = [networks.get_scheduler(cfgimizer, cfg) for cfgimizer in self.cfgimizers]

        if not cfg['isTrain'] or cfg.continue_train:
            self.load_networks(cfg.which_epoch)

    def set_input(self, input):
        self.input = input

    def forward(self):
        self.feat_syn = self.netB(self.input_syn, self.cfg.DA_feat)

        if self.cfg['use_DA'] and self.cfg['isTrain']:
            self.feat_real = self.netB(self.input_real, self.cfg.DA_feat)

    def backward_BH(self):
        # compute prediction
        self.norm_pred, self.dep_pred, self.edge_pred = netH(self.feat_syn)

        # compute loss
        self.loss_dep  = self.cfg['loss_dep_weight'] * criterion_dep(self.dep_pred, self.dep_gt)
        self.loss_norm = self.cfg['loss_norm_weight'] * criterion_norm(self.norm_pred, self.norm_gt)
        self.loss_edge = self.cfg['loss_edge_weight'] * criterion_edge(self.edge_pred, self.edge_gt)

        # backward from 3 tasks
        loss = self.loss_dep + self.loss_edge + self.loss_norm

        if self.cfg['use_DA']:
            pred_syn = self.netD(self.feat_syn.detach())
            self.loss_DA = self.criterionGAN(pred_syn, True)
            loss += self.loss_DA * self.cfg['loss_DA_weight']

        loss.backward()

    def backward_D(self):
        # Synthetic
        # stop backprop to netB by detaching
        _feat_s = self.syn_pool.query(self.feat_syn.detach())
        pred_syn = self.netD(_feat_s)
        self.loss_D_syn = self.criterionGAN(pred_syn, False)

        # Real
        _feat_r = torch.real_pool.query(self.feat_real.detach())
        pred_real = self.netD(_feat_r)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_syn + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def optimize(self):
        self.forward()
        # if DA, update on real data
        if self.cfg['use_DA']:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # update on synthetic data
        self.set_requires_grad(self.netB, True)
        self.set_requires_grad(self.netH, True)
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
        lr = self.cfgimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # load models from the disk
    def load_networks(self, which_epoch):
	    pass

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

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
import yaml
config_file = 'configs/alexnet.yaml'
with open(config_file, 'r') as f_in:
    cfg = yaml.load(f_in)
print(cfg)
model = Model()
model.initialize(cfg)
