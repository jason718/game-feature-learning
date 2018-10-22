import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from prototxt import *
from darknet import MaxPoolStride1

class FCView(nn.Module):
    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        nB = x.data.size(0)
        x = x.view(nB,-1)
        return x
    def __repr__(self):
        return 'view(nB, -1)'

class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def forward(self, x1, x2):
        if self.operation == '+' or self.operation == 'SUM':
            x = x1 + x2
        if self.operation == '*' or self.operation == 'MUL':
            x = x1 * x2
        if self.operation == '/' or self.operation == 'DIV':
            x = x1 / x2
        return x

class CaffeNet(nn.Module):
    def __init__(self, protofile):
        super(CaffeNet, self).__init__()
        self.net_info = parse_prototxt(protofile)
        self.models = self.create_network(self.net_info)
        for name,model in self.models.items():
            self.add_module(name, model)

        if self.net_info['props'].has_key('input_shape'):
            self.width = int(self.net_info['props']['input_shape']['dim'][3])
            self.height = int(self.net_info['props']['input_shape']['dim'][2])
        else:
            self.width = int(self.net_info['props']['input_dim'][3])
            self.height = int(self.net_info['props']['input_dim'][2])
        self.has_mean = False

    def forward(self, data):
        if self.has_mean:
            batch_size = data.data.size(0)
            data = data - torch.autograd.Variable(self.mean_img.repeat(batch_size, 1, 1, 1))

        blobs = OrderedDict()
        blobs['data'] = data
        
        layers = self.net_info['layers']
        layer_num = len(layers)
        i = 0
        tdata = None
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            ltype = layer['type']
            tname = layer['top']
            bname = layer['bottom']
            if ltype == 'Data' or ltype == 'Accuracy' or ltype == 'SoftmaxWithLoss' or ltype == 'Region':
                i = i + 1
                continue
            elif ltype == 'BatchNorm':
                i = i + 1
                tname = layers[i]['top']

            if ltype != 'Eltwise':
                bdata = blobs[bname]
                tdata = self._modules[lname](bdata)
                blobs[tname] = tdata
            else:
                bdata0 = blobs[bname[0]]
                bdata1 = blobs[bname[1]]
                tdata = self._modules[lname](bdata0, bdata1)
                blobs[tname] = tdata
            i = i + 1
        return tdata # blobs.values()[len(blobs)-1]

    def print_network(self):
        print(self)
        print_prototxt(self.net_info)

    def load_weights(self, caffemodel):
        if self.net_info['props'].has_key('mean_file'):
            import caffe_pb2
            self.has_mean = True
            mean_file = self.net_info['props']['mean_file']
            blob = caffe_pb2.BlobProto()
            blob.ParseFromString(open(mean_file, 'rb').read())
            mean_img = torch.from_numpy(np.array(blob.data)).float()

            if self.net_info['props'].has_key('input_shape'):
                channels = int(self.net_info['props']['input_shape']['dim'][1])
                height = int(self.net_info['props']['input_shape']['dim'][2])
                width = int(self.net_info['props']['input_shape']['dim'][3])
            else:
                channels = int(self.net_info['props']['input_dim'][1])
                height = int(self.net_info['props']['input_dim'][2])
                width = int(self.net_info['props']['input_dim'][3])
            mean_img = mean_img.view(channels, height, width)#.mean(0)
            #mean_img = mean_img.repeat(3, 1, 1)
            self.register_buffer('mean_img', torch.zeros(channels, height, width))
            self.mean_img.copy_(mean_img)

        model = parse_caffemodel(caffemodel)
        layers = model.layer
        if len(layers) == 0:
            print('Using V1LayerParameter')
            layers = model.layers

        lmap = {}
        for l in layers:
            lmap[l.name] = l

        layers = self.net_info['layers']
        layer_num = len(layers)
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            ltype = layer['type']
            if ltype == 'Convolution':
                convolution_param = layer['convolution_param']
                bias = True
                if convolution_param.has_key('bias_term') and convolution_param['bias_term'] == 'false':
                    bias = False
                self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                if bias and len(lmap[lname].blobs) > 1:
                    print('convlution %s has bias' % lname)
                    self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                i = i + 1
            elif ltype == 'BatchNorm':
                scale_layer = layers[i+1]
                self.models[lname].running_mean.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data) / lmap[lname].blobs[2].data[0]))
                self.models[lname].running_var.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data) / lmap[lname].blobs[2].data[0]))
                self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[scale_layer['name']].blobs[0].data)))
                self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[scale_layer['name']].blobs[1].data)))
                i = i + 2
            elif ltype == 'InnerProduct':
                if type(self.models[lname]) == nn.Sequential:
                    self.models[lname][1].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                    if len(lmap[lname].blobs) > 1:
                        self.models[lname][1].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                else:
                    self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                    if len(lmap[lname].blobs) > 1:
                        self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                i = i + 1
            elif ltype == 'Pooling' or ltype == 'Eltwise' or ltype == 'ReLU' or ltype == 'Region':
                i = i + 1
            else:
                print('load_weights: unknown type %s' % ltype)
                i = i + 1

    def create_network(self, net_info):
        models = OrderedDict()
        blob_channels = dict()
        blob_width = dict()
        blob_height = dict()

        layers = net_info['layers']
        props = net_info['props']
        layer_num = len(layers)

        if props.has_key('input_shape'):
            blob_channels['data'] = int(props['input_shape']['dim'][1])
            blob_height['data'] = int(props['input_shape']['dim'][2])
            blob_width['data'] = int(props['input_shape']['dim'][3])
        else:
            blob_channels['data'] = int(props['input_dim'][1])
            blob_height['data'] = int(props['input_dim'][2])
            blob_width['data'] = int(props['input_dim'][3])
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            ltype = layer['type']
            if ltype == 'Data':
                i = i + 1
                continue
            bname = layer['bottom']
            tname = layer['top']
            if ltype == 'Convolution':
                convolution_param = layer['convolution_param']
                channels = blob_channels[bname]
                out_filters = int(convolution_param['num_output'])
                kernel_size = int(convolution_param['kernel_size'])
                stride = int(convolution_param['stride']) if convolution_param.has_key('stride') else 1
                pad = int(convolution_param['pad']) if convolution_param.has_key('pad') else 0
                group = int(convolution_param['group']) if convolution_param.has_key('group') else 1
                bias = True
                if convolution_param.has_key('bias_term') and convolution_param['bias_term'] == 'false':
                    bias = False
                models[lname] = nn.Conv2d(channels, out_filters, kernel_size, stride,pad,group, bias=bias)
                blob_channels[tname] = out_filters
                blob_width[tname] = (blob_width[bname] + 2*pad - kernel_size)/stride + 1
                blob_height[tname] = (blob_height[bname] + 2*pad - kernel_size)/stride + 1
                i = i + 1
            elif ltype == 'BatchNorm':
                assert(i + 1 < layer_num)
                assert(layers[i+1]['type'] == 'Scale')
                momentum = 0.9
                if layer.has_key('batch_norm_param') and layer['batch_norm_param'].has_key('moving_average_fraction'):
                    momentum = float(layer['batch_norm_param']['moving_average_fraction'])
                channels = blob_channels[bname]
                models[lname] = nn.BatchNorm2d(channels, momentum=momentum)
                tname = layers[i+1]['top']
                blob_channels[tname] = channels
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 2
            elif ltype == 'ReLU':
                inplace = (bname == tname)
                if layer.has_key('relu_param') and layer['relu_param'].has_key('negative_slope'):
                    negative_slope = float(layer['relu_param']['negative_slope'])
                    models[lname] = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
                else:
                    models[lname] = nn.ReLU(inplace=inplace)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Pooling':
                kernel_size = int(layer['pooling_param']['kernel_size'])
                stride = int(layer['pooling_param']['stride'])
                padding = 0
                if layer['pooling_param'].has_key('pad'):
                    padding = int(layer['pooling_param']['pad'])
                pool_type = layer['pooling_param']['pool']
                if pool_type == 'MAX' and kernel_size == 2 and stride == 1: # for tiny-yolo-voc
                    models[lname] = MaxPoolStride1()
                    blob_width[tname] = blob_width[bname]
                    blob_height[tname] = blob_height[bname]
                else:
                    if pool_type == 'MAX':
                        models[lname] = nn.MaxPool2d(kernel_size, stride, padding=padding)
                    elif pool_type == 'AVE':
                        models[lname] = nn.AvgPool2d(kernel_size, stride, padding=padding)

                    if stride > 1:
                        blob_width[tname] = (blob_width[bname] - kernel_size + 1)/stride + 1
                        blob_height[tname] = (blob_height[bname] - kernel_size + 1)/stride + 1
                    else:
                        blob_width[tname] = blob_width[bname] - kernel_size + 1
                        blob_height[tname] = blob_height[bname] - kernel_size + 1
                blob_channels[tname] = blob_channels[bname]
                i = i + 1
            elif ltype == 'Eltwise':
                operation = 'SUM'
                if layer.has_key('eltwise_param') and layer['eltwise_param'].has_key('operation'):
                    operation = layer['eltwise_param']['operation']
                bname0 = bname[0]
                bname1 = bname[1]
                models[lname] = Eltwise(operation)
                blob_channels[tname] = blob_channels[bname0]
                blob_width[tname] = blob_width[bname0]
                blob_height[tname] = blob_height[bname0]
                i = i + 1
            elif ltype == 'InnerProduct':
                filters = int(layer['inner_product_param']['num_output'])
                if blob_width[bname] != -1 or blob_height[bname] != -1:
                    channels = blob_channels[bname] * blob_width[bname] * blob_height[bname]
                    models[lname] = nn.Sequential(FCView(), nn.Linear(channels, filters))
                else:
                    channels = blob_channels[bname]
                    models[lname] = nn.Linear(channels, filters)
                blob_channels[tname] = filters
                blob_width[tname] = -1
                blob_height[tname] = -1
                i = i + 1
            elif ltype == 'Softmax':
                models[lname] = nn.Softmax()
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = -1
                blob_height[tname] = -1
                i = i + 1
            elif ltype == 'SoftmaxWithLoss':
                loss = nn.CrossEntropyLoss()
                blob_width[tname] = -1
                blob_height[tname] = -1
                i = i + 1
            elif ltype == 'Region':
                anchors = layer['region_param']['anchors'].strip('"').split(',')
                self.anchors = [float(j) for j in anchors]
                self.num_anchors = int(layer['region_param']['num'])
                self.anchor_step = len(self.anchors)/self.num_anchors
                self.num_classes = int(layer['region_param']['classes'])
                i = i + 1
            else:
                print('create_network: unknown type #%s#' % ltype)
                i = i + 1
        return models

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print('Usage: python caffenet.py model.prototxt model.caffemodel imgfile')
        print('')
        print('e.g. python caffenet.py ResNet-50-deploy.prototxt ResNet-50-model.caffemodel test.png')
        exit()
    from torch.autograd import Variable
    protofile = sys.argv[1]
    net = CaffeNet(protofile)
    net.print_network()

    net.load_weights(sys.argv[2])
    from PIL import Image
    img = Image.open(sys.argv[3]) #.convert('RGB')
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3)#.transpose(0,1).transpose(0,2).contiguous()
    img = torch.stack([img[:,:,2], img[:,:,1], img[:,:,0]], 0)
    img = img.view(1, 3, height, width)
    img = img.float()
    img = torch.autograd.Variable(img)
    output = net(img)
