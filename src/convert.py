#######################################################
### This is function is deprecated manily because the 
### netowrk file changed. However, you can still see how
### did I convert pytorch model into a caffe model.
### convert() -> rename_caffemodel()
### Will make this working and merge into master when I 
### get free time.
########################################################


import torch
import torch.nn as nn
import numpy as np

# absorbing the batch norm
def absorb_bn(conv, bn):
    print '==> absorbing batch norm'
    # get the parameters
    bn_mean = bn.running_mean.float().numpy()
    eps = 1e-5
    bn_var = bn.running_var.float().numpy() + eps
    bn_gamma = bn.weight.data.float().numpy()
    bn_beta = bn.bias.data.float().numpy()
    conv_w = conv.weight.data.float().numpy()
    conv_b = conv.bias.data.float().numpy()
    #  print bn_mean.size(), bn_var.size(), bn_gamma.size(), bn_beta.size(), conv_w.size(), conv_b.size()

    # absorbing the weights
    slope = np.sqrt(1./bn_var)
    ch = conv_w.shape[0]
    new_conv_w = conv_w * slope.reshape(ch,1,1,1) * bn_gamma.reshape(ch,1,1,1)
    new_conv_b = (conv_b - bn_mean) * slope * bn_gamma + bn_beta

    conv.weight.data = torch.from_numpy(new_conv_w)
    conv.bias.data = torch.from_numpy(new_conv_b)
    return conv

# Change Fully_convolutional layer into fc
def conv2fc(conv, cin=9216, cout=4096):
    print '==> converting fc'
    conv_w = conv.weight.data
    conv_b = conv.bias.data
    #  print conv_w.size(), conv_b.size()
    fc = torch.nn.Linear(cin, cout)
    #  print fc.weight.data.size()
    fc.weight.data = conv_w.view(cout, cin)
    fc.weight.biad = conv_b
    #  print fc.weight.data.size(), fc.bias.data.size()
    return fc

#######################################################
### This is function is deprecated manily because the 
### netowrk file changed. However, you can still see how
### did I convert pytorch model into a caffe model.
### Will make this working and merge into master when I 
### get free time.
########################################################

def convert(checkpoint_file, l='conv1'):
    from network import alex_base_side_out, alex_wo_bn
    from convertor.pytorch2caffe import pytorch2caffe
    netB = alex_base_side_out(1, l)
    netB.eval()
    netB.load_state_dict(torch.load(checkpoint_file))
    net = alex_wo_bn(1)

    conv1 = absorb_bn(netB.conv1[0], netB.conv1[1])
    net.features[0].weight.data = conv1.weight.data
    net.features[0].bias.data = conv1.bias.data

    conv2 = absorb_bn(netB.conv2[2], netB.conv2[3])
    net.features[3].weight.data = conv2.weight.data
    net.features[3].bias.data = conv2.bias.data

    conv3 = absorb_bn(netB.conv3[2], netB.conv3[3])
    net.features[6].weight.data = conv3.weight.data
    net.features[6].bias.data = conv3.bias.data

    conv4 = absorb_bn(netB.conv4[1], netB.conv4[2])
    net.features[8].weight.data = conv4.weight.data
    net.features[8].bias.data = conv4.bias.data

    conv5 = absorb_bn(netB.conv5[1], netB.conv5[2])
    net.features[10].weight.data = conv5.weight.data
    net.features[10].bias.data = conv5.bias.data

    fc6_conv = absorb_bn(netB.fc6[2], netB.fc6[3])
    fc6 = conv2fc(fc6_conv, 9216, 4096)
    net.classifier[0].weight.data = fc6.weight.data
    net.classifier[0].bias.data = fc6.bias.data

    fc7_conv = absorb_bn(netB.fc7[1], netB.fc7[2])
    fc7 = conv2fc(fc7_conv, 4096, 4096)
    net.classifier[3].weight.data = fc7.weight.data
    net.classifier[3].bias.data = fc7.bias.data

    name = 'caffenet-pt2cf-gan'
    net.eval()
    #  torch.save(net.state_dict(), name+'.pth')
    input_var = Variable(torch.rand(1, 3, 227, 227))
    output_var = net(input_var)
    pytorch2caffe(input_var, output_var, name+'.prototxt', name+'.caffemodel')


# rename the caffemodel into a standard caffenent
def rename_caffemodel(p_alex, p_game, m_game, final_out, dummy_m):
    import caffe
    caffe.set_mode_gpu()
    net = caffe.Net(p_game, m_game, caffe.TEST)
    alexnet = caffe.Net(p_alex, caffe.TEST)

    params_alex = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']
    params_game = ['ConvNdBackward1', 'ConvNdBackward4', 'ConvNdBackward7', 'ConvNdBackward9', 'ConvNdBackward11',
            'AddmmBackward14', 'AddmmBackward17']
    #  params_game = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']

    game_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params_game}
    alex_params = {pr: (alexnet.params[pr][0].data, alexnet.params[pr][1].data) for pr in params_alex}

    for i in range(len(params_alex)):
        g_name = params_game[i]
        a_name = params_alex[i]
        game_params = (net.params[g_name][0].data, net.params[g_name][1].data)
        alex_params = (alexnet.params[a_name][0].data, alexnet.params[a_name][1].data)
        print ' \ngame: {} weight: {}  &   bias are {}'.format(g_name, game_params[0].shape, game_params[1].shape)
        print 'alexnet: {} weight: {}  &   bias are {}'.format(a_name, alex_params[0].shape, alex_params[1].shape)
        print 'params check before: weights: %.3f, bias: %.3f' % (np.sum(alex_params[0]) - np.sum(game_params[0]), np.sum(alex_params[1]) - np.sum(game_params[1]))

        alex_params[0].flat = game_params[0].flat
        alex_params[1][...] = game_params[1].flat

        print 'params check after:  weights: %f, bias: %f' % (np.sum(alex_params[0]) - np.sum(game_params[0]),np.sum(alex_params[1]) - np.sum(game_params[1]))

    print '\n==> saved to ' + final_out
    alexnet.save(final_out)