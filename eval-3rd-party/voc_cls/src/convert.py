import numpy as np
import sys

import caffe

root1 = '/data/chongruo-data/deepblur/exp/alexnet_br_large/'
root2 = '/home/chongruo/code/caffe/models/'


net = caffe.Net( root1+'deploy_new.prototxt','./caffe_alexnet_train_iter_90000.caffemodel',caffe.TEST)

alexnet = caffe.Net(root2+'bvlc_reference_caffenet/deploy.prototxt', root2+'bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
        caffe.TEST)

params = ['conv1','conv2','conv3','conv4','conv5']

game_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

alex_params = {pr: (alexnet.params[pr][0].data, alexnet.params[pr][1].data) for pr in params}

for l in params:
    print 'game: {} weights are {} dimensional and biases are {} dimensional'.format(l, game_params[l][0].shape, game_params[l][1].shape)
    print 'firt params check before: weights: {}, bias: {}'.format(np.sum(alex_params[l][0] - game_params[l][0]),
            np.sum(alex_params[l][1] - game_params[l][1]))
    alex_params[l][0].flat = game_params[l][0].flat
    alex_params[l][1][...] = game_params[l][1]
    print 'params check: weights: {}, bias: {}\n\n'.format(np.sum(alex_params[l][0] - game_params[l][0]),
            np.sum(alex_params[l][1] - game_params[l][1]))

alexnet.save('./new.caffemodel')
