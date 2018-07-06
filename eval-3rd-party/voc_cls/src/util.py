from __future__ import division
from __future__ import print_function
import numpy as np


def loadNet(basename):
    from os import path
    from glob import glob
    prototxt = basename + 'trainval.prototxt'
    weights = glob(basename + '*.caffemodel')
    caffemodel = weights[np.argmax([path.getmtime(c) for c in weights])]
    print("Loading", prototxt, caffemodel)
    from caffe_all import caffe
    return caffe.Net(prototxt, caffemodel, caffe.TEST)


def addCaffePath():
    # Find the caffe binary
    import sys
    import config
    PD = config.CAFFE_DIR + '/python'
    if PD not in sys.path:
        sys.path.append(PD)


def prepareImage(im, shape):
    from skimage import transform
    tim = transform.resize(im, (int(shape[0]), int(shape[1]), 3))
    mean_value = np.array([104, 117, 123])
    return tim.transpose([2, 0, 1])[::-1] - mean_value[:, None, None]


def getDataDir(basename):
    import re
    r = re.compile('source : "(.*)_.*?_lmdb"')
    f = open(basename + 'trainval.prototxt', 'r')
    m = [r.search(l) for l in f]
    f.close()
    m = [i.group(1) for i in m if i is not None]
    if len(m):
        return m[0]
    return None


def hdf5DataDims(filename):
    import h5py
    f = h5py.File(filename, 'r')
    r = {s: f[s]['0'].shape for s in f}
    f.close()
    return r


def netFromProtoString(s, weights=None, save_path=None):
    import tempfile
    from caffe_all import caffe
    # Save the encoder net
    if save_path is None:
        f_out = tempfile.NamedTemporaryFile(mode='w+')
        save_path = f_out.name
    else:
        f_out = open(save_path, 'w')
    f_out.write(s)
    f_out.flush()
    if weights is None:
        net = caffe.Net(save_path, caffe.TEST)
    else:
        net = caffe.Net(save_path, weights, caffe.TEST)
    f_out.close()
    return net


def solver(train_net, base_lr=0.001, lr_policy='step', gamma=0.01, display=100,
           stepsize=10000, max_iter=30000, momentum=0.9, weight_decay=5e-5, clip_gradients=100,
           solver_mode='GPU', solver_type='SGD', save_path=None):
    import tempfile
    from caffe_all import caffe
    SOLVER_STR = """train_net: "%s"
    base_lr: %f
    lr_policy: "%s"
    gamma: %f
    stepsize: %d
    display: %d
    max_iter: %d
    momentum: %f
    iter_size: 32
    weight_decay: %f
    clip_gradients: %f
    solver_mode: %s
    solver_type: %s""" % (train_net, base_lr, lr_policy, gamma, stepsize,
                          display, max_iter, momentum, weight_decay,
                          clip_gradients, solver_mode, solver_type)
    if save_path is None:
        f_out = tempfile.NamedTemporaryFile(mode='w+')
        save_path = f_out.name
    else:
        f_out = open(save_path, 'w')
    f_out.write(SOLVER_STR)
    f_out.flush()
    solver = caffe.get_solver(save_path)
    f_out.close()
    return solver


def sglob(s):
    import glob
    import os
    return list(reversed(sorted(glob.glob(s), key=os.path.getmtime)))

def waitKey(fig):
	from pylab import waitforbuttonpress
	r = [None]
	fig.canvas.mpl_connect('key_press_event', lambda e: r.__setitem__(0, e.key))
	while not waitforbuttonpress():
		pass
	return r[0]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



LAYERS_WITH_PARAM = ['Convolution', 'Deconvolution', 'InnerProduct', 'Bias', 'Scalar']


def setParam(tops, name, p0, p1=None):
    try:
        tops = tops.values()
    except AttributeError:
        pass
    if hasattr(tops, '__iter__'):
        for t in tops:
            setParam(t, name, p0, p1)
    elif tops.fn.type_name in LAYERS_WITH_PARAM:
        if 'param' not in tops.fn.params:
            tops.fn.params['param'] = [{}]
        if not isinstance(tops.fn.params['param'], list):
            tops.fn.params['param'] = [tops.fn.params['param']]
        tops.fn.params['param'][0][name] = p0
        if p1 is not None:
            if len(tops.fn.params['param']) < 2:
                tops.fn.params['param'].append({})
            tops.fn.params['param'][1][name] = p1


def listAllTops(*tops):
	r = []
	for t in list(tops):
		for tt in listAllTops(*t.fn.inputs):
			if not tt in r:
				r.append(tt)
	for tt in tops:
		if not tt in r:
			r.append(tt)
	return r


def setLR(tops, weight_lr, bias_lr=None):
    setParam(tops, 'lr_mult', weight_lr, bias_lr)


def setDecay(tops, weight_decay, bias_decay=None):
    setParam(tops, 'decay_mult', weight_decay, bias_decay)

PARAM_NAME = {'Convolution': 'convolution_param',
              'Deconvolution': 'convolution_param',
              'InnerProduct': 'inner_product_param'}


def setFiller(tops, name, type='constant', a=None, b=None):
    try:
        for t in tops:
            if tops[t].fn.type_name in PARAM_NAME:
                setFiller(tops[t], name, type, a, b)
        return
    except TypeError:
        pass
    param = PARAM_NAME[tops.fn.type_name]
    if param not in tops.fn.params:
        tops.fn.params[param] = {}
    filler = {}
    filler['type'] = type
    if type == 'constant' and a is not None:
        filler['value'] = a
    if type == 'uniform':
        if a is not None:
            filler['min'] = a
        if b is not None:
            filler['max'] = b
    if type == 'gaussian':
        if a is not None:
            filler['mean'] = a
        if b is not None:
            filler['std'] = b
    tops.fn.params[param][name.lower() + '_filler'] = filler


def setWeightFiller(tops, *args, **kwargs):
    setFiller(tops, 'weight', *args, **kwargs)

def setBiasFiller(tops, *args, **kwargs):
    setFiller(tops, 'bias', *args, **kwargs)

def makeFCN(f, kernel_size=6, pad=True):
	first_ip = True
	for t in listAllTops(f):
		if t.fn.type_name == 'Convolution':
			p = t.fn.params['convolution_param']
			if pad: p['pad'] = int(kernel_size/2)
		if t.fn.type_name == 'Pooling':
			p = t.fn.params['pooling_param']
			if pad: p['pad'] = int((p['kernel_size']-1)/2)
		if t.fn.type_name == 'InnerProduct':
			t.fn.type_name = 'Convolution'
			t.fn.params['convolution_param'] = {}
			if 'inner_product_param' in t.fn.params:
				t.fn.params['convolution_param'] = t.fn.params['inner_product_param']
			if first_ip:
				t.fn.params['convolution_param']['kernel_size'] = kernel_size
				if pad: t.fn.params['convolution_param']['pad'] = int(kernel_size/2)
				first_ip = False
			else:
				t.fn.params['convolution_param']['kernel_size'] = 1
			if 'inner_product_param' in t.fn.params:
				del t.fn.params['inner_product_param']


def copyFCN(net, old_net):
	def matchShapes(s1, s2):
		return np.prod(s1[0].shape) == np.prod(s2[0].shape)
	def matchNets(net1, net2):
		# Fetch all layers with parameters
		n1, p1 = list(zip(*[(n,l) for n, l in zip(net1._layer_names, net1.layers) if len(l.blobs)>0]))
		n2, p2 = list(zip(*[(n,l) for n, l in zip(net2._layer_names, net2.layers) if len(l.blobs)>0]))

		# Find the optimal mapping between input and output parameters
		C = 100*np.ones((len(p1)+2,len(p2)+2), dtype=int)
		D = 0*np.ones((len(p1)+2,len(p2)+2), dtype=int)
		C[0,0] = 0
		for i in range(len(p1)+1):
			for j in range(len(p2)+1):
				if i and C[i,j] < C[i+1,j]:
					C[i+1,j] = C[i,j]
					D[i+1,j] = 1
				if i < len(p1) and j < len(p2) and matchShapes(p1[i].blobs, p2[j].blobs) and C[i,j] < C[i+1,j+1]:
					C[i+1,j+1] = C[i,j]
					D[i+1,j+1] = 2
				if C[i,j]+1 < C[i,j+1]: # Pay a cost to skip
					C[i,j+1] = C[i,j]+1
					D[i,j+1] = 3

		i, j = len(p1), len(p2)
		match = -np.ones(len(p2), dtype=int)
		while i>0 and j>0:
			if D[i,j] == 2:
				match[j-1] = i-1
				i, j = i-1, j-1
			elif D[i,j] == 1:
				i -= 1
			elif D[i,j] == 3:
				j -= 1
			else:
				print( "Giving up", i, j )
				break
		return n1,p1,n2,p2,match
	
	n1,p1,n2,p2,match = matchNets(old_net, net)
	
	# Translate the model
	for i,m in enumerate(match):
		if m >= 0:
			for a, b in zip(p1[m].blobs, p2[i].blobs):
				b.data.flat[...] = a.data.ravel()
	
def initFCN(net, encoder, weights):
	from caffe_all import caffe, L, P

	t = encoder(data=L.DummyData(shape=dict(dim=[1]+encoder.input_dim)), clip='drop7')
	tmp_net = caffe.get_net_from_string(str(t.to_proto()), caffe.TEST)
	tmp_net.copy_from(weights)
	copyFCN(net, tmp_net)
