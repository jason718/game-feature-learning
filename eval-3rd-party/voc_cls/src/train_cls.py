from __future__ import division
from __future__ import print_function

# Train and evaluate a classification model for VOC2012
import argparse, pdb
import config
import numpy as np

parser = argparse.ArgumentParser(description='Train and evaluate a classification model for VOC')
# Model parameters
parser.add_argument('prototxt', help='prototxt file of the model to evaluate (deploy.prototxt preferred)')
parser.add_argument('caffemodel', help='model weights of the model to evaluate')

# Training parameters
parser.add_argument('--output-dir', help='Do we want to save the model?')
parser.add_argument('--voc-dir', default=config.VOC_DIR, help='The VOC2007 or VOC2012 data directory')
parser.add_argument('--gpu', type=int, help='What GPU do we train on')

# Solver parameters
parser.add_argument('--solver', default='SGD', help='Solver type')
parser.add_argument('-lr', type=float, default=0.001, help='Base LR')
parser.add_argument('-bs', type=int, default=16, help='batch size')
parser.add_argument('-nit', type=int, default=160000, help='Number of training iterations')
parser.add_argument('-step', type=int, default=10000, help='Step size')

# Initialization parameters
parser.add_argument('--no-mean', action='store_true', help='Do not mean center')
parser.add_argument('--clip', default='drop7', help='clip the network at this layer')
parser.add_argument('--train-from', default=None, help='Train only layers after this layer')
parser.add_argument('--random-from', default='drop7', help='Initialize all layers after this layer randomly')
parser.add_argument('--scale', type=float, default=1, help='Scale of the input data')
parser.add_argument('--min-scale', type=float, default=0.5, help='Minimum scale transformation of image')
parser.add_argument('--max-scale', type=float, default=2.0, help='Maximum scale transformation of image')
parser.add_argument('--resize', type=int, help='Resize the images before passing them to the CNN')

args = parser.parse_args()

assert args.voc_dir is not None, "VOC_DIR is required as parameter or as user_config.py value"

# Create the output directory
output_dir = args.output_dir
if output_dir is None:
	from tempfile import mkdtemp
	output_dir = mkdtemp()+'/'

from os import path, mkdir
if not path.isdir(output_dir):
	mkdir(output_dir)

# Define the classification model to use
from caffe_all import *
from data import *
from util import *
from python_layers import Py
import load

model = load.ProtoDesc(args.prototxt)

# Create the training net
ns = NetSpec()
mean_value = [104,117,123]
if args.no_mean: mean_value = [0,0,0]
ns.data, ns.cls = dataLayer(args.voc_dir, output_dir, batch_size=args.bs, transform_param=dict(crop_size=model.input_dim[-1], min_scale=args.min_scale, max_scale=args.max_scale, mean_value=mean_value, mirror=True, scale=args.scale), resize=args.resize)

ns.fc8  = L.InnerProduct( model(data=ns.data, clip=args.clip), num_output=20, name='fc8_cls')
ns.loss = Py.SigmoidCrossEntropyLoss(ns.fc8, ns.cls, ignore_label=255, loss_weight=1)

#ns.prnt = Py.Print(ns.cls)


#setLR(listAllTops(ns.fc8), 1, 2)
#setDecay(listAllTops(ns.fc8), 1, 1)
# Set the learning rates
all_tops = listAllTops(ns.fc8)
train_t = args.train_from is None
for t in all_tops:
    if not train_t:
        train_t = args.train_from == t.fn.params.get('name','')
    if train_t:
        if t.fn.params.get('name','') == 'fc8_cls':
            setLR(t, 10, 20)
        elif t.fn.params.get('name','') == 'fc7':
            setLR(t, 10, 20)
        elif t.fn.params.get('name','') == 'fc6':
            setLR(t, 10, 20)
        else:
            setLR(t, 1, 2)
        setDecay(t, 1, 1)
    else:
        setLR(t, 0, 0)
        setDecay(t, 0, 0)
if not train_t:
	print("Something went wrong, not training any layers!")


# Choose the GPU
caffe.set_mode_gpu()
if args.gpu is not None:
	caffe.set_device(args.gpu)

# Save the file
prototxt = output_dir+'trainval.prototxt'
f_out = open(prototxt, 'w')
f_out.write(str(ns.to_proto()))
f_out.close()
print('save train_val prototxt to: ',prototxt)

# Run the solver
if args.nit:
	from solver import Solver
	# There should be no need to tune the solver parameters
	s = Solver(prototxt, output_dir+'final.caffemodel', output_dir+'snap.caffemodel', solver=args.solver,
            base_lr=args.lr, weight_decay=1e-6, log_file=output_dir+'log.txt', clip_gradients=10,
            lr_policy="step",gamma=0.5, stepsize = args.step)

	s.solver.net.copy_from(args.caffemodel)
	#pdb.set_trace()
	if args.random_from is not None:
		sr = False
		for l,n in zip(s.solver.net.layers, s.solver.net._layer_names):
			if not sr:
				sr = n == args.random_from
			if sr and len(l.blobs)>0:
				l.blobs[0].data[...] = np.random.normal(0, 0.01, l.blobs[0].shape)
				if len(l.blobs)>1:
					l.blobs[1].data[...] = 0.1
	#		if n == 'conv1' or n == 'conv2':
	#			if len(l.blobs) > 0:
	#				l.blobs[0].data[...] = np.random.normal(0, 0.01, l.blobs[0].shape)
	#			if len(l.blobs)>1:
	#				l.blobs[1].data[...] = 0.1

	s.run(args.nit)


# Evaluate the model
for N_CROP in [1, 10]:
	print(N_CROP)
	for t in ['test', 'train']:
		# Specify the eval net
		ns = NetSpec()
		ns.data, ns.cls = dataLayer(args.voc_dir, output_dir, batch_size=1, transform_param=dict(crop_size=model.input_dim[-1], min_scale=args.min_scale, max_scale=args.max_scale, mean_value=mean_value, mirror=True, scale=args.scale), image_set=t, resize=args.resize)
		ns.fc8  = L.InnerProduct( model(data=ns.data, clip=args.clip), num_output=20, name='fc8_cls')

		# Create the eval net
		from util import sglob
		net = caffe.get_net_from_string(str(ns.to_proto()), caffe.TEST)
		files = sglob(output_dir+'*.caffemodel')
		if len(files)>0:
			net.copy_from(files[0])

		# Get the number of images
		N = nImages(args.voc_dir, t)

		# and evaluate
		gts, scr = [], []
		from progressbar import *
		progress = ProgressBar(widgets=['%-10s %2d '%(t, N_CROP), Percentage(), Bar(), ETA()])
		for i in progress(range(N_CROP*N)):
			r = net.forward()
			if i < N:
				scr.append(1*r['fc8'])
				gts.append(1*r['cls'])
			else:
				scr[i%N] += r['fc8']
		gts = np.concatenate(gts, axis=0).T
		scr = np.concatenate(scr, axis=0).T
		print(gts)
		from sklearn import metrics
		aps = []
		for i in range(20):
			# Subtract eps from score to make AP work for tied scores
			ap = metrics.average_precision_score(gts[i][gts[i]<=1], scr[i][gts[i]<=1]-1e-5*gts[i][gts[i]<=1])
			aps.append( ap )
		print( np.mean(aps), '  ', ' '.join(['%0.2f'%a for a in aps]) )


if args.output_dir is None:
	from shutil import rmtree
	rmtree(output_dir)

exit(0)
