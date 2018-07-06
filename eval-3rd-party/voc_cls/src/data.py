from __future__ import division
from __future__ import print_function
try:
	FileNotFoundError
except NameError:
	FileNotFoundError = OSError # Python 2

from python_layers import PyLayer
import numpy as np

class VOCData:
	def __init__(self, voc_dir, image_set='train'):
		from os import path
		from glob import glob
		from random import shuffle
		
		# Find the image sets
		image_set_dir = path.join(voc_dir, 'ImageSets', 'Main')
		trainval_sets = glob( path.join(image_set_dir, '*_trainval.txt') )
		test_sets = glob( path.join(image_set_dir, '*_test.txt') )
		if len(test_sets) == 0:
			print( "No test set found, evaluating on the val set" )
			#test_sets = glob( path.join(image_sets, '*_val.txt') )
			#trainval_sets = glob( path.join(image_sets, '*_train.txt') )
			test_sets = glob( path.join(image_set_dir, '*_val.txt') )
			trainval_sets = glob( path.join(image_set_dir, '*_train.txt') )
		assert len(test_sets) == len(trainval_sets), "the same number of training and testing classes required"
		assert len(test_sets) > 0, "at least one training class required 'voc_dir' might point to the wrong location"
		image_sets = trainval_sets if 'train' in image_set else test_sets
		
		# Read the labels
		self.n_labels = len(image_sets)
		self.labels = {}
		for k,s in enumerate(sorted(image_sets)):
			for l in open(s, 'r'):
				name, lbl = l.strip().split()
				lbl = int(lbl)
				if name not in self.labels:
					self.labels[name] = -np.ones(self.n_labels, dtype=np.uint8)
				# Switch the ignore label and 0 label (in VOC -1: not present, 0: ignore)
				if lbl < 0:
					lbl = 0
				elif lbl == 0:
					lbl = 255
				self.labels[name][k] = lbl
		
		self.image_names = sorted(self.labels)
		shuffle(self.image_names)
		self.image_path = {}
		for n in self.image_names:
			self.image_path[n] = path.join(voc_dir, 'JPEGImages', n+'.jpg')
		self.n_images = len(self.image_names)
	
	def __len__(self):
		return self.n_images
	
	def __getitem__(self, i):
		from PIL import Image
		if isinstance(i, int):
			i = self.image_names[i]
		return self.labels[i], np.array(Image.open(self.image_path[i]))


class LabelData(PyLayer):
	label = None
	batch_size = 1
	
	def setup(self, bottom, top):
		super().setup(bottom, top)
		assert len(bottom) == 0
		assert len(top) == 1
		assert self.label is not None
		
		self.labels = np.load(open(self.label, 'rb'))
		assert len(self.labels) > 0, "At least one label set required"
		
		self.step = 0
	
	def reshape(self, bottom, top):
		l = self.labels[self.step]
		top[0].reshape(self.batch_size, l.size)
	
	def forward(self, bottom, top):
		data = top[0].data
		for i in range(data.shape[0]):
			data[i] = self.labels[self.step]
			self.step += 1
			if self.step >= len(self.labels):
				self.step = 0
	
	def backward(self, *args):
		pass

def dataAsImageDataLayer(voc_dir, tmp_dir, image_set='train', **kwargs):
	from caffe_all import L
	from os import path
	from python_layers import PY
	Py = PY('data')
	voc_data = VOCData(voc_dir, image_set)
	
	# Create a text file with all the paths
	source_file = path.join(tmp_dir, image_set+"_images.txt")
	if not path.exists( source_file ):
		f = open(source_file, 'w')
		for n in voc_data.image_names:
			print('%s 0'% voc_data.image_path[n], file=f)
		f.close()
	
	# Create a label file
	lbl_file = path.join(tmp_dir, image_set+"_images.lbl")
	if not path.exists( lbl_file ):
		np.save(open(lbl_file, 'wb'), [voc_data.labels[n] for n in voc_data.image_names])
	cs = kwargs.get('transform_param',{}).get('crop_size',0)
	return L.ImageData(source=source_file, ntop=2, new_width=cs, new_height=cs, **kwargs)[0], Py.LabelData(label=lbl_file, batch_size=kwargs.get('batch_size',1))

def dataAsHDF5Layer(voc_dir, tmp_dir, image_set='train', resize=None, **kwargs):
	import atexit
	from os import path, remove
	from caffe_all import L
	def try_remove(*args, **kwargs):
		try:
			remove(*args, **kwargs)
		except FileNotFoundError:
			pass
	
	hf5_file = path.join(tmp_dir, image_set+"_data.hf5")
	if not path.exists(hf5_file):
		import h5py
		f = h5py.File(hf5_file, 'w')
		progress = lambda x: x
		try:
			from progressbar import ProgressBar, Percentage, Bar, ETA
			progress = ProgressBar(widgets=["Writing %s   "%path.basename(hf5_file), Percentage(), Bar(), ETA()])
		except:
			print("Writing %s"%path.basename(hf5_file))
		
		voc_data = VOCData(voc_dir, image_set)
		for i in progress(range(len(voc_data))):
			lbl, im = voc_data[i]
			if resize is not None:
				from skimage import transform
				try:
					W, H = resize
				except:
					W, H = resize, resize
				im = transform.resize(im, (H,W))
				im = (255*im).astype(np.uint8)
			
			# Read and write the image
			f.create_dataset('/data/%d'%i, data=im[:,:,::-1].transpose((2,0,1)))
			
			# Write classification labels
			f.create_dataset('/cls/%d'%i, data=lbl.astype(np.uint8))
		f.close()
		# Clean up the file
		atexit.register(try_remove, hf5_file)
	fast_hdf5_input_param = dict(source=hf5_file, batch_size=kwargs.get('batch_size', 1), group_name='data')
	data = L.TransformingFastHDF5Input(fast_hdf5_input_param=fast_hdf5_input_param, transform_param = kwargs.get('transform_param', {}))
	fast_hdf5_input_param = dict(source=hf5_file, batch_size=kwargs.get('batch_size', 1), group_name='cls')
	cls  = L.FastHDF5Input(fast_hdf5_input_param=fast_hdf5_input_param)
	return data, cls


def dataLayer(*args, **kwargs):
	from caffe_all import pb
	has_fast_hdf5 = hasattr(pb, 'FastHDF5InputParameter')
	if has_fast_hdf5:
		return dataAsHDF5Layer(*args, **kwargs)
	else:
		print( "---------------------------------------------------------------------------\n"
		       "Warning: TransformingFastHDF5InputLayer not found. It is HIGHLY recommended\n"
		       "that you get it, as ImageDataLayer doesn't support all the transformations \n"
		       "required to train a good classifier. You can get it from here:             \n"
		       "  https://github.com/philkr/caffe/tree/future                              \n"
		       "---------------------------------------------------------------------------\n" )
		from time import sleep
		sleep(1)
		tp = kwargs.get('transform_param', {})
		try:
			del kwargs['transform_param']['min_size']
		except:
			pass
		try:
			del kwargs['transform_param']['max_size']
		except:
			pass
		return dataAsImageDataLayer(*args, **kwargs)


def nImages(voc_dir, image_set='train'):
	return len(VOCData(voc_dir, image_set))
