from __future__ import division
from __future__ import print_function
from caffe_all import *
import numpy as np
import unittest

# Helper class that allows python layers to be added to NetSpec easily. Just use
#    Py.YourLayer(bottom1, bottom2, ..., parameter1=.., ...)
# parameter1 will automatically be passed to YourLayer defined below
class PY:
	def _parse_kwargs(self, layer, kwargs):
		l = getattr(self.py_module, layer)
		if not 'param_str' in kwargs:
			py_args = {}
			for a in list(kwargs.keys()):
				if hasattr(l, a):
					py_args[a] = kwargs.pop(a)
			kwargs['param_str'] = str(py_args)
		if hasattr(l, 'N_TOP'):
			kwargs['ntop'] = l.N_TOP
		return kwargs


	def __init__(self, module):
		import importlib
		self.module = module
		self.py_module = importlib.import_module(module)

	def __getattr__(self, name):
		return lambda *args, **kwargs: caffe.layers.Python(*args, module=self.module, layer=name, **self._parse_kwargs(name, kwargs))
Py = PY('python_layers')


class PyLayer(caffe.Layer):
	def setup(self, bottom, top):
		if self.param_str:
			params = eval(self.param_str)
			if isinstance(params, dict):
				for p,v in params.items():
					setattr(self, p, v)


class SigmoidCrossEntropyLoss(PyLayer):
	ignore_label = None
	def reshape(self, bottom, top):
		assert len(bottom) == 2
		assert len(top) == 1
		top[0].reshape()
	
	def forward(self, bottom, top):
		N = bottom[0].shape[0]
		f, df, t = bottom[0].data, bottom[0].diff, bottom[1].data
		mask = (self.ignore_label is None or t != self.ignore_label)
		lZ  = np.log(1+np.exp(-np.abs(f))) * mask
		dlZ = np.exp(np.minimum(f,0))/(np.exp(np.minimum(f,0))+np.exp(-np.maximum(f,0))) * mask
		top[0].data[...] = np.sum(lZ + ((f>0)-t)*f * mask) / N
		df[...] = (dlZ - t*mask) / N
	
	def backward(self, top, prop, bottom):
		bottom[0].diff[...] *= top[0].diff


class SigmoidCrossEntropyLossTest(unittest.TestCase):
	def _setupTestNet(self, n, m):
		from caffe_all import L
		ns = caffe.NetSpec()
		ns.f, ns.gt = L.DummyData(dummy_data_param = dict(shape=[dict(dim=[n,m])]*2, data_filler=[dict(type='gaussian'), dict(type='uniform')]), ntop=2)
		ns.caffe_s = L.SigmoidCrossEntropyLoss(ns.f, ns.gt, loss_weight=1)
		ns.python_s = Py.SigmoidCrossEntropyLoss(ns.f, ns.gt, loss_weight=1)
		net = caffe.get_net_from_string('force_backward:true\n'+str(ns.to_proto()), caffe.TEST)
		return net
	
	def test_forward(self):
		# Create a test net
		for n in range(1,10):
			for m in range(1,10):
				with self.subTest(n=n,m=m):
					net = self._setupTestNet(n,m)
					r = net.forward()
					self.assertAlmostEqual(r['caffe_s'], r['python_s'], 3)
	
	def test_backward(self):
		# Create a test net
		for n in range(1,10):
			for m in range(1,10):
				with self.subTest(n=n,m=m):
					net = self._setupTestNet(n,m)
					net.forward()
					net.blobs['f'].diff[...], net.blobs['caffe_s'].diff[...], net.blobs['python_s'].diff[...] = 0, 1, 0
					r1 = net.backward(['f'])['f']
					net.forward()
					net.blobs['f'].diff[...], net.blobs['caffe_s'].diff[...], net.blobs['python_s'].diff[...] = 0, 0, 1
					r2 = net.backward(['f'])['f']
					np.testing.assert_array_almost_equal( r1, r2, 3 )
					self.assertGreater( np.mean(np.abs(r1)), 0 )
	

class Print(PyLayer):
	def reshape(self, bottom, top):
		pass
	
	def forward(self, bottom, top):
		print( bottom[0].data )
	
	def backward(self, top, prop, bottom):
		pass


if __name__ == '__main__':
	unittest.main()
