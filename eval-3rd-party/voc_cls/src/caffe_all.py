from os import environ
from util import addCaffePath
addCaffePath()
# Disable glog
if not 'GLOG_minloglevel' in environ:
	environ['GLOG_minloglevel'] = '3'

import caffe  # nopep8
from caffe import layers as L, params as P, NetSpec  # nopep8
import caffe.proto.caffe_pb2 as pb  # nopep8
