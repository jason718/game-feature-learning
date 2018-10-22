require 'loadcaffe'
require 'optim'

prototxt = '../../models/caffenet_deploy.prototxt'
binary = '../../models/carl_nofc6.caffemodel'

net = loadcaffe.load(prototxt, binary, 'cudnn')
net = net:float() -- essential reference https://github.com/clcarwin/convert_torch_to_pytorch/issues/8
print(net)

torch.save('./carl_nofc6.t7', net)
