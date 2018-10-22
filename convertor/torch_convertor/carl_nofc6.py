
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


carl_nofc6 = nn.Sequential( # Sequential,
	nn.Conv2d(3,96,(11, 11),(4, 4),(5, 5)),
	nn.ReLU(),
	nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
	Lambda(lambda x,lrn=torch.legacy.nn.SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
	nn.Conv2d(96,256,(5, 5),(1, 1),(2, 2)),
	nn.ReLU(),
	nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
	Lambda(lambda x,lrn=torch.legacy.nn.SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
	nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(384,384,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(384,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
)