import numpy as np

from nunet.dtype import Tensor
from nunet.layer import Module


class Linear(Module):

    def __init__(self, in_features, out_features, bias=True, init_method='normal'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.init_method = init_method
        self.W = None
        self.b = None
        self.input = None
        self.init_weights()

    def init_weights(self):
        self.W = Tensor((self.in_features, self.out_features),
                        init_method=self.init_method)
        self.b = Tensor(self.out_features)

    def load_param(self, *params):
        self.W = params[0]
        if self.bias:
            self.b = params[1]

    def forward(self, input):
        self.input = input
        output = np.matmul(input, self.W)
        if self.bias:
            output += self.b
        return output

    def backward(self, gradwrtoutput):
        self.W.grad = np.matmul(self.input.T, gradwrtoutput)
        if self.bias:
            self.b.grad = np.sum(gradwrtoutput, axis=0)
        return np.matmul(gradwrtoutput, self.W.T)

    @property
    def param(self):
        return [self.W, self.b]

    @property
    def grad(self):
        return [self.W.grad, self.b.grad]

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, init_method=\"{self.init_method}\")"
