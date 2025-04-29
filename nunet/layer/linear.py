import numpy as np

from nunet.dtype import Tensor
from nunet.layer import Module


class Linear(Module):

    def __init__(self, in_features, out_features, bias=True, init_method='xavier'):
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
        if self.init_method == 'xavier':
            limit = np.sqrt(6 / (self.in_features + self.out_features))
            weights = np.random.uniform(-limit, limit, (self.in_features, self.out_features))
            self.W = Tensor((self.in_features, self.out_features), init_method='normal')
            self.W[:] = weights
        else:
            self.W = Tensor((self.in_features, self.out_features), init_method=self.init_method)
        
        if self.bias:
            self.b = Tensor(self.out_features)
            self.b[:] = 0.0

    def load_param(self, *params):
        self.W = params[0]
        if self.bias:
            self.b = params[1]

    def forward(self, input):
        self.input = input
        try:
            output = np.matmul(input, self.W)
            if self.bias:
                output += self.b
            if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                print("Warning: NaN or inf values detected in forward pass")
                output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            return output
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return np.zeros((input.shape[0], self.out_features))

    def backward(self, gradwrtoutput):
        try:
            gradwrtoutput = np.array(gradwrtoutput)
            if np.any(np.isnan(gradwrtoutput)):
                print("Warning: NaN values in input gradients")
                gradwrtoutput = np.nan_to_num(gradwrtoutput, nan=0.0)
                
            self.W.grad = np.matmul(self.input.T, gradwrtoutput)
            
            if np.any(np.isnan(self.W.grad)):
                print("Warning: NaN values in weight gradients")
                self.W.grad = np.nan_to_num(self.W.grad, nan=0.0)
                
            if self.bias:
                self.b.grad = np.sum(gradwrtoutput, axis=0)
                if np.any(np.isnan(self.b.grad)):
                    print("Warning: NaN values in bias gradients")
                    self.b.grad = np.nan_to_num(self.b.grad, nan=0.0)
            
            grad_wrt_input = np.matmul(gradwrtoutput, self.W.T)
            
            if np.any(np.isnan(grad_wrt_input)):
                print("Warning: NaN values in output gradients")
                grad_wrt_input = np.nan_to_num(grad_wrt_input, nan=0.0)
                
            return grad_wrt_input
            
        except Exception as e:
            print(f"Error in backward pass: {e}")
            return np.zeros_like(self.input)

    @property
    def param(self):
        return [self.W, self.b]

    @property
    def grad(self):
        return [self.W.grad, self.b.grad]

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, init_method=\"{self.init_method}\")"
