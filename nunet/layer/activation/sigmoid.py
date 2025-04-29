import numpy as np
from nunet.layer import Module


class Sigmoid(Module):
    def __init__(self):
        self._last_output = None

    def forward(self, input_data):
        input_array = np.array(input_data)
        
        output = np.zeros_like(input_array)
        
        pos_mask = input_array >= 0
        neg_mask = ~pos_mask
        
        output[pos_mask] = 1.0 / (1.0 + np.exp(-input_array[pos_mask]))
        
        neg_exp = np.exp(input_array[neg_mask])
        output[neg_mask] = neg_exp / (1.0 + neg_exp)
        
        self._last_output = output
        return output

    def backward(self, gradwrtoutput):
        gradwrtoutput_array = np.array(gradwrtoutput)
        local_gradient = self._last_output * (1.0 - self._last_output)
        grad_wrt_input = gradwrtoutput_array * local_gradient
        return grad_wrt_input
