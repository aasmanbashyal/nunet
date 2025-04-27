import numpy as np 
from nunet.layer import Module

class ReLU(Module):
    def __init__(self):
        self._last_input = None
        print("ReLU module created.")

    def forward(self, input_data):
        input_array = np.array(input_data)
        self._last_input = input_array
        output = np. maximum(0, input_array)
        return output
    
    def backward(self, gradwrtoutput):
        gradwrtoutput_array = np.array(gradwrtoutput)
        relu_derivative_mask = (self._last_input > 0).astype(gradwrtoutput_array.dtype)
        grad_wrt_input = gradwrtoutput_array * relu_derivative_mask
        return grad_wrt_input
    
