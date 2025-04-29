import numpy as np
from nunet.layer import Module


class ReLU(Module):
    def __init__(self):
        self._last_input = None

    def forward(self, input_data):
        input_array = np.array(input_data)
        
        if np.any(np.isnan(input_array)):
            print("Warning: NaN values detected in ReLU input")
            input_array = np.nan_to_num(input_array, nan=0.0)
            
        self._last_input = input_array
        output = np.maximum(0, input_array)
        return output

    def backward(self, gradwrtoutput):
        gradwrtoutput_array = np.array(gradwrtoutput)
        
        if np.any(np.isnan(gradwrtoutput_array)):
            print("Warning: NaN values detected in ReLU gradient input")
            gradwrtoutput_array = np.nan_to_num(gradwrtoutput_array, nan=0.0)
            
        relu_derivative_mask = (self._last_input > 0).astype(float)
        
        grad_wrt_input = gradwrtoutput_array * relu_derivative_mask
        
        if np.any(np.isnan(grad_wrt_input)):
            print("Warning: NaN values detected in ReLU gradient output")
            grad_wrt_input = np.nan_to_num(grad_wrt_input, nan=0.0)
            
        return grad_wrt_input
