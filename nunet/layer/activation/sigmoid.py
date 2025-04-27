import numpy as np
from nunet.layer import Module


class Sigmoid(Module):
    def __init__(self):
        self._last_output = None
        print("Sigmoid module created.")


    def forward(self, input_data):
        input_array = np.array(input_data)

        output = np.where(
            input_array >= 0,
            1 / (1 + np.exp(-input_array)),
            np.exp(input_array) / (1 + np.exp(input_array))
        )
        self._last_output = output
        return output

    def backward(self, gradwrtoutput):
        gradwrtoutput_array = np.array(gradwrtoutput) 
        local_gradient = self._last_output * (1.0 - self._last_output) 
        grad_wrt_input = gradwrtoutput_array * local_gradient 
        return grad_wrt_input 

