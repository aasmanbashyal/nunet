from nunet.layer import Module


class Sequential(Module):

    def __init__(self, *args):
        self.layers = args

    def forward(self, input):
        for module in self.layers:
            input = module(input)
        return input

    def backward(self, gradwrtoutput):
        for module in reversed(self.layers):
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput

    @property
    def param(self):
        params = []
        for module in self.layers:
            params.append(module.param)
        return params

    @property
    def grad(self):
        grads = []
        for module in self.layers:
            grads += module.grad
        return grads

    def load_param(self, *params):
        for module in self.layers:
            module.load_param(params)

    def __repr__(self):
        layers_str = "\n".join(
            [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)])
        return f"{self.__class__.__name__}(\n{layers_str}\n)"
