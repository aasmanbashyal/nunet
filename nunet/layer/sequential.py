from .module import Module


class Sequential(Module):

    def __init__(self, *args):
        if not all(isinstance(module, Module) for module in args):
            raise TypeError("Sequential layers must be Module instances")
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
        """Load the nested values returned by ``param``, or unpacked layer values."""
        if len(params) == 1 and self._matches_structure(params[0], self.param):
            params = params[0]
        if not self._matches_structure(params, self.param):
            raise ValueError("parameters must match the per-layer structure of Sequential.param")
        for module, layer_params in zip(self.layers, params):
            module.load_param(*layer_params)

    @staticmethod
    def _matches_structure(values, template):
        if isinstance(template, (list, tuple)):
            return (isinstance(values, (list, tuple)) and len(values) == len(template)
                    and all(Sequential._matches_structure(value, expected)
                            for value, expected in zip(values, template)))
        return not isinstance(values, (list, tuple))

    def __repr__(self):
        layers_str = "\n".join(
            [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)])
        return f"{self.__class__.__name__}(\n{layers_str}\n)"
