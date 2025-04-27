from abc import ABC, abstractmethod

class Module(ABC):
    
    @property
    def param(self):
        return []
    @property
    def grad(self):
        return []
    @abstractmethod
    def forward(self, input):
        pass
    def __call__(self, *input):
        return self.forward(*input)
    @abstractmethod
    def backward(self, gradwrtoutput):
        pass
    def load_param(self, *params):
        pass
    def __repr__(self):
        return f"{self.__class__.__name__}()"