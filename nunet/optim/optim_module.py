from abc import ABC, abstractmethod


class BaseOptimizer(ABC):

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass
