from abc import ABC, abstractmethod


class LossModule(ABC):
    @abstractmethod
    def forward(self, predictions, target):
        pass

    @abstractmethod
    def backward(self):
        pass

    def __call__(self, predictions, target):
        return self.forward(predictions, target)
