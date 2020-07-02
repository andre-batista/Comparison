from abc import ABC, abstractmethod

class Inverse(ABC):
    name = ''

    @abstractmethod
    def print_parametrization(self):
        pass

    def solve(self, instance):
        pass

