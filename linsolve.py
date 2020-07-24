"""Summarize the module."""

from abc import ABC, abstractmethod
import numpy as np
from numpy import linalg as lag
from numba import jit


class LinearSolver(ABC):
    """Summarize the class."""
    
    parameter = None

    def __init__(self, parameter):
        """Summarize the method."""
        self.parameter = parameter

    @abstractmethod
    def solve(self, A, b, parameter=None):
        """Summarize the method."""
        self.parameter = parameter
        return np.zeros(A.shape[1], dtype=complex)


class Tikhonov(LinearSolver):
    """Summarize the class."""

    @jit(nopython=True)
    def solve(self, A, b, parameter=None):
        super().solve(A, b, parameter=parameter)
        