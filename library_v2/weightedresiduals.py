"""Give a title for the module.

Define the module.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import linalg as lag
from numba import jit

import library_v2.inverse as inv
import library_v2.inputdata as ipt

TIKHONOV_METHOD = 'tikhonov'
LANDWEBER_METHOD = 'landweber'
CONJUGATED_GRADIENT_METHOD = 'cg'
LEAST_SQUARES_METHOD = 'lstsq'


class MethodOfWeightedResiduals(inv.Inverse):
    """Summarize class."""

    A = np.zeros((int(), int()), dtype=complex)
    b = np.zeros(int(), dtype=complex)
    linsolver = str()
    parameter = tuple()

    def __init__(self, configuration, linear_solver, parameter):
        """Give a title."""
        super().__init__(configuration)
        self.linsolver = linear_solver

        if linear_solver == LANDWEBER_METHOD:
            if isinstance(parameter, tuple):
                self.parameter = parameter
            else:
                self.parameter = (parameter, 1000)

    def solve(self, inputdata):
        """Summarize the method."""
        A = self._compute_A(inputdata)
        b = self._compute_b(inputdata)

        if self.linsolver == TIKHONOV_METHOD:
            alpha = tikhonov(A, b, self.parameter)
        elif self.linsolver == LANDWEBER_METHOD:
            x0 = np.zeros(b.size, dtype=complex)
            alpha = landweber(A, b, self.parameter[0], x0, self.parameter[1])
        elif self.linsolver == CONJUGATED_GRADIENT_METHOD:
            x0 = np.zeros(b.size, dtype=complex)
            alpha = conjugated_gradient(A, b, x0, self.parameter)
        elif self.linsolver == LEAST_SQUARES_METHOD:
            alpha = least_squares(A, b, self.parameter)

        self._recover_map(inputdata, alpha)

    @abstractmethod
    def _compute_A(self, inputdata):
        """Give a title."""
        pass

    @abstractmethod
    def _compute_b(self, inputdata):
        """Give a title."""
        pass

    @abstractmethod
    def _recover_map(self, inputdata, alpha):
        """Summarize the method."""
        pass

@jit(nopython=True)
def tikhonov(A, b, alpha):
    """Summarize the method."""
    x = np.linalg.solve(A.conj().T@A + alpha*np.eye(A.shape[1]),
                        A.conj().T@b)
    return x


@jit(nopython=True)
def landweber(A, b, a, x0, M, TOL=1e-2, print_info=False):
    """Summarize the method."""
    x = np.copy(x0)
    d = lag.norm(b-A@x)
    d_last = 2*d
    it = 0
    if print_info:
        print('Landweber Regularization')
    while it < M and (d_last-d)/d_last > TOL:
        x = x + a*A.T.conj()@(b-A@x)
        d_last = d
        d = lag.norm(b-A@x)
        it += 1
        if print_info:
            print('Iteration %d - Error: %.3e' % ((d_last-d)/d_last))
    return x


@jit(nopython=True)
def conjugated_gradient(A, b, x0, delta):
    """Summarize the method."""
    p = -A.conj().T@b
    x = np.copy(x0)
    it = 0
    while True:
        kp = A@p
        tm = np.vdot(A@x-b, kp)/lag.norm(kp)**2
        x_last = np.copy(x)
        x = x - tm*p

        if lag.norm(A.conj().T@(A@x-b)) < delta:
            break

        gamma = (lag.norm(A.conj().T@(A@x-b))**2
                 / lag.norm(A.conj().T@(A@x_last-b))**2)
        p = A.conj().T@(A@x-b)+gamma*p
        it += 1

    return x


@jit(nopython=True)
def least_squares(A, b, rcond):
    """Summarize the method."""
    return np.linalg.lstsq(A, b, rcond=rcond)[0]
