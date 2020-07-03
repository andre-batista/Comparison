"""Give a title for the module.

Define the module.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import linalg as lag
from numba import jit

import library_v2.inverse as inv

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

    def __init__(self, linear_solver, parameter):
        """Give a title."""
        self.linsolver = linear_solver
        self.parameter = parameter
        pass

    def compute_A(self):
        """Give a title."""
        pass

    def compute_b(self):
        """Give a title."""
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
