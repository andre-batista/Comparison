r"""The Method of Weighted Residuals.

This module implements the general structure of the Method of Weighted
Residual (MWR) [1]_ for solving linear inverse scattering problems. The
MWR is a large class of methods. Therefore, the method is implemented as
an abstract class.

The modules provides

    :class:`MethodOfWeightedResiduals`
        The class with the general structure of MWR.
    :func:`tikhonov`
        The Tikhonov Regularization for linear ill-posed systems [2]_.
    :func:`landweber`
        The Landweber Regularization for linear ill-posed systems [2]_.
    :func:`conjugated_gradient`
        The Conjugated Gradient Method for linear ill-posed
        systems [2]_.
    :func:`least_squares`
        The least-squares solution to a linear matrix equation.
    :func:`quick_guess`
        A simple initial solution for a given system.
    :func:`lavarello_choice`
        A strategy for chosing the regularization parameter for Tikhonov
        Regularization [3]_.
    :func:`mozorov_choice`
        A strategy for chosing the regularization parameter for Tikhonov
        Regularization based on the Discrepancy Principle of Mozorov
        [2]_.

References
----------
.. [1] Fletcher, Clive AJ. "Computational galerkin methods."
   Computational galerkin methods. Springer, Berlin, Heidelberg, 1984.
   72-85.
.. [2] Kirsch, Andreas. An introduction to the mathematical theory of
   inverse problems. Vol. 120. Springer Science & Business Media, 2011.
.. [3] Lavarello, Roberto, and Michael Oelze. "A study on the
   reconstruction of moderate contrast targets using the distorted Born
   iterative method." IEEE transactions on ultrasonics, ferroelectrics,
   and frequency control 55.1 (2008): 112-124.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import linalg as lag
from numba import jit
from scipy.linalg import svdvals

import library_v2.inverse as inv
import library_v2.inputdata as ipt
import library_v2.error as error
import library_v2.results as rst

TIKHONOV_METHOD = 'tikhonov'
LANDWEBER_METHOD = 'landweber'
CONJUGATED_GRADIENT_METHOD = 'cg'
LEAST_SQUARES_METHOD = 'lstsq'
MOZOROV_CHOICE = 'mozorov'
LAVARELLO_CHOICE = 'lavarello'
FIXED_CHOICE = 'fixed'


class MethodOfWeightedResiduals(inv.Inverse):
    """Summarize class."""

    A = np.zeros((int(), int()), dtype=complex)
    b = np.zeros(int(), dtype=complex)
    linsolver = str()
    parameter = None
    name = "Method of Weighted Residuals"
    alias_name = 'mwr'
    discretization_method_name = ''
    discretization_method_alias = ''

    def __init__(self, configuration, linear_solver, parameter):
        """Give a title."""
        super().__init__(configuration)

        if linear_solver == TIKHONOV_METHOD:
            self.linsolver = linear_solver
            if isinstance(parameter, str):
                if (parameter == MOZOROV_CHOICE
                        or parameter == LAVARELLO_CHOICE):
                    self.choice_strategy = parameter
                elif parameter == FIXED_CHOICE:
                    raise error.MissingInputError(
                        'MethodOfWeightedResiduals.__init__',
                        'parameter_value'
                    )
                else:
                    raise error.WrongValueInput(
                        'MethodOfWeightedResiduals.__init__',
                        'parameter',
                        "{'mozorov', 'lavarello', 'fixed'}",
                        parameter
                    )
            elif isinstance(parameter, float):
                self.parameter = parameter
                self.choice_strategy = FIXED_CHOICE
            elif isinstance(parameter, tuple) or isinstance(parameter, list):
                if parameter[0] == FIXED_CHOICE:
                    if isinstance(parameter[1], float):
                        self.choice_strategy = parameter[0]
                        self.parameter = parameter[1]
                    else:
                        raise error.WrongTypeInput(
                            'MethodOfWeightedResiduals.__init__',
                            "('fixed', parameter_value)",
                            'float',
                            type(parameter[1])
                        )
                elif (parameter[0] == LAVARELLO_CHOICE
                        or parameter[0] == MOZOROV_CHOICE):
                    self.choice_strategy = parameter[0]
                    self.parameter = parameter[1]
                else:
                    raise error.WrongValueInput(
                        'MethodOfWeightedResiduals.__init__',
                        '(choice_strategy,...)',
                        "{'mozorov', 'lavarello', 'fixed'}",
                        parameter[0]
                    )
            if self.choice_strategy == LAVARELLO_CHOICE:
                self.beta_approximation = None

        elif linear_solver == LANDWEBER_METHOD:
            self.linsolver = linear_solver
            if isinstance(parameter, tuple):
                self.parameter = parameter
            else:
                self.parameter = (parameter, 1000)
        elif linear_solver == CONJUGATED_GRADIENT_METHOD:
            self.linsolver = linear_solver
            if isinstance(parameter, float):
                self.parameter = parameter
            else:
                raise error.WrongTypeInput(
                    'MethodOfWeightedResiduals.__init__',
                    'parameter',
                    'float',
                    type(parameter)
                )
        elif linear_solver == LEAST_SQUARES_METHOD:
            self.linsolver = linear_solver
            if isinstance(parameter, float):
                self.parameter = parameter
            elif parameter is None:
                self.parameter = 1e-3
            else:
                raise error.WrongTypeInput(
                    'MethodOfWeightedResiduals.__init__',
                    'parameter',
                    'float',
                )
        else:
            raise error.WrongValueInput('MethodOfWeightedResiduals.__init__',
                                        'linear_solver',
                                        "{'tikhonov', 'landweber', 'cg',"
                                        + "'lstsq'}",
                                        linear_solver)

    def solve(self, inputdata):
        """Summarize the method."""
        A = self._compute_A(inputdata)
        beta = self._compute_b(inputdata)

        if self.linsolver == TIKHONOV_METHOD:
            if self.choice_strategy == MOZOROV_CHOICE:
                self.parameter = mozorov_choice(A, beta, inputdata.noise)
            if self.choice_strategy == LAVARELLO_CHOICE:
                if self.beta_approximation is None:
                    alpha0 = quick_guess(A, beta)
                    self.parameter = lavarello_choice(
                        A, inputdata.es, np.reshape(A@alpha0,
                                                    inputdata.es.shape)
                    )
                else:
                    self.parameter = lavarello_choice(
                        A, inputdata.es, np.reshape(self.beta_approximation,
                                                    inputdata.es.shape)
                    )
            alpha = tikhonov(A, beta, self.parameter)
            if self.choice_strategy == LAVARELLO_CHOICE:
                self.beta_approximation = A@alpha

        elif self.linsolver == LANDWEBER_METHOD:
            x0 = np.zeros(beta.size, dtype=complex)
            alpha = landweber(A, beta, self.parameter[0]/lag.norm(A)**2, x0,
                              self.parameter[1])
        elif self.linsolver == CONJUGATED_GRADIENT_METHOD:
            x0 = np.zeros(beta.size, dtype=complex)
            alpha = conjugated_gradient(A, beta, x0, self.parameter)
        elif self.linsolver == LEAST_SQUARES_METHOD:
            alpha = least_squares(A, beta, self.parameter)

        self._recover_map(inputdata, alpha)

        return rst.Results(inputdata.name + '_' + self.alias_name + '_'
                           + self.discretization_method_alias,
                           method_name=self.name + ' '
                           + self.discretization_method_name,
                           configuration_filename=(
                               inputdata.configuration_filename),
                           inputdata_filename=inputdata.name,
                           scattered_field=np.reshape(A@alpha,
                                                      inputdata.es.shape),
                           relative_permittivity_map=inputdata.epsilon_r,
                           conductivity_map=inputdata.sigma)

    @abstractmethod
    def _compute_A(self, inputdata):
        """Give a title."""
        pass

    @abstractmethod
    def _compute_beta(self, inputdata):
        """Give a title."""
        pass

    @abstractmethod
    def _recover_map(self, inputdata, alpha):
        """Summarize the method."""
        pass

    def reset_parameters(self):
        """Summarize the method."""
        if self.linsolver == TIKHONOV_METHOD:
            if self.choice_strategy == LAVARELLO_CHOICE:
                self.beta_approximation = None

    def __str__(self):
        """Print method information."""
        message = super().__str__()
        message = message + 'Linear solver: '
        if self.linsolver == TIKHONOV_METHOD:
            message = message + 'Tikhonov Method\n'
            message = (message + 'Parameter choice strategy: '
                       + self.choice_strategy)
            if self.choice_strategy == FIXED_CHOICE:
                message = message + ', value: $.3e' % self.parameter
        elif self.linsolver == LANDWEBER_METHOD:
            message = (message + 'Landweber Method, a = %.3e/||K||^2, '
                       % self.parameter[0] + 'Iter.: %d' % self.parameter[1])
        elif self.linsolver == CONJUGATED_GRADIENT_METHOD:
            message = (message + 'Conjugated-Gradient Method, delta = %.3e'
                       % self.parameter)
        elif self.linsolver == LEAST_SQUARES_METHOD:
            message = (message + 'Least Squares Method, rcond = %.3e'
                       % self.parameter)
        message = message + '\n'
        return message


@jit(nopython=True)
def tikhonov(A, beta, alpha):
    """Summarize the method."""
    x = lag.solve(A.conj().T@A + alpha*np.eye(A.shape[1]), A.conj().T@beta)
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
    return lag.lstsq(A, b, rcond=rcond)[0]


def quick_guess(A, beta):
    """Summarize the method."""
    return A.conj().T@beta


def lavarello_choice(A, scattered_field_o, scattered_field_r):
    """Compute the regularization parameter according to [1].

    This parameter strategy is based on the first singular value of the
    coefficient matrix.

    Parameters
    ----------
        A : :class:`numpy.ndarray`
            Coefficient matrix returned by `_compute_A()` routine.

        scattered_field_o :class:`numpy.ndarray`
            Original scattered field matrix given for the problem.

        scattered_field_r :class:`numpy.ndarray`
            Recovered scattered field matrix given for the problem.

    References
    ----------
    .. [1] Lavarello, Roberto, and Michael Oelze. "A study on the
           reconstruction of moderate contrast targets using the
           distorted Born iterative method." IEEE transactions on
           ultrasonics, ferroelectrics, and frequency control 55.1
           (2008): 112-124.
    """
    RRE = rst.compute_rre(scattered_field_o, scattered_field_r)
    s0 = svdvals(A)[0]

    if .5 < RRE:
        return s0**2/2
    elif .25 < RRE <= .5:
        return s0**2/20
    elif RRE <= .25:
        return s0**2/200


def mozorov_choice(K, y, delta):
    r"""Apply the Discrepancy Principle of Morozov [1].

    Compute the regularization parameter according to the starting guess
    of Newton's method for solving the Discrepancy Principle of Morozov
    defined in [1].

    Parameters
    ----------
        K : :class:`numpy.ndarray`
            Coefficient matrix returned by `_compute_A()` routine.

        y : :class:`numpy.ndarray`
            Right-hand-side array returned by `_compute_b()` routine.

        delta : float
            Noise level of problem.

    Notes
    -----
        The Discrepancy Principle of Morozov is defined according to
        the zero of the following monotone function:

        .. math:: \phi(\alpha) = ||Kx^{\alpha,\delta}-y^{\delta}||^2-\delta^2

        The initial guess of Newton's method to determine the zero is:

        .. math:: \alpha = \frac{\delta||K||^2}{||y^\delta-\delta}

    References
    ----------
    .. [1] Kirsch, Andreas. An introduction to the mathematical theory
           of inverse problems. Vol. 120. Springer Science & Business
           Media, 2011.
    """
    return delta*lag.norm(K)**2/(lag.norm(y)-delta)
