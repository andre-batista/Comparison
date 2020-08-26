r"""The Method of Weighted Residuals.

This module implements the general structure of the Method of Weighted
Residual (MWR) [1]_ for solving linear inverse scattering problems. The
MWR is a large class of methods. Therefore, the method is implemented as
an abstract class.

The MWR is based on the equation:

.. math:: (R, w_k(x)) = 0

In which :math:`R` is the residual function :math:`K{x}-y` and
:math:`w_k(x)` is an analytical function, often called as the *weight
function*. The solution `x` is assumed to of the form:

.. math:: x(u,v) = \sum_{i=1}^{N_X}\sum_{j=1}^{N_Y} a_{ij}\Phi_{ij}(u,v)

where :math:`\Phi_{ij}` is an analytical function often called trial
function.

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

# Standard libraries
from abc import ABC, abstractmethod
import numpy as np
from numpy import linalg as lag
from numba import jit, prange
from scipy.linalg import svdvals
from matplotlib import pyplot as plt

# Developed libraries
import inverse as inv
import inputdata as ipt
import error
import results as rst

# String constants
TIKHONOV_METHOD = 'tikhonov'
LANDWEBER_METHOD = 'landweber'
CONJUGATED_GRADIENT_METHOD = 'cg'
LEAST_SQUARES_METHOD = 'lstsq'
MOZOROV_CHOICE = 'mozorov'
LAVARELLO_CHOICE = 'lavarello'
LCURVE_CHOICE = 'lcurve'
FIXED_CHOICE = 'fixed'


class MethodOfWeightedResiduals(inv.Inverse):
    r"""The Method of Weighted Residuals (MWR) [1]_.

    The class is an implementation of MWR which is a large group of
    methods. Therefore, this class is supposed to be an abstract one, in
    which derived methods will following the general structure and other
    classes may contain an object of MWR withouting needing to know
    which one in order to work.

    Considering inverse scattering problems, this implementation is
    prepared to solve the integral equation allowing different types of
    discretization. All discretizations  will yield a matrix `A` and an
    array `beta` correspondent to a resulting linear system A*alpha=beta
    which will be solved by the methods implemented in this module.

    Attibutes
    ---------
        linsolver : {'tikhonov', 'landweber', 'cg', 'lstsq'}
            The name of the chosen linear system solver.

        parameter
            The regularization parameter involved in each type of linear
            system solver.

        name : 'Method of Weighted Residuals'
            The name of the method.

        alias_name : 'mwr'
            The alias which will be included in the results to indicate
            the method.

        discretization_method_name : str
            An attribute for derived classes.

        discretization_method_alias : str
            An attribute for derived classes.

    References
    ----------
    .. [1] Fletcher, Clive AJ. "Computational galerkin methods."
       Computational galerkin methods. Springer, Berlin, Heidelberg,
       1984. 72-85.
    """

    def __init__(self, configuration, linear_solver, parameter):
        r"""Create the object.

        Parameters
        ----------
            configuration : :class:`configuration.Configuration`
                The container with the problem configuration variables.

            linear_solver : {'tikhonov', 'landweber', 'cg', 'lstsq'}
                The name of the chosen linear system solver.

            parameter
                The parameter to configure the linear solver. It may be
                defined in different ways:

            * `linear_solver='tikhonov'`: in this case, `parameter` will
              depend on different strategies of determining it. There
              are three different choice strategies:
                * Fixed: when you want to define an arbitrary value. The
                  argument must be `parameter=('fixed', float())` or
                  `parameter=float()`.
                * SVD-based: a strategy according to [1]_, which is
                  based on Singular Value Decomposition and the
                  definition of a Relative Residual Error. In this case,
                  `parameter='lavarello'.
                * The Discrepancy Principle of Mozorov: a traditional
                  principle for defining the regularization parameter of
                  Tikhonov regularization [2]_. In this case,
                  `parameter='mozorov'`.
                * L-curve: the parameter value is chosen according to
                  the curve which relates error and solution norm.

            * `linear_solver='landweber'`: the method depend on two
              parameters: the regularization coefficient and the number
              of iterations. Therefore, `parameter=(float(), int())` for
              defining the regularization coefficient and the number of
              iterations, respectively; or `parameter=float()`, which
              will define the regularization coefficient and the number
              of iterations will be defined as 1000. The regularization
              parameter is defined as a proportion constant to:
              .. math:: \frac{1}{||A||^2}

            * `linear_solver='cg'`: The Conjugated-Gradient method
              depends on a parameter which means the noise level of the
              data. Therefore, `parameter=float()`.

            * `linear_solver='lstsq'`: The Least Squares Method depend
              on a parameter called `rcond` which means the truncation
              level of singular values of `A`. Therefore, singular
              values under this threshold will be truncated to zero.
              The argument is `parameter=float()`. Default: `1e-3`.

        References
        ----------
        .. [1] Lavarello, Roberto, and Michael Oelze. "A study on the
           reconstruction of moderate contrast targets using the
           distorted Born iterative method." IEEE transactions on
           ultrasonics, ferroelectrics, and frequency control 55.1
           (2008): 112-124.
        .. [2] Kirsch, Andreas. An introduction to the mathematical
           theory of inverse problems. Vol. 120. Springer Science &
           Business Media, 2011.
        """
        super().__init__(configuration)
        self.name = "Method of Weighted Residuals"
        self.alias_name = 'mwr'
        self.discretization_method_alias = ''
        self.discretization_method_name = ''

        if linear_solver == TIKHONOV_METHOD:
            self.linsolver = linear_solver
            if isinstance(parameter, str):
                if (parameter == MOZOROV_CHOICE
                        or parameter == LAVARELLO_CHOICE):
                    self._choice_strategy = parameter
                    self.parameter = None
                elif parameter == LCURVE_CHOICE:
                    self._choice_strategy = parameter
                    self._bounds = None
                    self._number_terms = None
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
                self._choice_strategy = FIXED_CHOICE
            elif isinstance(parameter, tuple) or isinstance(parameter, list):
                if parameter[0] == FIXED_CHOICE:
                    if isinstance(parameter[1], float):
                        self._choice_strategy = parameter[0]
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
                    self._choice_strategy = parameter[0]
                    self.parameter = parameter[1]
                elif parameter[0] == LCURVE_CHOICE:
                    self._choice_strategy = parameter[0]
                    if len(parameter) == 2:
                        self._number_terms = parameter[1]
                        self._bounds = None
                    elif len(parameter) == 3:
                        self._number_terms = parameter[1]
                        self._bounds = (parameter[2], 0)
                    elif len(parameter) == 4:
                        self._number_terms = parameter[1]
                        self._bounds = (parameter[2], parameter[3])
                    else:
                        raise error.WrongValueInput(
                            'MethodOfWeightedResiduals', 'parameter',
                            "'lcurve' or ('lcurve', number_terms) " +
                            "or ('lcurve', number_terms, lower_bound) " +
                            "or ('lcurve', number_terms, lower_bound, " +
                            " upper_bound)", "len(parameter) > 4"
                        )
                else:
                    raise error.WrongValueInput(
                        'MethodOfWeightedResiduals.__init__',
                        '(choice_strategy,...)',
                        "{'mozorov', 'lavarello', 'fixed', 'exhaustive'," +
                        " 'lcurve'}", parameter[0]
                    )
            if self._choice_strategy == LAVARELLO_CHOICE:
                self._beta_approximation = None

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
        """Solve an instance of linear inverse scattering problem.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                The instance of the problem containing the information
                on the scattered and total field.

        Returns
        -------
            :class:`results.Results`
        """
        # Compute the coefficient matrix
        A = self._compute_A(inputdata)

        # Compute right-hand-side of system of equations
        beta = self._compute_beta(inputdata)

        # Solve according the predefined method
        if self.linsolver == TIKHONOV_METHOD:
            if self._choice_strategy == MOZOROV_CHOICE:
                if inputdata.noise is None or inputdata.noise == 0.:
                    self.parameter = mozorov_choice(A, beta)
                else:
                    self.parameter = mozorov_choice(A, beta, inputdata.noise)
            elif self._choice_strategy == LAVARELLO_CHOICE:
                if self._beta_approximation is None:
                    alpha0 = quick_guess(A, beta)
                    self.parameter = lavarello_choice(
                        A, inputdata.es, np.reshape(A@alpha0,
                                                    inputdata.es.shape)
                    )
                else:
                    self.parameter = lavarello_choice(
                        A, inputdata.es, np.reshape(self._beta_approximation,
                                                    inputdata.es.shape)
                    )
            elif self._choice_strategy == LCURVE_CHOICE:
                if self._bounds is None and self._number_terms is None:
                    self.parameter = lcurve_choice(A, beta)
                elif self._bounds is None and self._number_terms is not None:
                    self.parameter = lcurve_choice(
                        A, beta, number_terms=self._number_terms
                    )
                else:
                    self.parameter = lcurve_choice(A, beta, self._bounds,
                                                   self._number_terms)
            alpha = tikhonov(A, beta, self.parameter)
            if self._choice_strategy == LAVARELLO_CHOICE:
                self._beta_approximation = A@alpha

        elif self.linsolver == LANDWEBER_METHOD:
            x0 = quick_guess(A, beta)
            alpha = landweber(A, beta, self.parameter[0]/lag.norm(A)**2, x0,
                              self.parameter[1])
        elif self.linsolver == CONJUGATED_GRADIENT_METHOD:
            x0 = quick_guess(A, beta)
            alpha = conjugated_gradient(A, beta, x0, self.parameter)
        elif self.linsolver == LEAST_SQUARES_METHOD:
            alpha = least_squares(A, beta, self.parameter)

        # Recover the relative permittivity and conductivity maps
        self._recover_map(inputdata, alpha)

        aux = rst.Results(inputdata.name + '_' + self.alias_name + '_'
                          + self.discretization_method_alias,
                          method_name=self.name + ' '
                          + self.discretization_method_name,
                          configuration_filename=self.configuration.name,
                          configuration_filepath=self.configuration.path,
                          input_filename=inputdata.name,
                          input_filepath=inputdata.path,
                          relative_permittivity_map=inputdata.epsilon_r,
                          conductivity_map=inputdata.sigma)

        return aux

    @abstractmethod
    def _compute_A(self, inputdata):
        """Compute the coefficient matrix.

        Every method must define its way to compute the coefficient
        matrix. The size of the matrix is defined according to the
        method.

        The method must return the matrix `A`.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                The instance of problem.

        Returns
        -------
            :class:`numpy.ndarray`
        """
        return np.zeros((int(), int()), dtype=complex)

    @abstractmethod
    def _compute_beta(self, inputdata):
        """Compute the right-hand-side.

        Every method must define its way to compute the right-hand-side
        of the equation. The size of the array is defined according to
        the method.

        It must returns a Numpy array representing `beta`.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                The instance of problem.

        Returns
        -------
            :class:`numpy.ndarray`
        """
        return np.zeros(int(), dtype=complex)

    @abstractmethod
    def _recover_map(self, inputdata, alpha):
        """Recover the contrast map.

        Based on the solution of the linear system, the method must
        define a way to compute the relative permittivity and the
        conductivity maps and store them on `inputdata`. It does not
        need to return anything.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                The instance of problem.

            alpha : :class:`numpy.ndarray`
                The solution of the system `A*alpha=beta`
        """
        pass

    def reset_parameters(self):
        """Reset parameters.

        A routine to reset parameters which vary with any change on the
        input resolution. So, if you want to prevent your derived method
        from unexpected changes, overload this method.
        """
        if self.linsolver == TIKHONOV_METHOD:
            if self._choice_strategy == LAVARELLO_CHOICE:
                self._beta_approximation = None

    def __str__(self):
        """Print method information."""
        message = super().__str__()
        message = message + 'Linear solver: '
        if self.linsolver == TIKHONOV_METHOD:
            message = message + 'Tikhonov Method\n'
            message = (message + 'Parameter choice strategy: '
                       + self._choice_strategy)
            if self._choice_strategy == FIXED_CHOICE:
                message = message + ', value: %.3e' % self.parameter
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
    r"""Perform the Tikhonov regularization.

    Solve the linear ill-posed system through Tikhonov regularization
    [1]_. The solution is given according to:

    .. math:: (A^*A + \alpha I)x = A^*beta

    Parameters
    ----------
        A : :class:`numpy.ndarray`
            The coefficient matrix.

        beta : :class:`numpy.ndarray`
            The right-hand-side array.

        alpha : float
            Regularization parameter.

    References
    ----------
    .. [1] Kirsch, Andreas. An introduction to the mathematical theory
       of inverse problems. Vol. 120. Springer Science & Business Media,
       2011.
    """
    x = lag.solve(A.conj().T@A + alpha*np.eye(A.shape[1]), A.conj().T@beta)
    return x


@jit(nopython=True, parallel=True)
def landweber(A, b, a, x0, maximum_iterations, TOL=1e-2):
    r"""Perform the Landweber regularization.

    Solve the linear ill-posed system through Landweber regularization
    [1]_. The algorithm formula is:

    .. math:: x_{n+1} = x_n + aA^{*}(b-Ax_n)

    Parameters
    ----------
        A : :class:`numpy.ndarray`
            The coefficient matrix.

        b : :class:`numpy.ndarray`
            The right-hand-side array.

        a : float
            Regularization parameter.

        x0 : :class:`numpy.ndarray`
            Initial guess of solution.

        maximum_iterations : int
            Maximum number of iterations.

        TOL : float
            Error tolerance level.

        print_info : bool
            Print iteration information.

    References
    ----------
    .. [1] Kirsch, Andreas. An introduction to the mathematical theory
       of inverse problems. Vol. 120. Springer Science & Business Media,
       2011.
    """
    x = np.copy(x0)
    d = lag.norm(b-A@x)
    d_last = 2*d
    it = 0
    while it < maximum_iterations and (d_last-d)/d_last > TOL:
        x = x + a*A.T.conj()@(b-A@x)
        d_last = d
        d = lag.norm(b-A@x)
        it += 1
    return x


@jit(nopython=True, parallel=True)
def conjugated_gradient(A, b, x0, delta, print_info=True):
    r"""Perform the Conjugated-Gradient (CG) regularization.

    Solve the linear ill-posed system through CG regularization [1]_.

    Parameters
    ----------
        A : :class:`numpy.ndarray`
            The coefficient matrix.

        b : :class:`numpy.ndarray`
            The right-hand-side array.

        x0 : :class:`numpy.ndarray`
            Initial guess of solution.

        delta : float
            Error tolerance level.

        print_info : bool
            Print iteration information.

    References
    ----------
    .. [1] Kirsch, Andreas. An introduction to the mathematical theory
       of inverse problems. Vol. 120. Springer Science & Business Media,
       2011.
    """
    p = -A.conj().T@b
    x = np.copy(x0)
    it = 0
    last = 1e20
    while True:
        kp = A@p
        tm = np.vdot(A@x-b, kp)/lag.norm(kp)**2
        x_last = np.copy(x)
        x = x - tm*p

        if lag.norm(A.conj().T@(A@x-b)) < delta:
            break
        elif lag.norm(A.conj().T@(A@x-b)) > last:
            break
        else:
            last = lag.norm(A.conj().T@(A@x-b))
        # print(last)
        gamma = (lag.norm(A.conj().T@(A@x-b))**2
                 / lag.norm(A.conj().T@(A@x_last-b))**2)
        p = A.conj().T@(A@x-b)+gamma*p
        it += 1

    return x


@jit(nopython=True)
def least_squares(A, b, rcond):
    """Return the least-squares solution to a linear matrix equation.

    See explanation at `<https://numpy.org/doc/stable/reference
    /generated/numpy.linalg.lstsq.html>`_

    Parameters
    ----------
        A : :class:`numpy.ndarray`
            The coefficient matrix.

        b : :class:`numpy.ndarray`
            The right-hand-side array.

        rcond : float
            Truncation level of singular values.
    """
    return lag.lstsq(A, b, rcond=rcond)[0]


def quick_guess(A, beta):
    r"""Provide an initial guess of solution of the linear system.

    Return a simple solution to the linear system `A*alpha=beta` through
    :math:\alpha=A^{*}\beta

    Parameters
    ----------
        A : :class:`numpy.ndarray`
            The coefficient matrix.

        beta : :class:`numpy.ndarray`
            The right-hand-side array.

    References
    ----------
    .. [1] Shah, Pratik, and Mahta Moghaddam. "A fast level set method
       for multimaterial recovery in microwave imaging." IEEE
       Transactions on Antennas and Propagation 66.6 (2018): 3017-3026.
    """
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


@jit(nopython=True, parallel=True)
def mozorov_choice(A, b, delta=1e-3):
    r"""Apply the Discrepancy Principle of Morozov [1].

    Compute the regularization parameter according to the starting guess
    of Newton's method for solving the Discrepancy Principle of Morozov
    defined in [1].

    Parameters
    ----------
        A : :class:`numpy.ndarray`
            Coefficient matrix returned by `_compute_A()` routine.

        b : :class:`numpy.ndarray`
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
    # Auxiliar variables
    AsA = np.conj(A.T)@A
    Asb = np.conj(A.T)@b
    eye = np.eye(A.shape[1])

    # Initial guess of frequency interval
    x0 = np.log10(delta*lag.norm(A)**2/(lag.norm(b)-delta))
    xmax = x0+5
    xmin = x0-5

    # Error of the initial guess
    fa = (lag.norm(b - A@lag.solve(AsA + 10**xmin*eye, Asb))-delta**2)**2
    fb = (lag.norm(b - A@lag.solve(AsA + 10**xmax*eye, Asb))-delta**2)**2

    # Find interval
    evals = 2
    while fb < fa:
        xmin = xmax
        fa = fb
        xmax = 2*xmax
        fb = (lag.norm(b - A@lag.solve(AsA + 10**xmax*eye, Asb))-delta**2)**2
        evals += 1
    if evals <= 3:
        xmin = np.log10(delta*lag.norm(A)**2/(lag.norm(b)-delta))-5
    else:
        xmin = xmin/2

    # Solve the frequency
    xa = xmax - .618*(xmax-xmin)
    xb = xmin + .618*(xmax-xmin)
    fa = (lag.norm(b - A@lag.solve(AsA + 10**xa*eye, Asb))-delta**2)**2
    fb = (lag.norm(b - A@lag.solve(AsA + 10**xb*eye, Asb))-delta**2)**2

    while (xmax-xmin) > 1e-3:
        if fa > fb:
            xmin = xa
            xa = xb
            xb = xmin + 0.618*(xmax-xmin)
            fa = fb
            fb = (lag.norm(b - A@lag.solve(AsA + 10**xb*eye, Asb))-delta**2)**2

        else:
            xmax = xb
            xb = xa
            xa = xmax - 0.618*(xmax-xmin)
            fb = fa
            fa = (lag.norm(b - A@lag.solve(AsA + 10**xa*eye, Asb))-delta**2)**2

    return 10**((xmin+xmax)/2)


@jit(nopython=True, parallel=True)
def lcurve_choice(A, b, bounds=(-20, 0), number_terms=21):
    """Determine the regularization parameter through L-curve.

    The regularization parameter is determined according to the L-curve.
    The L-curve is the graph between error and solution norms. The
    values are normalized and the chosen point is the one in which its
    distance from (0, 0) is minimum.

    Parameters
    ----------
        A : 2-d :class:`numpy.ndarray`
            Coefficient matrix of the linear system.

        b : 1-d :class:`numpy.ndarray`
            Right-hand-side of the linear system.

        bounds : 2-tuple
            Minimum and maximum value of the exponential form of the
            regularization parameter.

        number_terms : int
            Number of samples at the L-curve.
    """
    # Auxiliar variables
    AsA = np.conj(A.T)@A
    Asb = np.conj(A.T)@b
    eye = np.eye(A.shape[1])

    f1, f2 = np.zeros(number_terms), np.zeros(number_terms)
    alpha = 10**np.linspace(bounds[0], bounds[1], number_terms)

    # Compute objective-functions
    for i in prange(number_terms):
        x = lag.solve(AsA + alpha[i]*eye, Asb)
        f1[i] = lag.norm(b-A@x)
        f2[i] = lag.norm(x)

    # Normalization
    f1, f2 = f1/np.amax(f1), f2/np.amax(f2)

    # Best solution (Closest solution to the utopic one)
    knee = np.argmin(np.sqrt(f1**2 + f2**2))
    return alpha[knee]
