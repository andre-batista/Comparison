r"""The Collocation Method.

This module implements the Collocation Method for solving the linear
inverse scattering problem. In this method, the weight function is
defined as the Dirac :math:`\delta` function and the trial function
may define as another analytical. This method is also called as the
Point-Matching Method. The module provide one option of Finite-Element
discretization and one option of the Spectral Method discretization.

This modules provides

    :class:`CollocationMethod`
        The definition of the coefficient matrix and right-hand-side
        array computations of the Method of Weighted Residuals according
        to the Collocation Method.
    :func:`bilinear`
        The implementation of the bilinear trial function.
    :func:`computeA`
        An accelerated code to compute the coefficient matrix.
    :func:`get_elements_mesh`
        Define the meshgrid of elements in D-domain.

References
----------
.. [1] Fletcher, Clive AJ. "Computational galerkin methods."
   Computational galerkin methods. Springer, Berlin, Heidelberg, 1984.
   72-85.
"""

# Standard libraries
import numpy as np
from scipy import constants as ct
from scipy.special import hankel2
from numba import jit

# Developed libraries
import library_v2.weightedresiduals as wrm
import library_v2.configuration as cfg

# String constants
TRIAL_BILINEAR = 'bilinear'
TRIAL_MININUM_NORM = 'mininum_norm'


class CollocationMethod(wrm.MethodOfWeightedResiduals):
    r"""The Collocation Method.

    This class implements the matrix coefficient and right-hand-side
    array computations of the Method of Weighted Residuals according to
    the Collocation Method [1]_.

    The two available options for the trial functions are: the bilinear
    function [1]_ and the minimum norm definition [2]_.

    Attributes
    ----------
        trial_function : {'biliear', 'minimum_norm'}
            A string indicating which trial function should be used.

    Notes
    -----
        The Minimum Norm Formulation is defined as:

        .. math::\Phi_{pq}(u,v) = j\omega\mu_bE_n(u,v)\times\frac{j}{4}
        H_0^{(2)}(k_b\sqrt{(x_p-u)^2 + (y_q-v)^2})

    References
    ----------
    .. [1] Fletcher, Clive AJ. "Computational galerkin methods."
       Computational galerkin methods. Springer, Berlin, Heidelberg,
       1984. 72-85.
    .. [2] Kirsch, Andreas. An introduction to the mathematical theory
       of inverse problems. Vol. 120. Springer Science & Business Media,
       2011.
    """

    trial_function = ''
    discretization_method_name = 'Collocation Method'
    discretization_method_alias = 'collocation'

    def __init__(self, configuration, linear_solver, parameter,
                 trial_function, discretization):
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

                * `linear_solver='tikhonov'`: in this case, `parameter`
                  will depend on different strategies of determining it.
                  There are three different choice strategies:
                    * Fixed: when you want to define an arbitrary value.
                      The argument must be `parameter=('fixed',
                      float())` or `parameter=float()`.
                    * SVD-based: a strategy according to [1]_, which is
                      based on Singular Value Decomposition and the
                      definition of a Relative Residual Error. In this
                      case, `parameter='lavarello'.
                    * The Discrepancy Principle of Mozorov: a
                      traditional principle for defining the
                      regularization parameter of Tikhonov
                      regularization [2]_. In this case, `parameter=
                      'mozorov'`.

                * `linear_solver='landweber'`: the method depend on two
                  parameters: the regularization coefficient and the
                  number of iterations. Therefore, `parameter=(float(),
                  int())` for defining the regularization coefficient
                  and the number of iterations, respectively; or
                  `parameter=float()`, which will define the
                  regularization coefficient and the number of
                  iterations will be defined as 1000. The regularization
                  parameter is defined as a proportion constant to:
                  .. math:: \frac{1}{||A||^2}

                * `linear_solver='cg'`: The Conjugated-Gradient method
                  depends on a parameter which means the noise level of
                  the data. Therefore, `parameter=float()`.

                * `linear_solver='lstsq'`: The Least Squares Method
                  depend on a parameter called `rcond` which means the
                  truncation level of singular values of `A`. Therefore,
                  singular values under this threshold will be truncated
                  to zero. The argument is `parameter=float()`. Default:
                  `1e-3`.

            trial_function : {'biliear', 'minimum_norm'}
                A string indicating which trial function should be used.

            discretization : 2-tuple
                The number of elements in y and x directions (meshgrid
                format).

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
        super().__init__(configuration, linear_solver, parameter)
        self.trial_function = trial_function
        self.discretization = discretization
        self._not_valid_variables = True

    def reset_parameters(self):
        """Reset elements mesh variables."""
        super().reset_parameters()
        self._not_valid_variables = True

    def _compute_A(self, inputdata):
        """Compute the coefficient matrix.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                An instance of problem with resolution and total fields
                data.
        """
        # Verify if it is necessary to recompute meshgrid of elements.
        if (self.__not_valid_variables
                or inputdata.resolution[0]*inputdata.resolution[1]
                != self._fij.shape[1]):
            self._set_meshes(inputdata)
            self._not_valid_variables = False

        # Compute kernel
        K = self._get_kernel(inputdata.et, inputdata.resolution)

        # Minimum Norm formulation requires the evaluation of trial
        # function since it depends on the total field.
        if self.trial_function == TRIAL_MININUM_NORM:
            self._fij = self._minimum_norm(inputdata.et)

        A = computeA(self.configuration.NM,
                     self.configuration.NS,
                     self.discretization[1],
                     self.discretization[0],
                     inputdata.resolution[1],
                     inputdata.resolution[0],
                     K, self._fij, self._du, self._dv)
        return A

    def _compute_beta(self, inputdata):
        """Compute the right-hand-side.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                An instance of problem with scattered fields data.
        """
        return np.copy(inputdata.es.reshape(-1))

    def _recover_map(self, inputdata, alpha):
        """Recover the dielectric information.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                An instance of problem with resolution information.

            alpha : :class:`numpy.ndarray`
                Solution of `A*alpha=beta`.
        """
        NY, NX = inputdata.resolution
        chi = np.zeros((NY, NX), dtype=complex)
        omega = 2*np.pi*self.configuration.f
        for i in range(NX):
            for j in range(NY):
                chi[j, i] = np.sum(alpha*self._fij[:, j*NX+i])

        if (self.configuration.perfect_dielectric
                or not self.configuration.good_conductor):
            inputdata.epsilon_r = (np.imag(chi)/ct.epsilon_0/omega
                                   + self.configuration.epsilon_rb)
            inputdata.epsilon_r[inputdata.epsilon_r < 1] = 1

        if (self.configuration.good_conductor
                or not self.configuration.perfect_dielectric):
            inputdata.sigma = np.real(chi) + self.configuration.sigma_b
            inputdata.sigma[inputdata.sigma < 0] = 0

    def _set_meshes(self, inputdata):
        """Set elements meshgrid data.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                An instance of problem with resolution information.
        """
        # S-domain resolution
        NM, NS = self.configuration.NM, self.configuration.NS

        # Elements resolution
        NQ, NP = self.discretization

        # Image resolution
        NY, NX = inputdata.resolution
        dx = self.configuration.Lx/NX
        dy = self.configuration.Ly/NY

        # S-domain mesh
        xm, ym = cfg.get_coordinates_sdomain(self.configuration.Ro,
                                             self.configuration.NM)
        xms, yms = (np.reshape(np.tile(xm.reshape((-1, 1)), (1, NS)), (-1)),
                    np.reshape(np.tile(ym.reshape((-1, 1)), (1, NS)), (-1)))

        # Image mesh at D-domain
        x, y = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                           resolution=inputdata.resolution)
        self._u, self._v = x.reshape(-1), y.reshape(-1)
        self._du, self._dv = dx, dy

        # Elements meshgrid
        self._xpq, self._ypq = get_elements_mesh(NX, NY, dx, dy, NP, NQ)

        # Radius matrix
        self._R = np.zeros((NM*NS, self._u.size))
        for i in range(NM*NS):
            self._R[i, :] = np.sqrt((xms[i]-self._u)**2 + (yms[i]-self._v)**2)

        # Bilinear trial function does not depend on field information
        if self.trial_function == TRIAL_BILINEAR:
            self._fij = bilinear(self._u.reshape((NY, NX)),
                                 self._v.reshape((NY, NX)),
                                 self._xpq.reshape((NQ, NP)),
                                 self._ypq.reshape((NQ, NP)))

    def _get_kernel(self, et, resolution):
        r"""Compute kernel function.

        Evaluate the kernel function of the integral operator. The
        kernel is defined as:

        .. math:: K(x, y, u, v) = j\omega\mu_b E(\phi, u, v)\frac{j}{4}
        H_0^{(2)}(k_b|\sqrt{(x-u)^2 + (y-v)^2}|)
        """
        NM, NS = self.configuration.NM, self.configuration.NS
        NY, NX = resolution
        K = np.zeros((NM*NS, NX*NY), dtype=complex)
        mub = ct.mu_0
        omega = 2*np.pi*self.configuration.f
        s = 0
        for i in range(NM*NS):
            K[i, :] = (1j*omega*mub*et[:, s]*1j/4
                       * hankel2(0, self.configuration.kb*self._R[i, :]))
            # Matching the s-domain indexation
            if s == NS-1:
                s = 0
            else:
                s += 1
        return K

    def _minimum_norm(self, et):
        """Evaluate the Minimum Norm formulation.

        In this formulation, the trial function is defined as the kernel
        function.

        Parameters
        ----------
            et : :class:`numpy.ndarray`
                Total field matrix.
        """
        N = self._u.size
        Q, P = self.discretization
        omega = 2*np.pi*self.configuration.f
        mub = ct.mu_0
        Kpq = np.zeros((P*Q, N), dtype=complex)
        s = 0
        for i in range(P*Q):
            R = np.sqrt((self._xpq[i]-self._u)**2+(self._ypq[i]-self._v)**2)
            Kpq[i, :] = (1j*omega*mub*et[:, s]*1j/4
                         * hankel2(0, self.configuration.kb*R))
            if s == self.configuration.NS-1:
                s = 0
            else:
                s += 1
        return Kpq

    def __str__(self):
        """Print method information."""
        message = super().__str__()
        message = (message + 'Discretization: '
                   + self.discretization_method_name + ', size: %d'
                   % self.discretization[0] + 'x%d' % self.discretization[1]
                   + '\Trial function: ' + self.trial_function + '\n')
        return message


def bilinear(u, v, x, y):
    r"""Evaluate the bilinear function over rectangular elements.

    The bilinear function is an analogy of the triangular function in
    two-dimensions [1]_. Each element function is zero out of the
    vicinity nodes.

    Parameters
    ----------
        x, y : :class:`numpy.ndarray`
            Elements meshgrid coordinates. The shape must be NYxNX.

        u, v : :class:`numpy.ndarry`
            Evaluation points of bilinear function. The shape must be
            NVxNU.

    Returns
    -------
        :class:`numpy.ndarray`
            Shape: (NY*NX, NV*NU)

    References
    ----------
    .. [1] Fletcher, Clive AJ. "Computational galerkin methods."
       Computational galerkin methods. Springer, Berlin, Heidelberg, 1984.
       72-85.
    """
    # Dimensions
    NV, NU = u.shape
    NY, NX = x.shape

    # Evaluation matrix
    f = np.zeros((x.size, u.size))

    # Each evaluation is localized into the elements meshgrid
    for i in range(NU):
        for j in range(NV):

            nx = np.argwhere(u[j, i] >= x[0, :])[-1][0]
            ny = np.argwhere(v[j, i] >= y[:, 0])[-1][0]

            if nx+1 < NX and ny+1 < NY:
                eta = 2*(u[j, i]-x[ny, nx])/(x[ny, nx+1]-x[ny, nx]) - 1
                qsi = 2*(v[j, i]-y[ny, nx])/(y[ny+1, nx]-y[ny, nx]) - 1

                f[ny*NX+nx, j*NU+i] = .25*(1-qsi)*(1-eta)  # 1
                f[(ny+1)*NX+nx, j*NU+i] = .25*(1+qsi)*(1-eta)  # 2
                f[(ny+1)*NX+nx+1, j*NU+i] = .25*(1+qsi)*(1+eta)  # 3
                f[ny*NX+nx+1, j*NU+i] = .25*(1-qsi)*(1+eta)  # 4

            elif nx+1 < NX and ny == NY-1:
                eta = 2*(u[j, i]-x[ny, nx])/(x[ny, nx+1]-x[ny, nx]) - 1
                # qsi = -1

                f[ny*NX+nx, j*NU+i] = .25*2*(1-eta)  # 1
                f[ny*NX+nx+1, j*NU+i] = .25*2*(1+eta)  # 4

            elif nx == NX-1 and ny+1 < NY:
                # eta = -1
                qsi = 2*(v[j, i]-y[ny, nx])/(y[ny+1, nx]-y[ny, nx]) - 1

                f[ny*NX+nx, j*NU+i] = .25*(1-qsi)*2  # 1
                f[(ny+1)*NX+nx, j*NU+i] = .25*(1+qsi)*2  # 2

            elif nx == NX-1 and ny == NY-1:
                # qsi = -1
                # eta = -1

                f[ny*NX+nx, j*NU+i] = 1.  # 1

    return f


@jit(nopython=True)
def computeA(NM, NS, NP, NQ, NX, NY, K, fij, du, dv):
    r"""Accelarate the coefficient matrix computation.

    Parameters
    ----------
        NM, NS : int
            Number of measurements and sources, respectively.

        NP, NQ : int
            Number of elements at D-domain in x- and y-axis,
            respectively.

        NX, NY : int
            Image resolution.

        K, fij : :class:`numpy.ndarray`
            Evaluation of the kernel and trial function, respectively.

        du, dv : float
            Cell size of image meshgrid in x- and y-direction,
            respectively.

    Returns
    -------
        :class:`numpy.ndarray`

    Notes
    -----
        The coefficient matrix formula is:

        .. math:: A_{mn,pq} = \int_a^b\int_c^d K(\phi_n, x_{m}, y_{m},
        u, v) f_{pq} (u, v) dvdu
    """
    A = 1j*np.zeros((NM*NS, NP*NQ))
    for i in range(NM*NS):
        for j in range(NP*NQ):
            A[i, j] = np.trapz(np.trapz(K[i, :].reshape((NY, NX))
                                        * fij[j, :].reshape((NY, NX)), dx=du),
                               dx=dv)
    return A


def get_elements_mesh(NX, NY, dx, dy, NP, NQ):
    """Return the meshgrid of D-domain elements.

    The elements nodes are placed at the corner elements while the image
    cells coordinates are placed at the center. This may avoids
    singularities.

    Paremeters
    ----------
        NX, NY : int
            Image resolution.

        dx, dy : float
            Cell-size of image.

        NP, NQ : int
            Number of elements in x and y directions, respectively.
    """
    x_min, x_max = cfg.get_bounds(NX*dx)
    y_min, y_max = cfg.get_bounds(NY*dy)
    xpq, ypq = np.meshgrid(np.linspace(x_min, x_max, NP),
                           np.linspace(y_min, y_max, NQ))
    xpq, ypq = xpq.reshape(-1), ypq.reshape(-1)
    return xpq, ypq
