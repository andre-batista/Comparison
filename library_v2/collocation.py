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
    :func:`bilinear_basisf`
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
BASIS_BILINEAR = 'bilinear'
BASIS_MININUM_NORM = 'mininum_norm'


class CollocationMethod(wrm.MethodOfWeightedResiduals):
    r"""The Collocation Method.

    This class implements the matrix coefficient and right-hand-side
    array computations of the Method of Weighted Residuals according to
    the Collocation Method [1]_.

    The two available options for the trial functions are: the bilinear
    function [1]_ and the minimum norm definition [2]_.

    Attributes
    ----------
        basis_function : {'biliear', 'minimum_norm'}
            A string indicating which trial function should be used.

    Notes
    -----
        The Minimum Norm Formulation is defined as:

        .. math::\Phi_{ij}(u,v) = j\omega\mu_bE_j(u,v)\times\frac{j}{4}
        H_0^{(2)}(k_b\sqrt{(x_i-u)^2 + (y_i-v)^2})

    References
    ----------
    .. [1] Fletcher, Clive AJ. "Computational galerkin methods."
       Computational galerkin methods. Springer, Berlin, Heidelberg,
       1984. 72-85.
    .. [2] Kirsch, Andreas. An introduction to the mathematical theory
       of inverse problems. Vol. 120. Springer Science & Business Media,
       2011.
    """

    basis_function = ''
    discretization_method_name = 'Collocation Method'
    discretization_method_alias = 'collocation'

    def __init__(self, configuration, linear_solver, parameter,
                 basis_function, discretization):
        """Summarize the method."""
        super().__init__(configuration, linear_solver, parameter)
        self.basis_function = basis_function
        self.discretization = discretization
        self._not_valid_variables = True

    def reset_parameters(self):
        """Reset variables."""
        super().reset_parameters()
        self._not_valid_variables = True

    def _compute_A(self, inputdata):
        """Summarize the method."""
        if (self.__not_valid_variables
                or inputdata.resolution[0]*inputdata.resolution[1]
                != self._fij.shape[1]):
            self._get_meshes(inputdata)
            self._not_valid_variables = False
        K = self._get_kernel(inputdata.et, inputdata.resolution)
        if self.basis_function == BASIS_MININUM_NORM:
            self._fij = self._minimum_norm_basisf(inputdata.et)
        A = computeA(self.configuration.NM,
                     self.configuration.NS,
                     self.discretization[1],
                     self.discretization[0],
                     inputdata.resolution[1],
                     inputdata.resolution[0],
                     K, self._fij, self._du, self._dv)
        return A

    def _compute_beta(self, inputdata):
        """Summarize the method."""
        return np.copy(inputdata.es.reshape(-1))

    def _recover_map(self, inputdata, alpha):
        """Summarize the method."""
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

    def _get_meshes(self, inputdata):
        """Summarize the method."""
        NM, NS = self.configuration.NM, self.configuration.NS
        NQ, NP = self.discretization
        NY, NX = inputdata.resolution
        dx = self.configuration.Lx/NX
        dy = self.configuration.Ly/NY

        xm, ym = cfg.get_coordinates_sdomain(self.configuration.Ro,
                                             self.configuration.NM)
        x, y = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                           resolution=inputdata.resolution)
        xms, yms = (np.reshape(np.tile(xm.reshape((-1, 1)), (1, NS)), (-1)),
                    np.reshape(np.tile(ym.reshape((-1, 1)), (1, NS)), (-1)))
        self._xpq, self._ypq = get_elements_mesh(NX, NY, dx, dy, NP, NQ)
        self._u, self._v = x.reshape(-1), y.reshape(-1)
        self._du, self._dv = dx, dy

        self._R = np.zeros((NM*NS, self._u.size))
        for i in range(NM*NS):
            self._R[i, :] = np.sqrt((xms[i]-self._u)**2 + (yms[i]-self._v)**2)

        if self.basis_function == BASIS_BILINEAR:
            self._fij = bilinear_basisf(self._u.reshape((NY, NX)),
                                        self._v.reshape((NY, NX)),
                                        self._xpq.reshape((NQ, NP)),
                                        self._ypq.reshape((NQ, NP)))

    def _get_kernel(self, et, resolution):
        """Summarize the method."""
        NM, NS = self.configuration.NM, self.configuration.NS
        NY, NX = resolution
        K = np.zeros((NM*NS, NX*NY), dtype=complex)
        mub = ct.mu_0
        omega = 2*np.pi*self.configuration.f
        s = 0
        for i in range(NM*NS):
            K[i, :] = (1j*omega*mub*et[:, s]*1j/4
                       * hankel2(0, self.configuration.kb*self._R[i, :]))
            if s == NS-1:
                s = 0
            else:
                s += 1
        return K

    def _minimum_norm_basisf(self, et):
        """Summarize the method."""
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
                   + '\nBasis function: ' + self.basis_function + '\n')
        return message


def bilinear_basisf(u, v, x, y):
    """Redefine the summary.

    Evaluate the triangular function. Given the collocation points
    in x and y, the function returns the evaluation of the triangular
    function in points specificated by the variables u and v. Each of
    the four variables must be 2D (meshgrid format).
    """
    NQ, NP = u.shape
    NY, NX = x.shape
    f = np.zeros((x.size, u.size))

    for p in range(NP):
        for q in range(NQ):

            nx = np.argwhere(u[q, p] >= x[0, :])[-1][0]
            ny = np.argwhere(v[q, p] >= y[:, 0])[-1][0]

            if nx+1 < NX and ny+1 < NY:
                eta = 2*(u[q, p]-x[ny, nx])/(x[ny, nx+1]-x[ny, nx]) - 1
                qsi = 2*(v[q, p]-y[ny, nx])/(y[ny+1, nx]-y[ny, nx]) - 1

                f[ny*NX+nx, q*NP+p] = .25*(1-qsi)*(1-eta)  # 1
                f[(ny+1)*NX+nx, q*NP+p] = .25*(1+qsi)*(1-eta)  # 2
                f[(ny+1)*NX+nx+1, q*NP+p] = .25*(1+qsi)*(1+eta)  # 3
                f[ny*NX+nx+1, q*NP+p] = .25*(1-qsi)*(1+eta)  # 4

            elif nx+1 < NX and ny == NY-1:
                eta = 2*(u[q, p]-x[ny, nx])/(x[ny, nx+1]-x[ny, nx]) - 1
                # qsi = -1

                f[ny*NX+nx, q*NP+p] = .25*2*(1-eta)  # 1
                f[ny*NX+nx+1, q*NP+p] = .25*2*(1+eta)  # 4

            elif nx == NX-1 and ny+1 < NY:
                # eta = -1
                qsi = 2*(v[q, p]-y[ny, nx])/(y[ny+1, nx]-y[ny, nx]) - 1

                f[ny*NX+nx, q*NP+p] = .25*(1-qsi)*2  # 1
                f[(ny+1)*NX+nx, q*NP+p] = .25*(1+qsi)*2  # 2

            elif nx == NX-1 and ny == NY-1:
                # qsi = -1
                # eta = -1

                f[ny*NX+nx, q*NP+p] = 1.  # 1

    return f


@jit(nopython=True)
def computeA(NM, NS, NP, NQ, NX, NY, K, fij, du, dv):
    """Summarize the method."""
    A = 1j*np.zeros((NM*NS, NP*NQ))
    for i in range(NM*NS):
        for j in range(NP*NQ):
            A[i, j] = np.trapz(np.trapz(K[i, :].reshape((NY, NX))
                                        * fij[j, :].reshape((NY, NX)), dx=du),
                               dx=dv)
    return A


def get_elements_mesh(NX, NY, dx, dy, NP, NQ):
    """Summarize the method."""
    x_min, x_max = cfg.get_bounds(NX*dx)
    y_min, y_max = cfg.get_bounds(NY*dy)
    xpq, ypq = np.meshgrid(np.linspace(x_min, x_max, NP),
                           np.linspace(y_min, y_max, NQ))
    xpq, ypq = xpq.reshape(-1), ypq.reshape(-1)
    return xpq, ypq
