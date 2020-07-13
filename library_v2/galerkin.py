r"""The Galerkin Method.

This module provides the implementation of the Galerkin Method [1]_ to
solve the linear inverse scattering problem. This is a derivation of the
Method of Weighted Residuals. Two scheme of trial-weight functions are
available: bilinear (Finite-Element Method) and Legendre (Spectrum
Method) discretizations.

This module provides:

    :class:`GalerkinMethod`
        The implementation of the method.
    :func:`legendre`
        Evaluate the legendre element discretization.
    :func:`interpolate_scattered_field`
        Change the resolution of the S-domain.
    :func:`interpolate_intern_field`
        Change resolution of total field.
    :func:`computeA_nointerp`
        Accelerated routine for coefficient matrix compution without
        field interpolation.
    :func:`computeA_interp`
        Accelerated routine for coefficient matrix compution with field
        interpolation.
    :func:`compute_beta`
        Accelerated routine for right-hand-side computation.

References
----------
.. [1] Fletcher, Clive AJ. "Computational galerkin methods."
   Computational galerkin methods. Springer, Berlin, Heidelberg, 1984.
   72-85.
"""

# Standard libraries
import numpy as np
from scipy.special import eval_legendre as Pn
from scipy.interpolate import interp2d
from scipy.special import hankel2
from scipy import constants as ct
from numba import jit

# Standard libraries
import library_v2.weightedresiduals as wrm
import library_v2.configuration as cfg
import library_v2.collocation as clc

# String constants
BASIS_BILINEAR = 'bilinear'
BASIS_LEGENDRE = 'legendre'


class GalerkinMethod(wrm.MethodOfWeightedResiduals):
    r"""The Galerkin Method.

    This class implements the matrix coefficient and right-hand-side
    array computations of the Method of Weighted Residuals according to
    the Galerkin Method Method [1]_.

    The two available options for the trial functions are: the bilinear
    (Finite Element Method) and Legendre (Spectrum Method) [1]_
    functions.

    Attributes
    ----------
        basis_function : {'biliear', 'legendre'}
            A string indicating which basis function should be used.

    References
    ----------
    .. [1] Fletcher, Clive AJ. "Computational galerkin methods."
       Computational galerkin methods. Springer, Berlin, Heidelberg,
       1984. 72-85.
    """

    basis_function = ''
    discretization = (int, int)
    discretization_method_name = 'Galerkin Method'
    discretization_method_alias = 'galerkin'
    constant_iterpolation = 4

    def __init__(self, configuration, linear_solver, parameter,
                 basis_function, discretization):
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

            basis_function : {'biliear', 'minimum_norm'}
                A string indicating which basis function should be used.

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
        self.basis_function = basis_function
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
        # Number of elements in S-space
        NW, NZ = self.discretization[0], self.discretization[1]

        # Number of elements in D-domain
        NP, NQ = self.discretization[3], self.discretization[2]

        # Image resolution
        NY, NX = inputdata.resolution

        # Any change in the problem configuration or image resolution
        # requires a different mesh configuration
        if self._not_valid_variables or self._u.size != NX*NY:
            self._set_meshes(inputdata)
            self._not_valid_variables = False

        # In case we are using more elements in S-space than the number
        # than the number of measurements and sources
        if self._FLAG_INTERPOLATION:
            new_NM = self.constant_interpolation*self.configuration.NM
            new_NS = self.constant_interpolation*self.configuration.NS
            inputdata.es = interpolate_scattered_field(inputdata.es, new_NM,
                                                       new_NS)

        # In case of more elements in S-space, the total field must
        # contain more sources.
        if self._FLAG_INTERPOLATION:
            new_NS = self.constant_iterpolation*self.configuration.NS
            inputdata.et = interpolate_intern_field(inputdata.et, new_NS)

        K = self._get_kernel(inputdata.et, inputdata.resolution)
        A = computeA(self._theta.shape[0], self._theta.shape[1], NW, NZ, NP,
                     NQ, NX, NY, K, self._fij, self._gij, self._du, self._dv,
                     self._dtheta, self._dphi)

        return A

    def _compute_beta(self, inputdata):
        """Compute the right-hand-side.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                An instance of problem with scattered fields data.
        """
        return computebeta(inputdata.es, self._gij, self._dtheta, self._dphi)

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
        omega = 2*np.pi*self.configuration.f
        fa = np.zeros((NY, NX), dtype=complex)
        for i in range(NX):
            for j in range(NY):
                fa[j, i] = np.sum(alpha*self._fij[:, j*NX+i])

        if (self.configuration.perfect_dielectric
                or not self.configuration.good_conductor):
            inputdata.epsilon_r = (np.imag(fa)/ct.epsilon_0/omega
                                   + self.configuration.epsilon_rb)
            inputdata.epsilon_r[inputdata.epsilon_r < 1] = 1

        if (self.configuration.good_conductor
                or not self.configuration.perfect_dielectric):
            inputdata.sigma = np.real(fa) + self.configuration.sigma_b
            inputdata.sigma[inputdata.sigma < 0] = 0

    def __str__(self):
        """Print method information."""
        message = super().__str__()
        message = (message + 'Discretization: '
                   + self.discretization_method_name + ', S-domain: %d'
                   % self.discretization[0] + 'x%d' % self.discretization[1]
                   + ', D-domain: %dx' % self.discretization[3]
                   + '%d' % self.discretization[2]
                   + '\nBasis function: ' + self.basis_function + '\n')
        return message

    def _set_meshes(self, inputdata):
        """Set elements meshgrid data.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                An instance of problem with resolution information.
        """
        # Number of measurements and sources
        NM, NS = self.configuration.NM, self.configuration.NS

        # Discretization in S-domain
        NW, NZ = self.discretization[0], self.discretization[1]

        # Discretization in D-domain
        NP, NQ = self.discretization[3], self.discretization[2]

        # Image resolution
        NY, NX = inputdata.resolution
        dx, dy = self.configuration.Lx/NX, self.configuration.Ly/NY

        x, y = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                           resolution=inputdata.resolution)
        self._u, self._v = x.reshape(-1), y.reshape(-1)
        self._du, self._dv = dx, dy

        # Interpolation condition: if S-space discretization is greater
        # than the number of measurements and sources.
        # If the discretization is smaller, than S-space is integrated
        # in the points of the original data.
        # Otherwise, the original data is interpolated in order to have
        # more integration points than elements.
        self._FLAG_INTERPOLATION = not(NW <= NM and NZ <= NS)

        # If the original data have more information than discretization
        if not self._FLAG_INTERPOLATION:
            xm, ym = cfg.get_coordinates_sdomain(self.configuration.Ro,
                                                 self.configuration.NM)
            self._xms = np.reshape(np.tile(xm.reshape((-1, 1)), (1, NS)), (-1))
            self._yms = np.reshape(np.tile(ym.reshape((-1, 1)), (1, NS)), (-1))
            self._phi_ms, self._theta_ms = np.meshgrid(cfg.get_angles(NS),
                                                       cfg.get_angles(NM))

        # If the original data have less information than discretization
        else:
            new_NM = self.constant_iterpolation*NW
            new_NS = self.constant_iterpolation*NZ
            xm, ym = cfg.get_coordinates_sdomain(self.configuration.Ro, new_NM)
            self._xms = np.reshape(np.tile(xm.reshape((-1, 1)), (1, new_NS)),
                                   (-1))
            self._yms = np.reshape(np.tile(ym.reshape((-1, 1)), (1, new_NS)),
                                   (-1))
            self._phi_ms, self._theta_ms = np.meshgrid(cfg.get_angles(new_NS),
                                                       cfg.get_angles(new_NM))

        self._phi_wz, self._theta_wz = np.meshgrid(cfg.get_angles(NZ),
                                                   cfg.get_angles(NW))
        self._dtheta = self._theta_ms[1, 0]-self._theta_ms[0, 0]
        self._dphi = self._phi_ms[0, 1]-self._phi_ms[0, 0]
        self._R = np.zeros((self._theta_ms.size, self._u.size))
        for i in range(self._R.shape[0]):
            self._R[i, :] = np.sqrt((self._xms[i]-self._u)**2
                                    + (self._yms[i]-self._v)**2)

        # Coordinates of D-domain elements
        self._xpq, self._ypq = clc.get_elements_mesh(NX, NY, dx, dy, NP, NQ)

        if self.basis_function == BASIS_BILINEAR:
            self._fij = clc.bilinear(self._u.reshape((NY, NX)),
                                     self._v.reshape((NY, NX)),
                                     self._xpq.reshape((NQ, NP)),
                                     self._ypq.reshape((NQ, NP)))
            self._gij = clc.bilinear(self._phi_ms, self._theta_ms,
                                     self._phi_wz, self._theta_wz)

        elif self.basis_function == BASIS_LEGENDRE:
            xmin, xmax = cfg.get_bounds(self.configuration.Lx)
            ymin, ymax = cfg.get_bounds(self.configuration.Ly)
            self._fij = legendre(self._u.reshape((NY, NX)),
                                 self._v.reshape((NY, NX)),
                                 NQ, NP, xmin, xmax, ymin, ymax)
            self._gij = legendre(self._phi_ms, self._theta_ms,
                                 NW, NZ, 0, 2*np.pi, 0, 2*np.pi)

    def _get_kernel(self, et, resolution):
        r"""Compute kernel function.

        Evaluate the kernel function of the integral operator. The
        kernel is defined as:

        .. math:: K(x, y, u, v) = j\omega\mu_b E(\phi, u, v)\frac{j}{4}
        H_0^{(2)}(k_b|\sqrt{(x-u)^2 + (y-v)^2}|)

        Parameters
        ----------
            et : :class:`numpy.ndarray`
                Total field matrix.

            resolution : 2-tuple
                Image discretization in y and x directions,
                respectively.
        """
        NS = self.configuration.NS
        mub = ct.mu_0
        omega = 2*np.pi*self.configuration.f
        K = np.zeros(self._R.shape, dtype=complex)
        if self._FLAG_INTERPOLATION:
            L = self.constant_iterpolation*NS
        else:
            L = NS
        s = 0
        for i in range(K.shape[0]):
            K[i, :] = (1j*omega*mub*et[:, s]*1j/4
                       * hankel2(0, self.configuration.kb*self._R[i, :]))
            # Matching the measurement-source indexation
            if s == L-1:
                s = 0
            else:
                s += 1
        return K


def legendre(x, y, M, N, xmin=None, xmax=None, ymin=None, ymax=None):
    """Evaluate the Legendre basis function.

    Parameters
    ----------
        x, y : :class:`numpy.ndarray`
            Domain coordinates.

        M, N : int
            Number of elements in each axis (x and y, respectively).

        xmin, xmax, ymin, ymax : float, optional
            Bounds of the axis.
    """
    if xmin is None:
        xmin = np.min(x.flatten())
    if xmax is None:
        xmax = np.max(x.flatten())
    if ymin is None:
        ymin = np.min(y.flatten())
    if ymax is None:
        ymax = np.max(y.flatten())

    f = np.zeros((M*N, x.size))
    xp = -1 + 2*(x.reshape(-1)-xmin)/(xmax-xmin)
    yp = -1 + 2*(y.reshape(-1)-ymin)/(ymax-ymin)

    i = 0
    for m in range(M):
        for n in range(N):
            f[i, :] = Pn(m, xp)*Pn(n, yp)
            i += 1
    return f


def interpolate_scattered_field(es, new_NM, new_NS):
    """Interpolate the scattered field.

    Parameters
    ----------
        es : :class:`numpy.ndarray`
            Scattered field matrix.

        new_NM : int
            New number of measurement angles.

        new_NS : int
            New number of incidence angles.
    """
    NM, NS = es.shape
    theta = cfg.get_angles(NM)
    phi = cfg.get_angles(NS)
    freal = interp2d(phi, theta, np.real(es), kind='cubic')
    fimag = interp2d(phi, theta, np.imag(es), kind='cubic')
    new_theta = cfg.get_angles(new_NM)
    new_phi = cfg.get_angles(new_NS)
    esi = freal(new_phi, new_theta) + 1j*fimag(new_phi, new_theta)
    return esi


def interpolate_intern_field(et, N):
    """Interpolate intern field data.

    Increase the number of sources.

    Parameters
    ----------
        et : :class:`numpy.ndarray`
            Total field matrx.

        N : int
            New number of sources.
    """
    theta = np.linspace(0, 1, et.shape[0])
    phi = np.linspace(0, 2*np.pi, et.shape[1], endpoint=False)
    freal = interp2d(phi, theta, np.real(et), kind='cubic')
    fimag = interp2d(phi, theta, np.imag(et), kind='cubic')
    phi2 = np.linspace(0, 2*np.pi, N, endpoint=False)
    et2 = freal(phi2, theta) + 1j*fimag(phi2, theta)
    return et2


@jit(nopython=True)
def computeA(NM, NS, NW, NZ, NP, NQ, NX, NY, K, fij, gij, du, dv, dtheta,
             dphi):
    """Summarize method."""
    B = 1j*np.zeros((NM*NS, NP*NQ))
    for j in range(NP*NQ):
        for k in range(NM*NS):
            B[k, j] = np.trapz(np.trapz(K[k, :].reshape((NY, NX))
                                        * fij[j, :].reshape((NY, NX)), dx=du),
                               dx=dv)
    A = 1j*np.zeros((NW*NZ, NP*NQ))
    for i in range(NW*NZ):
        for j in range(NP*NQ):
            A[i, j] = np.trapz(np.trapz(np.reshape(np.copy(B[:, j]), (NM, NS))
                                        * gij[i, :].reshape((NM, NS)),
                                        dx=dphi), dx=dtheta)
    return A


@jit(nopython=True)
def computebeta(es, gij, dtheta, dphi):
    """Summarize the method."""
    beta = 1j*np.zeros(gij.shape[0])
    for i in range(gij.shape[0]):
        beta[i] = np.trapz(np.trapz(es*gij[i, :].reshape(es.shape),
                                    dx=dphi), dx=dtheta)
    return beta
