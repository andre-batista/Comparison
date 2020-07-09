"""Give a title to the module.

A brief explanation of the module.
"""

import library_v2.weightedresiduals as wrm
import library_v2.configuration as cfg
import library_v2.collocation as clc

import numpy as np
from scipy.special import eval_legendre as Pn
from scipy.interpolate import interp2d
from scipy.special import hankel2
from scipy import constants as ct
from numba import jit

BASIS_BILINEAR = 'bilinear'
BASIS_LEGENDRE = 'legendre'


class GalerkinMethod(wrm.MethodOfWeightedResiduals):
    """The Galerkin Method."""

    basis_function = ''
    discretization_method_name = 'Galerkin Method'
    discretization_method_alias = 'galerkin'

    def __init__(self, configuration, linear_solver, parameter,
                 basis_function, discretization):
        """Summarize the method."""
        super().__init__(configuration, linear_solver, parameter)
        self.basis_function = basis_function
        self.discretization = discretization
        self._not_valid_variables = True

    def reset_parameters(self):
        """Summarize the method."""
        super().reset_parameters()
        self._not_valid_variables = True

    def _compute_A(self, inputdata):
        """Summarize the method."""
        NM, NS = self.configuration.NM, self.configuration.NS
        NW, NZ = self.discretization[0], self.discretization[1]
        NP, NQ = self.discretization[3], self.discretization[2]
        NY, NX = inputdata.resolution

        if self._not_valid_variables or self._u.size != NX*NY:
            self._set_meshes(inputdata)
            self._not_valid_variables = False

        if self._FLAG_INTERPOLATION and self._xwz is None:
            (inputdata.es, self._theta_wz, self._phi_wz, self._dtheta,
             self._dphi, self._theta_ms, self._phi_ms) = (
                interpolate_scattered_field(inputdata.es, NW, NZ)
            )
            self._xwz = (self.configuration.Ro
                         * np.cos(self._theta_wz.reshape(-1)))
            self._ywz = (self.configuration.Ro
                         * np.sin(self._theta_wz.reshape(-1)))

            for i in range(NW*NZ):
                self._R[i, :] = np.sqrt((self._xwz[i]-self._u)**2
                                        + (self._ywz[i]-self._v)**2)

            if self.basis_function == BASIS_BILINEAR:
                self._gij = clc.bilinear_basisf(self._phi_wz, self._theta_wz,
                                                self._phi_ms, self._theta_ms)
            elif self.basis_function == BASIS_LEGENDRE:
                self._gij = legendre_basisf(self._phi_wz, self._theta_wz,
                                            NM, NS, 0, 2*np.pi, 0, 2*np.pi)

        if self._FLAG_INTERPOLATION:
            inputdata.et = interpolate_intern_field(inputdata.et, NZ)

        K = self._get_kernel(inputdata.et, inputdata.resolution)

        if self._FLAG_INTERPOLATION:
            A = computeA_interp(NM, NS, NW, NZ, NP, NQ, NX, NY, K, self._fij,
                                self._gij, self._du, self._dv, self._dtheta,
                                self._dphi)
        else:
            A = computeA_nointerp(NM, NS, NW, NZ, NP, NQ, NX, NY, K, self._fij,
                                  self._gij, self._du, self._dv, self._dtheta,
                                  self._dphi)
        return A

    def _compute_beta(self, inputdata):
        """Summarize the method."""
        return computebeta(inputdata.es, self._gij, self._dtheta, self._dphi)

    def _recover_map(self, inputdata, alpha):
        """Summarize the method."""
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
        """Summarize the method."""
        NM, NS = self.configuration.NM, self.configuration.NS
        NW, NZ = self.discretization[0], self.discretization[1]
        NP, NQ = self.discretization[3], self.discretization[2]
        NY, NX = inputdata.resolution
        dx, dy = self.configuration.Lx/NX, self.configuration.Ly/NY

        xm, ym = cfg.get_coordinates_sdomain(self.configuration.Ro,
                                             self.configuration.NM)
        x, y = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                           resolution=inputdata.resolution)

        self._u, self._v = x.reshape(-1), y.reshape(-1)
        self._du, self._dv = dx, dy
        self._FLAG_INTERPOLATION = not(NW <= NM and NZ <= NS)

        if not self._FLAG_INTERPOLATION:
            self._xms = np.reshape(np.tile(xm.reshape((-1, 1)), (1, NS)), (-1))
            self._yms = np.reshape(np.tile(ym.reshape((-1, 1)), (1, NS)), (-1))
            self._phi_ms, self._theta_ms = np.meshgrid(cfg.get_angles(NS),
                                                       cfg.get_angles(NM))
            self._phi_wz, self._theta_wz = np.meshgrid(cfg.get_angles(NZ),
                                                       cfg.get_angles(NW))
            self._dtheta = self._theta_ms[1, 0]-self._theta_ms[0, 0]
            self._dphi = self._phi_ms[0, 1]-self._phi_ms[0, 0]
            self._R = np.zeros((NM*NS, self._u.size))
            for i in range(NM*NS):
                self._R[i, :] = np.sqrt((self._xms[i]-self._u)**2
                                        + (self._yms[i]-self._v)**2)

        else:
            self._xwz, self._ywz = None, None
            self._phi_ms, self._theta_ms = None, None
            self._phi_wz, self._theta_wz = None, None
            self._dphi, self._dtheta = None, None
            self._R = None

        self._xpq, self._ypq = clc.get_elements_mesh(NX, NY, dx, dy, NP, NQ)

        if self.basis_function == BASIS_BILINEAR:
            self._fij = clc.bilinear_basisf(self._u.reshape((NY, NX)),
                                            self._v.reshape((NY, NX)),
                                            self._xpq.reshape((NQ, NP)),
                                            self._ypq.reshape((NQ, NP)))
            if not self._FLAG_INTERPOLATION:
                self._gij = clc.bilinear_basisf(self._phi_ms, self._theta_ms,
                                                self._phi_wz, self._theta_wz)
            else:
                self._gij = None

        elif self.basis_function == BASIS_LEGENDRE:
            xmin, xmax = cfg.get_bounds(self.configuration.Lx)
            ymin, ymax = cfg.get_bounds(self.configuration.Ly)
            self._fij = legendre_basisf(self._u.reshape((NY, NX)),
                                        self._v.reshape((NY, NX)),
                                        NQ, NP, xmin, xmax, ymin, ymax)
            if not self._FLAG_INTERPOLATION:
                self._gij = legendre_basisf(self._phi_ms, self._theta_ms,
                                            NW, NZ, 0, 2*np.pi, 0, 2*np.pi)
            else:
                self._gij = None

    def _get_kernel(self, et, resolution):
        """Summarize the method."""
        NM, NS = self.configuration.NM, self.configuration.NS
        NW, NZ = self.discretization[0], self.discretization[1]
        NY, NX = resolution
        mub = ct.mu_0
        omega = 2*np.pi*self.configuration.f
        if self._FLAG_INTERPOLATION:
            K = np.zeros((NW*NZ, NX*NY), dtype=complex)
            L = NZ
        else:
            K = np.zeros((NM*NS, NX*NY), dtype=complex)
            L = NS
        s = 0
        for i in range(K.shape[0]):
            K[i, :] = (1j*omega*mub*et[:, s]*1j/4
                       * hankel2(0, self.configuration.kb*self._R[i, :]))
            if s == L-1:
                s = 0
            else:
                s += 1
        return K


def legendre_basisf(x, y, M, N, xmin=None, xmax=None, ymin=None, ymax=None):
    """Summarize the method."""
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


def interpolate_scattered_field(es, NW, NZ):
    """Summarize method."""
    NM, NS = es.shape
    theta_mn = np.linspace(0, 2*np.pi, NM, endpoint=False)
    phi_mn = np.linspace(0, 2*np.pi, NS, endpoint=False)
    freal = interp2d(phi_mn, theta_mn, np.real(es), kind='cubic')
    fimag = interp2d(phi_mn, theta_mn, np.imag(es), kind='cubic')
    theta_wz = np.linspace(0, 2*np.pi, NW, endpoint=False)
    phi_wz = np.linspace(0, 2*np.pi, NZ, endpoint=False)
    esi = freal(phi_wz, theta_wz) + 1j*fimag(phi_wz, theta_wz)
    dtheta, dphi = theta_wz[1]-theta_wz[0], phi_wz[1]-phi_wz[0]
    phi_wz, theta_wz = np.meshgrid(phi_wz, theta_wz)
    phi_mn, theta_mn = np.meshgrid(phi_mn, theta_mn)
    return esi, theta_wz, phi_wz, dtheta, dphi, theta_mn, phi_mn


def interpolate_intern_field(et, N):
    """Summarize method."""
    theta = np.linspace(0, 1, et.shape[0])
    phi = np.linspace(0, 2*np.pi, et.shape[1], endpoint=False)
    freal = interp2d(phi, theta, np.real(et), kind='cubic')
    fimag = interp2d(phi, theta, np.imag(et), kind='cubic')
    phi2 = np.linspace(0, 2*np.pi, N, endpoint=False)
    et2 = freal(phi2, theta) + 1j*fimag(phi2, theta)
    return et2


@jit(nopython=True)
def computeA_nointerp(NM, NS, NW, NZ, NP, NQ, NX, NY, K, fij, gij, du, dv,
                      dtheta, dphi):
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
def computeA_interp(NM, NS, NW, NZ, NP, NQ, NX, NY, K, fij, gij, du, dv,
                    dtheta, dphi):
    """Summarize method."""
    B = 1j*np.zeros((NW*NZ, NP*NQ))
    for j in range(NP*NQ):
        for k in range(NW*NZ):
            B[k, j] = np.trapz(np.trapz(K[k, :].reshape((NY, NX))
                                        * fij[j, :].reshape((NY, NX)), dx=du),
                               dx=dv)
    A = 1j*np.zeros((NM*NS, NP*NQ))
    for i in range(NM*NS):
        for j in range(NP*NQ):
            A[i, j] = np.trapz(np.trapz(B[:, j].reshape((NW, NZ))
                                        * gij[i, :].reshape((NW, NZ)),
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
