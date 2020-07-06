"""Define the title of the module.

A brief explanation of the module.
"""

import numpy as np
from scipy import constants as ct
from scipy.special import hankel2
from numba import jit

import library_v2.weightedresiduals as wrm
import library_v2.configuration as cfg

BASIS_BILINEAR = 'bilinear'
BASIS_MININUM_NORM = 'mininum_norm'


class CollocationMethod(wrm.MethodOfWeightedResiduals):
    """Define the class."""

    basis_function = ''

    def __init__(self, configuration, linear_solver, parameter,
                 basis_function, discretization):
        """Summarize the method."""
        super().__init__(configuration, linear_solver, parameter)
        self.basis_function = basis_function
        self.discretization = discretization

    def _compute_A(self, inputdata):
        """Summarize the method."""
        K = self._get_kernel(inputdata.et, inputdata.resolution)
        if self.basis_function == BASIS_MININUM_NORM:
            self.fij = self._minimum_norm_basisf(inputdata.et)
        A = computeA(self.configuration.NM,
                     self.configuration.NS,
                     self.discretization[0],
                     self.discretization[1],
                     inputdata.resolution[1],
                     inputdata.resolution[0],
                     K, self.fij, self.du, self.dv)
        return A

# PRECISO REVISAR A QUESTAO DE COMO O USUARIO VAI PASSAR A DISCRETIZACAO
# SE VAI SER (NX, NY) OU (NY, NX)

    def _get_meshes(self, inputdata):
        """Summarize the method."""
        M, N = self.configuration.NM, self.configuration.NS
        P, Q = self.discretization
        NX, NY = inputdata.resolution
        dx = self.configuration.Lx/inputdata.resolution[0]
        dy = self.configuration.Ly/inputdata.resolution[1]

        xm, ym = cfg.get_coordinates_sdomain(self.configuration.Ro,
                                             self.configuration.NM)
        x, y = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                           resolution=inputdata.resolution)

        self.xmn, self.ymn = (np.reshape(np.tile(xm.reshape((-1, 1)),
                                                 (1, N)), (-1)),
                              np.reshape(np.tile(ym.reshape((-1, 1)),
                                                 (1, N)), (-1)))

        self.xpq, self.ypq = self._get_collocation_mesh(
            inputdata.resolution[0], inputdata.resolution[1],
            self.configuration.Lx/inputdata.resolution[0],
            self.configuration.Ly/inputdata.resolution[1],
            P, Q
        )

        self.u, self.v = x.reshape(-1), y.reshape(-1)
        self.du, self.dv = dx, dy

        self.R = np.zeros((M*N, self.u.size))
        for i in range(M*N):
            self.R[i, :] = np.sqrt((self.xmn[i]-self.u)**2
                                   + (self.ymn[i]-self.v)**2)

        if self.basis_function == BASIS_BILINEAR:
            self.fij = bilinear_basisf(self.u.reshape((NY, NX)),
                                       self.v.reshape((NY, NX)),
                                       self.xpq.reshape((Q, P)),
                                       self.ypq.reshape((Q, P)))

    def _get_kernel(self, et, resolution):
        """Summarize the method."""
        P, Q = self.discretization
        NX, NY = resolution
        K = np.zeros((P*Q, NX*NY), dtype=complex)
        mub = ct.mu_0
        omega = 2*np.pi*self.configuration.f
        s = 0
        for i in range(P*Q):
            K[i, :] = (1j*omega*mub*et[:, s]*1j/4
                       * hankel2(0, self.configuration.kb*self.R[i, :]))
            if s == self.configuration.NS-1:
                s = 0
            else:
                s += 1
        return K

    def _get_collocation_mesh(self, NX, NY, dx, dy, P, Q):
        """Summarize the method."""
        x_min, x_max = cfg.get_bounds(NX*dx)
        y_min, y_max = cfg.get_bounds(NY*dy)
        xpq, ypq = np.meshgrid(np.linspace(x_min, x_max, P),
                               np.linspace(y_min, y_max, Q))
        xpq, ypq = xpq.reshape(-1), ypq.reshape(-1)
        return xpq, ypq

    def _minimum_norm_basisf(self, et):
        """Summarize the method."""
        N = self.u.size
        P, Q = self.discretization
        omega = 2*np.pi*self.configuration.f
        mub = ct.mu_0
        Kpq = np.zeros((P*Q, N), dtype=complex)
        s = 0
        for i in range(P*Q):
            R = np.sqrt((self.xpq[i]-self.u)**2+(self.ypq[i]-self.v)**2)
            Kpq[i, :] = (1j*omega*mub*et[:, s]*1j/4
                         * hankel2(0, self.configuration.kb*R))
            if s == self.configuration.NS-1:
                s = 0
            else:
                s += 1
        return Kpq


def bilinear_basisf(u, v, x, y):
    """Redefine the summary.

    Evaluate the triangular function. Given the collocation points
    in x and y, the function returns the evaluation of the triangular
    function in points specificated by the variables u and v. Each of
    the four variables must be 2D (meshgrid format).
    """
    Q, P = u.shape
    N, M = x.shape
    f = np.zeros((x.size, u.size))

    for p in range(P):
        for q in range(Q):

            m = np.argwhere(u[q, p] >= x[0, :])[-1][0]
            n = np.argwhere(v[q, p] >= y[:, 0])[-1][0]

            if m+1 < M and n+1 < N:
                eta = 2*(u[q, p]-x[n, m])/(x[n, m+1]-x[n, m]) - 1
                qsi = 2*(v[q, p]-y[n, m])/(y[n+1, m]-y[n, m]) - 1

                f[n*M+m, q*P+p] = .25*(1-qsi)*(1-eta)  # 1
                f[(n+1)*M+m, q*P+p] = .25*(1+qsi)*(1-eta)  # 2
                f[(n+1)*M+m+1, q*P+p] = .25*(1+qsi)*(1+eta)  # 3
                f[n*M+m+1, q*P+p] = .25*(1-qsi)*(1+eta)  # 4

            elif m+1 < M and n == N-1:
                eta = 2*(u[q, p]-x[n, m])/(x[n, m+1]-x[n, m]) - 1
                # qsi = -1

                f[n*M+m, q*P+p] = .25*2*(1-eta)  # 1
                f[n*M+m+1, q*P+p] = .25*2*(1+eta)  # 4

            elif m == M-1 and n+1 < N:
                # eta = -1
                qsi = 2*(v[q, p]-y[n, m])/(y[n+1, m]-y[n, m]) - 1

                f[n*M+m, q*P+p] = .25*(1-qsi)*2  # 1
                f[(n+1)*M+m, q*P+p] = .25*(1+qsi)*2  # 2

            elif m == M-1 and n == N-1:
                # qsi = -1
                # eta = -1

                f[n*M+m, q*P+p] = 1.  # 1

    return f


@jit(nopython=True)
def computeA(M, N, P, Q, I, J, K, fij, du, dv):
    A = 1j*np.zeros((M*N, P*Q))
    for i in range(M*N):
        for j in range(P*Q):
            A[i, j] = np.trapz(np.trapz(K[i, :].reshape((J, I))
                                        * fij[j, :].reshape((J, I)),
                                        dx=du), dx=dv)
    return A
