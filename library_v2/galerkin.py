"""Give a title to the module.

A brief explanation of the module.
"""

import library_v2.weightedresiduals as wrm
import library_v2.configuration as cfg
import library_v2.collocation as clc

import numpy as np
from scipy.special import eval_legendre as Pn

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

        self.u, self.v = x.reshape(-1), y.reshape(-1)
        self.du, self.dv = dx, dy
        self.FLAG_INTERPOLATION = not(NW <= NM and NZ <= NS)

        if not self.FLAG_INTERPOLATION:
            self.xmn = np.reshape(np.tile(xm.reshape((-1, 1)), (1, NS)), (-1))
            self.ymn = np.reshape(np.tile(ym.reshape((-1, 1)), (1, NS)), (-1))
            self.phi_mn, self.theta_mn = np.meshgrid(cfg.get_angles(NS),
                                                     cfg.get_angles(NM))
            self.phi_wz, self.theta_wz = np.meshgrid(cfg.get_angles(NZ),
                                                     cfg.get_angles(NW))
            self.dtheta = self.theta_mn[1, 0]-self.theta_mn[0, 0]
            self.dphi = self.phi_mn[0, 1]-self.phi_mn[0, 0]
            self.R = np.zeros((NM*NS, self.u.size))
            for i in range(NM*NS):
                self.R[i, :] = np.sqrt((self.xmn[i]-self.u)**2
                                       + (self.ymn[i]-self.v)**2)

        else:
            self.xwz, self.ywz = None, None
            self.phi_mn, self.theta_mn = None, None
            self.phi_wz, self.theta_wz = None, None
            self.esi, self.eti = None, None
            self.dphi, self.dtheta = None, None
            self.R = None

        self.xpq, self.ypq = clc.get_elements_mesh(NX, NY, dx, dy, NP, NQ)

        if self.basis_function == BASIS_BILINEAR:
            self.fij = clc.bilinear_basisf(self.u.reshape((NY, NX)),
                                           self.v.reshape((NY, NX)),
                                           self.xpq.reshape((NQ, NP)),
                                           self.ypq.reshape((NQ, NP)))
            if not self.FLAG_INTERPOLATION:
                self.gij = clc.bilinear_basisf(self.phi_mn, self.theta_mn,
                                               self.phi_wz, self.theta_wz)
            else:
                self.gij = None

        elif self.basis_function == BASIS_LEGENDRE:
            xmin, xmax = cfg.get_bounds(self.configuration.Lx)
            ymin, ymax = cfg.get_bounds(self.configuration.Ly)
            self.fij = legendre_basisf(self.u.reshape((NY, NX)),
                                       self.v.reshape((NY, NX)),
                                       NQ, NP, xmin, xmax, ymin, ymax)
            if not self.FLAG_INTERPOLATION:
                self.gij = legendre_basisf(self.phi_mn, self.theta_mn,
                                           NW, NZ, 0, 2*np.pi, 0, 2*np.pi)
            else:
                self.gij = None


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
