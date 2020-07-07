"""Define the module.

A brief explanation of the module.
"""

import numpy as np
from scipy import constants as ct
from numba import jit

import library_v2.weightedresiduals as wrm
import library_v2.configuration as cfg


class SubDomainMethod(wrm.MethodOfWeightedResiduals):
    """A class for pulse-basis discretization."""

    def _compute_A(self, inputdata):
        """Summarize the method."""
        xm, ym = cfg.get_coordinates_sdomain(self.configuration.Ro,
                                             self.configuration.NM)
        x, y = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                           resolution=inputdata.resolution)
        GS = cfg.get_greenfunction(xm, ym, x, y, self.configuration.kb)
        A = get_operator_matrix(inputdata.et,
                                self.configuration.NM,
                                self.configuration.NS, GS,
                                inputdata.resolution[0]
                                * inputdata.resolution[1])
        return A

    def _compute_beta(self, inputdata):
        return np.reshape(inputdata.es, (-1))

    def _recover_map(self, inputdata, alpha):
        """Summarize method."""
        if (self.configuration.perfect_dielectric
                or not self.configuration.good_conductor):
            inputdata.epsilon_r = np.reshape(self.configuration.epsilon_rb
                                             * (np.real(alpha)+1),
                                             inputdata.resolution)
            inputdata.epsilon_r[inputdata.epsilon_r < 1] = 1

        if (self.configuration.good_conductor
                or not self.configuration.perfect_dielectric):
            omega = 2*np.pi*self.configuration.f
            inputdata.sigma = np.reshape(self.configuration.sigma_b
                                         - np.imag(alpha)*omega*ct.epsilon_0,
                                         inputdata.resolution)
            inputdata.sigma[inputdata.sigma < 0] = 0

    def __str__(self):
        message = super().__str__()
        message = message + 'Discretization: Subdomain Method'
        return 

@jit(nopython=True)
def get_operator_matrix(et, NM, NS, GS, N):
    """Compute the kernel.

    This method computes the kernel matrix of the integral equation.

    Parameters
    ----------
        et : :class:`numpy.ndarray`
            The total field matrix.

        NM, NS, N : int
            The number of measurements, sources and pixels,
            respectively.

        GS : :class:`numpy.ndarray`
            The Green function matrix.

    Returns
    -------
        K : :class:`numpy.ndarray`
            The [NM*NS] x N matrix with the kernel evaluation.
    """
    K = 1j*np.ones((NM*NS, N))
    row = 0
    for m in range(NM):
        for n in range(NS):
            K[row, :] = GS[m, :]*et[:, n]
            row += 1
    return K
