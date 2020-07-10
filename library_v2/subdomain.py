r"""The Subdomain Method.

This module provides the implementation of the Subdomain Method [1]_
which assumes the weight and trial functions as the Dirac :math:`\delta`
function. This corresponds to the pulse discretization and the Finite-
Volume Method.

This modules provides

    :class:`SubdomainMethod`
        The implementation of the method.
    :func:`get_operator_matrix`
        Compute the kernel operator.

References
----------
.. [1] Fletcher, Clive AJ. "Computational galerkin methods."
   Computational galerkin methods. Springer, Berlin, Heidelberg, 1984.
   72-85.
.. [2] Pastorino, Matteo. Microwave imaging. Vol. 208. John Wiley &
   Sons, 2010.
"""

import numpy as np
from scipy import constants as ct
from numba import jit

import library_v2.weightedresiduals as wrm
import library_v2.configuration as cfg


class SubdomainMethod(wrm.MethodOfWeightedResiduals):
    r"""The Subdomain Method.

    This class implements the definition of the coefficient matrix and
    right-hand-side of the Method of Weighted Residuals according to the
    Subdomain Method [1]_. The `_recover_map` function is also defined.

    References
    ----------
    .. [1] Fletcher, Clive AJ. "Computational galerkin methods."
       Computational galerkin methods. Springer, Berlin, Heidelberg,
       1984. 72-85.
    """

    discretization_method_name = 'Subdomain Method'
    discretization_method_alias = 'subdomain'
    _GS = None

    def _compute_A(self, inputdata):
        """Compute the coefficient matrix.

        The coefficient matrix for the Method of Weighted Residuals is
        compute.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                The instance of the problem with the definition of the
                scattered and total fields.

        Returns
        -------
            :class:`numpy.ndarray`
        """
        # Check if it is necessary to recompute the Green function or
        # if the one compute for the last execution is compatible with
        # the new instance.
        if (self._GS is None
                or self._GS.shape[1]
                != inputdata.resolution[0]*inputdata.resolution[1]):
            xm, ym = cfg.get_coordinates_sdomain(self.configuration.Ro,
                                                 self.configuration.NM)
            x, y = cfg.get_coordinates_ddomain(
                configuration=self.configuration,
                resolution=inputdata.resolution
            )
            self._GS = cfg.get_greenfunction(xm, ym, x, y,
                                             self.configuration.kb)
        A = get_operator_matrix(inputdata.et,
                                self.configuration.NM,
                                self.configuration.NS, self._GS,
                                inputdata.resolution[0]
                                * inputdata.resolution[1])
        return A

    def _compute_beta(self, inputdata):
        """Compute the right-hand-side array.

        The right-hand-side array for the Method of Weighted Residuals
        is compute.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                The instance of the problem with the definition of the
                scattered field.

        Returns
        -------
            :class:`numpy.ndarray`
        """
        return np.reshape(inputdata.es, (-1))

    def _recover_map(self, inputdata, alpha):
        """Recover the contrast map.

        Recover the relative permittivity and conductivity maps from
        the solution of the linear system.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                The instance of the problem with the definition of the
                image resolution.

            alpha : :class:`numpy.ndarray`
                The solution of the `A*alpha=beta`.
        """
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
        """Print discretization information."""
        message = super().__str__()
        message = message + 'Discretization: Subdomain Method'
        return message

    def reset_parameters(self):
        """Reset the Green function matrix."""
        super().reset_parameters()
        self._GS = None


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
