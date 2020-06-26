"""Brief definition of the module.

Brief explanation of the module.
"""

import numpy as np
import scipy.special as spc
import library_v2.forward as fwr
import library_v2.inputdata as ipt
import library_v2.configuration as cfg


class MoM_CG_FFT(fwr.ForwardSolver):
    """Define the class.

    Definition of attributes.
    """

    MAX_IT = int()
    TOL = float()

    def __init__(self, configuration, configuration_filepath='',
                 tolerance=1e-6, maximum_iterations=100):
        """Brief definition of the constructor.

        Brief definition of parameters.
        """
        super().__init__(configuration, configuration_filepath)
        self.TOL = tolerance
        self.MAX_IT = maximum_iterations

    def solve(self, scenario):
        """Brief definition of the function.

        Brief definition of parameters.
        """
        super().solve(scenario)
        # Quick access for configuration variables
        NM = self.config.NM
        NS = self.config.NS
        Ro = self.config.Ro
        epsilon_rb = self.config.epsilon_rb
        sigma_b = self.config.sigma_b
        E0 = self.config.E0
        f = self.config.f
        omega = 2*np.pi*f
        lambda_b = self.config.lambda_b
        kb = self.config.kb
        Lx, Ly = self.config.Lx, self.config.Ly
        NX, NY = scenario.resolution
        epsilon_r, sigma = scenario.epsilon_r, scenario.sigma
        xmin, xmax = cfg.get_bounds(Lx)
        ymin, ymax = cfg.get_bounds(Ly)
        dx, dy = Lx/NX, Ly/NY
        x, y = cfg.get_coordinates_ddomain(dx, dy, xmin, xmax, ymin,
                                           ymax)
        ei = self.incident_field(scenario.resolution)

        if isinstance(self.f, float):
            MONO_FREQUENCY = True
        else:
            MONO_FREQUENCY = False

        deltasn = dx*dy # area of the cell
        an = np.sqrt(deltasn/np.pi) # radius of the equivalent circle
        Xr = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b, omega)

        # Using circular convolution [extended domain (2N-1)x(2N-1)]
        [xe, ye] = np.meshgrid(np.arange(xmin-(NX/2-1)*dx, xmax+NY/2*dx, dx),
                               np.arange(ymin-(NY/2-1)*dy, ymax+NY/2*dy, dy))
        Rmn = np.sqrt(xe**2 + ye**2) # distance between the cells
        Z = self.__get_extended_matrix(Rmn, kb, an, NX, NY)

    def __get_extended_matrix(self, Rmn, kb, an, Nx, Ny):
        """Return the extended matrix of Method of Moments"""
        if isinstance(kb, float):

            # Matrix elements for off-diagonal entries (m=/n)
            Zmn = ((1j*np.pi*kb*an)/2)*spc.jv(1, kb*an)*spc.hankel2(0, kb*Rmn)
            # Matrix elements for diagonal entries (m==n)
            Zmn[Ny-1, Nx-1] = ((1j*np.pi*kb*an)/2)*spc.hankel2(1, kb*an)+1

            # Extended matrix (2N-1)x(2N-1)
            Z = np.zeros((2*Ny-1, 2*Nx-1), dtype=complex)
            Z[:Ny, :Nx] = Zmn[Ny-1:2*Ny-1, Nx-1:2*Nx-1]
            Z[Ny:2*Ny-1, Nx:2*Nx-1] = Zmn[:Ny-1, :Nx-1]
            Z[Ny:2*Ny-1, :Nx] = Zmn[:Ny-1, Nx-1:2*Nx-1]
            Z[:Ny, Nx:2*Nx-1] = Zmn[Ny-1:2*Ny-1, :Nx-1]

        else:

            Z = np.zeros((2*Ny-1, 2*Nx-1, kb.size),dtype=complex)

            for f in range(kb.size):

                # Matrix elements for off-diagonal entries
                Zmn = (((1j*np.pi*kb[f]*an)/2)*spc.jv(1, kb[f]*an)
                       * spc.hankel2(0, kb[f]*Rmn))  # m=/n
                # Matrix elements for diagonal entries (m==n)
                Zmn[Ny-1, Nx-1] = (((1j*np.pi*kb[f]*an)/2)
                                   * spc.hankel2(1, kb[f]*an)+1)

                Z[:Ny, :Nx, f] = Zmn[Ny-1:2*Ny-1, Nx-1:2*Nx-1]
                Z[Ny:2*Ny-1, Nx:2*Nx-1, f] = Zmn[:Ny-1, :Nx-1]
                Z[Ny:2*Ny-1, :Nx, f] = Zmn[:Ny-1, Nx-1:2*Nx-1]
                Z[:Ny, Nx:2*Nx-1, f] = Zmn[Ny-1:2*Ny-1, :Nx-1]

        return Z