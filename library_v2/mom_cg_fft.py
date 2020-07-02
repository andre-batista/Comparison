"""Method of Moments - Conjugate-Gradient FFT Method.

This module provides the implementation of Method of Moments (MoM) with
the Conjugated-Gradient FFT formulation. It solves the forward problem
following the Forward Solver abstract class.

References
----------
.. [1] P. Zwamborn and P. M. van den Berg, "The three dimensional weak
   form of the conjugate gradient FFT method for solving scattering
   problems," in IEEE Transactions on Microwave Theory and Techniques,
   vol. 40, no. 9, pp. 1757-1766, Sept. 1992, doi: 10.1109/22.156602.

.. [2] Chen, Xudong. "Computational methods for electromagnetic inverse
   scattering". John Wiley & Sons, 2018.
"""

import time
import numpy as np
from numpy import linalg as lag
import scipy.special as spc
from joblib import Parallel, delayed
import multiprocessing
import library_v2.forward as fwr
import library_v2.inputdata as ipt
import library_v2.configuration as cfg

# Predefined constants
MEMORY_LIMIT = 16e9  # [GB]


class MoM_CG_FFT(fwr.ForwardSolver):
    """Method of Moments - Conjugated-Gradient FFT Method.

    This class implements the Method of Moments following the
    Conjugated-Gradient FFT formulation.

    Attributes
    ----------
        MAX_IT : int
            Maximum number of iterations.
        TOL : float
            Tolerance level of error.
    """

    MAX_IT = int()
    TOL = float()

    def __init__(self, configuration, configuration_filepath='',
                 tolerance=1e-6, maximum_iterations=10000):
        """Create the object.

        Parameters
        ----------
            configuration : string or :class:`Configuration`:Configuration
                Either a configuration object or a string with the name
                of file in which the configuration is saved. In this
                case, the file path may also be provided.

            configuration_filepath : string, optional
                A string with the path to the configuration file (when
                the file name is provided).

            tolerance : float, default: 1e-6
                Minimum error tolerance.

            maximum_iteration : int, default: 10000
                Maximum number of iterations.
        """
        super().__init__(configuration, configuration_filepath)
        self.TOL = tolerance
        self.MAX_IT = maximum_iterations
        self.name = 'Method of Moments - CG-FFT'

    def solve(self, scenario, PRINT_INFO=False, COMPUTE_INTERN_FIELD=True):
        """Solve the forward problem.

        Parameters
        ----------
            scenario : :class:`inputdata:InputData`
                An object describing the dielectric property map.

            PRINT_INFO : boolean, default: False
                Print iteration information.

            COMPUTE_INTERN_FIELD : boolean, default: True
                Compute the total field in D-domain.

        Return
        ------
            es, et, ei : :class:`numpy:ndarray`
                Matrices with the scattered, total and incident field
                information.

        Examples
        --------
        >>> solver = MoM_CG_FFT(configuration)
        >>> es, et, ei = solver.solve(scenario)
        >>> es, ei = solver.solve(scenario, COMPUTE_INTERN_FIELD=False)
        """
        super().solve(scenario)
        # Quick access for configuration variables
        NM = self.configuration.NM
        NS = self.configuration.NS
        Ro = self.configuration.Ro
        epsilon_rb = self.configuration.epsilon_rb
        sigma_b = self.configuration.sigma_b
        f = self.configuration.f
        omega = 2*np.pi*f
        kb = self.configuration.kb
        Lx, Ly = self.configuration.Lx, self.configuration.Ly
        NX, NY = scenario.resolution
        N = NX*NY
        epsilon_r, sigma = scenario.epsilon_r, scenario.sigma
        xmin, xmax = cfg.get_bounds(Lx)
        ymin, ymax = cfg.get_bounds(Ly)
        dx, dy = Lx/NX, Ly/NY
        xm, ym = cfg.get_coordinates_sdomain(Ro, NM)
        x, y = cfg.get_coordinates_ddomain(dx=dx, dy=dy, xmin=xmin, xmax=xmax,
                                           ymin=ymin, ymax=ymax)
        ei = self.incident_field(scenario.resolution)
        scenario.ei = np.copy(ei)
        GS = cfg.get_greenfunction(xm, ym, x, y, kb)

        if isinstance(f, float):
            MONO_FREQUENCY = True
        else:
            MONO_FREQUENCY = False
            NF = f.size

        deltasn = dx*dy  # area of the cell
        an = np.sqrt(deltasn/np.pi)  # radius of the equivalent circle
        Xr = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b, omega)

        # Using circular convolution [extended domain (2N-1)x(2N-1)]
        [xe, ye] = np.meshgrid(np.arange(xmin-(NX/2-1)*dx, xmax+NY/2*dx, dx),
                               np.arange(ymin-(NY/2-1)*dy, ymax+NY/2*dy, dy))
        Rmn = np.sqrt(xe**2 + ye**2)  # distance between the cells
        Z = self.__get_extended_matrix(Rmn, kb, an, NX, NY)

        if MONO_FREQUENCY:
            b = np.tile(Xr.reshape((-1, 1)), (1, NS))*ei

        else:
            b = np.zeros((N, NS, NF), dtype=complex)
            for nf in range(NF):
                b[:, :, nf] = (np.tile(Xr[:, :, nf].reshape((-1, 1)), (1, NS))
                               * ei[:, :, nf])

        if MONO_FREQUENCY:
            tic = time.time()
            J, niter, error = self.__CG_FFT(Z, b, NX, NY, NS, Xr, self.MAX_IT,
                                            self.TOL, PRINT_INFO)
            time_cg_fft = time.time()-tic
            if PRINT_INFO:
                print('Execution time: %.2f' % time_cg_fft + ' [sec]')

        else:
            J = np.zeros((N, NS, NF), dtype=complex)
            niter = np.zeros(NF)
            error = np.zeros((self.MAX_IT, NF))
            num_cores = multiprocessing.cpu_count()

            results = (Parallel(n_jobs=num_cores)(delayed(self.__CG_FFT)
                                                  (np.squeeze(Z[:, :, nf]),
                                                   np.squeeze(b[:, :, nf]),
                                                   NX, NY, NS,
                                                   np.squeeze(Xr[:, :, nf]),
                                                   self.MAX_IT, self.TOL,
                                                   False)
                                                  for f in range(NF)))

            for nf in range(NF):
                J[:, :, nf] = results[nf][0]
                niter[nf] = results[nf][1]
                error[:, nf] = results[nf][2]
                print('Frequency: %.3f ' % (f[nf]/1e9) + '[GHz] - '
                      + 'Number of iterations: %d - ' % (niter[nf]+1)
                      + 'Error: %.3e' % error[int(niter[nf]), nf])

        if MONO_FREQUENCY:
            es = GS@J  # Scattered Field, NM x NS

        else:
            es = np.zeros((NM, NS, NF), dtype=complex)
            for nf in range(NF):
                es[:, :, nf] = GS[:, :, nf]@J[:, :, nf]

        if scenario.noise > 0:
            es = fwr.add_noise(es, scenario.noise)
        scenario.es = np.copy(es)

        if COMPUTE_INTERN_FIELD:

            if MONO_FREQUENCY:
                GD = cfg.get_greenfunction(x.reshape(-1), y.reshape(-1), x, y,
                                           kb)
                et = GD@J

            else:
                et = np.zeros((N, NS, NF), dtype=complex)
                if 8*(N)**2*NF < MEMORY_LIMIT:
                    GD = cfg.get_greenfunction(x.reshape(-1), y.reshape(-1), x,
                                               y, kb)
                    for nf in range(NF):
                        et[:, :, nf] = GD[:, :, nf]@J[:, :, nf]
                else:
                    for f in range(NF):
                        GD = cfg.get_greenfunction(x.reshape(-1),
                                                   y.reshape(-1), x, y, kb[nf])
                        et[:, :, nf] = GD@J[:, :, nf]

            et = et + ei
            scenario.et = np.copy(et)

            return es, et, ei

        else:
            return es, ei

    def __get_extended_matrix(self, Rmn, kb, an, NX, NY):
        """Return the extended matrix of Method of Moments.

        Parameters
        ----------
            Rmn : :class:`numpy:ndarray`
                Radius matrix.

            kb : float or :class:`numpy:ndarray`
                Wavenumber [1/m]

            an : float
                Radius of equivalent element radius circle.

            Nx, Ny : int
                Number of cells in each direction.

        Returns
        -------
            Z : :class:`numpy:ndarray`
                The extent matrix.
        """
        if isinstance(kb, float):

            # Matrix elements for off-diagonal entries (m=/n)
            Zmn = ((1j*np.pi*kb*an)/2)*spc.jv(1, kb*an)*spc.hankel2(0, kb*Rmn)
            # Matrix elements for diagonal entries (m==n)
            Zmn[NY-1, NX-1] = ((1j*np.pi*kb*an)/2)*spc.hankel2(1, kb*an)+1

            # Extended matrix (2N-1)x(2N-1)
            Z = np.zeros((2*NY-1, 2*NX-1), dtype=complex)
            Z[:NY, :NX] = Zmn[NY-1:2*NY-1, NX-1:2*NX-1]
            Z[NY:2*NY-1, NX:2*NX-1] = Zmn[:NY-1, :NX-1]
            Z[NY:2*NY-1, :NX] = Zmn[:NY-1, NX-1:2*NX-1]
            Z[:NY, NX:2*NX-1] = Zmn[NY-1:2*NY-1, :NX-1]

        else:

            Z = np.zeros((2*NY-1, 2*NX-1, kb.size), dtype=complex)

            for f in range(kb.size):

                # Matrix elements for off-diagonal entries
                Zmn = (((1j*np.pi*kb[f]*an)/2)*spc.jv(1, kb[f]*an)
                       * spc.hankel2(0, kb[f]*Rmn))  # m=/n
                # Matrix elements for diagonal entries (m==n)
                Zmn[NY-1, NX-1] = (((1j*np.pi*kb[f]*an)/2)
                                   * spc.hankel2(1, kb[f]*an)+1)

                Z[:NY, :NX, f] = Zmn[NY-1:2*NY-1, NX-1:2*NX-1]
                Z[NY:2*NY-1, NX:2*NX-1, f] = Zmn[:NY-1, :NX-1]
                Z[NY:2*NY-1, :NX, f] = Zmn[:NY-1, NX-1:2*NX-1]
                Z[:NY, NX:2*NX-1, f] = Zmn[NY-1:2*NY-1, :NX-1]

        return Z

    def __CG_FFT(self, Z, b, NX, NY, NS, Xr, MAX_IT, TOL, PRINT_CONVERGENCE):
        """Apply the Conjugated-Gradient Method to the forward problem.

        Parameters
        ----------
            Z : :class:`numpy.ndarray`
                Extended matrix, (2NX-1)x(2NY-1)

            b : :class:`numpy.ndarray`
                Excitation source, (NX.NY)xNi

            NX : int
                Contrast map in x-axis.

            NY : int
                Contrast map in x-axis.

            NS : int
                Number of incidences.

            Xr : :class:`numpy.ndarray`
                Contrast map, NX x NY

            MAX_IT : int
                Maximum number of iterations

            TOL : float
                Error tolerance

            PRINT_CONVERGENCE : boolean
                Print error information per iteration.

        Returns
        -------
            J : :class:`numpy:ndarray`
                Current density, (NX.NY)xNS
        """
        Jo = np.zeros((NX*NY, NS), dtype=complex)  # initial guess
        ro = self.__fft_A(Jo, Z, NX, NY, NS, Xr)-b  # ro = A.Jo - b;
        go = self.__fft_AH(ro, Z, NX, NY, NS, Xr)  # Complex conjugate AH
        po = -go
        error_res = np.zeros(MAX_IT)

        for n in range(MAX_IT):

            alpha = -1*(np.sum(np.conj(self.__fft_A(po, Z, NX, NY, NS, Xr))
                               * (self.__fft_A(Jo, Z, NX, NY, NS, Xr)-b),
                               axis=0)
                        / lag.norm(np.reshape(self.__fft_A(po, Z, NX, NY, NS,
                                                           Xr), (NX*NY*NS, 1),
                                              order='F'), ord='fro')**2)

            J = Jo + np.tile(alpha, (NX*NY, 1))*po
            r = self.__fft_A(J, Z, NX, NY, NS, Xr)-b
            g = self.__fft_AH(r, Z, NX, NY, NS, Xr)

            error = lag.norm(r)/lag.norm(b)  # error tolerance
            error_res[n] = error

            if PRINT_CONVERGENCE:
                print('Iteration %d ' % (n+1) + ' - Error: %.3e' % error)

            if error < TOL:  # stopping criteria
                break

            beta = np.sum(np.conj(g)*(g-go), axis=0)/np.sum(np.abs(go)**2,
                                                            axis=0)
            p = -g + np.tile(beta, (NX*NY, 1))*po

            po = p
            Jo = J
            go = g

        return J, n, error_res
