"""Brief definition of the module.

Brief explanation of the module.
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
MEMORY_LIMIT = 16e9


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

    def solve(self, scenario, PRINT_INFO=False, COMPUTE_INTERN_FIELD=False):
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
        f = self.config.f
        omega = 2*np.pi*f
        kb = self.config.kb
        Lx, Ly = self.config.Lx, self.config.Ly
        NX, NY = scenario.resolution
        N = NX*NY
        epsilon_r, sigma = scenario.epsilon_r, scenario.sigma
        xmin, xmax = cfg.get_bounds(Lx)
        ymin, ymax = cfg.get_bounds(Ly)
        dx, dy = Lx/NX, Ly/NY
        xm, ym = cfg.get_coordinates_sdomain(Ro, NM)
        x, y = cfg.get_coordinates_ddomain(dx, dy, xmin, xmax, ymin,
                                           ymax)
        ei = self.incident_field(scenario.resolution)
        scenario.ei = np.copy(ei)
        GS = cfg.get_greenfunction(xm, ym, x, y, kb)

        if isinstance(self.f, float):
            MONO_FREQUENCY = True
        else:
            MONO_FREQUENCY = False
            NF = self.f.size

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
            for f in range(NF):
                b[:, :, f] = (np.tile(Xr[:, :, f].reshape((-1, 1)), (1, NS))
                              * ei[:, :, f])

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
                                                  (np.squeeze(Z[:, :, f]),
                                                   np.squeeze(b[:, :, f]),
                                                   NX, NY, NS,
                                                   np.squeeze(Xr[:, :, f]),
                                                   self.MAX_IT, self.TOL,
                                                   False)
                                                  for f in range(NF)))

            for f in range(NF):
                J[:, :, f] = results[f][0]
                niter[f] = results[f][1]
                error[:, f] = results[f][2]
                print('Frequency: %.3f ' % (self.f[f]/1e9) + '[GHz] - '
                      + 'Number of iterations: %d - ' % (niter[f]+1)
                      + 'Error: %.3e' % error[int(niter[f]), f])

        if MONO_FREQUENCY:
            es = GS@J  # Scattered Field, NM x NS

        else:
            es = np.zeros((NM, NS, NF), dtype=complex)
            for f in range(NF):
                es[:, :, f] = GS[:, :, f]@J[:, :, f]

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
                    for f in range(NF):
                        et[:, :, f] = GD[:, :, f]@J[:, :, f]
                else:
                    for f in range(NF):
                        GD = cfg.get_greenfunction(x.reshape(-1),
                                                   y.reshape(-1), x, y, kb[f])
                        et[:, :, f] = GD@J[:, :, f]

            et = et + ei
            scenario.et = np.copy(et)

            return es, et, ei

        else:
            return es, ei

    def __get_extended_matrix(self, Rmn, kb, an, NX, NY):
        """Return the extended matrix of Method of Moments."""
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
            J : :class:`numpy.ndarray`
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
