r"""Method of Moments - Conjugate-Gradient FFT Method.

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
from numpy import fft
from scipy import constants as ct
import scipy.special as spc
from joblib import Parallel, delayed
import multiprocessing
import forward as fwr
import inputdata as ipt
import configuration as cfg
from matplotlib import pyplot as plt

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

    def incident_field(self, resolution):
        """Compute the incident field matrix.

        Given the configuration information stored in the object, it
        computes the incident field matrix considering plane waves in
        different from different angles.

        Parameters
        ----------
            resolution : 2-tuple
                The image size of D-domain in pixels (y and x).

        Returns
        -------
            ei : :class:`numpy.ndarray`
                Incident field matrix. The rows correspond to the points
                in the image following `C`-order and the columns
                corresponds to the sources.
        """
        NY, NX = resolution
        phi = cfg.get_angles(self.configuration.NS)
        x, y = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                           resolution=resolution)
        kb = self.configuration.kb
        E0 = self.configuration.E0

        if isinstance(kb, float) or isinstance(kb, complex):
            ei = E0*np.exp(-1j*kb*(x.reshape((-1, 1))
                                   @ np.cos(phi.reshape((1, -1)))
                                   + y.reshape((-1, 1))
                                   @ np.sin(phi.reshape((1, -1)))))
        else:
            ei = np.zeros((NX*NY, self.configuration.NS, kb.size),
                          dtype=complex)
            for f in range(kb.size):
                ei[:, :, f] = E0*np.exp(-1j*kb[f]*(x.reshape((-1, 1))
                                                   @ np.cos(phi.reshape((1,
                                                                         -1)))
                                                   + y.reshape((-1, 1))
                                                   @ np.sin(phi.reshape((1,
                                                                         -1))))
                                        )
        return ei

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
        epsilon_r, sigma = super().solve(scenario)
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
        NY, NX = scenario.resolution
        N = NX*NY
        xmin, xmax = cfg.get_bounds(Lx)
        ymin, ymax = cfg.get_bounds(Ly)
        dx, dy = Lx/NX, Ly/NY
        xm, ym = cfg.get_coordinates_sdomain(Ro, NM)
        x, y = cfg.get_coordinates_ddomain(dx=dx, dy=dy, xmin=xmin, xmax=xmax,
                                           ymin=ymin, ymax=ymax)
        ei = self.incident_field(scenario.resolution)
        scenario.ei = np.copy(ei)
        GS = get_greenfunction(xm, ym, x, y, kb)

        if isinstance(f, float):
            MONO_FREQUENCY = True
        else:
            MONO_FREQUENCY = False
            NF = f.size

        deltasn = dx*dy  # area of the cell
        an = np.sqrt(deltasn/np.pi)  # radius of the equivalent circle
        Xr = get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b, omega)

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
                GD = get_greenfunction(x.reshape(-1), y.reshape(-1), x, y, kb)
                et = GD@J

            else:
                et = np.zeros((N, NS, NF), dtype=complex)
                if 8*(N)**2*NF < MEMORY_LIMIT:
                    GD = get_greenfunction(x.reshape(-1), y.reshape(-1), x, y,
                                           kb)
                    for nf in range(NF):
                        et[:, :, nf] = GD[:, :, nf]@J[:, :, nf]
                else:
                    for f in range(NF):
                        GD = get_greenfunction(x.reshape(-1), y.reshape(-1), x,
                                               y, kb[nf])
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
        if isinstance(kb, float) or isinstance(kb, complex):

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

    def __fft_A(self, J, Z, NX, NY, NS, Xr):
        """Compute Matrix-vector product by using two-dimensional FFT."""
        J = np.reshape(J, (NY, NX, NS))
        Z = np.tile(Z[:, :, np.newaxis], (1, 1, NS))
        e = fft.ifft2(fft.fft2(Z, axes=(0, 1))
                      * fft.fft2(J, axes=(0, 1), s=(2*NY-1, 2*NX-1)),
                      axes=(0, 1))
        e = e[:NY, :NX, :]
        e = np.reshape(e, (NX*NY, NS))
        e = np.reshape(J, (NX*NY, NS)) + np.tile(Xr.reshape((-1, 1)),
                                                 (1, NS))*e

        return e

    def __fft_AH(self, J, Z, NX, NY, NS, Xr):
        """Summarize the method."""
        J = np.reshape(J, (NY, NX, NS))
        Z = np.tile(Z[:, :, np.newaxis], (1, 1, NS))
        e = fft.ifft2(fft.fft2(np.conj(Z), axes=(0, 1))
                      * fft.fft2(J, axes=(0, 1), s=(2*NY-1, 2*NX-1)),
                      axes=(0, 1))
        e = e[:NY, :NX, :]
        e = np.reshape(e, (NX*NY, NS))
        e = (np.reshape(J, (NX*NY, NS))
             + np.conj(np.tile(Xr.reshape((-1, 1)), (1, NS)))*e)
        return e

    def __str__(self):
        """Print method parametrization."""
        message = super().__str__()
        message = message + "Number of iterations: %d, " % self.MAX_IT
        message = message + "Tolerance level: %.3e" % self.TOL
        return message


def get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b, omega):
    """Compute the contrast function for a given image.

    Parameters
    ----------
        epsilon_r : `:class:numpy.ndarray`
            A matrix with the relative permittivity map.

        sigma : `:class:numpy.ndarray`
            A matrix with the conductivity map [S/m].

        epsilon_rb : float
            Background relative permittivity of the medium.

        sigma_b : float
            Background conductivity of the medium [S/m].

        frequency : float
            Linear frequency of operation [Hz].
    """
    return ((epsilon_r - 1j*sigma/omega/ct.epsilon_0)
            / (epsilon_rb - 1j*sigma_b/omega/ct.epsilon_0) - 1)


def get_greenfunction(xm, ym, x, y, kb):
    r"""Compute the Green function matrix for pulse basis discre.

    The routine computes the Green function based on a discretization of
    the integral equation using pulse basis functions [1]_.

    Parameters
    ----------
        xm : `numpy.ndarray`
            A 1-d array with the x-coordinates of measumerent points in
            the S-domain [m].

        ym : `numpy.ndarray`
            A 1-d array with the y-coordinates of measumerent points in
            the S-domain [m].

        x : `numpy.ndarray`
            A meshgrid matrix of x-coordinates in the D-domain [m].

        y : `numpy.ndarray`
            A meshgrid matrix of y-coordinates in the D-domain [m].

        kb : float or complex
            Wavenumber of background medium [1/m].

    Returns
    -------
        G : `numpy.ndarray`, complex
            A matrix with the evaluation of Green function at D-domain
            for each measument point, considering pulse basis
            discretization. The shape of the matrix is NM x (Nx.Ny),
            where NM is the number of measurements (size of xm, ym) and
            Nx and Ny are the number of points in each axis of the
            discretized D-domain (shape of x, y).

    References
    ----------
    .. [1] Pastorino, Matteo. Microwave imaging. Vol. 208. John Wiley
       & Sons, 2010.
    """
    Ny, Nx = x.shape
    M = xm.size
    dx, dy = x[0, 1]-x[0, 0], y[1, 0]-y[0, 0]
    an = np.sqrt(dx*dy/np.pi)  # radius of the equivalent circle

    xg = np.tile(xm.reshape((-1, 1)), (1, Nx*Ny))
    yg = np.tile(ym.reshape((-1, 1)), (1, Nx*Ny))
    R = np.sqrt((xg-np.tile(np.reshape(x, (Nx*Ny, 1)).T, (M, 1)))**2
                + (yg-np.tile(np.reshape(y, (Nx*Ny, 1)).T, (M, 1)))**2)

    G = (-1j*kb*np.pi*an/2*spc.jv(1, kb*an)*spc.hankel2(0, kb*R))
    G[R == 0] = 1j/2*(np.pi*kb*an*spc.hankel2(1, kb*an)-2j)

    return G
