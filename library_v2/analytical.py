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

import numpy as np
from numpy import pi
from scipy.constants import epsilon_0, mu_0
from scipy.special import jv, jvp, hankel2, h2vp
import forward as fwr
import inputdata as ipt
import configuration as cfg
import error
from matplotlib import pyplot as plt

PERFECT_DIELECTRIC_PROBLEM = 'perfect_dieletric'
PERFECT_CONDUCTOR_PROBLEM = 'perfect_conductor'


class Analytical(fwr.ForwardSolver):
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

    def __init__(self, configuration, configuration_filepath='',
                 number_terms=25):
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
        self.name = "Analytical Solution to Cylinder Scattering"
        self.NT = number_terms
        self.et, self.ei, self.es = None, None, None
        self.epsilon_r, self.sigma = None, None
        self.radius_proportion = None
        self.constrast = None
        self.problem = None

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

    def solve(self, scenario, problem=PERFECT_DIELECTRIC_PROBLEM,
              radius_proportion=.5, PRINT_INFO=False, SAVE_INTERN_FIELD=True,
              SAVE_MAP=False, contrast=2.):
        """Summarize the method."""
        if problem == PERFECT_DIELECTRIC_PROBLEM:
            self.dielectric_cylinder(scenario,
                                     radius_proportion=radius_proportion,
                                     contrast=contrast,
                                     SAVE_INTERN_FIELD=SAVE_INTERN_FIELD,
                                     SAVE_MAP=SAVE_MAP)
        elif problem == PERFECT_CONDUCTOR_PROBLEM:
            self.conductor_cylinder(scenario,
                                    radius_proportion=radius_proportion,
                                    SAVE_INTERN_FIELD=SAVE_INTERN_FIELD,
                                    SAVE_MAP=SAVE_MAP)
        else:
            raise error.WrongValueInput('Analytical.solve', 'problem',
                                        "'perfect_dielectric' or "
                                        + "'perfect_conductor'", problem)

    def dielectric_cylinder(self, scenario, radius_proportion=0.5,
                            contrast=2., SAVE_INTERN_FIELD=True,
                            SAVE_MAP=False):
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
        # Main constants
        omega = 2*pi*self.configuration.f  # Angular frequency [rad/s]
        epsilon_rd = cfg.get_relative_permittivity(
            contrast, self.configuration.epsilon_rb
        )
        epsd = epsilon_rd*epsilon_0  # Cylinder's permittivity [F/m]
        epsb = self.configuration.epsilon_rb*epsilon_0
        mud = mu_0  # Cylinder's permeability [H/m]
        kb = self.configuration.kb  # Wavenumber of background [rad/m]
        kd = omega*np.sqrt(mud*epsd)  # Wavenumber of cylinder [rad/m]
        lambdab = 2*pi/kb  # Wavelength of background [m]
        a = radius_proportion*lambdab  # Sphere's radius [m]
        thetal = cfg.get_angles(self.configuration.NS)
        thetam = cfg.get_angles(self.configuration.NM)

        # Summing coefficients
        an, cn = get_coefficients(self.NT, kb, kd, a, epsd, epsb)

        # Mesh parameters
        x, y = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                           resolution=scenario.resolution)

        # Incident field
        ei = self.incident_field(scenario.resolution)

        # Total field array
        et = compute_total_field(x, y, a, an, cn, self.NT, kb, kd,
                                 self.configuration.E0, thetal)

        # Map of parameters
        epsilon_r, _ = get_map(x, y, a, self.configuration.epsilon_rb,
                               epsilon_rd)

        # Scatered field
        rho = self.configuration.Ro
        xm, ym = rho*np.cos(thetam), rho*np.sin(thetam)
        es = compute_scattered_field(xm, ym, an, kb, thetal,
                                     self.configuration.E0)

        if scenario.noise > 0:
            es = fwr.add_noise(es, scenario.noise)

        scenario.es = np.copy(es)
        scenario.ei = np.copy(ei)
        if SAVE_INTERN_FIELD:
            scenario.et = np.copy(et)
        if SAVE_MAP:
            scenario.epsilon_r = np.copy(epsilon_r)
        self.et = et
        self.ei = ei
        self.es = es
        self.epsilon_r = epsilon_r
        self.sigma = None
        self.radius_proportion = radius_proportion
        self.contrast = contrast
        self.problem = PERFECT_DIELECTRIC_PROBLEM

    def conductor_cylinder(self, scenario, radius_proportion=0.5,
                           SAVE_INTERN_FIELD=True, SAVE_MAP=False):
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
        # Main constants
        omega = 2*pi*self.configuration.f  # Angular frequency [rad/s]
        kb = self.configuration.kb  # Wavenumber of background [rad/m]
        a = radius_proportion*self.configuration.lambda_b  # Sphere's radius
        thetal = cfg.get_angles(self.configuration.NS)
        thetam = cfg.get_angles(self.configuration.NM)

        # Summing coefficients
        n = np.arange(-self.NT, self.NT+1)
        an = -jv(n, kb*a)/hankel2(n, kb*a)
        cn = np.zeros(n.size)

        # Mesh parameters
        x, y = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                           resolution=scenario.resolution)

        # Incident field
        ei = self.incident_field(scenario.resolution)

        # Total field array
        et = compute_total_field(x, y, a, an, cn, self.NT, kb, 1.,
                                 self.configuration.E0, thetal)

        # Map of parameters
        sigma = np.zeros(x.shape)
        sigma[x**2 + y**2 <= a**2] = 1e10

        # Scatered field
        rho = self.configuration.Ro
        xm, ym = rho*np.cos(thetam), rho*np.sin(thetam)
        es = compute_scattered_field(xm, ym, an, kb, thetal,
                                     self.configuration.E0)

        if scenario.noise > 0:
            es = fwr.add_noise(es, scenario.noise)

        scenario.es = np.copy(es)
        scenario.ei = np.copy(ei)
        if SAVE_INTERN_FIELD:
            scenario.et = np.copy(et)
        if SAVE_MAP:
            scenario.sigma = np.copy(sigma)
        self.et = et
        self.ei = ei
        self.es = es
        self.epsilon_r = None
        self.sigma = sigma
        self.radius_proportion = radius_proportion
        self.problem = PERFECT_DIELECTRIC_PROBLEM

    def __str__(self):
        """Print method parametrization."""
        message = super().__str__()
        message = message + "Number of summing terms: %d" % self.NT
        if self.radius_proportion is not None:
            message = (message + '\nRadius proportion: %.2f [wavelengths]'
                       % self.radius_proportion)
        if self.problem == PERFECT_DIELECTRIC_PROBLEM:
            message = message + '\nProblem: Perfect Dielectric Cylinder'
            message = message + '\nConstrast: %.2f' % self.contrast 
        elif self.problem == PERFECT_CONDUCTOR_PROBLEM:
            message = message + '\nProblem: Perfect Conductor Cylinder'
        return message


def cart2pol(x, y):
    """Summarize the method."""
    rho = np.sqrt(x**2+y**2)
    phi = np.arctan2(y, x)
    phi[phi < 0] = 2*pi + phi[phi < 0]
    return rho, phi


def get_coefficients(Nterms, wavenumber_b, wavenumber_d, radius, epsilon_d,
                     epsilon_b):
    """Summarize the method."""
    n = np.arange(-Nterms, Nterms+1)
    kb, kd = wavenumber_b, wavenumber_d
    a = radius

    an = (-jv(n, kb*a)/hankel2(n, kb*a)*(
        (epsilon_d*jvp(n, kd*a)/(epsilon_b*kd*a*jv(n, kd*a))
         - jvp(n, kb*a)/(kb*a*jv(n, kb*a)))
        / (epsilon_d*jvp(n, kd*a)/(epsilon_b*kd*a*jv(n, kd*a))
           - h2vp(n, kb*a)/(kb*a*hankel2(n, kb*a)))
    ))

    cn = 1/jv(n, kd*a)*(jv(n, kb*a)+an*hankel2(n, kb*a))

    return an, cn


def rotate_axis(theta, x, y):
    """Summarize the method."""
    T = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    r = np.vstack((x.reshape(-1), y.reshape(-1)))
    rp = T@r
    xp, yp = np.vsplit(rp, 2)
    xp = np.reshape(np.squeeze(xp), x.shape)
    yp = np.reshape(np.squeeze(yp), y.shape)
    return xp, yp


def compute_total_field(x, y, radius, an, cn, N, wavenumber_b, wavenumber_d,
                        magnitude, theta=None):
    """Summarize the method."""
    E0 = magnitude
    kb, kd = wavenumber_b, wavenumber_d
    a = radius

    if theta is None:
        rho, phi = cart2pol(x, y)
        et = np.zeros(rho.shape, dtype=complex)
        i = 0
        for n in range(-N, N+1):

            et[rho > a] = et[rho > a] + (
                E0*1j**(-n)*(jv(n, kb*rho[rho > a])
                             + an[i]*hankel2(n, kb*rho[rho > a]))
                * np.exp(1j*n*phi[rho > a])
            )

            et[rho <= a] = et[rho <= a] + (
                E0*1j**(-n)*cn[i]*jv(n, kd*rho[rho <= a])
                * np.exp(1j*n*phi[rho <= a])
            )

            i += 1

    else:
        S = theta.size
        et = np.zeros((x.size, S), dtype=complex)
        for s in range(S):
            xp, yp = rotate_axis(theta[s], x.reshape(-1), y.reshape(-1))
            rho, phi = cart2pol(xp, yp)
            i = 0
            for n in range(-N, N+1):

                et[rho > a, s] = et[rho > a, s] + (
                    E0*1j**(-n)*(jv(n, kb*rho[rho > a])
                                 + an[i]*hankel2(n, kb*rho[rho > a]))
                    * np.exp(1j*n*phi[rho > a])
                )

                et[rho <= a, s] = et[rho <= a, s] + (
                    E0*1j**(-n)*cn[i]*jv(n, kd*rho[rho <= a])
                    * np.exp(1j*n*phi[rho <= a])
                )

                i += 1
    return et


def get_map(x, y, radius, epsilon_rb, epsilon_rd):
    """Summarize the method."""
    epsilon_r = epsilon_rb*np.ones(x.shape)
    sigma = np.zeros(x.shape)
    epsilon_r[x**2+y**2 <= radius**2] = epsilon_rd
    return epsilon_r, sigma


def compute_scattered_field(xm, ym, an, kb, theta, magnitude):
    """Summarize the method."""
    M, S, N = xm.size, theta.size, round((an.size-1)/2)
    E0 = magnitude
    es = np.zeros((M, S), dtype=complex)
    n = np.arange(-N, N+1)
    for s in range(S):
        xp, yp = rotate_axis(theta[s], xm, ym)
        rho, phi = cart2pol(xp, yp)
        for j in range(phi.size):
            es[j, s] = E0*np.sum(1j**(-n)*an*hankel2(n, kb*rho[j])
                                 * np.exp(1j*n*phi[j]))

    return es
