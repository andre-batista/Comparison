"""A module for fixed parameters of problems.

Each problem addressed by a method may share common parameters, such as
the number of sources, measuments etc. This module contains a class for
storing information which will be shared by different scenarios. We call
this problem configuration and it determines information regarding
parameters of image and measurement domains.

The :class:`Configuration` implements the container. In addition,
auxiliary routines are provided in order to standardize some kind of
information which may be necessary for other modules.

The following class is defined

    :class:`Configuration`
        The container of all fixed information shared between scenarios.

The following routines are defined

    :func:`import_dict`
        A function for importing a dictionary with configuration
        attributes.
    :func:`compute_wavelength`
        Determine the wavelength for a given media and frequency.
    :func:`compute_frequency`
        Determine the frequency for a given media and wavelength.
    :func:`compute_wavenumber`
        Determine the wavenumber for a given media and frequency.
    :func:`get_angles`
        Return angles for a given number of points in S-domain.
    :func:`get_coordinates_sdomain`
        Return rectangular coordinates for a given number of points in
        S-domain.
    :func:`get_bounds`
        Return the standard bounds of the interval of D-domain for a
        given axis length.
    :func:`get_coordinates_ddomain`
        Return the standard meshgrid for D-domain.
    :func:`get_contrast_map`
        Return the contrast values for a given mesh of values of
        relative permittivity and conductivity.
    :func:`get_greenfunction`
        Return the Green function matrix for pulse-based discretization.
    :func:`solve_frequency`
        Estimate the frequency for a given complex media and wavelength.
    :func:`plot_ddomain_limits`
        Return a plot with D-domain bounds.
    :func:`plot_antennas`
        Return a plot with antennas points.
"""

import pickle
import numpy as np
from scipy import constants as ct
import matplotlib.pyplot as plt
from numba import jit
import error
import results as rst

# Constants for easy access of saved pickle file
NAME = 'name'
PATH = 'path'
NUMBER_MEASUREMENTS = 'NM'
NUMBER_SOURCES = 'NS'
OBSERVATION_RADIUS = 'Ro'
IMAGE_SIZE = 'image_size'
BACKGROUND_RELATIVE_PERMITTIVITY = 'epsilon_rb'
BACKGROUND_CONDUCTIVITY = 'sigma_b'
FREQUENCY = 'f'
BACKGROUND_WAVELENGTH = 'lambda_b'
BACKGROUND_WAVENUMBER = 'kb'
MAGNITUDE = 'E0'
PERFECT_DIELECTRIC_FLAG = 'perfect_dielectric'
GOOD_CONDUCTOR_FLAG = 'good_conductor'

# Other constants
TITLE_PROBLEM_CONFIGURATION = 'Problem Configuration'


class Configuration:
    """The container of all fixed information shared between scenarios.

    It stores all the information regarding domains sizes, background
    media and flags for indicating simplifications.

    Attributes
    ----------
        name
            A string naming the problem configuration.

        path
            The path where the object file was saved.

        NM
            Number of measurements

        NS
            Number of sources

        Ro
            Radius of observation (S-domain) [m]

        Lx
            Size of image domain (D-domain) in x-axis [m]

        epsilon_rb
            Background relative permittivity

        sigma_b
            Background conductivity [S/m]

        frequency
            Linear frequency of operation [Hz]

        lambda_b
            Background wavelength [m]

        kb
            Background wavenumber [1/m]

        perfect_dielectric
            Flag for assuming perfect dielectric objects

        good_conductor
            Flag for assuming good conductor objects

        E0
            Magnitude of incident field [V/m].
    """

    name = ''
    path = ''
    NM, NS = int(), int()
    Ro, Lx, Ly = float(), float(), float()
    epsilon_rb, sigma_b = float(), float()
    f, lambda_b, kb = float(), float(), float()
    perfect_dielectric, good_conductor = bool(), bool()
    E0 = float()

    def __init__(self, name=None, number_measurements=10, number_sources=10,
                 observation_radius=None, frequency=None, wavelength=None,
                 background_permittivity=1., background_conductivity=.0,
                 image_size=[1., 1.], wavelength_unit=True, magnitude=1.,
                 perfect_dielectric=False, good_conductor=False,
                 import_filename=None, import_filepath=''):
        """Build or import a configuration object.

        You may either give the parameters or import information from a
        save object. In addition, you must give either the frequency of
        operation or the wavelength.

        Parameters
        ----------
            name : string
                A string naming the problem configuration.

            number_measurements : int, default: 10
                Receivers in S-domain.

            number_sources : int, default: 10
                Sources in S-domain

            observation_radius : float, default: 1.1*sqrt(2)*max([Lx,Ly])
                Radius for circular array of sources and receivers at
                S-domain [m]

            frequency : float
                Linear frequency of operation [Hz]

            wavelength : float
                Background wavelength [m]

            background_permittivity : float, default: 1.
                Relative permittivity.

            background_conductivity : float, default: .0 [S/m]

            image_size : 2-tuple of int, default: (1., 1.)
                Side length of the image domain (D-domain). It may be
                given in meters or in wavelengths.

            wavelength_unit : boolean, default: True
                If `True`, the variable image_size will be interpreted
                as given in wavelegnths. Otherwise, it will be
                interpreted in meters.

            magnitude : float, default: 1.0
                Magnitude of the incident field.

            perfect_dielectric : boolean, default: False
                If `True`, it indicates the assumption of only perfect
                dielectric objects. Then, only the relative permittivity
                map will be recovered.

            good_conductor : boolean, default: False
                If `True`, it indicates the assumption of only good
                conductors objects. Then, only the conductivity map will
                be recovered.

            import_filename : string, default: None
                A string with the name of the pickle file containing the
                information of a previous Configuration object.

            import_filepath : string, default: ''
                A string containing the path to the object to be
                imported.

        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)

        else:

            if name is None:
                raise error.MissingInputError('Configuration.__init__()',
                                              'name')
            if wavelength is None and frequency is None:
                raise error.MissingInputError('Configuration.__init__()',
                                              'frequency or wavelength')
            elif wavelength is not None and frequency is not None:
                raise error.ExcessiveInputsError('Configuration.__init__()',
                                                 ['frequency', 'wavelength'])
            if perfect_dielectric and good_conductor:
                raise error.ExcessiveInputsError('Configuration.__init__()',
                                                 ['perfect_dielectric',
                                                  'good_conductor'])

            self.name = name
            self.NM = number_measurements
            self.NS = number_sources
            self.epsilon_rb = background_permittivity
            self.sigma_b = background_conductivity
            self.perfect_dielectric = perfect_dielectric
            self.good_conductor = good_conductor
            self.E0 = magnitude
            self.path = None

            if frequency is not None:
                self.f = frequency
                self.lambda_b = compute_wavelength(frequency, self.epsilon_rb,
                                                   sigma=self.sigma_b)
            else:
                self.lambda_b = wavelength
                self.f = compute_frequency(self.lambda_b, self.epsilon_rb)

            self.kb = compute_wavenumber(self.f, epsilon_r=self.epsilon_rb,
                                         sigma=self.sigma_b)

            if wavelength_unit:
                self.Lx = image_size[1]*self.lambda_b
                self.Ly = image_size[0]*self.lambda_b
            else:
                self.Lx = image_size[1]
                self.Ly = image_size[0]

            if observation_radius is None:
                self.Ro = 1.1*np.sqrt(2)*max([self.Lx, self.Ly])
            else:
                if wavelength_unit:
                    self.Ro = observation_radius*self.lambda_b
                else:
                    self.Ro = observation_radius

    def save(self, file_path=''):
        """Save the problem configuration within a pickle file.

        It will only be saved the attribute variables, not the object
        itself. If you want to load these variables, you may use the
        constant string variables for a more friendly usage.
        """
        self.path = file_path
        data = {
            NAME: self.name,
            PATH: self.path,
            NUMBER_MEASUREMENTS: self.NM,
            NUMBER_SOURCES: self.NS,
            OBSERVATION_RADIUS: self.Ro,
            IMAGE_SIZE: (self.Lx, self.Ly),
            BACKGROUND_RELATIVE_PERMITTIVITY: self.epsilon_rb,
            BACKGROUND_CONDUCTIVITY: self.sigma_b,
            FREQUENCY: self.f,
            BACKGROUND_WAVELENGTH: self.lambda_b,
            BACKGROUND_WAVENUMBER: self.kb,
            MAGNITUDE: self.E0,
            PERFECT_DIELECTRIC_FLAG: self.perfect_dielectric,
            GOOD_CONDUCTOR_FLAG: self.good_conductor
        }

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def draw(self, epsr=None, sig=None, file_path='', file_format='eps',
             show=False):
        """Draw domain, sources and probes.

        Plot an illustration of the problem configuration based on the
        values defined in the object. You may plot an empty problem or
        give a map.

        Parameters
        ----------
            epsr : :class:`numpy.ndarray`, default: None
                A matrix with the relative permittivity map that you may
                want to draw as example.

            sig : :class:`numpy.ndarray`, default: None
                A matrix with the conductivity [S/m] map that you may
                want to draw as example.

            show : boolean, default: False
                If `True`, a window with the figure will appear. If
                `False`, the figure will be saved as a file.

            file_path : string, default: ''
                A string indicating the path where you want to save the
                figure.

            file_format : string, default: 'eps'
                Format of the saved figure. It must be the same
                supported formats of `matplotlib.pyplot.savefig()`.

        """
        if epsr is None and sig is None:
            Nx, Ny = 100, 100
        elif epsr is not None:
            Ny, Nx = epsr.shape
        else:
            Ny, Nx = sig.shape

        dx, dy = self.Lx/Nx, self.Ly/Ny
        xmin, xmax = get_bounds(self.Lx)
        ymin, ymax = get_bounds(self.Ly)
        min_radius = np.sqrt(((xmax-xmin)/2)**2+((ymax-ymin)/2)**2)

        if self.Ro > min_radius:
            NXG = int((self.Ro-(xmax-xmin)/2)/dx)  # Cells in the gap
            NYG = int((self.Ro-(ymax-ymin)/2)/dy)  # Cells in the gap
            xmin, xmax = xmin-(NXG+5)*dx, xmax+(NXG+5)*dx
            ymin, ymax = ymin-(NYG+5)*dy, ymax+(NYG+5)*dy

        xm, ym = get_coordinates_sdomain(self.Ro, self.NM)
        xl, yl = get_coordinates_sdomain(self.Ro, self.NS)
        x, y = get_coordinates_ddomain(dx=dx, dy=dy, xmin=xmin, xmax=xmax,
                                       ymin=ymin, ymax=ymax)
        bounds = (xmin/self.lambda_b, xmax/self.lambda_b, ymin/self.lambda_b,
                  ymax/self.lambda_b)
        map_bounds = [-self.Lx/2/self.lambda_b, self.Lx/2/self.lambda_b,
                      -self.Ly/2/self.lambda_b, self.Ly/2/self.lambda_b]

        epsilon_r = self.epsilon_rb*np.ones(x.shape)
        sigma = self.sigma_b*np.ones(x.shape)

        if epsr is not None:
            epsilon_r[np.ix_(np.logical_and(y[:, 0] > -self.Ly/2,
                                            y[:, 0] < self.Ly/2),
                             np.logical_and(x[0, :] > -self.Lx/2,
                                            x[0, :] < self.Lx/2))] = epsr

        if sig is not None:
            sigma[np.ix_(np.logical_and(y[:, 0] > -self.Ly/2,
                                        y[:, 0] < self.Ly/2),
                         np.logical_and(x[0, :] > -self.Lx/2,
                                        x[0, :] < self.Lx/2))] = sig

        if self.perfect_dielectric or self.good_conductor:
            figure = plt.figure(figsize=rst.IMAGE_SIZE_SINGLE)
            axes = rst.get_single_figure_axes(figure)
            if self.perfect_dielectric:
                rst.add_image(axes, epsilon_r,
                              TITLE_PROBLEM_CONFIGURATION,
                              rst.COLORBAR_REL_PERMITTIVITY,
                              bounds=bounds)
            else:
                rst.add_image(axes, sigma,
                              TITLE_PROBLEM_CONFIGURATION,
                              rst.COLORBAR_CONDUCTIVITY, bounds=bounds)
            plot_ddomain_limits(axes, map_bounds)
            lg_m = plot_antennas(axes, xm, ym, self.lambda_b, 'r', 'Probe')
            lg_l = plot_antennas(axes, xl, yl, self.lambda_b, 'g', 'Source')
            plt.legend(handles=[lg_m, lg_l], loc='upper right')

        else:
            figure = plt.figure(figsize=rst.IMAGE_SIZE_1x2)
            rst.set_subplot_size(figure)

            axes = figure.add_subplot(1, 2, 1)
            rst.add_image(axes, epsilon_r,
                          rst.TITLE_REL_PERMITTIVITY,
                          rst.COLORBAR_REL_PERMITTIVITY, bounds=bounds)
            plot_ddomain_limits(axes, map_bounds)
            lg_m = plot_antennas(axes, xm, ym, self.lambda_b, 'r', 'Probe')
            lg_l = plot_antennas(axes, xl, yl, self.lambda_b, 'g', 'Source')
            plt.legend(handles=[lg_m, lg_l], loc='upper right')

            axes = figure.add_subplot(1, 2, 2)
            rst.add_image(axes, sigma,
                          rst.TITLE_CONDUCTIVITY,
                          rst.COLORBAR_CONDUCTIVITY, bounds=bounds)
            plot_ddomain_limits(axes, map_bounds)
            lg_m = plot_antennas(axes, xm, ym, self.lambda_b, 'r', 'Probe')
            lg_l = plot_antennas(axes, xl, yl, self.lambda_b, 'g', 'Source')
            plt.legend(handles=[lg_m, lg_l], loc='upper right')

        if show:
            plt.show()
        else:
            plt.savefig(file_path + self.name + '.' + file_format,
                        format=file_format)
            plt.close()

    def importdata(self, file_name, file_path=''):
        """Import data from a saved object."""
        data = import_dict(file_name, file_path)
        self.name = data[NAME]
        self.path = data[PATH]
        self.NM = data[NUMBER_MEASUREMENTS]
        self.NS = data[NUMBER_SOURCES]
        self.Ro = data[OBSERVATION_RADIUS]
        self.Lx, self.Ly = data[IMAGE_SIZE]
        self.epsilon_rb = data[BACKGROUND_RELATIVE_PERMITTIVITY]
        self.sigma_b = data[BACKGROUND_CONDUCTIVITY]
        self.f = data[FREQUENCY]
        self.lambda_b = data[BACKGROUND_WAVELENGTH]
        self.kb = data[BACKGROUND_WAVENUMBER]
        self.perfect_dielectric = data[PERFECT_DIELECTRIC_FLAG]
        self.good_conductor = data[GOOD_CONDUCTOR_FLAG]
        self.E0 = data[MAGNITUDE]

    def __str__(self):
        """Print object information."""
        message = 'Configuration name: ' + self.name
        if self.path is not None:
            message = message + '\nFilepath = ' + self.path
        message = message + '\nNumber of measurements: %d' % self.NM
        message = message + '\nNumber of sources: %d' % self.NS
        message = message + '\nObservation radius: %.3e [m]' % self.Ro
        message = (message + '\nImage domain size: %.3e x ' % self.Lx
                   + '%.3e [m]' % self.Ly)
        message = (message + '\nBackground relative permittivity: %.1f'
                   % self.epsilon_rb)
        message = (message + '\nBackground conductivity: %.3e [S/m]'
                   % self.sigma_b)
        message = message + '\nFrequency: %.3e [Hz]' % self.f
        message = message + '\nBackground wavelength: %.3e [m]' % self.lambda_b
        message = (message
                   + '\nBackground wavenumber: {:.3e} [1/m]'.format(self.kb))
        if self.perfect_dielectric:
            message = message + '\nAssumption: perfect dielectrics only'
        elif self.good_conductor:
            message = message + '\nAssumption: good conductors only'
        return message


def import_dict(file_name, file_path=''):
    """Import dictionary with configuration data."""
    with open(file_path + file_name, 'rb') as datafile:
        data = pickle.load(datafile)
    return data


def compute_wavelength(frequency, epsilon_r=1., mu_r=1., sigma=.0):
    """Compute wavelength.

    Parameters
    ----------
        frequency : float
            Linear frequency of operation [Hz].

        epsilon_r : float, default: 1.
            Relative permittivity of the medium.

        mu_r : float, default: 1.
            Relative permeability of the medium.

        sigma : float, default: .0
            Conductivity of the medium [S/m].

    Returns
    -------
        wavelength : float
            In meters.
    """
    omega = 2*np.pi*frequency
    return 1/np.real(frequency*np.sqrt(ct.mu_0*(epsilon_r*ct.epsilon_0
                                                - 1j*sigma/omega)))


def compute_frequency(wavelength, epsilon_r=1., mu_r=1., sigma=.0):
    """Compute frequency.

    Parameters
    ----------
        wavelength : float
            In meters.

        epsilon_r : float, default: 1.
            Relative permittivity of the medium.

        mu_r : float, default: 1.
            Relative permeability of the medium.

        sigma : float, default: .0
            Conductivity of the medium [S/m].

    Returns
    -------
        frequency : float
            In Hertz.
    """
    if sigma == 0.:
        return 1/np.sqrt(mu_r*ct.mu_0*epsilon_r*ct.epsilon_0)/wavelength
    else:
        return solve_frequency(wavelength, mu_r, epsilon_r, sigma)


def compute_wavenumber(frequency, epsilon_r=1., mu_r=1., sigma=0.):
    """Compute wavenumber.

    Parameters
    ----------
        frequency : float
            Linear frequency of operation [Hz].

        epsilon_r : float, default: 1.
            Relative permittivity of the medium.

        mu_r : float, default: 1.
            Relative permeability of the medium.

        sigma : float, default: .0
            Conductivity of the medium [S/m].

    Returns
    -------
        wavenumber : float
            In [1/m].
    """
    omega = 2*np.pi*frequency
    mu, epsilon = mu_r*ct.mu_0, epsilon_r*ct.epsilon_0
    return np.sqrt(-1j*omega*mu*sigma + omega**2*mu*epsilon)


def get_angles(n_samples):
    """Compute angles [rad] in a circular array of points equaly spaced.

    Parameter
    ---------
        n_samples : int
            Number of samples.

    Returns
    -------
        `numpy.ndarray`
    """
    return np.arange(0, 2*np.pi, 2*np.pi/n_samples)


def get_coordinates_sdomain(radius, n_samples, shift=0.):
    """Compute coordinates of points in a circular array equaly spaced.

    Parameters
    ----------
        radius : float
            Observation radius [m].

        n_samples : int
            Number of sampled points.

        shift : float, default: .0
            A radial shift in array position.

    Returns
    -------
        x, y
            Coordinates of the sampled points.
    """
    phi = get_angles(n_samples)
    return (radius*np.cos(phi + np.deg2rad(shift)),
            radius*np.sin(phi + np.deg2rad(shift)))


def get_bounds(length):
    """Compute the standard bound coordinates."""
    return -length/2, length/2


def get_coordinates_ddomain(configuration=None, resolution=None,
                            dx=None, dy=None, xmin=None, xmax=None, ymin=None,
                            ymax=None):
    """Return the meshgrid of the image domain.

    Examples
    --------
        The function must be called in only one of the two different
        ways:

        >>> get_coordinates_ddomain(configuration=Configuration()
                                    resolution=(100, 100))
        >>> get_coordinates_ddomain(dx=.1, dy=.1, xmin=-1., xmax=1.,
                                    ymin=-1., ymax=1.)

    Parameters
    ----------
        configuration : :class:`Configuration`
            A configuration object.

        resolution : 2-tuple
            Discretization size in y- and x-coordinates (this order).

        dx, dy : float
            Cell size.

        xmin, xmax : float
            Limits of the interval in x-axis.

        ymin, ymax : float
            Limits of the interval in y-axis.

    Notes
    -----
        The D-domain are such that (x,y) in [xmin,xmax] times [ymin,
        ymax]. The coordinates are positioned at the center of the
        cells.
    """
    function_name = 'get_coordinates_ddomain'
    if configuration is not None and resolution is None:
        raise error.MissingInputError(function_name, 'resolution')
    elif configuration is None and resolution is not None:
        raise error.MissingInputError(function_name, 'configuration')
    elif configuration is None and (dx is None or dy is None or xmin is None
                                    or xmax is None or ymin is None
                                    or ymax is None):
        inputs = []
        if dx is None:
            inputs.append('dx')
        if dy is None:
            inputs.append('dy')
        if xmin is None:
            inputs.append('xmin')
        if xmax is None:
            inputs.append('xmax')
        if ymin is None:
            inputs.append('ymin')
        if ymax is None:
            inputs.append('ymax')
        raise error.MissingInputError(function_name, inputs)

    if configuration is not None:
        NY, NX = resolution
        xmin, xmax = get_bounds(configuration.Lx)
        ymin, ymax = get_bounds(configuration.Ly)
        dx, dy = configuration.Lx/NX, configuration.Ly/NY

    return np.meshgrid(np.arange(xmin + .5*dx, xmax, dx),
                       np.arange(ymin + .5*dy, ymax, dy))


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
    return (epsilon_r/epsilon_rb - 1
            - 1j*(sigma-sigma_b)/(omega*epsilon_rb*ct.epsilon_0))


def get_relative_permittivity(chi, epsilon_rb):
    """Compute the relative permittivity for a given contrast value.

    Parameters
    ----------
        chi : float or :class:`numpy.ndarray`
            Contrast value or array.

        epsilon_rb : float
            Background relative permittivity

    """
    return epsilon_rb*(np.real(chi)+1)


def get_conductivity(chi, omega, epsilon_rb, sigma_b):
    """Compute the conductvity for a given contrast value.

    Parameters
    ----------
        chi : float or :class:`numpy.ndarray`
            Contrast value or array.

        omega : float
            Angular frequency [rad/s].

        epsilon_rb : float
            Background relative permittivity

        sigma_b : float
            Background conductivity
    """
    return sigma_b-np.imag(chi)*omega*epsilon_rb*ct.epsilon_0


@jit(nopython=True)
def solve_frequency(lambda_b, mu_r, epsilon_r, sigma):
    r"""Approximate the frequency.

    The routine estimates the corresponding frequency for a given
    combination of wavelength [1/m], relative permeability, relative
    permittivity and conductivity [S/m] values.

    Parameters
    ----------
        lambda_b : float
            Wavelength [m].

        mu_r : float
            Relative permeability.

        epsilon_r : float
            Relative permittivity

        sigma : float
            Conductivity [S/m]

    Notes
    -----
        The estimation is based on the Golden Section Method for
        unidimensional problems. The solution is the one which minimizes
        the following objective-function:

        .. math:: \phi(f) = \(\lambda_b - \frac{1}{fR\{\sqrt{\mu(\epsilon
                  - j\frac{\sigma}{2\pi f})}\}}
    """
    # Constants
    mu = mu_r*ct.mu_0
    epsilon = epsilon_r*ct.epsilon_0

    # Initial guess of frequency interval
    a, b = 1e0, 2e0

    # Error of the initial guess
    fa = (lambda_b-1/a/np.real(np.sqrt(mu*(epsilon-1j*sigma/(2*np.pi*a)))))**2
    fb = (lambda_b-1/b/np.real(np.sqrt(mu*(epsilon-1j*sigma/(2*np.pi*b)))))**2

    # Find interval
    evals = 2
    while fb < fa:
        a = b
        fa = fb
        b = 2*b
        fb = (lambda_b-1/b/np.real(np.sqrt(mu*(epsilon-1j*sigma/(2*np.pi*b))))
              )**2
        evals += 1
    if evals <= 3:
        a = 1e0
    else:
        a = a/2

    # Solve the frequency
    xa = b - .618*(b-a)
    xb = a + .618*(b-a)
    fa = (lambda_b-1/xa/np.real(np.sqrt(mu*(epsilon-1j*sigma/(2*np.pi*xa))))
          )**2
    fb = (lambda_b-1/xb/np.real(np.sqrt(mu*(epsilon-1j*sigma/(2*np.pi*xb))))
          )**2
    while (b-a) > 1e-3:
        if fa > fb:
            a = xa
            xa = xb
            xb = a + 0.618*(b-a)
            fa = fb
            fb = (lambda_b-1/xb/np.real(np.sqrt(mu*(epsilon-1j*sigma/(2*np.pi
                                                                      * xb))))
                  )**2
        else:
            b = xb
            xb = xa
            xa = b - 0.618*(b-a)
            fb = fa
            fa = (lambda_b-1/xa/np.real(np.sqrt(mu*(epsilon-1j*sigma/(2*np.pi
                                                                      * xa))))
                  )**2
    return (a+b)/2


def plot_ddomain_limits(axes, bounds):
    """Plot lines of D-domain limits.

    Parameters
    ----------
        axes : :class:`matplotlib.pyplot.Figure.axes.Axes`
            Axes object.

        bounds : 4-tuple
            x and y axis bounds.
    """
    axes.plot(np.array([bounds[0], bounds[0], bounds[1], bounds[1],
                        bounds[0]]),
              np.array([bounds[2], bounds[3], bounds[3], bounds[2],
                        bounds[2]]), 'k--')


def plot_antennas(axes, x, y, wavelength, color, label):
    """Plot antennas array.

    Parameters
    ----------
        axes : :class:`matplotlib.pyplot.Figure.axes.Axes`
            Axes object.

        x : :class:`numpy.ndarray`
            Array with the x-coordinates.

        y : :class:`numpy.ndarray`
            Array with the y-coordinates.

        wavelength : float

        color : {'r', 'g'}
            Color of the points.

        label : string
            Name of the antenna array.
    """
    return axes.plot(x/wavelength, y/wavelength, color + 'o', label=label)[0]
