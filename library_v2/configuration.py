"""Class and functions to define the configuration of problem."""
import pickle
import numpy as np
import scipy.constants as ct
import scipy.special as spc
import matplotlib.pyplot as plt
import library_v2.error as error

# Constants for easy access of saved pickle file
NAME = 'name'
NUMBER_MEASUREMENTS = 'NM'
NUMBER_SOURCES = 'NS'
OBSERVATION_RADIUS = 'Ro'
IMAGE_SIZE = 'image_size'
BACKGROUND_RELATIVE_PERMITTIVITY = 'epsilon_rb'
BACKGROUND_CONDUCTIVITY = 'sigma_b'
FREQUENCY = 'f'
BACKGROUND_WAVELENGTH = 'lambda_b'
BACKGROUND_WAVENUMBER = 'kb'
PERFECT_DIELECTRIC_FLAG = 'perfect_dielectric'
GOOD_CONDUCTOR_FLAG = 'good_conductor'


class Configuration:
    """Problem configuration class.

    Attributes:
        name -- a string naming the problem configuration
        NM -- number of measurements
        NS -- number of sources
        Ro -- radius of observation (S-domain) [m]
        Lx -- size of image domain (D-domain) in x-axis [m]
        epsilon_rb -- background relative permittivity
        sigma_b -- background conductivity [S/m]
        frequency -- linear frequency of operation [Hz]
        lambda_b -- background wavelength [m]
        kb -- background wavenumber [1/m]
        perfect_dielectric -- flag for assuming perfect dielectric
            objects
        good_conductor -- flag for assuming good conductor objects
    """

    name = ''
    NM, NS = int(), int()
    Ro, Lx, Ly = float(), float(), float()
    epsilon_rb, sigma_b = float(), float()
    f, lambda_b, kb = float(), float(), float()
    perfect_dielectric, good_conductor = bool(), bool()

    def __init__(self, name=None, number_measurements=10, number_sources=10,
                 observation_radius=None, frequency=None, wavelength=None,
                 background_permittivity=1., background_conductivity=.0,
                 image_size=[1., 1.], wavelength_unit=True,
                 perfect_dielectric=False, good_conductor=False,
                 import_filename=None, import_filepath=''):
        """Build or import a configuration object.

        Keyword arguments:
            name -- a string naming the problem configuration
            number_measurements -- receivers in S-domain (default 10)
            number_sources -- sources in S-domain (default 10)
            observation_radius -- radius for circular array of sources
                and receivers at S-domain [m]
                (default 1.1*sqrt(2)*max([Lx,Ly]))
            frequency -- linear frequency of operation [Hz]
            wavelength -- background wavelength [m]
            background_permittivity -- Relative permittivity (default 1.0)
            background_conductivity -- [S/m] (default 0.0)
            image_size -- a tuple with the side sizes of image domain
                (D-domain). It may be given in meters or in wavelength
                proportion (default (1.,1.))
            wavelength_unit -- a flag to indicate if image_size is given
                in wavelength or not (default True)
            perfect_dielectric -- a flag to indicate the assumption of
                only perfect dielectric objects (default False)
            good_conductor -- a flag to indicate the assumption of
                only good conductors objects (default False)
            import_filename -- a string with the name of object to be
                imported.
            import_filepath -- a string containing the path to the
                object to be imported. (default '')

        Obs.: you must give either the name and path to the imported
        object or give the parameters to create one.
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

            self.name = name
            self.NM = number_measurements
            self.NS = number_sources
            self.Ro = observation_radius
            self.epsilon_rb = background_permittivity
            self.sigma_b = background_conductivity
            self.perfect_dielectric = perfect_dielectric
            self.good_conductor = good_conductor

            if frequency is not None:
                self.f = frequency
                self.lambda_b = compute_wavelength(frequency, self.epsilon_rb)
            else:
                self.lambda_b = wavelength
                self.f = compute_frequency(self.lambda_b, self.epsilon_rb)

            self.kb = compute_wavenumber(self.f, epsilon_r=self.epsilon_rb,
                                         sigma=self.sigma_b)

            if wavelength_unit:
                self.Lx = image_size[0]*self.lambda_b
                self.Ly = image_size[1]*self.lambda_b

    def save(self, file_path=''):
        """Save the problem configuration within a pickle file."""
        data = {
            NAME: self.name,
            NUMBER_MEASUREMENTS: self.NM,
            NUMBER_SOURCES: self.NS,
            OBSERVATION_RADIUS: self.Ro,
            IMAGE_SIZE: (self.Lx, self.Ly),
            BACKGROUND_RELATIVE_PERMITTIVITY: self.epsilon_rb,
            BACKGROUND_CONDUCTIVITY: self.sigma_b,
            FREQUENCY: self.f,
            BACKGROUND_WAVELENGTH: self.lambda_b,
            BACKGROUND_WAVENUMBER: self.kb,
            PERFECT_DIELECTRIC_FLAG: self.perfect_dielectric,
            GOOD_CONDUCTOR_FLAG: self.good_conductor
        }

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def draw(self, epsr=None, sig=None, file_path='', file_format='eps',
             show=False):
        """Draw domain, sources and probes."""
        if epsr is None and sig is None:
            Nx, Ny = 100, 100
        elif epsr is not None:
            Ny, Nx = epsr.shape
        else:
            Ny, Nx = sig.shape

        dx, dy = self.Lx/Nx, self.Ly/Ny
        min_radius = np.sqrt((self.Lx/2)**2+(self.Ly/2)**2)

        if self.Ro > min_radius:
            xmin, xmax = -1.05*self.Ro, 1.05*self.Ro
            ymin, ymax = -1.05*self.Ro, 1.05*self.Ro
        else:
            xmin, xmax = -self.Lx/2, self.Lx/2
            ymin, ymax = -self.Ly/2, self.Ly/2

        xm, ym = get_coordinates_sdomain(self.Ro, self.NM)
        xl, yl = get_coordinates_sdomain(self.Ro, self.NS)
        x, y = get_coordinates_ddomain(dx, dy, xmin, xmax, ymin, ymax)

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

        if self.perfect_dielectric:

            if isinstance(self.f, float):

                im1 = plt.imshow(epsilon_r, extent=[xmin/self.lambda_b,
                                                    xmax/self.lambda_b,
                                                    ymin/self.lambda_b,
                                                    ymax/self.lambda_b],
                                 origin='lower')

                plt.plot(np.array([-self.Lx/2/self.lambda_b,
                                   -self.Lx/2/self.lambda_b,
                                   self.Lx/2/self.lambda_b,
                                   self.Lx/2/self.lambda_b,
                                   -self.Lx/2/self.lambda_b]),
                         np.array([-self.Ly/2/self.lambda_b,
                                   self.Ly/2/self.lambda_b,
                                   self.Ly/2/self.lambda_b,
                                   -self.Ly/2/self.lambda_b,
                                   -self.Ly/2/self.lambda_b]), 'k--')

                lg_m, = plt.plot(xm/self.lambda_b, ym/self.lambda_b, 'ro',
                                 label='Probe')
                lg_l, = plt.plot(xl/self.lambda_b, yl/self.lambda_b, 'go',
                                 label='Source')

                plt.xlabel(r'x [$\lambda_b$]')
                plt.ylabel(r'y [$\lambda_b$]')

            else:

                im1 = plt.imshow(epsilon_r, extent=[xmin, xmax, ymin, ymax],
                                 origin='lower')

                plt.plot(np.array([-self.Lx/2, -self.Lx/2,
                                   self.Lx/2, self.Lx/2,
                                   -self.Lx/2]),
                         np.array([-self.Ly/2, self.Ly/2,
                                   self.Ly/2, -self.Ly/2,
                                   -self.Ly/2]), 'k--')

                lg_m, = plt.plot(xm, ym, 'ro', label='Probe')
                lg_l, = plt.plot(xl, yl, 'go', label='Source')

                plt.set_xlabel('x [m]')
                plt.set_ylabel('y [m]')

            plt.legend(handles=[lg_m, lg_l], loc='upper right')
            cbar = plt.colorbar()
            cbar.set_label(r'$\epsilon_r$')
            plt.title('Relative Permittivity')

        elif self.good_conductor:

            if isinstance(self.f, float):

                im1 = plt.imshow(sigma, extent=[xmin/self.lambda_b,
                                                xmax/self.lambda_b,
                                                ymin/self.lambda_b,
                                                ymax/self.lambda_b],
                                 origin='lower')

                plt.plot(np.array([-self.Lx/2/self.lambda_b,
                                   -self.Lx/2/self.lambda_b,
                                   self.Lx/2/self.lambda_b,
                                   self.Lx/2/self.lambda_b,
                                   -self.Lx/2/self.lambda_b]),
                         np.array([-self.Ly/2/self.lambda_b,
                                   self.Ly/2/self.lambda_b,
                                   self.Ly/2/self.lambda_b,
                                   -self.Ly/2/self.lambda_b,
                                   -self.Ly/2/self.lambda_b]), 'k--')

                lg_m, = plt.plot(xm/self.lambda_b, ym/self.lambda_b, 'ro',
                                 label='Probe')
                lg_l, = plt.plot(xl/self.lambda_b, yl/self.lambda_b, 'go',
                                 label='Source')

                plt.xlabel(r'x [$\lambda_b$]')
                plt.ylabel(r'y [$\lambda_b$]')

            else:

                im1 = plt.imshow(sigma, extent=[xmin, xmax, ymin, ymax])

                plt.plot(np.array([-self.Lx/2, -self.Lx/2,
                                   self.Lx/2, self.Lx/2,
                                   -self.Lx/2]),
                         np.array([-self.Ly/2, self.Ly/2,
                                   self.Ly/2, -self.Ly/2,
                                   -self.Ly/2]), 'k--')

                lg_m, = plt.plot(xm, ym, 'ro', label='Probe')
                lg_l, = plt.plot(xl, yl, 'go', label='Source')

                plt.xlabel('x [m]')
                plt.ylabel('y [m]')

            cbar = plt.colorbar()
            cbar.set_label(r'$\sigma$ [S/m]')
            plt.title('Conductivity')
            plt.legend(handles=[lg_m, lg_l], loc='upper right')

        else:

            fig = plt.figure(figsize=(10, 4))
            fig.subplots_adjust(left=.125, bottom=.1, right=.9, top=.9,
                                wspace=.5, hspace=.2)

            ax = fig.add_subplot(1, 2, 1)

            if isinstance(self.f, float):

                im1 = ax.imshow(epsilon_r, extent=[xmin/self.lambda_b,
                                                   xmax/self.lambda_b,
                                                   ymin/self.lambda_b,
                                                   ymax/self.lambda_b],
                                origin='lower')

                ax.plot(np.array([-self.Lx/2/self.lambda_b,
                                  -self.Lx/2/self.lambda_b,
                                  self.Lx/2/self.lambda_b,
                                  self.Lx/2/self.lambda_b,
                                  -self.Lx/2/self.lambda_b]),
                        np.array([-self.Ly/2/self.lambda_b,
                                  self.Ly/2/self.lambda_b,
                                  self.Ly/2/self.lambda_b,
                                  -self.Ly/2/self.lambda_b,
                                  -self.Ly/2/self.lambda_b]), 'k--')

                lg_m, = ax.plot(xm/self.lambda_b, ym/self.lambda_b, 'ro',
                                label='Probe')
                lg_l, = ax.plot(xl/self.lambda_b, yl/self.lambda_b, 'go',
                                label='Source')

                ax.set_xlabel(r'x [$\lambda_b$]')
                ax.set_ylabel(r'y [$\lambda_b$]')

            else:

                im1 = ax.imshow(epsilon_r, extent=[xmin, xmax, ymin, ymax],
                                origin='lower')

                ax.plot(np.array([-self.Lx/2, -self.Lx/2,
                                  self.Lx/2, self.Lx/2,
                                  -self.Lx/2]),
                        np.array([-self.Ly/2, self.Ly/2,
                                  self.Ly/2, -self.Ly/2,
                                  -self.Ly/2]), 'k--')

                lg_m, = ax.plot(xm, ym, 'ro', label='Probe')
                lg_l, = ax.plot(xl, yl, 'go', label='Source')

                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')

            plt.legend(handles=[lg_m, lg_l], loc='upper right')
            cbar = fig.colorbar(im1, fraction=0.046, pad=0.04)
            cbar.set_label(r'$\epsilon_r$')
            ax.set_title('Relative Permittivity')

            ax = fig.add_subplot(1, 2, 2)

            if isinstance(self.f, float):

                im1 = ax.imshow(sigma, extent=[xmin/self.lambda_b,
                                               xmax/self.lambda_b,
                                               ymin/self.lambda_b,
                                               ymax/self.lambda_b],
                                origin='lower')

                ax.plot(np.array([-self.Lx/2/self.lambda_b,
                                  -self.Lx/2/self.lambda_b,
                                  self.Lx/2/self.lambda_b,
                                  self.Lx/2/self.lambda_b,
                                  -self.Lx/2/self.lambda_b]),
                        np.array([-self.Ly/2/self.lambda_b,
                                  self.Ly/2/self.lambda_b,
                                  self.Ly/2/self.lambda_b,
                                  -self.Ly/2/self.lambda_b,
                                  -self.Ly/2/self.lambda_b]), 'k--')

                lg_m, = ax.plot(xm/self.lambda_b, ym/self.lambda_b, 'ro',
                                label='Probe')
                lg_l, = ax.plot(xl/self.lambda_b, yl/self.lambda_b, 'go',
                                label='Source')

                ax.set_xlabel(r'x [$\lambda_b$]')
                ax.set_ylabel(r'y [$\lambda_b$]')

            else:

                im1 = ax.imshow(sigma, extent=[xmin, xmax, ymin, ymax],
                                origin='lower')

                ax.plot(np.array([-self.Lx/2, -self.Lx/2,
                                  self.Lx/2, self.Lx/2,
                                  -self.Lx/2]),
                        np.array([-self.Ly/2, self.Ly/2,
                                  self.Ly/2, -self.Ly/2,
                                  -self.Ly/2]), 'k--')

                lg_m, = ax.plot(xm, ym, 'ro', label='Probe')
                lg_l, = ax.plot(xl, yl, 'go', label='Source')

                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')

            cbar = fig.colorbar(im1, fraction=0.046, pad=0.04)
            cbar.set_label(r'$\sigma$ [S/m]')
            ax.set_title('Conductivity')
            plt.legend(handles=[lg_m, lg_l], loc='upper right')

        if show:
            plt.show()
        else:
            plt.savefig(file_path + self.name + '.' + file_format,
                        format=file_format)
            plt.close()

    def importdata(self, file_name, file_path=''):
        """Import data from a saved object."""
        with open(file_path + file_name, 'rb') as datafile:
            data = pickle.load(datafile)
        self.name = data[NAME]
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


def compute_wavelength(frequency, epsilon_r=1., mu_r=1.):
    """Compute wavelength [m]."""
    return 1/np.sqrt(epsilon_r*ct.epsilon_0*mu_r*ct.mu_0)/frequency


def compute_frequency(wavelength, epsilon_r=1., mu_r=1.):
    """Compute frequency [Hz]."""
    return 1/np.sqrt(epsilon_r*ct.epsilon_0*mu_r*ct.mu_0)/wavelength


def compute_wavenumber(frequency, epsilon_r=1., mu_r=1., sigma=0.):
    """Compute real part of wavenumber."""
    omega, mu, epsilon = 2*np.pi*frequency
    mu, epsilon = mu_r*ct.mu_0, epsilon_r*ct.epsilon_0
    return np.sqrt(-1j*omega*mu*sigma + omega**2*mu*epsilon)


def get_angles(n_samples):
    """Compute angles [rad] in a circular array of points equaly spaced."""
    return np.arange(0, 2*np.pi, 2*np.pi/n_samples)


def get_coordinates_sdomain(radius, n_samples, shift=0.):
    """Compute coordinates of points in a circular array equaly spaced."""
    phi = get_angles(n_samples)
    return (radius*np.cos(phi + np.deg2rad(shift)),
            radius*np.sin(phi + np.deg2rad(shift)))


def get_bounds(length):
    """Compute the standard bound coordinates."""
    return -length/2, length/2


def get_coordinates_ddomain(dx, dy, xmin, xmax, ymin, ymax):
    """Return the meshgrid of the image domain."""
    return np.meshgrid(np.arange(xmin + .5*dx, xmax + .5*dx, dx),
                       np.arange(ymin + .5*dy, ymax + .5*dy, dy))


def get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b, omega):
    """Compute the contrast function for a given image."""
    if isinstance(omega, float):
        return ((epsilon_r - 1j*sigma/omega/ct.epsilon_0)
                / (epsilon_rb - 1j*sigma_b/omega/ct.epsilon_0) - 1)

    else:
        Xr = np.zeros((epsilon_r.shape[0], epsilon_r.shape[1], omega.size),
                      dtype=complex)
        for f in range(omega.size):
            Xr[:, :, f] = ((epsilon_r - 1j*sigma/omega[f]/ct.epsilon_0)
                           / (epsilon_rb - 1j*sigma_b/omega[f]/ct.epsilon_0)
                           - 1)
        return Xr


def get_greenfunction(xm, ym, x, y, kb):
    """Compute the Green function."""
    Ny, Nx = x.shape
    M = xm.size
    dx, dy = x[0, 1]-x[0, 0], y[1, 0]-y[0, 0]
    an = np.sqrt(dx*dy/np.pi)  # radius of the equivalent circle

    if isinstance(kb, float):
        MONO_FREQUENCY = True
    else:
        MONO_FREQUENCY = False

    xg = np.tile(xm.reshape((-1, 1)), (1, Nx*Ny))
    yg = np.tile(ym.reshape((-1, 1)), (1, Nx*Ny))
    R = np.sqrt((xg-np.tile(np.reshape(x, (Nx*Ny, 1)).T, (M, 1)))**2
                + (yg-np.tile(np.reshape(y, (Nx*Ny, 1)).T, (M, 1)))**2)

    if MONO_FREQUENCY:
        G = (-1j*kb*np.pi*an/2*spc.jv(1, kb*an) * spc.hankel2(0, kb*R))
        G[R == 0] = 1j/2*(np.pi*kb*an*spc.hankel2(1, kb*an)-2j)

    else:
        G = np.zeros((M, Nx*Ny, kb.size), dtype=complex)
        for f in range(kb.size):
            aux = (-1j*kb[f]*np.pi*an/2*spc.jv(1, kb[f]*an)
                   * spc.hankel2(0, kb[f]*R))
            aux[R == 0] = 1j/2*(np.pi*kb[f]*an*spc.hankel2(1, kb[f]*an)-2j)
            G[:, :, f] = aux

    return G
