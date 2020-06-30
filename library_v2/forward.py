"""The abstract class for forward solvers.

This module provides the abstract class for any forward solver used in
this package, with the basic methods and attributes. Therefore,
different forward methods may be implemented and coupled to inverse
solvers without to specify which one is implemented.

The following routine is also provided:

`add_noise(x, delta)`
    Add noise to an array.
"""

from abc import ABC, abstractmethod
import copy as cp
import numpy as np
from numpy import random as rnd
import pickle

import library_v2.configuration as cfg
import library_v2.inputdata as ipt
import library_v2.error as error

# String constants for easier access dictionary from saved file
TOTAL_FIELD = 'et'
SCATTERED_FIELD = 'es'
INCIDENT_FIELD = 'ei'
RELATIVE_PERMITTIVITY_MAP = 'epsilon_r'
CONDUCTIVITY_MAP = 'sigma'
CONFIGURATION_FILENAME = 'config_filename'


class ForwardSolver(ABC):
    """The abstract class for Forward Solvers.

    This class provides the expected attributes and methods of any
    implementation of a forward solver.

    Attributes
    ----------
        name : str
            The name of the method. It should be defined within the
            implementation of the method.
        et, ei : :class:`numpy.ndarray`
            Matrices containing the total and incident field
            information. The rows are points in D-domain following 'C'
            order. The columns are the sources.

        es : :class:`numpy.ndarray`
            Matrix containing the scattered field information. The rows
            correspond to the measurement points and the columns
            correspond to the sources.

        epsilon_r, sigma : :class:`numpy.ndarray`
            Matrices representing the dielectric properties in each
            pixel of the image. *epsilon_r* stands for the relative
            permitivitty and *sigma* stands for the conductivity (S/m).
            `Obs.:` the rows correspond to the y-coordinates, and the
            columns, to the x-ones.

        configuration : :class:`configuration:Configuration`
            Configuration object.
    """

    name = ''
    et = np.array([], dtype=complex)
    ei = np.array([], dtype=complex)
    es = np.array([], dtype=complex)
    epsilon_r = np.array([])
    sigma = np.array([])

    @property
    def configuration(self):
        """Get routine of configuration attribute."""
        return cp.deepcopy(self._configuration)

    @configuration.setter
    def configuration(self, new_configuration):
        """Set configuration.

        Parameters
        ----------
            new_configuration : :class:`configuration:Configuration` or
                                string or 2-tuple
                The attribute is set in one of the three following ways:
                (i) an object of :class:`configuration:Configuration`;
                (ii) a string with the name of the file to be read with
                the configuration information; and (iii) a 2-tuple with
                the name and the path to the file to be read.

        Examples
        --------
        The following examples explain the ways to set the attribute:

        >>> solver.configuration = Configuration(...)
        >>> solver.configuration = 'myfile'
        >>> solver.configuration = ('myfile', './folder/')
        """
        if isinstance(new_configuration, tuple):
            file_name, file_path = new_configuration
            self._configuration = cfg.Configuration(
                import_filename=file_name,
                configuration_filepath=file_path
            )
        elif isinstance(new_configuration, str):
            self.configuration = cfg.Configuration(
                import_filename=new_configuration
            )
        else:
            self.configuration = cp.deepcopy(new_configuration)

    def __init__(self, configuration, configuration_filepath=''):
        """Create a forward solver object.

        Parameters
        ----------
            configuration : string or :class:`Configuration`:Configuration
                Either a configuration object or a string with the name
                of file in which the configuration is saved. In this
                case, the file path may also be provided.

            configuration_filepath : string, optional
                A string with the path to the configuration file (when
                the file name is provided).
        """
        self.configuration(configuration, configuration_filepath)

    @abstractmethod
    def solve(self, input):
        """Execute the method given a problem input.

        This is the basic model of the simulation routine.

        Parameters
        ----------
            input : :class:`inputdata:InputData`
                An object of InputData type which must contains the
                following attributes: `resolution`, `epsilon_r` and
                `sigma`.

        Returns
        -------
            es, et, ei : :class:`numpy.ndarray`
                Matrices with the computed scattered, total and incident
                fields, respectively.
        """
        if input.resolution is None:
            error.MissingAttributesError('InputData', 'resolution')
        if input.epsilon_r is None:
            error.MissingAttributesError('InputData', 'epsilon_r')
        if input.sigma is None:
            error.MissingAttributesError('InputData', 'sigma')

        return np.copy(self.es), np.copy(self.et), np.copy(self.ei)

    def incident_field(self, resolution):
        """Compute the incident field matrix.

        Given the configuration information stored in the object, it
        computes the incident field matrix considering plane waves in
        different from different angles.

        Parameters
        ----------
            resolution : 2-tuple
                The image size of D-domain in pixels.

        Returns
        -------
            ei : :class:`numpy.ndarray`
                Incident field matrix. The rows correspond to the points
                in the image following `C`-order and the columns
                corresponds to the sources.
        """
        NX, NY = resolution
        phi = cfg.get_angles(self.config.NS)
        xmin, xmax = cfg.get_bounds(self.config.Lx)
        ymin, ymax = cfg.get_bounds(self.config.Ly)
        dx, dy = self.config.Lx/NX, self.config.Ly/NY
        x, y = cfg.get_domain_coordinates(dx, dy, xmin, xmax, ymin,
                                          ymax)
        kb = self.config.kb
        E0 = self.config.E0

        if isinstance(kb, float):
            ei = E0*np.exp(-1j*kb*(x.reshape((-1, 1))
                                   @ np.cos(phi.reshape((1, -1)))
                                   + y.reshape((-1, 1))
                                   @ np.sin(phi.reshape((1, -1)))))
        else:
            ei = np.zeros((NX*NY, self.config.NS, kb.size), dtype=complex)
            for f in range(kb.size):
                ei[:, :, f] = E0*np.exp(-1j*kb[f]*(x.reshape((-1, 1))
                                                   @ np.cos(phi.reshape((1,
                                                                         -1)))
                                                   + y.reshape((-1, 1))
                                                   @ np.sin(phi.reshape((1,
                                                                         -1))))
                                        )
        return ei

    def save(self, file_name, file_path=''):
        """Save simulation data."""
        data = {
            CONFIGURATION_FILENAME: self.config.name,
            TOTAL_FIELD: self.et,
            INCIDENT_FIELD: self.ei,
            SCATTERED_FIELD: self.es,
            RELATIVE_PERMITTIVITY_MAP: self.epsilon_r,
            CONDUCTIVITY_MAP: self.sigma
        }

        with open(file_path + file_name, 'wb') as datafile:
            pickle.dump(data, datafile)


def add_noise(x, delta):
    r"""Add noise to data.

    Parameters
    ----------
        x : array_like
            Data to receive noise.
        delta : float
            Level of noise.

    Notes
    -----
        The noise level follows the following rule:

        .. math:: ||x-x^\delta|| \le \delta
    """
    originalshape = x.shape
    N = x.size
    xd = (rnd.normal(loc=np.real(x.reshape(-1)),
                     scale=delta/1.9/np.sqrt(N),
                     size=x.size)
          + 1j*rnd.normal(loc=np.imag(x.reshape(-1)),
                          scale=delta/1.9/np.sqrt(N),
                          size=x.size))
    return np.reshape(xd, originalshape)
