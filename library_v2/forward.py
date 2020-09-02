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

import configuration as cfg
import inputdata as ipt
import error as error

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
                import_filepath=file_path
            )
        elif isinstance(new_configuration, str):
            self._configuration = cfg.Configuration(
                import_filename=new_configuration
            )
        else:
            self._configuration = cp.deepcopy(new_configuration)

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
        if isinstance(configuration, str):
            self.configuration = (configuration, configuration_filepath)
        else:
            self.configuration = configuration
        self.name = None
        self.et = None
        self.ei = None
        self.es = None
        self.epsilon_r = None
        self.sigma = None


    @abstractmethod
    def solve(self, inputdata, noise=None, PRINT_INFO=False,
              SAVE_INTERN_FIELD=True):
        """Execute the method given a problem input.

        This is the basic model of the simulation routine.

        Parameters
        ----------
            input : :class:`inputdata:InputData`
                An object of InputData type which must contains the
                `resolution` attribute and either `epsilon_r` or
                `sigma` or both.

        Returns
        -------
            es, et, ei : :class:`numpy.ndarray`
                Matrices with the computed scattered, total and incident
                fields, respectively.
        """
        if inputdata.resolution is None:
            raise error.MissingAttributesError('InputData', 'resolution')
        if inputdata.epsilon_r is None and inputdata.sigma is None:
            raise error.MissingAttributesError('InputData',
                                               'epsilon_r or sigma')
        if inputdata.epsilon_r is None:
            epsilon_r = (self.configuration.epsilon_rb
                         * np.ones(inputdata.resolution))
        else:
            epsilon_r = np.copy(inputdata.epsilon_r)

        if inputdata.sigma is None:
            sigma = self.configuration.sigma_b*np.ones(inputdata.resolution)
        else:
            sigma = np.copy(inputdata.sigma)

        if SAVE_INTERN_FIELD:
            inputdata.total_field_resolution = inputdata.resolution

        return epsilon_r, sigma

    @abstractmethod
    def incident_field(self, resolution):
        """Return the incident field for a given resolution."""
        return np.zeros((int, int), dtype=complex)

    def save(self, file_name, file_path=''):
        """Save simulation data."""
        data = {
            CONFIGURATION_FILENAME: self.configuration.name,
            TOTAL_FIELD: self.et,
            INCIDENT_FIELD: self.ei,
            SCATTERED_FIELD: self.es,
            RELATIVE_PERMITTIVITY_MAP: self.epsilon_r,
            CONDUCTIVITY_MAP: self.sigma
        }

        with open(file_path + file_name, 'wb') as datafile:
            pickle.dump(data, datafile)

    @abstractmethod
    def __str__(self):
        """Print information of the method object."""
        return "Foward Solver: " + self.name + "\n"


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
