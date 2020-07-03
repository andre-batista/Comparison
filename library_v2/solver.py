"""Abstract Inverse Scattering Solver model.

This module provides the abstract class for implementation of any method
which solve the nonlinear inverse scattering problem. Therefore, this
class aims to compute the dielectric map and the total intern field.
"""

from abc import ABC, abstractmethod
import numpy as np
from numba import jit
import copy as cp

import library_v2.error as error
import library_v2.configuration as cfg
import library_v2.inputdata as ipt
import library_v2.results as rst


class Solver(ABC):
    """Abstract inverse solver class.

    This class defines the basic defintion of any implementation of
    inverse solver.

    Attributes
    ----------
        name : str
            The name of the solver.

        version : str
            The version of method.

        config : :class:`configuration.Configuration`
            An object of problem configuration.

        execution_time : float
            The amount of time for a single execution of the method.

    Notes
    -----
        The name of the method should be defined by default.
    """

    name = ''
    version = ''
    execution_time = float()

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

    def __init__(self, configuration):
        """Create the object.

        Parameters
        ----------
            configuration : :class:`configuration.Configuration`
                An object of problem configuration.
        """
        self.configuration = configuration

    @abstractmethod
    def solve(self, inputdata):
        """Solve the inverse scattering problem.

        This is the model routine for any method implementation. The
        input may include other arguments. But the output must always be
        an object of :class:`results.Results`.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                An object of the class which defines an instance.

        Returns
        -------
            :class:`results.Results`
        """
        return rst.Results(inputdata.name)

    def _print_title(self, instance):
        """Print the title of the execution.

        Parameters
        ----------
            instance : :class:`results.Results`
        """
        print("==============================================================")
        print('Method: ' + self.name)
        print('Problem configuration: ' + self.configuration.name)
        print('Instance: ' + instance.name)


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
