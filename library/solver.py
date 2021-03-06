"""Abstract Inverse Scattering Solver model.

This module provides the abstract class for implementation of any method
which solve the nonlinear inverse scattering problem. Therefore, this
class aims to compute the dielectric map and the total intern field.
"""

# Standard libraries
from abc import ABC, abstractmethod
import numpy as np
import copy as cp
import sys

# Developed libraries
import error
import configuration as cfg
import inputdata as ipt
import results as rst


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

    def __init__(self, configuration):
        """Create the object.

        Parameters
        ----------
            configuration : :class:`configuration.Configuration`
                An object of problem configuration.
        """
        self.configuration = configuration
        self.name = ''
        self.alias = ''
        self.execution_time = None


    @abstractmethod
    def solve(self, inputdata, print_info=True, print_file=sys.stdout):
        """Solve the inverse scattering problem.

        This is the model routine for any method implementation. The
        input may include other arguments. But the output must always be
        an object of :class:`results.Results`.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                An object of the class which defines an instance.

            print_info : bool
                A flag to indicate if information should be displayed or
                not on the screen.

        Returns
        -------
            :class:`results.Results`
        """
        if print_info:
            self._print_title(inputdata, print_file=print_file)

        return rst.Results(inputdata.name)

    def _print_title(self, instance, print_file=sys.stdout):
        """Print the title of the execution.

        Parameters
        ----------
            instance : :class:`results.Results`
        """
        print("==============================================================",
              file=print_file)
        print('Method: ' + self.name, file=print_file)
        if self.alias != '':
            print('Alias: ' + self.alias, file=print_file)
        print('Problem configuration: ' + self.configuration.name,
              file=print_file)
        print('Instance: ' + instance.name, file=print_file)
