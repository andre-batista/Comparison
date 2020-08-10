"""Basic class structure of linear inverse methods.

This module defines the abstract class of linear inverse methods. Any
implementation must contain the attributes and methods defined in
:class:`Inverse`.
"""

# Standard libraries
from abc import ABC, abstractmethod
import copy as cp

# Developed libraries
import configuration as cfg
import results as rst


class Inverse(ABC):
    """Abstract class for linear inverse methods.

    This is an abstract class for any method implementation for solving
    the linear inverse scattering problem. Therefore, based on
    information of the scattered and total fields, the method should be
    able to recover the contrast information.

    Attributes
    ----------
        name : str
            The name of the method. Each implementation should define it
            within the class definition.

        configuration : :class:`configuration.Configuration`
            An object of problem configuration.
    """

    name = ''

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
        """Initialize the method object."""
        self.configuration = configuration

    @abstractmethod
    def solve(self, inputdata):
        """Solve a linear inverse scattering problem.

        This is the routine in which the method is implemented, i.e.,
        the method receives an instance of an inverse scattering problem
        containing the scattered and total fields and it returns a
        solution.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                An object of :class:`InputData` defining the instance of
                the problem. It must contains scattered and total fields
                data.

        Returns
        -------
            :class:`results.Result`
        """
        return rst.Results('')

    @abstractmethod
    def __str__(self):
        """Print method information."""
        return "Inverse solver: " + self.name + "\n"
