"""Give a title for the module.

Define the module.
"""

from abc import ABC, abstractmethod
import copy as cp

import library_v2.configuration as cfg


class Inverse(ABC):
    """Summarize the class."""

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
        """Summarize the method."""
        self.configuration = configuration

    @abstractmethod
    def solve(self, inputdata):
        """Summarize the method."""
        pass

    @abstractmethod
    def __str__(self):
        """Print method information."""
        return "Inverse solver: " + self.name + "\n"
