"""The definition of the module.

A brief description.
"""

from abc import ABC, abstractmethod
import copy as cp
import numpy as np
import library_v2.configuration as cfg
import library_v2.inputdata as ipt


class ForwardSolver(ABC):
    """The model class of Foward Solver.

    A brief description.

    Attributes
    ----------
        lalala
    """

    def __init__(self, configuration, configuration_filepath=''):
        """Create a forward solver object."""
        self.configuration(configuration, configuration_filepath)

    @property
    def configuration(self):
        """Define the configuration attribute."""
        return cp.deepcopy(self._configuration)

    @configuration.setter
    def configuration(self, new_configuration):
        """Set configuration."""
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
            self.config = cp.deepcopy(new_configuration)

    @abstractmethod
    def solve(self, input):
        """Solve the forward problem."""
        pass

    def incident_field(self, resolution):
        """Brief definition of the function."""
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
