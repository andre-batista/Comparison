"""Class and functions to define input data of class Solver."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import error

# Constants for easier access to fields of the saved pickle file
NAME = 'name'
CONFIGURATION_FILENAME = 'configuration_filename'
RESOLUTION = 'resolution'
SCATTERED_FIELD = 'es'
TOTAL_FIELD = 'total_field'
INCIDENT_FIELD = 'incident_field'
RELATIVE_PERMITTIVITY_MAP = 'epsilon_r'
CONDUCTIVITY_MAP = 'sigma'
NOISE = 'noise'


class InputData:
    """Class for storing information required for solvers.

    Attributes:
        name -- a string naming the problem.
        configuration_filename -- a string for referencing the problem
            configuration.
        resolution -- a tuple with the size, in pixels, of the recovered
            image.
        scattered_field -- matrix containing the scattered field
            information at S-domain.
        total_field -- matrix containing the total field information
                at D-domain.
        incident_field -- matrix containing the incident field
                information at D-domain.
        relative_permittivity_map -- matrix with the discretized image
                of the relative permittivity map.
        conductivity_map -- matrix with the discretized image of the
                conductivity map.
        noise -- noise level of scattered field data.
    """

    name = ''
    configuration_filename = ''
    resolution = (int(), int())
    es = np.array((int(), int()), dtype=complex)
    et = np.array((int(), int()), dtype=complex)
    ei = np.array((int(), int()), dtype=complex)
    epsilon_r = np.array((int(), int()), dtype=complex)
    sigma = np.array((int(), int()), dtype=complex)
    noise = float()

    def __init__(self, name=None, configuration_filename=None, resolution=None,
                 scattered_field=None, total_field=None, incident_field=None,
                 relative_permittivity_map=None, conductivity_map=None,
                 noise=None, import_filename=None, import_filepath=''):
        """Build or import an object.

        Keyword arguments:
            name -- a string naming the problem.
            configuration_filename -- a string with the name of the
                problem configuration file.
            resolution -- the size, in pixels, of the final image.
            scattered_field -- matrix containing the scattered field
                information at S-domain. (optional)
            total_field -- matrix containing the total field information
                at D-domain. (optional)
            incident_field -- matrix containing the incident field
                information at D-domain. (optional)
            relative_permittivity_map -- matrix with the discretized
                image of the relative permittivity map. (optional)
            conductivity_map -- matrix with the discretized image of the
                conductivity map. (optional)
            noise -- noise level of scattered field data.
            import_filename -- a string with the name of the saved file.
            import_filepath -- a string with the path to the saved file.

        Obs.: you must give either the import file name and path or the
        required variables.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)

        else:

            if name is None:
                raise error.MissingInputError('InputData.__init__()', 'name')
            if configuration_filename is None:
                raise error.MissingInputError('InputData.__init__()',
                                              'configuration_filename')
            if resolution is None:
                raise error.MissingInputError('InputData.__init__()',
                                              'resolution')

            self.name = name
            self.configuration_filename = configuration_filename
            self.resolution = resolution

            if scattered_field is not None:
                self.es = np.copy(scattered_field)
            else:
                self.es = None

            if total_field is not None:
                self.et = np.copy(total_field)
            else:
                self.et = None

            if incident_field is not None:
                self.ei = np.copy(incident_field)
            else:
                self.ei = None

            if relative_permittivity_map is not None:
                self.epsilon_r = relative_permittivity_map
            else:
                self.epsilon_r = None

            if conductivity_map is not None:
                self.sigma = conductivity_map
            else:
                self.sigma = None

            if noise is not None:
                self.noise = noise
            else:
                self.noise = None

    def save(self, file_path=''):
        """Save object information."""
        data = {
            NAME: self.name,
            CONFIGURATION_FILENAME: self.configuration_filename,
            RESOLUTION: self.resolution,
            SCATTERED_FIELD: self.es,
            TOTAL_FIELD: self.et,
            INCIDENT_FIELD: self.ei,
            NOISE: self.noise,
            RELATIVE_PERMITTIVITY_MAP: self.epsilon_r,
            CONDUCTIVITY_MAP: self.sigma
        }

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import data from a saved object."""
        with open(file_path + file_name, 'rb') as datafile:
            data = pickle.load(datafile)
        self.name = data[NAME]
        self.configuration_filename = data[CONFIGURATION_FILENAME]
        self.resolution = data[RESOLUTION]
        self.et = data[TOTAL_FIELD]
        self.es = data[SCATTERED_FIELD]
        self.ei = data[INCIDENT_FIELD]
        self.epsilon_r = data[RELATIVE_PERMITTIVITY_MAP]
        self.sigma = data[CONDUCTIVITY_MAP]
        self.noise = data[NOISE]

    def draw(self, figure_title=None, file_path='', file_format='eps',
             show=False):
        """Draw figure with the relative permittivity/conductivity map."""
        if self.epsilon_r is not None and self.sigma is None:
            plt.imshow(self.epsilon_r, origin='lower')
            plt.xlabel('x [Pixels]')
            plt.ylabel('y [Pixels]')
            if figure_title is None:
                plt.title('Relative Permittivity Map')
            else:
                plt.title(figure_title)
            cbar = plt.colorbar()
            cbar.set_label(r'$\epsilon_r$')

        elif self.epsilon_r is None and self.sigma is not None:
            plt.imshow(self.sigma, origin='lower')
            plt.xlabel('x [Pixels]')
            plt.ylabel('y [Pixels]')
            if figure_title is None:
                plt.title('Conductivity Map')
            else:
                plt.title(figure_title)
            cbar = plt.colorbar()
            cbar.set_label(r'$\sigma$ [S/m]')

        elif self.epsilon_r is not None and self.sigma is not None:
            fig = plt.figure(figsize=(10, 4))
            fig.subplots_adjust(left=.125, bottom=.1, right=.9, top=.9,
                                wspace=.5, hspace=.2)
            ax = fig.add_subplot(1, 2, 1)
            im1 = ax.imshow(self.epsilon_r, origin='lower')
            ax.set_xlabel('x [Pixels]')
            ax.set_ylabel('y [Pixels]')
            cbar = fig.colorbar(im1, fraction=0.046, pad=0.04)
            cbar.set_label(r'$\epsilon_r$')
            ax.set_title('Relative Permittivity')
            ax = fig.add_subplot(1, 2, 2)
            im2 = ax.imshow(self.sigma, origin='lower')
            ax.set_xlabel('x [Pixels]')
            ax.set_ylabel('y [Pixels]')
            cbar = fig.colorbar(im2, fraction=0.046, pad=0.04)
            cbar.set_label(r'$\sigma$')
            ax.set_title('Conductivity')

        else:
            raise error.MissingInputError('InputData.draw()',
                                          'relative_permittivity or '
                                          + 'conductivity')

        if self.epsilon_r is not None or self.sigma is not None:
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '.' + file_format,
                            format=file_format)
                plt.close()

    def set_scattered_field(self, scattered_field, noise=0.):
        """Set scattered field and noise data."""
        self.es = np.copy(scattered_field)
        self.noise = noise

    def set_total_field(self, total_field):
        """Set total field data."""
        self.et = np.copy(total_field)

    def set_incident_field(self, incident_field):
        """Set incident field data."""
        self.ei = np.copy(incident_field)

    def set_relative_permittivity_map(self, relative_permittivity_map):
        """Set relative permittivity map."""
        self.epsilon_r = np.copy(relative_permittivity_map)

    def set_conductivity_map(self, conductivity_map):
        """Set conductivity map."""
        self.sigma = np.copy(conductivity_map)
