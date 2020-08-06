"""A module to represent problem case.

Based on the same problem configuration, there may be infinite scenarios
describing different geometries and resolutions. So, this module
provides a class in which we may store information about a scenario,
i.e., a problem case in which we may the scattered field measurements
and some other information which will be received by the solver
describing the problem to be solved.

The :class:`InputData` implements a container which will be the standard
input to solvers and include all the information necessary to solve a
inverse scattering problem.

The following class is defined

:class:`InputData`
    The container representing an instance of a inverse scattering
    problem.
"""

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
COMPUTE_RESIDUAL_ERROR = 'residual_error'
COMPUTE_MAP_ERROR = 'map_error'
COMPUTE_TOTALFIELD_ERROR = 'totalfield_error'


class InputData:
    """The container representing an instance of a problem.

    Attributes
    ----------
        name
            A string naming the instance.

        configuration_filename
            A string for referencing the problem configuration.

        resolution
            A tuple with the size, in pixels, of the recovered image.
            Y-X ordered.

        scattered_field
            Matrix containing the scattered field information at
            S-domain.

        total_field
            Matrix containing the total field information at D-domain.

        incident_field
            Matrix containing the incident field information at
            D-domain.

        relative_permittivity_map
            Matrix with the discretized image of the relative
            permittivity map.

        conductivity_map
            Matrix with the discretized image of the conductivity map.

        noise
            noise level of scattered field data.

        homogeneous_objects : bool
            A flag to indicate if the instance only contains
            homogeneous objects.

        compute_residual_error : bool
            A flag to indicate the measurement of the residual error
            throughout or at the end of the solver executation.

        compute_map_error : bool
            A flag to indicate the measurement of the error in
            predicting the dielectric properties of the image.

        compute_totalfield_error : bool
            A flag to indicate the measurement of the estimation error
            of the total field throughout or at the end of the solver
            executation.
    """

    name = ''
    configuration_filename = ''
    resolution = (int(), int())
    es = np.array((int(), int()), dtype=complex)
    et = np.array((int(), int()), dtype=complex)
    ei = np.array((int(), int()), dtype=complex)
    epsilon_r = np.array((int(), int()), dtype=complex)
    sigma = np.array((int(), int()), dtype=complex)
    homogeneous_objects = bool()
    noise = float()
    compute_residual_error = bool()
    compute_map_error = bool()
    compute_totalfield_error = bool()

    def __init__(self, name=None, configuration_filename=None, resolution=None,
                 scattered_field=None, total_field=None, incident_field=None,
                 relative_permittivity_map=None, conductivity_map=None,
                 noise=None, import_filename=None, import_filepath='',
                 homogeneous_objects=True, compute_residual_error=True,
                 compute_map_error=False, compute_totalfield_error=False):
        r"""
        Build or import an object.

        You must give either the import file name and path or the
        required variables.

        Call signatures::

            InputData(import_filename='my_file',
                      import_filepath='./data/')
            InputData(name='instance00',
                      configuration_filename='setup00', ...)

        Parameters
        ----------
            name : string
                The name of the instance.

            `configuration_filename` : string
                A string with the name of the problem configuration
                file.

            resolution : 2-tuple
                The size, in pixels, of the image to be recovered. Y-X
                ordered.

            scattered_field : :class:`numpy.ndarray`
                A matrix containing the scattered field information at
                S-domain.

            total_field : :class:`numpy.ndarray`
                A matrix containing the total field information at
                D-domain.

            incident_field : :class:`numpy.ndarray`
                A matrix containing the incident field information at
                D-domain.

            relative_permittivity_map : :class:`numpy.ndarray`
                A matrix with the discretized image of the relative
                permittivity map.

            conductivity_map : :class:`numpy.ndarray`
                A matrix with the discretized image of the conductivity
                map.

            noise : float
                Noise level of scattered field data.

            homogeneous_objects : bool
                A flag to indicate if the instance only contains
                homogeneous objects.

            compute_residual_error : bool
                A flag to indicate the measurement of the residual error
                throughout or at the end of the solver executation.

             compute_map_error : bool
                A flag to indicate the measurement of the error in
                predicting the dielectric properties of the image.

             compute_totalfield_error : bool
                A flag to indicate the measurement of the estimation
                error of the total field throughout or at the end of the
                solver executation.

            import_filename : string
                A string with the name of the saved file.

            import_filepath : string
                A string with the path to the saved file.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)

        else:

            if name is None:
                raise error.MissingInputError('InputData.__init__()', 'name')
            if configuration_filename is None:
                raise error.MissingInputError('InputData.__init__()',
                                              'configuration_filename')
            if (resolution is None and relative_permittivity_map is None
                    and conductivity_map is None):
                raise error.MissingInputError('InputData.__init__()',
                                              'resolution')

            self.name = name
            self.configuration_filename = configuration_filename
            self.homogeneous_objects = homogeneous_objects
            self.compute_residual_error = compute_residual_error
            self.compute_map_error = compute_map_error
            self.compute_totalfield_error = compute_totalfield_error

            if resolution is not None:
                self.resolution = None

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
                if resolution is None:
                    self.resolution = relative_permittivity_map.shape
            else:
                self.epsilon_r = None

            if conductivity_map is not None:
                self.sigma = conductivity_map
                if resolution is None:
                    self.resolution = conductivity_map.shape
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
            CONDUCTIVITY_MAP: self.sigma,
            COMPUTE_RESIDUAL_ERROR: self.compute_residual_error,
            COMPUTE_MAP_ERROR: self.compute_map_error,
            COMPUTE_TOTALFIELD_ERROR: self.compute_totalfield_error
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
        self.compute_residual_error = data[COMPUTE_RESIDUAL_ERROR]
        self.compute_map_error = data[COMPUTE_MAP_ERROR]
        self.compute_totalfield_error = data[COMPUTE_TOTALFIELD_ERROR]

    def draw(self, figure_title=None, file_path='', file_format='eps',
             show=False):
        """Draw the relative permittivity/conductivity map.

        Parameters
        ----------
            figure_title : str, optional
                A title that you want to give to the figure.

            show : boolean, default: False
                If `True`, a window will be raised to show the image. If
                `False`, the image will be saved.

            file_path : str, default: ''
                A path where you want to save the figure.

            file_format : str, default: 'eps'
                The file format. It must be one of the available ones by
                `matplotlib.pyplot.savefig()`.
        """
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
            if type(figure_title) is str:
                fig.suptitle(figure_title, fontsize=16)
            elif figure_title is None:
                fig.suptitle(self.name, fontsize=16)
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

    def __str__(self):
        """Print information."""
        message = 'Input name: ' + self.name
        message = (message + '\nConfiguration file: '
                   + self.configuration_filename)
        message = (message + '\nResolution: %dx' % self.resolution[0]
                   + '%d' % self.resolution[1])
        if self.es is not None:
            message = (message + '\nScattered field - measurement samples: %d'
                       % self.es.shape[0]
                       + '\nScattered field - source samples: %d'
                       % self.es.shape[1])
        if self.et is not None:
            message = (message + '\nTotal field - measurement samples: %d'
                       % self.et.shape[0]
                       + '\nTotal field - source samples: %d'
                       % self.et.shape[1])
        if self.ei is not None:
            message = (message + '\nIncident field - measurement samples: %d'
                       % self.ei.shape[0]
                       + '\nIncident field - source samples: %d'
                       % self.ei.shape[1])
        if self.noise is not None:
            message = message + '\nNoise level: %.3e' % self.noise
        if self.epsilon_r is not None:
            message = (message + '\nRelative Permit. map shape: %dx'
                       % self.epsilon_r.shape[0] + '%d'
                       % self.epsilon_r.shape[1])
        if self.sigma is not None:
            message = (message + '\nConductivity map shape: %dx'
                       % self.sigma.shape[0] + '%d'
                       % self.sigma.shape[1])
        message = (message + '\nCompute residual error: '
                   + str(self.compute_residual_error))
        message = (message + '\nCompute map error: '
                   + str(self.compute_map_error))
        message = (message + '\nCompute total field error: '
                   + str(self.compute_totalfield_error))
        return message
