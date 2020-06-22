"""A class for results of the solver."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import library_v2.error as error
import library_v2.configuration as cfg


class Results:
    """A class for storing results information of a single execution.

    Atributtes
        method name --
    """

    method_name = str()
    configuration_filename, configuration_filepath = str(), str()
    scenario_filename, scenario_filepath = str(), str()
    et = np.array([])
    epsilon_r, sigma = np.array([]), np.array([])
    es = np.array([])

    def __init__(self, method_name, configuration_filename=None,
                 configuration_filepath=None, scenario_filename=None,
                 scenario_filepath=None, scattered_field=None,
                 total_field=None, relative_permittivity_map=None,
                 conductivity_map=None):
        """Build the object.

        Arguments:
            method_name --
        """
        self.method_name = method_name
        self.configuration_filename = configuration_filename
        self.configuration_filepath = configuration_filepath
        self.scenario_filename = scenario_filename
        self.scenario_filepath = scenario_filepath
        self.et = total_field
        self.es = scattered_field
        self.epsilon_r = relative_permittivity_map
        self.sigma = conductivity_map

    def plot_map(self, show=False, file_path='', file_format='eps'):
        """Plot map results."""
        if self.configuration_filename is None:
            raise error.MissingAttributesError('Results',
                                               'configuration_filename')

        with open(self.configuration_filepath + self.configuration_filename,
                  'rb') as datafile:
            data = pickle.load(datafile)

        lambda_b = data[cfg.BACKGROUND_WAVELENGTH]
        Lx, Ly = data[cfg.IMAGE_SIZE]
        xmin, xmax = cfg.get_bounds(Lx)
        ymin, ymax = cfg.get_bounds(Ly)
        xmin, xmax = xmin/lambda_b, xmax/lambda_b
        ymin, ymax = ymin/lambda_b, ymax/lambda_b

        if self.scenario_filename is None:

            if data[cfg.PERFECT_DIELECTRIC_FLAG]:
                if self.epsilon_r is None:
                    raise error.MissingAttributesError('Results',
                                                       'relative_permittivity'
                                                       + '_map')
                plt.imshow(self.epsilon_r, extent=[xmin, xmax, ymin, ymax],
                           origin='lower')
                plt.xlabel(r'x [$\lambda_b$]')
                plt.ylabel(r'y [$\lambda_b$]')
                plt.title('Recovered Relative Permittivity')
                cbar = plt.colorbar()
                cbar.set_label(r'$\epsilon_r$')

            elif data[cfg.GOOD_CONDUCTOR_FLAG]:
                if self.sigma is None:
                    raise error.MissingAttributesError('Results',
                                                       'conductivity_map')
                plt.imshow(self.sigma, extent=[xmin, xmax, ymin, ymax],
                           origin='lower')
                plt.xlabel(r'x [$\lambda_b$]')
                plt.ylabel(r'y [$\lambda_b$]')
                plt.title('Recovered Conductivity')
                cbar = plt.colorbar()
                cbar.set_label(r'$\sigma$')

            else:
                fig = plt.figure(figsize=(10, 4))
                fig.subplots_adjust(left=.125, bottom=.1, right=.9, top=.9,
                                    wspace=.5, hspace=.2)
                ax = fig.add_subplot(1, 2, 1)
                im1 = ax.imshow(self.epsilon_r, extent=[xmin, xmax, ymin, ymax],
                                origin='lower')
                ax.set_xlabel(r'x [$\lambda_b$]')
                ax.set_ylabel(r'y [$\lambda_b$]')
                ax.set_title('Recovered Relative Permittivity')
                cbar = fig.colorbar()
                cbar.set_label(r'$\epsilon_r$')
                im2 = ax.imshow(self.sigma, extent=[xmin, xmax, ymin, ymax],
                                origin='lower')
                ax.set_xlabel(r'x [$\lambda_b$]')
                ax.set_ylabel(r'y [$\lambda_b$]')
                ax.set_title('Recovered Conductivity')
                cbar = fig.colorbar()
                cbar.set_label(r'$\sigma$ [S/m]')