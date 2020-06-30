"""A module for results information.

The results module provides the :class:`Results` which contains the
resultant information of a single execution of a method for a given
scenario and the corresponding problem configuration. The class is also
a tool for plotting results. The following class is defined

:class:`Results`
    a class for storing results information of a single execution.

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import library_v2.error as error
import library_v2.configuration as cfg
import library_v2.inputdata as ipt

# Strings for easier implementation of plots
XLABEL_STANDARD = r'x [$\lambda_b$]'
YLABEL_STANDARD = r'y [$\lambda_b$]'
COLORBAR_RELATIVE_PERMITTIVITY = r'$\epsilon_r$'
COLORBAR_CONDUCTIVITY = r'$\sigma$ [S/m]'
TITLE_RELATIVE_PERMITTIVITY = 'Relative Permittivity'
TITLE_CONDUCTIVITY = 'Conductivity'
TITLE_RECOVERED_RELATIVE_PERMITTIVITY = ('Recovered '
                                         + TITLE_RELATIVE_PERMITTIVITY)
TITLE_RECOVERED_CONDUCTIVITY = 'Recovered ' + TITLE_CONDUCTIVITY
TITLE_ORIGINAL_RELATIVE_PERMITTIVITY = ('Original '
                                        + TITLE_RELATIVE_PERMITTIVITY)
TITLE_ORIGINAL_CONDUCTIVITY = 'Original ' + TITLE_CONDUCTIVITY
IMAGE_SIZE_SINGLE = (6., 5.)
IMAGE_SIZE_1x2 = (9., 5.)
IMAGE_SIZE_2X2 = (9., 9.)


class Results:
    """A class for storing results information of a single execution.

    Each executation of method for a giving scenario with corresponding
    configuration will result in information which will be stored in
    this class.

    Attributes
    ----------
        name
            A string identifying the stored result. It may be a
            combination of the method, scenario and configuration names.

        method_name
            A string with the name of the method which yield this result.

        configuration_filename
            A string containing the file name in which configuration
            info is stored.

        configuration_filepath
            A string containing the path to the file which stores the
            configuration info.

        scenario_filename
            A string containing the file name in which scenario info is
            stored.

        scenario_filepath
            A string containing the path to the file which stores the
            scenario info.

        et
            The total field matrix which is estimated by the inverse
            nonlinear solver. Unit: [V/m]

        es
            The scattered field matrix resulting from the estimation of
            the total field and contrast map. Unit: [V/m]

        epsilon_r
            The image matrix containing the result of the relative
            permittivity map estimation.

        sigma
            The image matrix containing the result of the conductivity
            map estimation. Unit: [S/m]


    """

    name = str()
    method_name = str()
    configuration_filename, configuration_filepath = str(), str()
    scenario_filename, scenario_filepath = str(), str()
    et = np.array([])
    es = np.array([])
    epsilon_r, sigma = np.array([]), np.array([])

    def __init__(self, name, method_name=None, configuration_filename=None,
                 configuration_filepath='', scenario_filename=None,
                 scenario_filepath='', scattered_field=None,
                 total_field=None, relative_permittivity_map=None,
                 conductivity_map=None):
        """Build the object.

        You may provide here the value of all attributes. But only name
        is required.
        """
        self.name = name
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
        """Plot map results.

        Call signatures::

            plot_map(show=False, filepath='', file_format='eps')

        Parameters
        ----------
            show : boolean
                If `False`, a figure will be saved with the name
                attribute of the object will be save at the specified
                path with the specified format. If `True`, a plot window
                will be displayed.

            file_path : string
                The location in which you want save the figure. Default:
                ''

            file_format : string
                The format of the figure to be saved. The possible
                formats are the same of the command `pyplot.savefig()`.
                Default: 'eps'

        """
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
        bounds = (xmin/lambda_b, xmax/lambda_b, ymin/lambda_b, ymax/lambda_b)

        if self.scenario_filename is None:

            if data[cfg.PERFECT_DIELECTRIC_FLAG]:
                if self.epsilon_r is None:
                    raise error.MissingAttributesError('Results',
                                                       'relative_permittivity'
                                                       + '_map')
                figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
                axes = get_single_figure_axes(figure)
                add_image(axes, self.epsilon_r,
                          TITLE_RELATIVE_PERMITTIVITY,
                          COLORBAR_RELATIVE_PERMITTIVITY, bounds=bounds)

            elif data[cfg.GOOD_CONDUCTOR_FLAG]:
                if self.sigma is None:
                    raise error.MissingAttributesError('Results',
                                                       'conductivity_map')
                figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
                axes = get_single_figure_axes(figure)
                add_image(axes, self.sigma,
                          TITLE_CONDUCTIVITY,
                          COLORBAR_CONDUCTIVITY, bounds=bounds)

            else:
                figure = plt.figure(figsize=IMAGE_SIZE_1x2)
                set_subplot_size(figure)

                axes = figure.add_subplot(1, 2, 1)
                add_image(axes, self.epsilon_r, TITLE_RELATIVE_PERMITTIVITY,
                          COLORBAR_RELATIVE_PERMITTIVITY, bounds=bounds)

                axes = figure.add_subplot(1, 2, 2)
                add_image(axes, self.sigma, TITLE_CONDUCTIVITY,
                          COLORBAR_CONDUCTIVITY, bounds=bounds)

        else:

            with open(self.scenario_filepath
                      + self.scenario_filename, 'rb') as datafile:
                scenario = pickle.load(datafile)

            if data[cfg.PERFECT_DIELECTRIC_FLAG]:
                if self.epsilon_r is None:
                    raise error.MissingAttributesError('Results',
                                                       'relative_permittivity'
                                                       + '_map')

                figure = plt.figure(figsize=IMAGE_SIZE_1x2)
                set_subplot_size(figure)

                axes = figure.add_subplot(1, 2, 1)
                add_image(axes, scenario[ipt.RELATIVE_PERMITTIVITY_MAP],
                          TITLE_ORIGINAL_RELATIVE_PERMITTIVITY,
                          COLORBAR_RELATIVE_PERMITTIVITY, bounds=bounds)

                axes = figure.add_subplot(1, 2, 2)
                add_image(axes, self.epsilon_r,
                          TITLE_RECOVERED_RELATIVE_PERMITTIVITY,
                          COLORBAR_RELATIVE_PERMITTIVITY, bounds=bounds)

            elif data[cfg.GOOD_CONDUCTOR_FLAG]:
                if self.sigma is None:
                    raise error.MissingAttributesError('Results',
                                                       'conductivity_map')

                figure = plt.figure(figsize=IMAGE_SIZE_1x2)
                set_subplot_size(figure)

                axes = figure.add_subplot(1, 2, 1)
                add_image(axes, scenario[ipt.CONDUCTIVITY_MAP],
                          TITLE_ORIGINAL_CONDUCTIVITY, COLORBAR_CONDUCTIVITY,
                          bounds=bounds)

                axes = figure.add_subplot(1, 2, 2)
                add_image(axes, self.sigma,
                          TITLE_ORIGINAL_CONDUCTIVITY, COLORBAR_CONDUCTIVITY,
                          bounds=bounds)

            else:
                if self.epsilon_r is None:
                    raise error.MissingAttributesError('Results',
                                                       'relative_permittivity'
                                                       + '_map')
                if self.sigma is None:
                    raise error.MissingAttributesError('Results',
                                                       'conductivity_map')

                figure = plt.figure(figsize=IMAGE_SIZE_2X2)
                set_subplot_size(figure)

                axes = figure.add_subplot(2, 2, 1)
                add_image(axes, scenario[ipt.RELATIVE_PERMITTIVITY_MAP],
                          TITLE_ORIGINAL_RELATIVE_PERMITTIVITY,
                          COLORBAR_RELATIVE_PERMITTIVITY, bounds=bounds)

                axes = figure.add_subplot(2, 2, 2)
                add_image(axes, scenario[ipt.CONDUCTIVITY_MAP],
                          TITLE_ORIGINAL_CONDUCTIVITY, COLORBAR_CONDUCTIVITY,
                          bounds=bounds)

                axes = figure.add_subplot(2, 2, 3)
                add_image(axes, self.epsilon_r,
                          TITLE_RECOVERED_RELATIVE_PERMITTIVITY,
                          COLORBAR_RELATIVE_PERMITTIVITY, bounds=bounds)

                axes = figure.add_subplot(2, 2, 4)
                add_image(axes, self.sigma,
                          TITLE_ORIGINAL_CONDUCTIVITY, COLORBAR_CONDUCTIVITY,
                          bounds=bounds)

        if show:
            plt.show()
        else:
            plt.savefig(file_path + self.name + '.' + file_format,
                        format=file_format)


def add_image(axes, image, title, colorbar_name, bounds=(-1., 1., -1., 1.),
              origin='lower', xlabel=XLABEL_STANDARD, ylabel=YLABEL_STANDARD):
    """Add a image to the axes.

    A predefined function for plotting image. This is useful for
    standardize plots involving contrast maps and fields.

    Paramaters
    ----------
        axes : :class:`matplotlib.pyplot.Figure.axes.Axes`
            The axes object.

        image : :class:`numpy.ndarray`
            A matrix with image to be displayed. If complex, the
            magnitude will be displayed.

        title : string
            The title to be displayed in the figure.

        colorbar_name : string
            The label for color bar.

        bounds : 4-tuple of floats, default: (-1., 1., -1., 1.)
            The value of the bounds of each axis. Example: (xmin, xmax,
            ymin, ymax).

        origin : {'lower', 'upper'}, default: 'lower'
            Origin of the y-axis.

        xlabel : string, default: XLABEL_STANDARD
            The label of the x-axis.

        ylabel : string, default: YLABEL_STANDARD
            The label of the y-axis.

    """
    if image.dtype == complex:
        im = axes.imshow(np.abs(image),
                         extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
                         origin=origin)
    else:
        im = axes.imshow(image,
                         extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
                         origin=origin)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    cbar = plt.colorbar(ax=axes, mappable=im, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_name)


def set_subplot_size(figure):
    """Set subplot sizes.

    A standard definition for setting images in subplot figures.

    Parameters
    ----------
        figure : `:class:matplotlib.pyplot.Figure`
            A figure object.
    """
    figure.subplots_adjust(left=.125, bottom=.0, right=.9, top=.9, wspace=.5,
                           hspace=.2)


def get_single_figure_axes(figure):
    """Define axes for single plots.

    Parameters
    ----------
        figure : :class:`matplotlib.pyplot.Figure`
            A figure object.

    """
    return figure.add_axes([0.125, 0.15, .7, .7])
