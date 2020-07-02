"""A module for results information.

The results module provides the :class:`Results` which contains the
resultant information of a single execution of a method for a given
input data and the corresponding problem configuration. The class is also
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

    Each executation of method for a giving input data with 
    corresponding configuration will result in information which will be
    stored in this class.

    Attributes
    ----------
        name
            A string identifying the stored result. It may be a
            combination of the method, input data and configuration
            names.

        method_name
            A string with the name of the method which yield this result.

        configuration_filename
            A string containing the file name in which configuration
            info is stored.

        configuration_filepath
            A string containing the path to the file which stores the
            configuration info.

        inputdata_filename
            A string containing the file name in which instance info is
            stored.

        inputdata_filepath
            A string containing the path to the file which stores the
            instance info.

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
    inputdata_filename, inputdata_filepath = str(), str()
    et = np.array([])
    es = np.array([])
    epsilon_r, sigma = np.array([]), np.array([])
    zeta_rn, zeta_rpad = list(), list()
    zeta_epad, zeta_spad = list(), list()
    zeta_be = list()
    zeta_ebe, zeta_sbe = list(), list()
    zeta_eoe, zeta_soe = list(), list()
    zeta_tfmpad, zeta_tfppad = list(), list()

    def __init__(self, name, method_name=None, configuration_filename=None,
                 configuration_filepath='', inputdata_filename=None,
                 inputdata_filepath='', scattered_field=None,
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
        self.inputdata_filename = inputdata_filename
        self.inputdata_filepath = inputdata_filepath
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

        if self.inputdata_filename is None:

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

            with open(self.inputdata_filepath
                      + self.inputdata_filename, 'rb') as datafile:
                inputdata = pickle.load(datafile)

            if data[cfg.PERFECT_DIELECTRIC_FLAG]:
                if self.epsilon_r is None:
                    raise error.MissingAttributesError('Results',
                                                       'relative_permittivity'
                                                       + '_map')

                figure = plt.figure(figsize=IMAGE_SIZE_1x2)
                set_subplot_size(figure)

                axes = figure.add_subplot(1, 2, 1)
                add_image(axes, inputdata[ipt.RELATIVE_PERMITTIVITY_MAP],
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
                add_image(axes, inputdata[ipt.CONDUCTIVITY_MAP],
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
                add_image(axes, inputdata[ipt.RELATIVE_PERMITTIVITY_MAP],
                          TITLE_ORIGINAL_RELATIVE_PERMITTIVITY,
                          COLORBAR_RELATIVE_PERMITTIVITY, bounds=bounds)

                axes = figure.add_subplot(2, 2, 2)
                add_image(axes, inputdata[ipt.CONDUCTIVITY_MAP],
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

    def update_error(self, inputdata, scattered_field=None, total_field=None,
                     relative_permittivity_map=None, conductivity_map=None):
        """Compute errors for a given set of variables.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                An object of InputData representing an instance.

            scattered_field, total_field : :class:`numpy.ndarray`
                Fields estimated by the solver.

            relative_permittivity_map : :class:`numpy.ndarray`
                Relative permittivity image recovered by the solver.

            conductivity_map : :class:`numpy.ndarray`
                Conductivity image recovered by the solver.
        """
        if scattered_field is not None and inputdata.es is not None:
            self.zeta_rn.append(compute_zeta_rn(inputdata.es, scattered_field))
            self.zeta_rpad.append(compute_zeta_rpad(inputdata.es,
                                                    scattered_field))

        if total_field is not None and inputdata.et is not None:
            self.zeta_tfmpad.append(compute_zeta_tfmpad(inputdata.et,
                                                        total_field))
            self.zeta_tfppad.append(compute_zeta_tfppad(inputdata.et,
                                                        total_field))

        if self.configuration_filename is None:
            raise error.MissingAttributesError('Results',
                                               'configuration_filename')
        else:
            config = cfg.import_dict(self.configuration_filename,
                                     self.configuration_filepath)
            epsilon_rb = config[cfg.BACKGROUND_RELATIVE_PERMITTIVITY]
            sigma_b = config[cfg.BACKGROUND_CONDUCTIVITY]
            omega = 2*np.pi*config[cfg.FREQUENCY]

        if (inputdata.epsilon_r is not None
                and relative_permittivity_map is not None):
            self.zeta_epad.append(compute_zeta_epad(inputdata.epsilon_r,
                                                    relative_permittivity_map))
            self.zeta_ebe.append(compute_zeta_ebe(inputdata.epsilon_r,
                                                  relative_permittivity_map,
                                                  epsilon_rb))
            self.zeta_eoe.append(compute_zeta_eoe(inputdata.epsilon_r,
                                                  relative_permittivity_map,
                                                  epsilon_rb))

        if inputdata.sigma is not None and conductivity_map is not None:
            self.zeta_spad.append(compute_zeta_spad(inputdata.sigma,
                                                    conductivity_map))
            self.zeta_sbe.append(compute_zeta_sbe(inputdata.sigma,
                                                  conductivity_map,
                                                  sigma_b))
            self.zeta_soe.append(compute_zeta_soe(inputdata.sigma,
                                                  conductivity_map,
                                                  sigma_b))

        if inputdata.homogeneous_objects:
            if (relative_permittivity_map is not None
                    and conductivity_map is None):
                conductivity_map = (
                    sigma_b*np.zeros(relative_permittivity_map.shape)
                )
            elif (relative_permittivity_map is None
                    and conductivity_map is not None):
                relative_permittivity_map = (
                    epsilon_rb*np.zeros(conductivity_map.shape)
                )
            x, y = cfg.get_coordinates_ddomain(
                configuration=cfg.Configuration(
                    import_filename=self.configuration_filename,
                    import_filepath=self.configuration_filepath
                ), resolution=relative_permittivity_map.shape
            )
            self.zeta_be.append(
                compute_zeta_be(cfg.get_contrast_map(relative_permittivity_map,
                                                     conductivity_map,
                                                     epsilon_rb, sigma_b,
                                                     omega), x, y)
            )


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


def compute_zeta_rn(es_o, es_a):
    r"""Compute the residual norm error.

    The zeta_rn error is the residual norm error of the scattered
    field approximation.

    Parameters
    ----------
        es_o : :class:`numpy.ndarray`
            Original scattered field matrix.

        es_a : :class:`numpy.ndarray`
            Approximated scattered field matrix.

    Notes
    -----
        The error is computed through the following relation:

        .. math:: ||E^s-E^{s,\delta}|| = \sqrt{\iint_S(y-y^\delta)
        \overline{(y-y^\delta)}d\theta
    """
    NM, NS = es_o.shape
    theta = cfg.get_angles(NM)
    phi = cfg.get_angles(NS)
    y = (es_o-es_a)*np.conj(es_o-es_a)
    return np.sqrt(np.trapz(np.trapz(y, x=phi), x=theta))


def compute_zeta_rpad(es_o, es_r):
    r"""Compute the residual percentage average deviation.

    The zeta_padr error is the residual percentage average deviation
    of the scattered field approximation.

    Parameters
    ----------
        es_o : :class:`numpy.ndarray`
            Original scattered field matrix.

        es_a : :class:`numpy.ndarray`
            Approximated scattered field matrix.
    """
    y = np.hstack((np.real(es_o.flatten()), np.imag(es_o.flatten())))
    yd = np.hstack((np.real(es_r.flatten()), np.imag(es_r.flatten())))
    return np.mean(np.abs((y-yd)/y))


def compute_zeta_epad(epsilon_ro, epsilon_rr):
    """Compute the percent. aver. deviation of relative permit. map.

    The zeta_epad error is the evaluation of the relative
    permittivity estimation error per pixel.

    Parameters
    ----------
        epsilon_ro, epsilon_rr : :class:`numpy.ndarray`
            Original and recovered relative permittivity maps,
            respectively.
    """
    y = epsilon_ro.flatten()
    yd = epsilon_rr.flatten()
    return np.mean(np.abs((y-yd)/y))


def compute_zeta_spad(sigma_o, sigma_r):
    """Compute the percentage average deviation of conductivity map.

    The zeta_epad error is the evaluation of the conductivity
    estimation error per pixel.

    Parameters
    ----------
        sigma_o, sigma_r : :class:`numpy.ndarray`
            Original and recovered conductivity maps, respectively.
    """
    y = sigma_o.flatten()
    yd = sigma_r.flatten()
    return np.mean(np.abs((y-yd)))


def compute_zeta_be(chi, x, y):
    """Compute the boundary estimation error.

    The zeta_be is an estimation of the error at the boundaries of
    objects. It is a kind of variational functional [1].

    Parameters
    ----------
        chi : :class:`numpy.ndarray`
            Constrast map.

        x, y : :class:`numpy.ndarray`
            Meshgrid arrays of x and y coordinates.

    References
    ----------
    .. [1] Lobel, P., et al. "A new regularization scheme for
       inverse scattering." Inverse Problems 13.2 (1997): 403.
    """
    grad_chi = np.gradient(chi, y[:, 0], x[0, :])
    X = np.sqrt(np.abs(grad_chi[1])**2 + np.abs(grad_chi[0])**2)
    return np.trapz(np.trapz(X**2/(X**2+1), x=x[0, :]), x=y[:, 0])


def compute_zeta_ebe(epsilon_ro, epsilon_rr, epsilon_rb):
    """Compute the background relative permit. estimation error.

    The zeta_ebe is an estimation of the error of predicting the
    background region considering specifically the relative
    permittivity information. It is an analogy to the false-positive
    rate.

    Parameters
    ----------
        epsilon_ro, epsilon_rr : :class:`numpy.ndarray`
            Original and recovered relative permittivity maps.
        epsilon_rb : float
            Background relative permittivity.
    """
    background = np.zeros(epsilon_ro.shape, dtype=bool)
    background[epsilon_ro == epsilon_rb] = True
    y = epsilon_ro[background]
    yd = epsilon_rr[background]
    return np.mean(np.abs(y-yd)/y)


def compute_zeta_sbe(sigma_o, sigma_r, sigma_b):
    """Compute the background conductivity estimation error.

    The zeta_sbe is an estimation of the error of predicting the
    background region considering specifically the conductivity
    information. It is an analogy to the false-positive
    rate.

    Parameters
    ----------
        sigma_o, sigma_r : :class:`numpy.ndarray`
            Original and recovered conductivity maps.
        sigma_b : float
            Background conductivity.
    """
    background = np.zeros(sigma_o.shape, dtype=bool)
    background[sigma_o == sigma_b] = True
    y = sigma_o[background]
    yd = sigma_r[background]
    return np.mean(np.abs(y-yd)/y)


def compute_zeta_eoe(epsilon_ro, epsilon_rr, epsilon_rb):
    """Compute the object relative permit. estimation error.

    The zeta_eoe is an estimation of the error of predicting the
    object region considering specifically the relative
    permittivity information. It is an analogy to the false-negative
    rate.

    Parameters
    ----------
        epsilon_ro, epsilon_rr : :class:`numpy.ndarray`
            Original and recovered relative permittivity maps.
        epsilon_rb : float
            Background relative permittivity.
    """
    not_background = np.zeros(epsilon_ro.shape, dtype=bool)
    not_background[epsilon_ro != epsilon_rb] = True
    y = epsilon_ro[not_background]
    yd = epsilon_rr[not_background]
    return np.mean(np.abs(y-yd)/y)


def compute_zeta_soe(sigma_o, sigma_r, sigma_b):
    """Compute the object conductivity estimation error.

    The zeta_soe is an estimation of the error of predicting the
    object region considering specifically the conductivity
    information. It is an analogy to the false-negative
    rate.

    Parameters
    ----------
        sigma_o, sigma_r : :class:`numpy.ndarray`
            Original and recovered conductivity maps.
        sigma_b : float
            Background conductivity.
    """
    not_background = np.zeros(sigma_o.shape, dtype=bool)
    not_background[sigma_o != sigma_b] = True
    y = sigma_o[not_background]
    yp = sigma_r[not_background]
    return np.mean(np.abs(y-yp)/y)


def compute_zeta_tfmpad(et_o, et_r):
    """Compute the percen. aver. devi. of the total field magnitude.

    The measure estimates the error in the estimation of the
    magnitude of total field.

    Parameters
    ----------
        et_o, et_r : :class:`numpy.ndarray`
            Original and recovered total field, respectively.
    """
    y = np.abs(et_o.flatten())
    yd = np.abs(et_r.flatten())
    return np.mean(np.abs((y-yd)/yd))


def compute_zeta_tfppad(et_o, et_r):
    """Compute the percen. aver. devi. of the total field phase.

    The measure estimates the error in the estimation of the
    phase of total field.

    Parameters
    ----------
        et_o, et_r : :class:`numpy.ndarray`
            Original and recovered total field, respectively.
    """
    y = np.angle(et_o.flatten())
    yd = np.angle(et_r.flatten())
    return np.mean(np.abs((y-yd)/yd))
