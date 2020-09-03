"""A module for results information.

The results module provides the :class:`Results` which contains the
resultant information of a single execution of a method for a given
input data and the corresponding problem configuration. The class is also
a tool for plotting results. The following class is defined

:class:`Results`
    a class for storing results information of a single execution.

The list of routines...
"""

import pickle
import copy as cp
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import error
import configuration as cfg
import inputdata as ipt

# Strings for easier implementation of plots
XLABEL_STANDARD = r'x [$\lambda_b$]'
YLABEL_STANDARD = r'y [$\lambda_b$]'
XLABEL_UNDEFINED = r'x [$L_x$]'
YLABEL_UNDEFINED = r'y [$L_y$]'
COLORBAR_REL_PERMITTIVITY = r'$\epsilon_r$'
COLORBAR_CONDUCTIVITY = r'$\sigma$ [S/m]'
TITLE_REL_PERMITTIVITY = 'Relative Permittivity'
TITLE_CONDUCTIVITY = 'Conductivity'
TITLE_RECOVERED_REL_PERMITTIVITY = ('Recovered '
                                         + TITLE_REL_PERMITTIVITY)
TITLE_RECOVERED_CONDUCTIVITY = 'Recovered ' + TITLE_CONDUCTIVITY
TITLE_ORIGINAL_REL_PERMITTIVITY = ('Original '
                                        + TITLE_REL_PERMITTIVITY)
TITLE_ORIGINAL_CONDUCTIVITY = 'Original ' + TITLE_CONDUCTIVITY
LABEL_ZETA_RN = r'$\zeta_{RN}$'
LABEL_ZETA_RPAD = r'$\zeta_{RPAD}$'
LABEL_ZETA_EPAD = r'$\zeta_{\epsilon PAD}$'
LABEL_ZETA_EBE = r'$\zeta_{\epsilon BE}$'
LABEL_ZETA_EOE = r'$\zeta_{\epsilon OE}$'
LABEL_ZETA_SAD = r'$\zeta_{\sigma AD}$'
LABEL_ZETA_SBE = r'$\zeta_{\sigma BE}$'
LABEL_ZETA_SOE = r'$\zeta_{\sigma OE}$'
LABEL_ZETA_BE = r'$\zeta_{BE}$'
LABEL_ZETA_TFMPAD = r'$\zeta_{TFMPAD}$'
LABEL_ZETA_TFPPAD = r'$\zeta_{TFPPAD}$'


IMAGE_SIZE_SINGLE = (6., 5.)
IMAGE_SIZE_1x2 = (9., 4.) # 9 x 5
IMAGE_SIZE_2X2 = (9., 9.)

# Constant string for easier access of dictionary fields
NAME = 'name'
CONFIGURATION_FILENAME = 'configuration_filename'
CONFIGURATION_FILEPATH = 'configuration_filepath'
INPUT_FILENAME = 'input_filename'
INPUT_FILEPATH = 'input_filepath'
METHOD_NAME = 'method_name'
TOTAL_FIELD = 'et'
SCATTERED_FIELD = 'es'
RELATIVE_PERMITTIVITY_MAP = 'epsilon_r'
CONDUCTIVITY_MAP = 'sigma'
EXECUTION_TIME = 'execution_time'
RESIDUAL_NORM_ERROR = 'zeta_rn'
RESIDUAL_PAD_ERROR = 'zeta_pad'
REL_PERMITTIVITY_PAD_ERROR = 'zeta_epad'
CONDUCTIVITY_AD_ERROR = 'zeta_sad'
BOUNDARY_ERROR = 'zeta_be'
REL_PERMITTIVITY_BACKGROUND_ERROR = 'zeta_ebe'
REL_PERMITTIVITY_OBJECT_ERROR = 'zeta_eoe'
CONDUCTIVITY_BACKGROUND_ERROR = 'zeta_sbe'
CONDUCTIVITY_OBJECT_ERROR = 'zeta_soe'
TOTALFIELD_MAGNITUDE_PAD = 'zeta_tfmpad'
TOTALFIELD_PHASE_PAD = 'zeta_tfppad'


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

        input_filename
            A string containing the file name in which instance info is
            stored.

        input_filepath
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

        execution_time
            The amount of time for running the method.
    """

    def __init__(self, name=None, method_name=None,
                 configuration_filename=None, configuration_filepath='',
                 input_filename=None, input_filepath='', scattered_field=None,
                 total_field=None, relative_permittivity_map=None,
                 conductivity_map=None, execution_time=None,
                 import_filename=None, import_filepath=''):
        """Build the object.

        You may provide here the value of all attributes. But only name
        is required.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            if name is None:
                raise error.MissingInputError('Results.__init__()', 'name')
            self.name = name
            self.method_name = method_name
            self.configuration_filename = configuration_filename
            self.configuration_filepath = configuration_filepath
            self.input_filename = input_filename
            self.input_filepath = input_filepath
            self.et = total_field
            self.es = scattered_field
            self.epsilon_r = relative_permittivity_map
            self.sigma = conductivity_map
            self.execution_time = execution_time
            self.zeta_rn, self.zeta_rpad = list(), list()
            self.zeta_epad, self.zeta_sad = list(), list()
            self.zeta_be = list()
            self.zeta_ebe, self.zeta_sbe = list(), list()
            self.zeta_eoe, self.zeta_soe = list(), list()
            self.zeta_tfmpad, self.zeta_tfppad = list(), list()

    def save(self, file_path=''):
        """Save object information."""
        data = {
            NAME: self.name,
            CONFIGURATION_FILENAME: self.configuration_filename,
            CONFIGURATION_FILEPATH: self.configuration_filepath,
            INPUT_FILENAME: self.input_filename,
            INPUT_FILEPATH: self.input_filepath,
            METHOD_NAME: self.method_name,
            TOTAL_FIELD: self.et,
            SCATTERED_FIELD: self.es,
            RELATIVE_PERMITTIVITY_MAP: self.epsilon_r,
            CONDUCTIVITY_MAP: self.sigma,
            EXECUTION_TIME: self.execution_time,
            RESIDUAL_NORM_ERROR: self.zeta_rn,
            RESIDUAL_PAD_ERROR: self.zeta_rpad,
            REL_PERMITTIVITY_PAD_ERROR: self.zeta_epad,
            REL_PERMITTIVITY_BACKGROUND_ERROR: self.zeta_ebe,
            REL_PERMITTIVITY_OBJECT_ERROR: self.zeta_eoe,
            CONDUCTIVITY_AD_ERROR: self.zeta_sad,
            CONDUCTIVITY_BACKGROUND_ERROR: self.zeta_sbe,
            CONDUCTIVITY_OBJECT_ERROR: self.zeta_soe,
            BOUNDARY_ERROR: self.zeta_be,
            TOTALFIELD_MAGNITUDE_PAD: self.zeta_tfmpad,
            TOTALFIELD_PHASE_PAD: self.zeta_tfppad
        }

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import data from a saved object."""
        with open(file_path + file_name, 'rb') as datafile:
            data = pickle.load(datafile)
        self.name = data[NAME]
        self.configuration_filename = data[CONFIGURATION_FILENAME]
        self.configuration_filepath = data[CONFIGURATION_FILEPATH]
        self.input_filename = data[INPUT_FILENAME]
        self.input_filepath = data[INPUT_FILEPATH]
        self.method_name = data[METHOD_NAME]
        self.et = data[TOTAL_FIELD]
        self.es = data[SCATTERED_FIELD]
        self.epsilon_r = data[RELATIVE_PERMITTIVITY_MAP]
        self.sigma = data[CONDUCTIVITY_MAP]
        self.execution_time = data[EXECUTION_TIME]
        self.zeta_rn = data[RESIDUAL_NORM_ERROR]
        self.zeta_rpad = data[RESIDUAL_PAD_ERROR]
        self.zeta_epad = data[REL_PERMITTIVITY_PAD_ERROR]
        self.zeta_ebe = data[REL_PERMITTIVITY_BACKGROUND_ERROR]
        self.zeta_eoe = data[REL_PERMITTIVITY_OBJECT_ERROR]
        self.zeta_sad = data[CONDUCTIVITY_AD_ERROR]
        self.zeta_sbe = data[CONDUCTIVITY_BACKGROUND_ERROR]
        self.zeta_soe = data[CONDUCTIVITY_OBJECT_ERROR]
        self.zeta_be = data[BOUNDARY_ERROR]
        self.zeta_tfmpad = data[TOTALFIELD_MAGNITUDE_PAD]
        self.zeta_tfppad = data[TOTALFIELD_PHASE_PAD]

    def plot_map(self, show=False, file_path='', file_format='eps', benchmark=None):
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
        try:
            with open(self.configuration_filepath + self.configuration_filename,
                      'rb') as datafile:
                data = pickle.load(datafile)
            lambda_b = data[cfg.BACKGROUND_WAVELENGTH]
            Lx, Ly = data[cfg.IMAGE_SIZE]
            xmin, xmax = cfg.get_bounds(Lx)
            ymin, ymax = cfg.get_bounds(Ly)
            bounds = (xmin/lambda_b, xmax/lambda_b,
                      ymin/lambda_b, ymax/lambda_b)
            perfect_dieletric = data[cfg.PERFECT_DIELECTRIC_FLAG]
            good_conductor = data[cfg.GOOD_CONDUCTOR_FLAG]
            xlabel, ylabel = XLABEL_STANDARD, YLABEL_STANDARD
        except Exception:
            bounds = (0., 1., 0., 1.)
            if self.epsilon_r is not None and self.sigma is None:
                perfect_dieletric, good_conductor = True, False
            elif self.epsilon_r is None and self.sigma is not None:
                perfect_dieletric, good_conductor = False, True
            elif self.epsilon_r is not None and self.sigma is not None:
                perfect_dieletric, good_conductor = False, False
            else:
                raise error.MissingAttributesError('Results',
                                                   'epsilon_r or/and sigma')
            xlabel, ylabel = XLABEL_UNDEFINED, YLABEL_UNDEFINED

        if benchmark is None:
            try:
                with open(self.input_filepath
                          + self.input_filename, 'rb') as datafile:
                    inputdata = pickle.load(datafile)
                epsilon_rgoal = inputdata[ipt.RELATIVE_PERMITTIVITY_MAP]
                sigma_goal = inputdata[ipt.CONDUCTIVITY_MAP]
            except Exception:
                epsilon_rgoal = sigma_goal = None
        elif not isinstance(benchmark, ipt.InputData):
            raise error.WrongTypeInput('Results.plot_map', 'benchmark',
                                       benchmark.__class__.__name__,
                                       'InputData')
        else:
            epsilon_rgoal = cp.deepcopy(benchmark.epsilon_r)
            sigma_goal = cp.deepcopy(benchmark.sigma)

        if epsilon_rgoal is None and sigma_goal is None:

            if perfect_dieletric:
                if self.epsilon_r is None:
                    raise error.MissingAttributesError('Results',
                                                       'relative_permittivity'
                                                       + '_map')
                figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
                axes = get_single_figure_axes(figure)
                add_image(axes, self.epsilon_r,
                          TITLE_REL_PERMITTIVITY,
                          COLORBAR_REL_PERMITTIVITY, bounds=bounds,
                          xlabel=xlabel, ylabel=ylabel)

            elif good_conductor:
                if self.sigma is None:
                    raise error.MissingAttributesError('Results',
                                                       'conductivity_map')
                figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
                axes = get_single_figure_axes(figure)
                add_image(axes, self.sigma,
                          TITLE_CONDUCTIVITY,
                          COLORBAR_CONDUCTIVITY, bounds=bounds,
                          xlabel=xlabel, ylabel=ylabel)

            else:
                figure = plt.figure(figsize=IMAGE_SIZE_1x2)
                set_subplot_size(figure)

                axes = figure.add_subplot(1, 2, 1)
                add_image(axes, self.epsilon_r, TITLE_REL_PERMITTIVITY,
                          COLORBAR_REL_PERMITTIVITY, bounds=bounds,
                          xlabel=xlabel, ylabel=ylabel)

                axes = figure.add_subplot(1, 2, 2)
                add_image(axes, self.sigma, TITLE_CONDUCTIVITY,
                          COLORBAR_CONDUCTIVITY, bounds=bounds,
                          xlabel=xlabel, ylabel=ylabel)

        else:

            if perfect_dieletric:
                if self.epsilon_r is None:
                    raise error.MissingAttributesError('Results',
                                                       'relative_permittivity'
                                                       + '_map')

                figure = plt.figure(figsize=IMAGE_SIZE_1x2)
                set_subplot_size(figure)

                axes = figure.add_subplot(1, 2, 1)
                add_image(axes, epsilon_rgoal,
                          TITLE_ORIGINAL_REL_PERMITTIVITY,
                          COLORBAR_REL_PERMITTIVITY, bounds=bounds,
                          xlabel=xlabel, ylabel=ylabel)

                axes = figure.add_subplot(1, 2, 2)
                add_image(axes, self.epsilon_r,
                          TITLE_RECOVERED_REL_PERMITTIVITY,
                          COLORBAR_REL_PERMITTIVITY, bounds=bounds,
                          xlabel=xlabel, ylabel=ylabel)

            elif good_conductor:
                if self.sigma is None:
                    raise error.MissingAttributesError('Results',
                                                       'conductivity_map')

                figure = plt.figure(figsize=IMAGE_SIZE_1x2)
                set_subplot_size(figure)

                axes = figure.add_subplot(1, 2, 1)
                add_image(axes, sigma_goal,
                          TITLE_ORIGINAL_CONDUCTIVITY, COLORBAR_CONDUCTIVITY,
                          bounds=bounds, xlabel=xlabel, ylabel=ylabel)

                axes = figure.add_subplot(1, 2, 2)
                add_image(axes, sigma_goal,
                          TITLE_RECOVERED_CONDUCTIVITY, COLORBAR_CONDUCTIVITY,
                          bounds=bounds, xlabel=xlabel, ylabel=ylabel)

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
                add_image(axes, epsilon_rgoal,
                          TITLE_ORIGINAL_REL_PERMITTIVITY,
                          COLORBAR_REL_PERMITTIVITY, bounds=bounds,
                          xlabel=xlabel, ylabel=ylabel)

                axes = figure.add_subplot(2, 2, 2)
                add_image(axes, sigma_goal,
                          TITLE_ORIGINAL_CONDUCTIVITY, COLORBAR_CONDUCTIVITY,
                          bounds=bounds, xlabel=xlabel, ylabel=ylabel)

                axes = figure.add_subplot(2, 2, 3)
                add_image(axes, self.epsilon_r,
                          TITLE_RECOVERED_REL_PERMITTIVITY,
                          COLORBAR_REL_PERMITTIVITY, bounds=bounds,
                          xlabel=xlabel, ylabel=ylabel)

                axes = figure.add_subplot(2, 2, 4)
                add_image(axes, self.sigma,
                          TITLE_RECOVERED_CONDUCTIVITY, COLORBAR_CONDUCTIVITY,
                          bounds=bounds, xlabel=xlabel, ylabel=ylabel)

        if show:
            plt.show()
        else:
            plt.savefig(file_path + self.name + '.' + file_format,
                        format=file_format)
            plt.close()

    def update_error(self, inputdata, scattered_field=None, total_field=None,
                     relative_permittivity_map=None, conductivity_map=None,
                     resolution=None):
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
        if inputdata.compute_residual_error:
            if scattered_field is None and self.es is None:
                raise error.MissingInputError('Result.update_error',
                                              'scattered_field')
            elif inputdata.es is None:
                raise error.MissingAttributesError('InputData', 'es')
            if scattered_field is not None:
                self.zeta_rn.append(compute_zeta_rn(inputdata.es,
                                                    scattered_field))
                self.zeta_rpad.append(compute_zeta_rpad(inputdata.es,
                                                        scattered_field))
            else:
                self.zeta_rn.append(compute_zeta_rn(inputdata.es,
                                                    self.es))
                self.zeta_rpad.append(compute_zeta_rpad(inputdata.es,
                                                        self.es))

        if inputdata.compute_totalfield_error:
            if total_field is None and self.et is None:
                raise error.MissingInputError('Result.update_error',
                                              'total_field')
            elif inputdata.et is None:
                raise error.MissingAttributesError('InputData', 'et')
            if total_field is not None:
                if total_field.shape[0] != inputdata.et.shape[0]:
                    if (resolution is None
                            and relative_permittivity_map is None
                            and conductivity_map is None):
                        raise error.MissingInputError('Results.update_error',
                                                      'resolution')
                    elif (resolution is None
                            and relative_permittivity_map is not None):
                        resolution = relative_permittivity_map.shape
                    elif resolution is None and conductivity_map is not None:
                        resolution = conductivity_map.shape

                et = resize_totalfield(inputdata.et,
                                       inputdata.total_field_resolution,
                                       resolution)

                self.zeta_tfmpad.append(compute_zeta_tfmpad(et,
                                                            total_field))
                self.zeta_tfppad.append(compute_zeta_tfppad(et,
                                                            total_field))
            else:
                if self.et.shape[0] != inputdata.et.shape[0]:
                    if (resolution is None
                            and relative_permittivity_map is None
                            and conductivity_map is None):
                        raise error.MissingInputError('Results.update_error',
                                                      'resolution')
                    elif (resolution is None
                            and relative_permittivity_map is not None):
                        resolution = relative_permittivity_map.shape
                    elif resolution is None and conductivity_map is not None:
                        resolution = conductivity_map.shape

                et = resize_totalfield(inputdata.et,
                                       inputdata.total_field_resolution,
                                       resolution)

                self.zeta_tfmpad.append(compute_zeta_tfmpad(et,
                                                            self.et))
                self.zeta_tfppad.append(compute_zeta_tfppad(et,
                                                            self.et))

        if inputdata.compute_map_error:
            if self.configuration_filename is None:
                raise error.MissingAttributesError('Results',
                                                   'configuration_filename')
            config = cfg.import_dict(self.configuration_filename,
                                     self.configuration_filepath)
            epsilon_rb = config[cfg.BACKGROUND_RELATIVE_PERMITTIVITY]
            sigma_b = config[cfg.BACKGROUND_CONDUCTIVITY]
            omega = 2*np.pi*config[cfg.FREQUENCY]

            if inputdata.epsilon_r is not None:

                if (relative_permittivity_map is None
                        and self.epsilon_r is None):
                    raise error.MissingInputError('Results.update_error',
                                                  'relative_permittivity_map')

                if relative_permittivity_map is not None:

                    if (relative_permittivity_map.shape[0] !=
                            inputdata.epsilon_r.shape[0] 
                            or relative_permittivity_map.shape[1] !=
                            inputdata.epsilon_r.shape[1]):
                        epsilon_r = resize_image(relative_permittivity_map,
                                                 inputdata.epsilon_r.shape)
                    else:
                        epsilon_r = relative_permittivity_map

                elif self.epsilon_r is not None:

                    if (self.epsilon_r.shape[0] != inputdata.epsilon_r.shape[0] 
                            or self.epsilon_r.shape[1] !=
                            inputdata.epsilon_r.shape[1]):
                        epsilon_r = resize_image(self.epsilon_r,
                                                 inputdata.epsilon_r.shape)
                    else:
                        epsilon_r = self.epsilon_r

                self.zeta_epad.append(compute_zeta_epad(inputdata.epsilon_r,
                                                        epsilon_r))

                if inputdata.homogeneous_objects:
                    self.zeta_ebe.append(compute_zeta_ebe(inputdata.epsilon_r,
                                                          epsilon_r,
                                                          epsilon_rb))
                    self.zeta_eoe.append(compute_zeta_eoe(inputdata.epsilon_r,
                                                          epsilon_r,
                                                          epsilon_rb))

            else:
                epsilon_r = epsilon_rb*np.zeros(sigma.shape)

            if inputdata.sigma is not None:

                if conductivity_map is None and self.sigma is None:
                    raise error.MissingInputError('Results.update_error',
                                                  'conductivity_map')
                if conductivity_map is not None:
                    if (conductivity_map.shape[0] != inputdata.sigma.shape[0] 
                            or conductivity_map.shape[1] !=
                            inputdata.sigma.shape[1]):
                        sigma = resize_image(conductivity_map,
                                             inputdata.sigma.shape)
                    else:
                        sigma = conductivity_map
                else:
                    if (self.sigma.shape[0] != inputdata.sigma.shape[0] 
                            or self.sigma.shape[1] !=
                            inputdata.sigma.shape[1]):
                        sigma = resize_image(self.sigma, inputdata.sigma.shape)
                    else:
                        sigma = self.sigma

                self.zeta_sad.append(compute_zeta_sad(inputdata.sigma,
                                                      sigma))

                if inputdata.homogeneous_objects:
                    self.zeta_sbe.append(compute_zeta_sbe(inputdata.sigma,
                                                          sigma, sigma_b))
                    self.zeta_soe.append(compute_zeta_soe(inputdata.sigma,
                                                          sigma, sigma_b))

            else:
                sigma = sigma_b*np.zeros(epsilon_r.shape)

            x, y = cfg.get_coordinates_ddomain(
                configuration=cfg.Configuration(
                    import_filename=self.configuration_filename,
                    import_filepath=self.configuration_filepath
                ), resolution=epsilon_r.shape
            )

            self.zeta_be.append(
                compute_zeta_be(cfg.get_contrast_map(epsilon_r=epsilon_r,
                                                     sigma=sigma,
                                                     epsilon_rb=epsilon_rb,
                                                     sigma_b=sigma_b,
                                                     omega=omega), x, y)
            )

    def last_error_message(self, inputdata, pre_message=None):
        """Summarize the method."""
        if pre_message is not None:
            message = pre_message
        else:
            message = ''

        if inputdata.compute_residual_error:
            message = message + 'Residual norm: %.3e, ' % self.zeta_rn[-1]
            message = message + 'PAD: %.2f%%' % self.zeta_rpad[-1]

        if inputdata.compute_map_error:
            if inputdata.compute_residual_error:
                message = message + ' - '
            if len(self.zeta_epad) != 0:
                message = (message
                           + 'Rel. Per. PAD: %.2f%%' % self.zeta_epad[-1])
                if inputdata.homogeneous_objects:
                    message = message + ', Back.: %.2f%%, ' % self.zeta_ebe[-1]
                    message = message + 'Ob.: %.2f%%' % self.zeta_eoe[-1]
            if len(self.zeta_sad) != 0:
                if len(self.zeta_epad) != 0:
                    message = message + ' - '
                message = message + 'Con. AD: %.3e' % self.zeta_sad[-1]
                if inputdata.homogeneous_objects:
                    message = message + ' Back.: %.3e,' % self.zeta_sbe[-1]
                    message = message + 'Ob.: %.3e' % self.zeta_soe[-1]
            if inputdata.homogeneous_objects:
                message = message + ' - Bound.: %.3e' % self.zeta_be[-1]

        if inputdata.compute_totalfield_error:
            if inputdata.compute_residual_error or inputdata.compute_map_error:
                message = message + ' - '
            message = (message
                       + 'To. Field Mag. PAD: %.2f%%' % self.zeta_tfmpad[-1])
            message = (message
                       + ' To. Field Phase PAD: %.2f%%' % self.zeta_tfppad[-1])

        return message

    def plot_convergence(self, show=False, file_path='', file_format='eps'):
        """Summarize the method."""
        number_plots = 0
        if len(self.zeta_be) > 0:
            number_plots += 1
        if len(self.zeta_ebe) > 0:
            number_plots += 1
        if len(self.zeta_eoe) > 0:
            number_plots += 1
        if len(self.zeta_epad) > 0:
            number_plots += 1
        if len(self.zeta_rn) > 0:
            number_plots += 1
        if len(self.zeta_rpad) > 0:
            number_plots += 1
        if len(self.zeta_sad) > 0:
            number_plots += 1
        if len(self.zeta_sbe) > 0:
            number_plots += 1
        if len(self.zeta_soe) > 0:
            number_plots += 1
        if len(self.zeta_tfmpad) > 0:
            number_plots += 1
        if len(self.zeta_tfppad) > 0:
            number_plots += 1
        nrows = int(np.sqrt(number_plots))
        ncols = int(np.ceil(number_plots/nrows))
        image_size = (5.+2*ncols, 5.+1*nrows)
        figure = plt.figure(figsize=image_size)
        figure.subplots_adjust(wspace=.5, hspace=.5)
        i = 1
        if len(self.zeta_be) > 0:
            axes = figure.add_subplot(nrows, ncols, i)
            self.plot_boundary_error(axes=axes)
            i += 1
        if len(self.zeta_ebe) > 0:
            axes = figure.add_subplot(nrows, ncols, i)
            self.plot_relpermittivity_be(axes=axes)
            i += 1
        if len(self.zeta_eoe) > 0:
            axes = figure.add_subplot(nrows, ncols, i)
            self.plot_relpermittivity_oe(axes=axes)
            i += 1
        if len(self.zeta_epad) > 0:
            axes = figure.add_subplot(nrows, ncols, i)
            self.plot_relpermittivity_pad(axes=axes)
            i += 1
        if len(self.zeta_rn) > 0:
            axes = figure.add_subplot(nrows, ncols, i)
            self.plot_residual_norm(axes=axes)
            i += 1
        if len(self.zeta_rpad) > 0:
            axes = figure.add_subplot(nrows, ncols, i)
            self.plot_residual_pad(axes=axes)
            i += 1
        if len(self.zeta_sad) > 0:
            axes = figure.add_subplot(nrows, ncols, i)
            self.plot_conductivity_ad(axes=axes)
            i += 1
        if len(self.zeta_sbe) > 0:
            axes = figure.add_subplot(nrows, ncols, i)
            self.plot_conductivity_be(axes=axes)
            i += 1
        if len(self.zeta_soe) > 0:
            axes = figure.add_subplot(nrows, ncols, i)
            self.plot_conductivity_oe(axes=axes)
            i += 1
        if len(self.zeta_tfmpad) > 0:
            axes = figure.add_subplot(nrows, ncols, i)
            self.plot_totalfield_mpad(axes=axes)
            i += 1
        if len(self.zeta_tfppad) > 0:
            axes = figure.add_subplot(nrows, ncols, i)
            self.plot_totalfield_ppad(axes=axes)
            i += 1
        if show:
            plt.show()
        else:
            plt.savefig(file_path + self.name + '_convergence.' + file_format,
                        format=file_format)
            plt.close()

    def plot_residual_norm(self, show=False, file_path='', file_format='eps',
                           axes=None):
        """Summarize the method."""
        if len(self.zeta_rn) == 0:
            raise error.EmptyAttribute('Results', 'zeta_rn')
        if axes is None:
            figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
            axes = get_single_figure_axes(figure)
            single_plot = True
        else:
            single_plot = False

        add_plot(axes, self.zeta_rn, title='Residual Norm Error',
                 ylabel=LABEL_ZETA_RN)

        if single_plot:
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_zeta_rn.' + file_format,
                            format=file_format)
                plt.close()
        else:
            return axes

    def plot_residual_pad(self, show=False, file_path='', file_format='eps',
                          axes=None):
        """Summarize the method."""
        if len(self.zeta_rpad) == 0:
            raise error.EmptyAttribute('Results', 'zeta_rpad')
        if axes is None:
            figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
            axes = get_single_figure_axes(figure)
            single_plot = True
        else:
            single_plot = False

        add_plot(axes, self.zeta_rpad, title='Residual PAD Error',
                 ylabel=LABEL_ZETA_RPAD)

        if single_plot:
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_zeta_rpad.'
                            + file_format, format=file_format)
                plt.close()
        else:
            return axes

    def plot_relpermittivity_pad(self, show=False, file_path='',
                                 file_format='eps', axes=None):
        """Summarize the method."""
        if len(self.zeta_epad) == 0:
            raise error.EmptyAttribute('Results', 'zeta_epad')
        if axes is None:
            figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
            axes = get_single_figure_axes(figure)
            single_plot = True
        else:
            single_plot = False

        add_plot(axes, self.zeta_epad, title='Rel. Permittivity PAD Error',
                 ylabel=LABEL_ZETA_EPAD)

        if single_plot:
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_zeta_epad.' + file_format,
                            format=file_format)
                plt.close()
        else:
            return axes

    def plot_relpermittivity_be(self, show=False, file_path='',
                                file_format='eps', axes=None):
        """Summarize the method."""
        if len(self.zeta_ebe) == 0:
            raise error.EmptyAttribute('Results', 'zeta_ebe')
        if axes is None:
            figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
            axes = get_single_figure_axes(figure)
            single_plot = True
        else:
            single_plot = False

        add_plot(axes, self.zeta_ebe, ylabel=LABEL_ZETA_EBE,
                 title='Rel. Permittivity Background Error')

        if single_plot:
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_zeta_ebe.' + file_format,
                            format=file_format)
                plt.close()
        else:
            return axes

    def plot_relpermittivity_oe(self, show=False, file_path='',
                                file_format='eps', axes=None):
        """Summarize the method."""
        if len(self.zeta_eoe) == 0:
            raise error.EmptyAttribute('Results', 'zeta_eoe')
        if axes is None:
            figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
            axes = get_single_figure_axes(figure)
            single_plot = True
        else:
            single_plot = False

        add_plot(axes, self.zeta_eoe, ylabel=LABEL_ZETA_EOE,
                 title='Rel. Permittivity Object Error')

        if single_plot:
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_zeta_eoe.' + file_format,
                            format=file_format)
                plt.close()
        else:
            return axes

    def plot_conductivity_ad(self, show=False, file_path='',
                             file_format='eps', axes=None):
        """Summarize the method."""
        if len(self.zeta_sad) == 0:
            raise error.EmptyAttribute('Results', 'zeta_sad')
        if axes is None:
            figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
            axes = get_single_figure_axes(figure)
            single_plot = True
        else:
            single_plot = False

        add_plot(axes, self.zeta_sad, ylabel=LABEL_ZETA_SAD,
                 title='Conductivity AD Error')

        if single_plot:
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_zeta_sad.' + file_format,
                            format=file_format)
                plt.close()
        else:
            return axes

    def plot_conductivity_be(self, show=False, file_path='',
                             file_format='eps', axes=None):
        """Summarize the method."""
        if len(self.zeta_sbe) == 0:
            raise error.EmptyAttribute('Results', 'zeta_sbe')
        if axes is None:
            figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
            axes = get_single_figure_axes(figure)
            single_plot = True
        else:
            single_plot = False

        add_plot(axes, self.zeta_sbe, ylabel=LABEL_ZETA_SBE,
                 title='Conductivity Background Error')

        if single_plot:
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_zeta_sbe.' + file_format,
                            format=file_format)
                plt.close()
        else:
            return axes

    def plot_conductivity_oe(self, show=False, file_path='',
                             file_format='eps', axes=None):
        """Summarize the method."""
        if len(self.zeta_soe) == 0:
            raise error.EmptyAttribute('Results', 'zeta_soe')
        if axes is None:
            figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
            axes = get_single_figure_axes(figure)
            single_plot = True
        else:
            single_plot = False

        add_plot(axes, self.zeta_soe, ylabel=LABEL_ZETA_SOE,
                 title='Conductivity Object Error')

        if single_plot:
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_zeta_soe.' + file_format,
                            format=file_format)
                plt.close()
        else:
            return axes

    def plot_boundary_error(self, show=False, file_path='', file_format='eps',
                            axes=None):
        """Summarize the method."""
        if len(self.zeta_be) == 0:
            raise error.EmptyAttribute('Results', 'zeta_be')
        if axes is None:
            figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
            axes = get_single_figure_axes(figure)
            single_plot = True
        else:
            single_plot = False

        add_plot(axes, self.zeta_be, ylabel=LABEL_ZETA_BE,
                 title='Boundary Error')

        if single_plot:
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_zeta_be.' + file_format,
                            format=file_format)
                plt.close()
        else:
            return axes

    def plot_totalfield_mpad(self, show=False, file_path='', file_format='eps',
                             axes=None):
        """Summarize the method."""
        if len(self.zeta_tfmpad) == 0:
            raise error.EmptyAttribute('Results', 'zeta_tfmpad')
        if axes is None:
            figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
            axes = get_single_figure_axes(figure)
            single_plot = True
        else:
            single_plot = False

        add_plot(axes, self.zeta_tfmpad, ylabel=LABEL_ZETA_TFMPAD,
                 title='Total Field Mag. PAD Error')

        if single_plot:
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_zeta_tfmpad.'
                            + file_format, format=file_format)
                plt.close()
        else:
            return axes

    def plot_totalfield_ppad(self, show=False, file_path='', file_format='eps',
                             axes=None):
        """Summarize the method."""
        if len(self.zeta_tfppad) == 0:
            raise error.EmptyAttribute('Results', 'zeta_tfppad')
        if axes is None:
            figure = plt.figure(figsize=IMAGE_SIZE_SINGLE)
            axes = get_single_figure_axes(figure)
            single_plot = True
        else:
            single_plot = False

        add_plot(axes, self.zeta_tfppad, ylabel=LABEL_ZETA_TFPPAD,
                 title='Total Field Phase PAD Error')

        if single_plot:
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_zeta_tfppad.'
                            + file_format, format=file_format)
                plt.close()
        else:
            return axes

    def __str__(self):
        """Print object information."""
        message = 'Results name: ' + self.name
        message = (message + '\nConfiguration filename: '
                   + self.configuration_filename)
        if self.configuration_filepath is not None:
            message = (message + '\nConfiguration file path: '
                       + self.configuration_filepath)
        if self.input_filename is not None:
            message = (message + '\nInput file name: '
                       + self.input_filename)
        if self.input_filepath is not None:
            message = (message + '\nInput file path: '
                       + self.input_filepath)
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
        if self.epsilon_r is not None:
            message = (message + '\nRelative Permit. map resolution: %dx'
                       % self.epsilon_r.shape[0] + '%d'
                       % self.epsilon_r.shape[1])
        if self.sigma is not None:
            message = (message + '\nConductivity map resolution: %dx'
                       % self.sigma.shape[0] + '%d'
                       % self.sigma.shape[1])
        if self.execution_time is not None:
            print('Execution time: %.2f [sec]' % self.execution_time)
        if len(self.zeta_rn) > 0:
            if len(self.zeta_rn) == 1:
                info = '%.3e' % self.zeta_rn[0]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_rn) + ']')
            message = message + '\nResidual norm error: ' + info
        if len(self.zeta_rpad) > 0:
            if len(self.zeta_rpad) == 1:
                info = '%.2f%%' % self.zeta_rpad[0]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_rpad) + ']')
            message = message + '\nPercent. Aver. Devi. of Residuals: ' + info
        if len(self.zeta_epad) > 0:
            if len(self.zeta_epad) == 1:
                info = '%.2f%%' % self.zeta_epad[0]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_epad) + ']')
            message = (message + '\nPercent. Aver. Devi. of Rel. Permittivity:'
                       + ' ' + info)
        if len(self.zeta_sad) > 0:
            if len(self.zeta_sad) == 1:
                info = '%.3e' % self.zeta_sad[0]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_sad) + ']')
            message = (message + '\nAver. Devi. of Conductivity: '
                       + info)
        if len(self.zeta_be) > 0:
            if len(self.zeta_be) == 1:
                info = '%.3e' % self.zeta_be[0]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_be) + ']')
            message = message + '\nBoundary error: ' + info
        if len(self.zeta_ebe) > 0:
            if len(self.zeta_ebe) == 1:
                info = '%.2f%%' % self.zeta_ebe[0]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_ebe) + ']')
            message = message + '\nBackground Rel. Permit. error: ' + info
        if len(self.zeta_sbe) > 0:
            if len(self.zeta_sbe) == 1:
                info = '%.3e' % self.zeta_sbe[0]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_sbe) + ']')
            message = message + '\nBackground Conductivity error: ' + info
        if len(self.zeta_eoe) > 0:
            if len(self.zeta_eoe) == 1:
                info = '%.2f%%' % self.zeta_eoe[0]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_eoe) + ']')
            message = message + '\nObject Rel. Permit. error: ' + info
        if len(self.zeta_soe) > 0:
            if len(self.zeta_soe) == 1:
                info = '%.3e' % self.zeta_soe[0]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_soe) + ']')
            message = message + '\nObject Conduc. error: ' + info
        if len(self.zeta_tfmpad) > 0:
            if len(self.zeta_tfmpad) == 1:
                info = '%.2f%%' % self.zeta_tfmpad[0]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_tfmpad) + ']')
            message = (message + '\nTotal Field Mag. Per. Aver. Devi. error: '
                       + info)
        if len(self.zeta_tfppad) > 0:
            if len(self.zeta_tfppad) == 1:
                info = '%.2f%%' % self.zeta_tfppad[0]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_tfppad) + ']')
            message = (message + '\nTotal Field Phase Per. Aver. Devi. error:'
                       + ' ' + info)
        return message


def resize_totalfield(et, original_resolution, new_resolution):
    I1, J1 = original_resolution
    I2, J2 = new_resolution
    NS = et.shape[1]
    x, y = np.arange(.5/I1, 1., 1/I1), np.arange(.5/J1, 1., 1/J1)
    w, z = np.arange(.5/I2, 1., 1/I2), np.arange(.5/J2, 1., 1/J2)
    new_et = np.zeros((I2*J2, NS), dtype=complex)
    for ns in range(NS):
        freal = interp2d(x, y, np.real(et[:, ns].reshape((J1, I1))))
        fimag = interp2d(x, y, np.imag(et[:, ns].reshape((J1, I1))))
        new_et[:, ns] = np.reshape(freal(w, z) + fimag(w, z), -1)
    return new_et


def resize_image(image, new_resolution):
    I1, J1 = image.shape
    I2, J2 = new_resolution
    x, y = np.arange(.5/I1, 1., 1/I1), np.arange(.5/J1, 1., 1/J1)
    w, z = np.arange(.5/I2, 1., 1/I2), np.arange(.5/J2, 1., 1/J2)
    f = interp2d(x, y, image)
    new_image = f(w, z)
    return new_image


def add_image(axes, image, title, colorbar_name, bounds=(-1., 1., -1., 1.),
              origin='lower', xlabel=XLABEL_STANDARD, ylabel=YLABEL_STANDARD,
              aspect='equal', interpolation=None):
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
                         extent=[bounds[0], bounds[1],
                                 bounds[2], bounds[3]],
                         origin=origin, aspect=aspect,
                         interpolation=interpolation)
    else:
        im = axes.imshow(image,
                         extent=[bounds[0], bounds[1],
                                 bounds[2], bounds[3]],
                         origin=origin, aspect=aspect,
                         interpolation=interpolation)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    cbar = plt.colorbar(ax=axes, mappable=im, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_name)


def add_plot(axes, data, x=None, title=None, xlabel='Iterations', ylabel=None,
             style='--*', xticks=None):
    """Add a plot to the axes.

    A predefined function for plotting curves. This is useful for
    standardize plots involving convergence data.

    Paramaters
    ----------
        axes : :class:`matplotlib.pyplot.Figure.axes.Axes`
            The axes object.

        data : :class:`numpy.ndarray`
            The y-data.

        x : :class:`numpy.ndarray`, default: None
            The x-data.

        title : string, default: None
            The title to be displayed in the plot.

        xlabel : string, default: 'Iterations'
            The label of the x-axis.

        ylabel : string, default: None
            The label of the y-axis.

        style : string, default: '--*'
            The style of the curve (line, marker, color).
    """
    if x is None:
        if type(data) is list:
            length = len(data)
        else:
            length = data.size
        x = np.arange(1, length+1)

    axes.plot(x, data, style)
    axes.set_xlabel(xlabel)
    if xticks is not None:
        axes.set_xticks(xticks)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if title is not None:
        axes.set_title(title)
    axes.grid()


def set_subplot_size(figure):
    """Set subplot sizes.

    A standard definition for setting images in subplot figures.

    Parameters
    ----------
        figure : `:class:matplotlib.pyplot.Figure`
            A figure object.
    """
    figure.subplots_adjust(left=.125, bottom=-.3, right=.9, top=1.3, wspace=.7,
                           hspace=-.7)


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
    return np.real(np.sqrt(np.trapz(np.trapz(y, x=phi), x=theta)))


def compute_rre(es_o, es_a):
    """Compute the Relative Residual Error (RRE).

    The RRE is a definition found in [1] and it is useful for
    determining the parameter of Tikhonov regularization.

    Parameters
    ----------
        es_o : :class:`numpy.ndarray`
            Original scattered field matrix.

        es_a : :class:`numpy.ndarray`
            Approximated scattered field matrix.

    References
    ----------
    .. [1] Lavarello, Roberto, and Michael Oelze. "A study on the
           reconstruction of moderate contrast targets using the
           distorted Born iterative method." IEEE transactions on
           ultrasonics, ferroelectrics, and frequency control 55.1
           (2008): 112-124.
    """
    return compute_zeta_rn(es_o, es_a)/compute_zeta_rn(es_o,
                                                       np.zeros(es_o.shape,
                                                                dtype=complex))


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
    return np.mean(np.abs((y-yd)/y))*100


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
    return np.mean(np.abs((y-yd)/y))*100


def compute_zeta_sad(sigma_o, sigma_r):
    """Compute the average deviation of conductivity map.

    The zeta_epad error is the evaluation of the conductivity
    estimation error per pixel.

    Parameters
    ----------
        sigma_o, sigma_r : :class:`numpy.ndarray`
            Original and recovered conductivity maps, respectively.
    """
    y = sigma_o.flatten()
    yd = sigma_r.flatten()
    return np.mean(np.abs((y-yd)))*100


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
    return np.mean(np.abs(y-yd)/y)*100


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
    return np.mean(np.abs(y-yd))


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
    return np.mean(np.abs(y-yd)/y)*100


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
    return np.mean(np.abs(y-yp))


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
    return np.mean(np.abs((y-yd)/yd))*100


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
    return np.mean(np.abs((y-yd)/yd))*100
