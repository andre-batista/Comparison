"""Experiments Module

This module is intended to provide tools to analyse the perfomance of
solvers. According to the definition of some parameters, simulations may
be carried out and there are tools for statistical studies.

This module provides:

    :class:`Experiment`
        A container for joining methods, inputs and configurations for
        statistical analysis of performance.
    :func:`create_scenario`
        A routine to create random scenarios for experiments.
    :func:`contrast_density`
        Evaluate the contrast density of a given map.
    :func:`isleft`
        Determine if a point is on the left of a line.
    :func:`winding_number`
        Determine if a point is inside a polygon.

A set of routines for drawing geometric figures is provided:

    :func:`draw_triangle`
    :func:`draw_square`
    :func:`draw_rhombus`
    :func:`draw_trapezoid`
    :func:`draw_parallelogram`
    :func:`draw_4star`
    :func:`draw_5star`
    :func:`draw_6star`
    :func:`draw_circle`
    :func:`draw_ring`
    :func:`draw_ellipse`
    :func:`draw_cross`
    :func:`draw_line`
    :func:`draw_polygon`
    :func:`draw_random`

A set of routines for defining surfaces is also provided:

    :func:`draw_wave`
    :func:`draw_random_waves`
    :func:`draw_random_gaussians`
"""

# Standard libraries
import pickle
import copy as cp
import numpy as np
from numpy import random as rnd
from numpy import pi, logical_and
import time as tm
from joblib import Parallel, delayed
import multiprocessing
from matplotlib import pyplot as plt
from statsmodels import api as sm
from statsmodels import stats
from statsmodels.stats import oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels import sandbox as snd
import scipy
import pingouin as pg
import warnings
from numba import jit
from scipy.optimize import curve_fit

# Developed libraries
import error
import configuration as cfg
import inputdata as ipt
import solver as slv
import results as rst
import forward as frw
import mom_cg_fft as mom

# Constants
STANDARD_SYNTHETIZATION_RESOLUTION = 25
STANDARD_RECOVER_RESOLUTION = 20
GEOMETRIC_PATTERN = 'geometric'
SURFACES_PATTERN = 'surfaces'
LABEL_INSTANCE = 'Instance Index'

# PICKLE DICTIONARY STRINGS
NAME = 'name'
CONFIGURATIONS = 'configurations'
SCENARIOS = 'scenarios'
METHODS = 'methods'
MAXIMUM_CONTRAST = 'maximum_contrast'
MAXIMUM_OBJECT_SIZE = 'maximum_object_size'
MAXIMUM_CONTRAST_DENSITY = 'maximum_contrast_density'
NOISE = 'noise'
MAP_PATTERN = 'map_pattern'
SAMPLE_SIZE = 'sample_size'
SYNTHETIZATION_RESOLUTION = 'synthetization_resolution'
RECOVER_RESOLUTION = 'recover_resolution'
FORWARD_SOLVER = 'forward_solver'
STUDY_RESIDUAL = 'study_residual'
STUDY_MAP = 'study_map'
STUDY_INTERNFIELD = 'study_internfield'
RESULTS = 'results'


class Experiment:
    """Experiments container.

    Define and execute an experiment with methods as well as analyses
    its results.

    An experiment has three parameters: maximum contrast allowed,
    maximum length allowed of objects and maximum contrast density in
    the image. These parameters were thought as effect factors on the
    performance of the methods. Then they need to be fixed for running
    statistical analyses.

    Attributes
    ----------
        name : str
            A name for the experiment.
        maximum_contrast : list
            A list with maximum contrast values allowed in the
            experiments.
        maximum_object_size : list
            A list with maximum values of the size of objects.
        maximum_contrast_density : list
            A list with the maximum value of contrast density.
        map_pattern : {'geometric', 'surfaces'}
            Define the kind of contrast pattern in the image.
        sample_size : int
            Number of scenarios for experiments.
        synthetization_resolution : 2-tuple
            Synthetization image resolution.
        recover_resoluton : 2-tuple
            Recovered image resolution.
        configurations : list
            List of objects of Configuration class.
        scenarios : list
            Instances which will be considered.
        methods : list
            Set of solvers.
        results : list
            List of outputs of executions.
        forward_solver : :class:`forward.Forward`
            An object of forward solver for synthetizing data.
    """

    @property
    def configurations(self):
        """Get the configurations list."""
        return self._configurations

    @configurations.setter
    def configurations(self, configurations):
        """Set the configurations attribute.

        There are three options to set this attribute:

        >>> self.configurations = cfg.Configuration
        >>> self.configurations = [cfg.Configuration, cfg.Configuration]
        >>> self.configurations = None
        """
        if type(configurations) is cfg.Configuration:
            self._configurations = [cp.deepcopy(configurations)]
        elif type(configurations) is list:
            self._configurations = cp.deepcopy(configurations)
        else:
            self._configurations = None

    @property
    def scenarios(self):
        """Get the scenario list."""
        return self._scenarios

    @scenarios.setter
    def scenarios(self, new_scenario):
        """Set the scenarios attribute.

        There are three options to set this attribute:

        >>> self.scenarios = ipt.InputData
        >>> self.scenarios = [ipt.InputData, ipt.InputData]
        >>> self.scenarios = None
        """
        if type(new_scenario) is ipt.InputData:
            self._scenarios = [cp.deepcopy(new_scenario)]
        elif type(new_scenario) is list:
            self._scenarios = cp.deepcopy(new_scenario)
        else:
            self._scenarios = None

    @property
    def methods(self):
        """Get the list of methods."""
        return self._methods

    @methods.setter
    def methods(self, methods):
        """Set the methods attribute.

        There are three options to set this attribute:

        >>> self.methods = slv.Solver
        >>> self.methods = [slv.Solver, slv.Solver]
        >>> self.methods = None
        """
        if type(methods) is slv.Solver:
            self._methods = [cp.deepcopy(methods)]
        elif type(methods) is list:
            self._methods = cp.deepcopy(methods)
        else:
            self._methods = None

    @property
    def maximum_contrast(self):
        """Get the list of maximum contrast values."""
        return self._maximum_contrast

    @maximum_contrast.setter
    def maximum_contrast(self, maximum_contrast):
        """Set the maximum contrast attribute.

        There are three options to set this attribute:

        >>> self.maximum_contrast = float()
        >>> self.maximum_contrast = complex()
        >>> self.maximum_contrast = [complex(), complex()]
        """
        if type(maximum_contrast) is float:
            self._maximum_contrast = [maximum_contrast + 0j]
        elif type(maximum_contrast) is complex:
            self._maximum_contrast = [maximum_contrast]
        elif type(maximum_contrast) is list:
            self._maximum_contrast = list.copy(maximum_contrast)

    @property
    def maximum_object_size(self):
        """Get the list of maximum value of objects sizes."""
        return self._maximum_object_size

    @maximum_object_size.setter
    def maximum_object_size(self, maximum_object_size):
        """Set the maximum value of objects sizes.

        There are two options to set this attribute:

        >>> self.maximum_contrast = float()
        >>> self.maximum_contrast = [float(), float()]
        """
        if type(maximum_object_size) is float:
            self._maximum_object_size = [maximum_object_size]
        elif type(maximum_object_size) is list:
            self._maximum_object_size = list.copy(maximum_object_size)

    @property
    def maximum_contrast_density(self):
        """Get the list of maximum values of contrast density."""
        return self._maximum_average_contrast

    @maximum_contrast_density.setter
    def maximum_contrast_density(self, maximum_contrast_density):
        """Set the maximum value of contrast density.

        There are three options to set this attribute:

        >>> self.maximum_contrast = float()
        >>> self.maximum_contrast = complex()
        >>> self.maximum_contrast = [complex(), complex()]
        """
        if type(maximum_contrast_density) is float:
            self._maximum_average_contrast = [maximum_contrast_density + 0j]
        elif type(maximum_contrast_density) is complex:
            self._maximum_average_contrast = [maximum_contrast_density]
        elif type(maximum_contrast_density) is list:
            self._maximum_average_contrast = list.copy(
                maximum_contrast_density
            )

    @property
    def noise(self):
        """Get the list of noise."""
        return self._noise

    @noise.setter
    def noise(self, value):
        """Set the noise level.

        There are three options to set this attribute:

        >>> self.noise = float()
        >>> self.noise = [float(), ...]
        >>> self.noise = None
        """
        if type(value) is float:
            self._noise = [value]
        elif type(value) is list:
            self._noise = value
        elif value is None:
            self._noise = [0.]
        else:
            self._noise = None

    @property
    def map_pattern(self):
        """Get the map pattern."""
        return self._map_pattern

    @map_pattern.setter
    def map_pattern(self, map_pattern):
        """Set the map pattern."""
        if map_pattern == GEOMETRIC_PATTERN or map_pattern == SURFACES_PATTERN:
            self._map_pattern = map_pattern
        else:
            raise error.WrongValueInput('Experiment', 'map_pattern',
                                        GEOMETRIC_PATTERN + ' or '
                                        + SURFACES_PATTERN, map_pattern)

    def __init__(self, name=None, maximum_contrast=None,
                 maximum_object_size=None, maximum_contrast_density=None,
                 map_pattern=None, sample_size=30,
                 synthetization_resolution=None, recover_resolution=None,
                 configurations=None, scenarios=None, methods=None,
                 forward_solver=None, noise=None, study_residual=True,
                 study_map=False, study_internfield=False,
                 import_filename=None, import_filepath=''):
        """Create the experiment object.

        The object should be defined with one of the following
        possibilities of combination of parameters (maximum_contrast,
        maximum_object_size, maximum_contrast_density): (i) all are
        single values; (ii) one is list and the others are single
        values; and (iii) all are list of same size.

        Parameters
        ----------
            name : str
                The name of the experiment.
            maximum_contrast : float or complex of list
                The
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
            return

        if name is None:
            raise error.MissingInputError('Experiment.__init__', 'name')
        elif maximum_contrast is None:
            raise error.MissingInputError('Experiment.__init__',
                                          'maximum_contrast')
        elif maximum_object_size is None:
            raise error.MissingInputError('Experiment.__init__',
                                          'maximum_object_size')
        elif maximum_contrast_density is None:
            raise error.MissingInputError('Experiment.__init__',
                                          'maximum_contrast_density')
        elif map_pattern is None:
            raise error.MissingInputError('Experiment.__init__', 'map_pattern')

        self.name = name
        self.maximum_contrast = maximum_contrast
        self.maximum_object_size = maximum_object_size
        self.maximum_contrast_density = maximum_contrast_density
        self.noise = noise
        self.map_pattern = map_pattern
        self.sample_size = sample_size
        self.synthetization_resolution = synthetization_resolution
        self.recover_resolution = recover_resolution
        self.configurations = configurations
        self.scenarios = scenarios
        self.methods = methods
        self.forward_solver = forward_solver
        self.study_residual = study_residual
        self.study_map = study_map
        self.study_internfield = study_internfield
        self.results = None

        # Enforcing that all experimentation parameters are of same length
        if (len(self.maximum_contrast) == len(self.maximum_object_size)
                and len(self.maximum_object_size)
                == len(self.maximum_contrast_density)
                and len(self.maximum_contrast_density) == len(self.noise)):
            pass
        elif (len(self.maximum_contrast) > 1
                and len(self.maximum_object_size) == 1
                and len(self.maximum_contrast_density) == 1
                and len(self.noise) == 1):
            N = len(self.maximum_contrast)
            self.maximum_object_size = N * self.maximum_object_size
            self.maximum_contrast_density = N * self.maximum_contrast_density
            self.noise = N * self.noise
        elif (len(self.maximum_contrast) == 1
                and len(self.maximum_object_size) > 1
                and len(self.maximum_contrast_density) == 1
                and len(self.noise) == 1):
            N = len(self.maximum_object_size)
            self.maximum_contrast = N * self.maximum_contrast
            self.maximum_contrast_density = N * self.maximum_contrast_density
            self.noise = N * self.noise
        elif (len(self.maximum_contrast) == 1
                and len(self.maximum_object_size) == 1
                and len(self.maximum_contrast_density) > 1
                and len(self.noise) == 1):
            N = len(self.maximum_contrast_density)
            self.maximum_contrast = N*self.maximum_contrast
            self.maximum_object_size = N*self.maximum_object_size
            self.noise = N * self.noise
        elif (len(self.maximum_contrast) == 1
                and len(self.maximum_object_size) == 1
                and len(self.maximum_contrast_density) == 1
                and len(self.noise) > 1):
            N = len(self.noise)
            self.maximum_contrast = N*self.maximum_contrast
            self.maximum_object_size = N*self.maximum_object_size
            self.maximum_contrast_density = N * self.maximum_contrast_density
        else:
            raise error.WrongValueInput('Experiment.__init__',
                                        'maximum_contrast and ' +
                                        'maximum_object_size and ' +
                                        'maximum_contrast_density',
                                        'all float/complex or ' +
                                        'one list and float/complex',
                                        'More than one are list')

    def run(self, configurations=None, scenarios=None, methods=None):
        """Summarize the method."""
        if self.configurations is None and configurations is None:
            raise error.MissingInputError('Experiment.run', 'configurations')
        elif configurations is not None:
            self.configurations = configurations
        if self.methods is None and methods is None:
            raise error.MissingInputError('Experiment.run', 'methods')
        elif methods is not None:
            self.methods = methods
        if scenarios is not None:
            self.scenarios = scenarios

        if self.synthetization_resolution is None:
            self.define_synthetization_resolution()

        if self.recover_resolution is None:
            self.define_recover_resolution()

        if self.scenarios is None:
            self.randomize_scenarios(self.synthetization_resolution)

        if self.forward_solver is None:
            self.forward_solver = mom.MoM_CG_FFT(self.configurations[0])

        self.synthesize_scattered_field()

        self.solve_scenarios()

    def define_synthetization_resolution(self):
        """Set synthetization resolution attribute."""
        if self.maximum_contrast is None:
            raise error.MissingAttributesError('Experiment',
                                               'maximum_contrast')
        if self.configurations is None or len(self.configurations) == 0:
            raise error.MissingAttributesError('Experiment',
                                               'configurations')
        self.synthetization_resolution = []
        for i in range(len(self.maximum_contrast)):
            self.synthetization_resolution.append(list())
            for j in range(len(self.configurations)):
                epsilon_rd = cfg.get_relative_permittivity(
                    self.maximum_contrast[i],
                    self.configurations[j].epsilon_rb
                )
                sigma_d = cfg.get_conductivity(
                    self.maximum_contrast[i],
                    self.configurations[j].epsilon_rb,
                    2*pi*self.configurations[j].f,
                    self.configurations[j].sigma_b
                )
                lam_d = cfg.compute_wavelength(self.configurations[j].f,
                                               epsilon_r=epsilon_rd,
                                               sigma=sigma_d)
                resolution = compute_resolution(
                    lam_d, self.configurations[j].Ly,
                    self.configurations[j].Lx,
                    STANDARD_SYNTHETIZATION_RESOLUTION
                )
                self.synthetization_resolution[i].append(resolution)

    def define_recover_resolution(self):
        """Set recover resolution variable."""
        if self.maximum_contrast is None:
            raise error.MissingAttributesError('Experiment',
                                               'maximum_contrast')
        if self.configurations is None or len(self.configurations) == 0:
            raise error.MissingAttributesError('Experiment',
                                               'configurations')
        self.recover_resolution = []
        for i in range(len(self.maximum_contrast)):
            self.recover_resolution.append(list())
            for j in range(len(self.configurations)):
                epsilon_rd = cfg.get_relative_permittivity(
                    self.maximum_contrast[i],
                    self.configurations[j].epsilon_rb
                )
                sigma_d = cfg.get_conductivity(
                    self.maximum_contrast[i],
                    self.configurations[j].epsilon_rb,
                    2*pi*self.configurations[j].f,
                    self.configurations[j].sigma_b
                )
                lam_d = cfg.compute_wavelength(self.configurations[j].f,
                                               epsilon_r=epsilon_rd,
                                               sigma=sigma_d)
                resolution = compute_resolution(
                    lam_d, self.configurations[j].Ly,
                    self.configurations[j].Lx,
                    STANDARD_RECOVER_RESOLUTION
                )
                self.recover_resolution[i].append(resolution)

    def randomize_scenarios(self, resolution=None):
        """Create random scenarios."""
        if self.maximum_contrast is None:
            raise error.MissingAttributesError('Experiment',
                                               'maximum_contrast')
        if self.configurations is None or len(self.configurations) == 0:
            raise error.MissingAttributesError('Experiment',
                                               'configurations')
        if self.sample_size is None:
            raise error.MissingAttributesError('Experiment',
                                               'configurations')
        if resolution is None and self.synthetization_resolution is None:
            raise error.MissingAttributesError('Experiment',
                                               'configurations')
        if resolution is None:
            resolution = self.synthetization_resolution
        self.scenarios = []
        for i in range(len(self.maximum_contrast)):
            self.scenarios.append(list())
            for j in range(len(self.configurations)):
                self.scenarios[i].append(list())
                num_cores = multiprocessing.cpu_count()
                output = Parallel(n_jobs=num_cores)(delayed(create_scenario)(
                    'rand' + '%d' % i + '%d' % j + '%d' % k,
                    self.configurations[j], resolution[i][j], self.map_pattern,
                    self.maximum_contrast[i], self.maximum_contrast_density[i],
                    maximum_object_size=self.maximum_object_size[i],
                    noise=self.noise[i],
                    compute_residual_error=self.study_residual,
                    compute_map_error=self.study_map,
                    compute_totalfield_error=self.study_internfield
                ) for k in range(self.sample_size))
                for k in range(self.sample_size):
                    new_scenario = output[k]
                    self.scenarios[i][j].append(cp.deepcopy(new_scenario))

    def synthesize_scattered_field(self, PRINT_INFO=True):
        if self.maximum_contrast is None:
            raise error.MissingAttributesError('Experiment',
                                               'maximum_contrast')
        if self.configurations is None or len(self.configurations) == 0:
            raise error.MissingAttributesError('Experiment', 'configurations')
        if self.sample_size is None:
            raise error.MissingAttributesError('Experiment', 'sample_size')
        if self.forward_solver is None:
            self.forward_solver = mom.MoM_CG_FFT(self.configurations[0])
        if self.scenarios is None:
            raise error.MissingAttributesError('Experiment', 'scenarios')
        if self.study_internfield:
            SAVE_INTERN_FIELD = True
        else:
            SAVE_INTERN_FIELD = False
        N = (len(self.maximum_contrast)*len(self.configurations)
             * self.sample_size)
        n = 0
        for i in range(len(self.maximum_contrast)):
            for j in range(len(self.configurations)):
                self.forward_solver.configuration = self.configurations[j]
                for k in range(self.sample_size):
                    if PRINT_INFO:
                        print('Synthesizing scattered field: %d' % (n+1)
                              + ' of %d' % N + ' scenarios', end='\r',
                              flush=True)
                    self.forward_solver.solve(
                        self.scenarios[i][j][k],
                        noise=self.scenarios[i][j][k].noise,
                        SAVE_INTERN_FIELD=SAVE_INTERN_FIELD
                    )
                    n += 1
        print('Synthesizing scattered field: %d' % N + ' of %d' % N
              + ' scenarios')

    def solve_scenarios(self, parallelization=False):
        """Run inverse solvers."""
        if self.maximum_contrast is None:
            raise error.MissingAttributesError('Experiment',
                                               'maximum_contrast')
        if self.configurations is None or len(self.configurations) == 0:
            raise error.MissingAttributesError('Experiment', 'configurations')
        if self.sample_size is None:
            raise error.MissingAttributesError('Experiment', 'sample_size')
        if self.methods is None:
            raise error.MissingAttributesError('Experiment', 'methods')
        if self.scenarios is None:
            raise error.MissingAttributesError('Experiment', 'scenarios')
        self.results = []
        for i in range(len(self.maximum_contrast)):
            self.results.append(list())
            for j in range(len(self.configurations)):
                self.results[i].append(list())
                for m in range(len(self.methods)):
                    self.methods[m].configuration = self.configurations[j]
                for k in range(self.sample_size):
                    self.results[i][j].append(list())
                    self.scenarios[i][j][k].resolution = (
                        self.recover_resolution[i][j]
                    )
                    self.results[i][j][k] = (
                        run_methods(self.methods, self.scenarios[i][j][k],
                                    parallelization=parallelization)
                    )

    def fixed_sampleset_plot(self, group_idx=0, config_idx=0, method_idx=0,
                             show=False, file_path='', file_format='eps'):
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')
        if type(method_idx) is int:
            method_idx = [method_idx]

        g, c = group_idx, config_idx
        y = np.zeros((self.sample_size, len(method_idx)))
        measures = self.get_measure_set(config_idx)
        nplots = len(measures)

        nrows = int(np.sqrt(nplots))
        ncols = int(np.ceil(nplots/nrows))
        image_size = (3.*ncols, 3.*nrows)
        figure = plt.figure(figsize=image_size)
        rst.set_subplot_size(figure)
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        x = range(1, self.sample_size+1)
        i = 1
        for j in range(len(measures)):
            for m in range(len(method_idx)):
                y[:, m] = self.get_final_value_over_samples(
                    group_idx=g, config_idx=c, method_idx=method_idx[m],
                    measure=measures[j]
                )
            axes = figure.add_subplot(nrows, ncols, i)
            rst.add_plot(axes, y, x=x, title=get_title(measures[j]),
                         xlabel=LABEL_INSTANCE, ylabel=get_label(measures[j]),
                         xticks=x, legend=method_names)
            i += 1

        if show:
            plt.show()
        else:
            plt.savefig(file_path + self.name + '_single.' + file_format,
                        format=file_format)
            plt.close()

    def fixed_sampleset_boxplot(self, group_idx=0, config_idx=0,
                                method_idx=[0], show=False,
                                file_path='', file_format='eps'):
        """Summarize the class method."""
        if type(group_idx) is not int:
            raise error.WrongTypeInput('fixed_sampleset_boxplot', 'group_idx',
                                       'int', type(group_idx))
        if type(config_idx) is not int:
            raise error.WrongTypeInput('fixed_sampleset_boxplot', 'config_idx',
                                       'int', type(config_idx))

        g, c = group_idx, config_idx
        measures = self.get_measure_set(config_idx)
        nplots = len(measures)

        nrows = int(np.sqrt(nplots))
        ncols = int(np.ceil(nplots/nrows))
        image_size = (3.*ncols, 3.*nrows)
        figure = plt.figure(figsize=image_size)
        rst.set_subplot_size(figure)
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        n = 1
        for i in range(len(measures)):
            data = []
            for m in range(len(method_idx)):
                data.append(
                    self.get_final_value_over_samples(group_idx=g,
                                                      config_idx=c,
                                                      method_idx=method_idx[m],
                                                      measure=measures[i])
                )
            axes = figure.add_subplot(nrows, ncols, n)
            violinplot(data, axes=axes, labels=method_names, xlabel='Methods',
                       ylabel=get_label(measures[i]),
                       title=get_title(measures[i]), show=show,
                       file_path=file_path, file_format=file_format)
            n += 1

        if show:
            plt.show()
        else:
            plt.savefig(file_path + self.name + '_%d' + config_idx + '%d'
                        + group_idx + '.' + file_format, format=file_format)
            plt.close()

    def fixed_measure_boxplot(self, group_idx=[0], config_idx=[0],
                              measure=None, method_idx=[0], show=False,
                              file_path='', file_format='eps'):
        """Summarize the class method."""
        if measure is None:
            raise error.MissingInputError('Experiments.fixed_measure_boxplot',
                                          'measure')
        if type(group_idx) is int:
            group_idx = [group_idx]
        if type(config_idx) is int:
            config_idx = [config_idx]

        ylabel = get_label(measure)
        nplots = len(group_idx)*len(config_idx)
        if nplots > 1:
            nrows = int(np.sqrt(nplots))
            ncols = int(np.ceil(nplots/nrows))
            image_size = (3.*ncols, 3.*nrows)
            figure = plt.figure(figsize=image_size)
            rst.set_subplot_size(figure)
        else:
            axes = rst.get_single_figure_axes(plt.figure())

        n = 1
        for i in range(len(group_idx)):
            for j in range(len(config_idx)):
                data = []
                labels = []
                if nplots > 1:
                    axes = figure.add_subplot(nrows, ncols, n)
                    n += 1
                for k in range(len(method_idx)):
                    data.append(
                        self.get_final_value_over_samples(
                            group_idx=group_idx[i], config_idx=config_idx[j],
                            method_idx=method_idx[k], measure=measure
                        )
                    )
                    labels.append(self.methods[k].alias)
                if nplots > 1:
                    if len(group_idx) == 1:
                        title = 'Con. %d' % config_idx[j]
                    elif len(config_idx) == 1:
                        title = 'Group %d' % group_idx[i]
                    else:
                        title = ('Group %d' % group_idx[i]
                                 + ', Con. %d' % config_idx[j])
                    figure.suptitle(get_title(measure))
                else:
                    title = get_title(measure)
                violinplot(data, axes=axes, labels=labels, xlabel='Methods',
                           ylabel=ylabel, title=title, show=show,
                           file_path=file_path, file_format=file_format)
        if show:
            plt.show()
        else:
            plt.savefig(file_path + self.name + '_' + measure + '.'
                        + file_format, format=file_format)
            plt.close()

    def evolution_boxplot(self, group_idx=[0], config_idx=[0],
                          measure=None, method_idx=[0], show=False,
                          file_path='', file_format='eps'):
        if type(group_idx) is int:
            group_idx = [group_idx]
        if type(config_idx) is int:
            config_idx = [config_idx]
        if measure is None:
            none_measure = True
        else:
            none_measure = False
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)
        colors = ['cornflowerblue', 'indianred', 'seagreen', 'mediumorchid',
                  'chocolate', 'palevioletred', 'teal', 'rosybrown', 'tan',
                  'crimson']

        if len(group_idx) > 1:

            labels = []
            for j in group_idx:
                labels.append('g%d' % j)
            
            for i in config_idx:
                if none_measure:
                    measure = self.get_measure_set(i)
                if len(measure) == 1:
                    figure = plt.figure()
                    axes = rst.get_single_figure_axes(figure)
                else:
                    nplots = len(measure)
                    nrows = int(np.sqrt(nplots))
                    ncols = int(np.ceil(nplots/nrows))
                    image_size = (3.*ncols, 3.*nrows)
                    figure = plt.figure(figsize=image_size)
                    rst.set_subplot_size(figure)
                k = 1
                for mea in measure:
                    if len(measure) > 1:
                        axes = figure.add_subplot(nrows, ncols, k)
                    n = 0
                    for m in method_idx:
                        data = []
                        for j in group_idx:
                            data.append(self.get_final_value_over_samples(
                                group_idx=j, config_idx=i, method_idx=m,
                                measure=mea
                            ))
                        boxplot(data, axes=axes, meanline=True, labels=labels,
                                xlabel='Groups', ylabel=get_label(mea),
                                color=colors[n], legend=method_names[n],
                                title=get_title(mea))
                        n += 1
                    k += 1
                plt.suptitle('Con. ' + self.configurations[i].name)

                if show:
                    plt.show()
                else:
                    plt.savefig(file_path + self.name + '_evolution_c%d' % i
                                + '.' + file_format, format=file_format)
                    plt.close()

        else:

            labels = []
            for i in config_idx:
                labels.append('c%d' % i)
            j = group_idx[0]

            if none_measure:
                measure = self.get_measure_set(config_idx[0])
            if len(measure) == 1:
                figure = plt.figure()
                axes = rst.get_single_figure_axes(figure)
            else:
                nplots = len(measure)
                nrows = int(np.sqrt(nplots))
                ncols = int(np.ceil(nplots/nrows))
                image_size = (3.*ncols, 3.*nrows)
                figure = plt.figure(figsize=image_size)
                rst.set_subplot_size(figure)
            k = 1
            for mea in measure:
                if len(measure) > 1:
                    axes = figure.add_subplot(nrows, ncols, k)
                n = 0
                for m in method_idx:
                    data = []
                    for i in config_idx:
                        data.append(self.get_final_value_over_samples(
                            group_idx=j, config_idx=i, method_idx=m,
                            measure=mea
                        ))
                    boxplot(data, axes=axes, meanline=True, labels=labels,
                            xlabel='Configuration', ylabel=get_label(mea),
                            color=colors[n], legend=method_names[n],
                            title=get_title(mea))
                    n += 1
                k += 1
            plt.suptitle('Group %d' % j)

            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_evolution_c%d' % i
                            + '.' + file_format, format=file_format)
                plt.close()

    def plot_sampleset_results(self, group_idx=[0], config_idx=[0],
                               method_idx=[0], show=False, file_path='',
                               file_format='eps'):
        """Summarize the method."""
        if type(group_idx) is int:
            group_idx = [group_idx]
        if type(config_idx) is int:
            config_idx = [config_idx]
        if type(method_idx) is int:
            method_idx = [method_idx]

        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        if len(method_idx) > 1:
            nplots = 1 + len(method_idx)
        else:
            nplots = self.sample_size

        nrows = int(np.sqrt(nplots))
        ncols = int(np.ceil(nplots/nrows))
        image_size = (3.*ncols, 3.*nrows)
        bounds = (0, 1, 0, 1)
        xlabel, ylabel = r'$L_x$', r'$L_y$'

        for i in group_idx:
            for j in config_idx:

                omega = 2*pi*self.configurations[j].f
                epsilon_rb = self.configurations[j].epsilon_rb
                sigma_b = self.configurations[j].sigma_b

                if len(method_idx) == 1:
                    figure = plt.figure(figsize=image_size)
                    rst.set_subplot_size(figure)
                    n = 1

                for k in range(self.sample_size):

                    if len(method_idx) > 1:
                        figure = plt.figure(figsize=image_size)
                        rst.set_subplot_size(figure)

                        axes = figure.add_subplot(nrows, ncols, 1)
                        chi = cfg.get_contrast_map(
                            epsilon_r=self.scenarios[i][j][k].epsilon_r,
                            sigma=self.scenarios[i][j][k].sigma,
                            epsilon_rb=epsilon_rb,
                            sigma_b=sigma_b,
                            omega=omega
                        )

                        rst.add_image(axes, np.abs(chi), title='Original',
                                      colorbar_name=r'$|\chi|$', bounds=bounds,
                                      xlabel=xlabel, ylabel=ylabel)
                        n = 2

                    p = 0
                    for m in method_idx:

                        chi = cfg.get_contrast_map(
                            epsilon_r=self.results[i][j][k][m].epsilon_r,
                            sigma=self.results[i][j][k][m].sigma,
                            epsilon_rb=epsilon_rb,
                            sigma_b=sigma_b,
                            omega=omega
                        )

                        if len(method_idx) > 1:
                            title = method_names[p]
                        else:
                            title = self.scenarios[i][j][k].name

                        axes = figure.add_subplot(nrows, ncols, n)
                        rst.add_image(axes, np.abs(chi), title=title,
                                      colorbar_name=r'$|\chi|$', bounds=bounds,
                                      xlabel=xlabel, ylabel=ylabel)
                        n += 1

                    if len(method_idx) > 1:
                        if show:
                            plt.show()
                        else:
                            plt.savefig(file_path + self.name
                                        + 'recoverd_images' + str(i) + str(j)
                                        + str(k) + '.' + file_format,
                                        format=file_format)
                            plt.close()

                if len(method_idx) == 1:
                    if show:
                        plt.show()
                    else:
                        plt.savefig(file_path + self.name + 'recoverd_images'
                                    + str(i) + str(j) + '.' + file_format,
                                    format=file_format)
                        plt.close()

    def plot_nbest_results(self, n, measure, group_idx=[0], config_idx=[0],
                           method_idx=None, show=False, file_path='',
                           file_format='eps'):
        if method_idx is None:
            if len(self.methods) == 1:
                method_idx = [0]
            else:
                method_idx = range(len(self.methods))
        else:
            if type(method_idx) is list:
                pass
            else:
                single_method = True
                method_idx = [method_idx]

        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        nplots = n
        nrows = int(np.sqrt(nplots))
        ncols = int(np.ceil(nplots/nrows))
        image_size = (3.*ncols, 3.*nrows)
        bounds = (0, 1, 0, 1)
        xlabel, ylabel = r'$L_x$', r'$L_y$'

        for j in config_idx:

            omega = 2*pi*self.configurations[j].f
            epsilon_rb = self.configurations[j].epsilon_rb
            sigma_b = self.configurations[j].sigma_b

            for i in group_idx:
                for m in method_idx:

                    figure = plt.figure(figsize=image_size)
                    rst.set_subplot_size(figure)
                    y = self.get_final_value_over_samples(group_idx=i,
                                                          config_idx=j,
                                                          method_idx=m,
                                                          measure=measure)
                    yi = np.argsort(y)

                    for k in range(nplots):

                        chi = cfg.get_contrast_map(
                            epsilon_r=self.results[i][j][yi[k]][m].epsilon_r,
                            sigma=self.results[i][j][yi[k]][m].sigma,
                            epsilon_rb=epsilon_rb,
                            sigma_b=sigma_b,
                            omega=omega
                        )

                        axes = figure.add_subplot(nrows, ncols, k+1)
                        title = (self.scenarios[i][j][yi[k]].name
                                 + ' - %.2e' % y[yi[k]])
                        rst.add_image(axes, np.abs(chi), title=title,
                                      colorbar_name=r'$|\chi|$', bounds=bounds,
                                      xlabel=xlabel, ylabel=ylabel)

                    title = ('C%d,' % j + ' G%d,' % i + ' '
                             + get_label(measure) + ' - '
                             + self.methods[m].alias)
                    plt.suptitle(title)

                    if show:
                        plt.show()
                    else:
                        plt.savefig(file_path + self.name + 'nbeast' + str(i)
                                    + str(j) + str(m) + '.' + file_format,
                                    format=file_format)
                        plt.close()

    def study_single_mean(self, measure=None, group_idx=[0], config_idx=[0],
                          method_idx=[0], show=False, file_path='',
                          file_format='eps', printscreen=False, write=False):
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        if write or printscreen:
            title = 'Confidence Interval of Means - *' + self.name + '*'
            text = ''.join(['*' for _ in range(len(title))]) + '\n'
            text = text + title + '\n' + text

        if measure is None or (type(measure) is list and len(measure) > 1):

            if measure is None:
                none_measure = True
            else:
                none_measure = False

            for i in config_idx:

                if none_measure:
                    measure = self.get_measure_set(i)
                nplots = len(measure)
                nrows = int(np.sqrt(nplots))
                ncols = int(np.ceil(nplots/nrows))
                image_size = (3.*ncols, 3.*nrows)

                for j in group_idx:

                    if write or printscreen:
                        subtitle = 'Configuration %d' % i + ', Group %d' % j
                        text = (text + '\n' + subtitle + '\n'
                                + ''.join(['=' for _ in range(len(subtitle))])
                                + '\n')

                    figure = plt.figure(figsize=image_size)
                    rst.set_subplot_size(figure)
                    k = 1

                    for mea in measure:

                        y = np.zeros((self.sample_size, len(method_idx)))
                        n = 0

                        if write or printscreen:
                            subsubtitle = 'Measure: ' + mea
                            text = (text + '\n' + subsubtitle + '\n'
                                    + ''.join(['-'
                                               for _ in range(len(subtitle))])
                                    + '\n')

                        for m in method_idx:

                            y[:, n] = self.get_final_value_over_samples(
                                measure=mea, group_idx=j, config_idx=i,
                                method_idx=m
                            )

                            if stats.diagnostic.normal_ad(y[:, n])[1] < .05:
                                message = ('The sample from method '
                                           + method_names[n] + ', config. %d, '
                                           % i + 'group %d, ' % j
                                           + ' and measure ' + mea
                                           + ' is not from a normal '
                                           + ' distribution!')
                                warnings.warn(message)
                                if printscreen or write:
                                    text = text + message + '\n'

                            if write or printscreen:
                                info = stats.weightstats.DescrStatsW(y[:, n])
                                cf = info.tconfint_mean()
                                text = (text + method_names[n] + ': [%.2e, '
                                        % cf[0] + '%.2e]' % cf[1] + '\n')

                    else:
                        plt.savefig(file_path + self.name + '_confint_'
                                    + str(i) + str(j) + '.' + file_format,
                                    format=file_format)
                        plt.close()

                        axes = figure.add_subplot(nrows, ncols, k)
                        confintplot(y, axes=axes, xlabel=get_label(mea),
                                    ylabel=method_names, title=get_title(mea))

                        k += 1

                    plt.suptitle('c%d' % i + 'g%d' % j)

                    if show:
                        plt.show()
                    else:
                        plt.savefig(file_path + self.name + '_confint_'
                                    + str(i) + str(j) + '.' + file_format,
                                    format=file_format)
                        plt.close()

        else:
            if type(measure) is list:
                mea = measure[0]
            else:
                mea = measure

            if write or printscreen:
                subsubtitle = 'Measure: ' + mea
                text = (text + '\n' + subsubtitle + '\n'
                        + ''.join(['=' for _ in range(len(subsubtitle))])
                        + '\n')

            if len(group_idx) == 1 and len(config_idx) == 1:
                i, j = config_idx[0], group_idx[0]
                y = np.zeros((self.sample_size, len(method_idx)))

                if write or printscreen:
                    subtitle = 'Configuration %d' % i + ', Group %d' % j
                    text = (text + '\n' + subtitle + '\n'
                            + ''.join(['-' for _ in range(len(subtitle))])
                            + '\n')

                for m in method_idx:
                    y[:, n] = self.get_final_value_over_samples(measure=mea,
                                                                group_idx=j,
                                                                config_idx=i,
                                                                method_idx=m)

                    if stats.diagnostic.normal_ad(y[:, n])[1] < .05:
                        message = ('The sample from method '
                                   + method_names[n] + ', config. %d, '
                                   % i + 'group %d, ' % j
                                   + ' and measure ' + mea
                                   + ' is not from a normal '
                                   + ' distribution!')
                        warnings.warn(message)
                        if printscreen or write:
                            text = text + message + '\n'

                    if write or printscreen:
                        info = stats.weightstats.DescrStatsW(y[:, n])
                        cf = info.tconfint_mean()
                        text = (text + '* ' + method_names[m] + ': [%.2e, '
                                % cf[0] + '%.2e]' % cf[1] + '\n')

                confintplot(y, xlabel=get_label(mea), ylabel=method_names,
                            title=get_title(mea))
                if show:
                    plt.show()
                else:
                    plt.savefig(file_path + self.name + '_confint_' + mea + '_'
                                + str(i) + str(j) + '.' + file_format,
                                format=file_format)
                    plt.close()

            elif len(group_idx) == 1 and len(config_idx) > 1:

                nplots = len(config_idx)
                nrows = int(np.sqrt(nplots))
                ncols = int(np.ceil(nplots/nrows))
                image_size = (3.*ncols, 3.*nrows)
                figure = plt.figure(figsize=image_size)
                rst.set_subplot_size(figure)
                j = group_idx[0]

                k = 1
                for i in config_idx:

                    if write or printscreen:
                        subtitle = 'Configuration %d' % i + ', Group %d' % j
                        text = (text + '\n' + subtitle + '\n'
                                + ''.join(['=' for _ in range(len(subtitle))])
                                + '\n')

                    y = np.zeros((self.sample_size, len(method_idx)))
                    n = 0
                    for m in method_idx:
                        y[:, n] = self.get_final_value_over_samples(
                            measure=mea, group_idx=j, config_idx=i,
                            method_idx=m
                        )

                        if stats.diagnostic.normal_ad(y[:, n])[1] < .05:
                            message = ('The sample from method '
                                       + method_names[n] + ', config. %d, '
                                       % i + 'group %d, ' % j
                                       + ' and measure ' + mea
                                       + ' is not from a normal '
                                       + ' distribution!')
                            warnings.warn(message)
                            if printscreen or write:
                                text = text + message + '\n'

                        if write or printscreen:
                            info = stats.weightstats.DescrStatsW(y[:, n])
                            cf = info.tconfint_mean()
                            text = (text + '* ' + method_names[m] + ': [%.2e, '
                                    % cf[0] + '%.2e]' % cf[1] + '\n')

                    axes = figure.add_subplot(nrows, ncols, k)
                    confintplot(y, axes=axes, xlabel=get_label(mea),
                                ylabel=method_names,
                                title=self.configurations[i].name)
                    k += 1

                if show:
                    plt.show()
                else:
                    plt.savefig(file_path + self.name + '_confint_' + mea + '_'
                                + 'g%d' % j + '.' + file_format,
                                format=file_format)
                    plt.close()

            else:

                nplots = len(group_idx)
                nrows = int(np.sqrt(nplots))
                ncols = int(np.ceil(nplots/nrows))
                image_size = (3.*ncols, 3.*nrows)
                figure = plt.figure(figsize=image_size)
                rst.set_subplot_size(figure)

                for i in config_idx:

                    k = 1
                    for j in group_idx:

                        if write or printscreen:
                            subtitle = ('Configuration %d' % i
                                        + ', Group %d' % j)
                            text = (text + '\n' + subtitle + '\n'
                                    + ''.join(['='
                                               for _ in range(len(subtitle))])
                                    + '\n')

                        y = np.zeros((self.sample_size, len(method_idx)))
                        n = 0
                        for m in method_idx:
                            y[:, n] = self.get_final_value_over_samples(
                                measure=mea, group_idx=j, config_idx=i,
                                method_idx=m
                            )

                            if stats.diagnostic.normal_ad(y[:, n])[1] < .05:
                                message = ('The sample from method '
                                           + method_names[n] + ', config. %d, '
                                           % i + 'group %d, ' % j
                                           + ' and measure ' + mea
                                           + ' is not from a normal '
                                           + ' distribution!')
                                warnings.warn(message)
                                if printscreen or write:
                                    text = text + message + '\n'

                            if write or printscreen:
                                info = stats.weightstats.DescrStatsW(y[:, n])
                                cf = info.tconfint_mean()
                                text = (text + '* ' + method_names[m] 
                                        + ': [%.2e, ' % cf[0] + '%.2e]' % cf[1]
                                        + '\n')

                        axes = figure.add_subplot(nrows, ncols, k)
                        confintplot(y, axes=axes, xlabel=get_label(mea),
                                    ylabel=method_names,
                                    title='g. %d' % j)
                        k += 1

                    if show:
                        plt.show()
                    else:
                        plt.savefig(file_path + self.name + '_confint_' + mea
                                    + '_' + 'c%d' % i + '.' + file_format,
                                    format=file_format)
                        plt.close()

        if printscreen:
            print(text)
        if write:
            file = open(file_path + self.name + '_confint.txt', 'w')
            file.write(text)
            file.close()

    def plot_normality(self, measure=None, group_idx=[0], config_idx=[0],
                       method_idx=[0], show=False, file_path='',
                       file_format='eps'):
        if type(group_idx) is int:
            group_idx = [group_idx]
        if type(config_idx) is int:
            config_idx = [config_idx]
        if type(measure) is str:
            measure = [measure]
        if measure is None:
            none_measure = True
        else:
            none_measure = False

        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        for i in config_idx:

            if none_measure:
                measure = self.get_measure_set(i)

            for j in group_idx:

                if len(measure) > 1 and len(method_idx) == 1:
                    m = method_idx[0]
                    nplots = len(measure)
                    nrows = int(np.sqrt(nplots))
                    ncols = int(np.ceil(nplots/nrows))
                    image_size = (3.*ncols, 3.*nrows)
                    fig = plt.figure(figsize=image_size)
                    rst.set_subplot_size(fig)
                    data = np.zeros((self.sample_size, len(measure)))
                    for k in range(len(measure)):
                        data[:, k] = self.get_final_value_over_samples(
                            group_idx=j, config_idx=i, method_idx=m,
                            measure=measure[k])
                        axes = fig.add_subplot(nrows, ncols, k+1)
                        normalitiyplot(data[:, k], axes, measure[k])
                    plt.suptitle('c%d' % i + 'g%d - ' % j + method_names[0])
                    if show:
                        plt.show()
                    else:
                        plt.savefig(file_path + self.name + '_normality_'
                                    + 'c%d' % i + 'g%d' % j + '_'
                                    + method_names[0] + '.' + file_format,
                                    format=file_format)
                        plt.close()

                elif len(measure) == 1 and len(method_idx) > 1:
                    nplots = len(method_idx)
                    nrows = int(np.sqrt(nplots))
                    ncols = int(np.ceil(nplots/nrows))
                    image_size = (3.*ncols, 3.*nrows)
                    fig = plt.figure(figsize=image_size)
                    rst.set_subplot_size(fig)
                    data = np.zeros((self.sample_size, len(method_idx)))
                    for k in range(len(method_idx)):
                        data[:, k] = self.get_final_value_over_samples(
                            group_idx=j, config_idx=i,
                            method_idx=method_idx[k], measure=measure[0])
                        axes = fig.add_subplot(nrows, ncols, k+1)
                        normalitiyplot(data[:, k], axes, method_names[k])
                    plt.suptitle('c%d' % i + 'g%d - ' % j
                                 + get_title(measure[0]))
                    if show:
                        plt.show()
                    else:
                        plt.savefig(file_path + self.name + '_normality_'
                                    + 'c%d' % i + 'g%d' % j + '_'
                                    + measure[0] + '.' + file_format,
                                    format=file_format)
                        plt.close()

                else:

                    nplots = len(method_idx)
                    nrows = int(np.sqrt(nplots))
                    ncols = int(np.ceil(nplots/nrows))
                    image_size = (3.*ncols, 3.*nrows)

                    for mea in measure:

                        fig = plt.figure(figsize=image_size)
                        rst.set_subplot_size(fig)
                        data = np.zeros((self.sample_size, len(method_idx)))
                        for k in range(len(method_idx)):
                            data[:, k] = self.get_final_value_over_samples(
                                group_idx=j, config_idx=i, measure=mea,
                                method_idx=method_idx[k])
                            axes = fig.add_subplot(nrows, ncols, k+1)
                            normalitiyplot(data[:, k], axes, method_names[k])
                        plt.suptitle('c%d' % i + 'g%d - ' % j + get_title(mea))
                        if show:
                            plt.show()
                        else:
                            plt.savefig(file_path + self.name + '_normality_'
                                        + 'c%d' % i + 'g%d' % j + '_'
                                        + measure[0] + '.' + file_format,
                                        format=file_format)
                            plt.close()

    def compare_two_methods(self, measure=None, group_idx=[0], config_idx=[0],
                            method_idx=[0, 1], printscreen=False, write=False,
                            file_path=''):
        if type(group_idx) is int:
            group_idx = [group_idx]
        if type(config_idx) is int:
            config_idx = [config_idx]
        if measure is None:
            none_measure = True
        else:
            none_measure = False
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        if write or printscreen:
            title = 'Paired Study - *' + self.name + '*'
            subtitle = 'Methods: ' + method_names[0] + ', ' + method_names[1]
            text = ''.join(['*' for _ in range(len(title))])
            text = text + '\n' + title + '\n' + text + '\n\n'
            aux = ''.join(['#' for _ in range(len(subtitle))])
            text = text + subtitle + '\n' + aux + '\n\n'
            text = text + 'Significance level: %.2f\n' % 0.05
            text = text + 'Power: %.2f\n\n' % 0.8
        else:
            text = ''

        results = []
        for i in config_idx:
            if none_measure:
                measure = self.get_measure_set(i)

            if write or printscreen:
                section = 'Configuration ' + self.configurations[i].name
                aux = ''.join(['=' for _ in range(len(section))])
                text = text + section + '\n' + aux + '\n\n'

            results.append(list())
            for j in group_idx:

                if write or printscreen:
                    subsection = 'Group %d' % j
                    aux = ''.join(['-' for _ in range(len(subsection))])
                    text = text + subsection + '\n' + aux + '\n'

                results[-1].append(list())
                for k in range(len(measure)):

                    y1 = self.get_final_value_over_samples(
                        group_idx=j, config_idx=i, method_idx=method_idx[0],
                        measure=measure[k]
                    )
                    y2 = self.get_final_value_over_samples(
                        group_idx=j, config_idx=i, method_idx=method_idx[1],
                        measure=measure[k]
                    )

                    if write or printscreen:
                        topic = '* ' + measure[k]
                        text = text + topic

                    if scipy.stats.shapiro(y1-y2)[1] > .05:
                        pvalue, lower, upper = stats.weightstats.ttost_paired(
                            y1, y2, 0, 0
                        )
                        delta = stats.power.tt_solve_power(
                            nobs=self.sample_size, alpha=0.05, power=.80
                        ) / np.std(y1-y2)

                        result, text = self._pairedtest_result(pvalue, lower,
                                                               upper,
                                                               method_names,
                                                               delta,
                                                               text + ': ')

                    elif scipy.stats.shapiro(np.log(y1)
                                                    - np.log(y2))[1] > .05:
                        pvalue, lower, upper = stats.weightstats.ttost_paired(
                            np.log(y1), np.log(y2), 0, 0
                        )
                        delta = stats.power.tt_solve_power(
                            nobs=self.sample_size, alpha=0.05, power=.80
                        ) / np.std(np.log(y1)-np.log(y2))

                        result, text = self._pairedtest_result(
                            pvalue, lower, upper, method_names, delta,
                            text + ' (Log Transformation): '
                        )

                    elif scipy.stats.shapiro(np.sqrt(y1)
                                                    - np.sqrt(y2))[1] > .05:
                        pvalue, lower, upper = stats.weightstats.ttost_paired(
                            np.log(y1), np.log(y2), 0, 0
                        )
                        delta = stats.power.tt_solve_power(
                            nobs=self.sample_size, alpha=0.05, power=.80
                        ) / np.std(np.sqrt(y1)-np.sqrt(y2))

                        result, text = self._pairedtest_result(
                            pvalue, lower, upper, method_names, delta, text
                            + ' (Square-root Transformation): '
                        )

                    else:
                        pvalue = scipy.stats.wilcoxon(y1, y2)[1]
                        text = text + ' (Wilcoxon-Test): '
                        if pvalue > .05:
                            text = (text + 'Equality hypothesis not rejected '
                                    '(pvalue: %.2e)' % pvalue + '\n')
                            result = '1=2'
                        else:
                            text = (text + 'Equality hypothesis rejected '
                                    '(pvalue: %.2e)' % pvalue + '\n')
                            _, lower = scipy.stats.wilcoxon(
                                y1, y2, alternative='less'
                            )
                            _, upper = scipy.stats.wilcoxon(
                                y1, y2, alternative='greater'
                            )
                            if lower < .05:
                                text = (text + '  Better performance of '
                                        + method_names[0]
                                        + ' has been detected (pvalue: %.2e).'
                                        % lower + '\n')
                                result = '1<2'
                            if upper < .05:
                                text = (text + '  Better performance of '
                                        + method_names[1]
                                        + ' has been detected (pvalue: %.2e).'
                                        % upper + '\n')
                                result = '1>2'

                    results[-1][-1].append(result)

                if write or printscreen:
                    text = text + '\n'

        mtd1, mtd2, equal = 0, 0, 0
        for i in range(len(results)):
            for j in range(len(results[i])):
                for k in range(len(results[i][j])):
                    if results[i][j][k] == '1=2':
                        equal += 1
                    elif results[i][j][k] == '1<2':
                        mtd1 += 1
                    else:
                        mtd2 += 2

        if printscreen or write:
            text = (text + 'Number of equality results: %d ' % equal
                    + '(%.1f%%)\n' % (equal/(equal+mtd1+mtd2)*100))
            text = (text + 'Number of times than ' + method_names[0]
                    + ' outperformed ' + method_names[1] + ': %d ' % mtd1
                    + '(%.1f%%)\n' % (mtd1/(equal+mtd1+mtd2)*100))
            text = (text + 'Number of times than ' + method_names[1]
                    + ' outperformed ' + method_names[0] + ': %d ' % mtd2
                    + '(%.1f%%)\n' % (mtd2/(equal+mtd1+mtd2)*100))

        if printscreen:
            print(text)
        if write:
            file = open(file_path + self.name + '_compare2mtd_%d'
                        % method_idx[0] + '_%d_' % method_idx[1] + '.txt', 'w')
            file.write(text)
            file.close()

        return results, mtd1, mtd2, equal

    def compare_multiple_methods(self, measure=None, group_idx=[0],
                                 config_idx=[0], method_idx=[0, 1],
                                 printscreen=False, write=False, file_path='',
                                 all2all=True, one2all=None):
        if type(group_idx) is int:
            group_idx = [group_idx]
        if type(config_idx) is int:
            config_idx = [config_idx]
        if measure is None:
            none_measure = True
        else:
            none_measure = False
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        for i in config_idx:

            if none_measure:
                measure = self.get_measure_set(i)

            for j in config_idx:
                k = 0
                for mea in measure:
                    data = []
                    for m in method_idx:
                        data.append(self.get_final_value_over_samples(j, i, m,
                                                                      mea))
                    residuals = np.zeros((len(method_idx), self.sample_size))
                    for p in range(len(method_idx)):
                        for q in range(self.sample_size):
                            residuals[i, j] = data[i][j]-np.mean(data[i])
                    not_normal = False
                    if scipy.stats.shapiro(residuals.flatten())[1] < .05:
                        output = data_transformation(residuals.flatten())
                        if output is None:
                            message = 'Not normal data'
                            not_normal = True
                        else:
                            for m in range(len(method_idx)):
                                if output[1] == 'log':
                                    data[m] = np.log(data[m])
                                elif output[1] == 'sqrt':
                                    data[m] = np.sqrt(data[m])
                    if not not_normal:
                        if scipy.stats.fligner(*data)[1] > .05:
                            output = oneway.anova_oneway(data, use_var='equal')
                            homoscedasticity = True
                        else:
                            output = oneway.anova_oneway(data, use_var='unequal')
                            homoscedasticity = False

                        if output.pvalue > .05:
                            message = 'Fails to reject null hypothesis of equality of means'
                        else:
                            message = 'Reject equality of means'
                            if all2all and homoscedasticity:
                                data2 = np.zeros(residuals.shape)
                                groups = []
                                for m in range(len(method_idx)):
                                    data2[m, :] = data[m]
                                    groups = groups + [method_names[m]] * self.sample_size
                                data2 = data2.flatten()
                                output = snd.stats.multicomp.MultiComparison(data, groups).tukeyhsd()
                                message = message + '\n' + str(output)
                            elif all2all and not homoscedasticity:
                                a = len(method_idx)
                                alpha = 0.05/(a*(a-1)/2)
                                for i in range(len(method_idx)-1):
                                    for j in range(i, len(method_idx)):
                                        y1, y2 = data[i], data[j]
                                        if ttest_ind_nonequalvar(y1, y2, alpha):
                                            message = message + '\nNo difference between' + method_names[i] + ' and ' + method_names[j]
                                        else:
                                            message = message + '\nEvidence for difference between' + method_names[i] + ' and ' + method_names[j]

                            if one2all is not None and homoscedasticity:
                                y0 = data[one2all]
                                y = []
                                j = []
                                for m in range(len(method_idx)):
                                    if method_idx[m] != one2all:
                                        y.append(data[m])
                                        j.append(m)
                                output = dunnettest(y0, y)
                                for i in range(len(output)):
                                    if output[i]:
                                        message = message + '\nNo difference between' + method_names[one2all] + ' and ' + method_names[j[i]]
                                    else:
                                        message = message + '\nEvidence for difference between' + method_names[one2all] + ' and ' + method_names[j[i]]

                            elif one2all is not None and not homoscedasticity:
                                a = len(method_idx)
                                alpha = 0.05/(a-1)
                                y0 = data[one2all]
                                y = []
                                j = []
                                for m in range(len(method_idx)):
                                    if method_idx[m] != one2all:
                                        y.append(data[m])
                                        j.append(m)
                                for i in range(a-1):
                                    if ttest_ind_nonequalvar(y0, y[i], alpha):
                                        message = message + '\nNo difference between' + method_names[one2all] + ' and ' + method_names[j[i]]
                                    else:
                                        message = message + '\nEvidence for difference between' + method_names[one2all] + ' and ' + method_names[j[i]]

                    else:
                        _, pvalue = scipy.stats.kruskal(*data)
                        if pvalue > 0.05:
                            message += '\nEqual distributions not reject!'
                        else:
                            if all2all:
                                for i in range(len(method_idx)-1):
                                    for j in range(i, len(method_idx)):
                                        y1, y2 = data[i], data[j]
                                        if scipy.stats.mannwhitneyu(y1, y2):
                                            message = message + '\nThe probability of ' + method_names[i] + ' having a better performance than ' + method_names[j] + ' is the same of otherwise.'
                                        else:
                                            message = message + '\nThe probability of ' + method_names[i] + ' having a better performance than ' + method_names[j] + ' is NOT the same of otherwise.'
                            if one2all is not None:
                                y0 = data[one2all]
                                y = []
                                j = []
                                for m in range(len(method_idx)):
                                    if method_idx[m] != one2all:
                                        y.append(data[m])
                                        j.append(m)
                                for i in range(a-1):
                                    if scipy.stats.mannwhitneyu(y0, y[i]):
                                        message = message + '\nThe probability of ' + method_names[one2all] + ' having a better performance than ' + method_names[j[i]] + ' is the same of otherwise.'
                                    else:
                                        message = message + '\nThe probability of ' + method_names[one2all] + ' having a better performance than ' + method_names[j[i]] + ' is NOT the same of otherwise.'

    def _pairedtest_result(self, pvalue, lower, upper, method_names,
                           effect_size=None, text=None):
        if text is None:
            text = ''

        if effect_size is None:
            aux = ''
        else:
            aux = ', effect size: %.3e' % effect_size

        if pvalue < .05:
            text = (text + 'Difference hypothesis rejected (pvalue: %.2e'
                    % pvalue + aux + ').\n')
            result = '1=2'
        else:
            text = (text + 'No evidence against difference in performance '
                    + '(pvalue: %.2e' % pvalue + aux + ').\n')

            if lower[2] > .05:
                text = (text + '  No evidence against a better performance of '
                        + method_names[0] + ' (pvalue: %.2e).' % lower[2]
                        + '\n')
                result = '1<2'
            else:
                text = (text + '  No evidence against a better performance of '
                        + method_names[1] + ' (pvalue: %.2e).' % upper[2]
                        + '\n')
                result = '1>2'
        return result, text   

    def __str__(self):
        """Print the object information."""
        message = 'Experiment name: ' + self.name
        if all(i == self.maximum_contrast[0] for i in self.maximum_contrast):
            message = (message
                       + '\nMaximum Contrast: %.2f'
                       % np.real(self.maximum_contrast[0]) + ' %.2ej'
                       % np.imag(self.maximum_contrast[0]))
        else:
            message = (message + '\nMaximum Contrast: '
                       + str(self.maximum_contrast))
        if all(i == self.maximum_object_size[0]
               for i in self.maximum_object_size):
            message = (message + '\nMaximum Object Size: %.1f [lambda]'
                       % self.maximum_object_size[0])
        else:
            message = (message + '\nMaximum Object Size: '
                       + str(self.maximum_object_size))
        if all(i == self.maximum_contrast_density[0]
               for i in self.maximum_contrast_density):
            message = (message + '\nMaximum Constrast Density: %.1f'
                       % np.real(self.maximum_contrast_density[0]) + ' + %.2ej'
                       % np.imag(self.maximum_contrast_density[0]))
        else:
            message = (message + '\nMaximum Contrast Density: '
                       + str(self.maximum_contrast_density))
        if all(i == 0 for i in self.noise):
            message = message + '\nNoise levels: None'
        elif all(i == self.noise[0] for i in self.noise):
            message = message + '\nNoise levels: %.1e' % self.noise[0]
        else:
            message = message + '\nNoise levels: ' + str(self.noise)
        message = message + '\nMap pattern: ' + self.map_pattern
        if self.sample_size is not None:
            message = message + '\nSample Size: %d' % self.sample_size
        message = message + 'Study residual error: ' + str(self.study_residual)
        message = message + 'Study map error: ' + str(self.study_map)
        message = (message + 'Study intern field error: '
                   + str(self.study_internfield))
        if self.configurations is not None and len(self.configurations) > 0:
            message = message + '\nConfiguration names:'
            for i in range(len(self.configurations)-1):
                message = message + ' ' + self.configurations[i].name + ','
            message = message + ' ' + self.configurations[-1].name
        if self.methods is not None and len(self.methods) > 0:
            message = message + '\nMethods:'
            for i in range(len(self.methods)-1):
                message = message + ' ' + self.methods[i].alias + ','
            message = message + ' ' + self.methods[-1].alias
        if self.forward_solver is not None:
            message = message + '\nForward solver: ' + self.forward_solver.name
        if self.synthetization_resolution is not None:
            message = message + '\nSynthetization resolutions: '
            for j in range(len(self.configurations)):
                message = message + 'Configuration %d: [' % (j+1)
                for i in range(len(self.synthetization_resolution)-1):
                    message = (message + '%dx'
                               % self.synthetization_resolution[i][j][0]
                               + '%d, '
                               % self.synthetization_resolution[i][j][1])
                message = (message + '%dx'
                           % self.synthetization_resolution[-1][j][0]
                           + '%d], '
                           % self.synthetization_resolution[-1][j][1])
            message = message[:-2]
        if self.recover_resolution is not None:
            message = message + '\nRecover resolutions: '
            for j in range(len(self.configurations)):
                message = message + 'Configuration %d: [' % (j+1)
                for i in range(len(self.recover_resolution)-1):
                    message = (message + '%dx'
                               % self.recover_resolution[i][j][0]
                               + '%d, '
                               % self.recover_resolution[i][j][1])
                message = (message + '%dx'
                           % self.recover_resolution[-1][j][0]
                           + '%d], '
                           % self.recover_resolution[-1][j][1])
            message = message[:-2]
        if self.scenarios is not None:
            message = (message + '\nNumber of scenarios: %d'
                       % (len(self.scenarios)*len(self.scenarios[0])
                          * len(self.scenarios[0][0])))
        return message

    def save(self, file_path=''):
        data = {
            NAME: self.name,
            CONFIGURATIONS: self.configurations,
            SCENARIOS: self.scenarios,
            METHODS: self.methods,
            MAXIMUM_CONTRAST: self.maximum_contrast,
            MAXIMUM_OBJECT_SIZE: self.maximum_object_size,
            MAXIMUM_CONTRAST_DENSITY: self.maximum_contrast_density,
            NOISE: self.noise,
            MAP_PATTERN: self.map_pattern,
            SAMPLE_SIZE: self.sample_size,
            SYNTHETIZATION_RESOLUTION: self.synthetization_resolution,
            RECOVER_RESOLUTION: self.recover_resolution,
            FORWARD_SOLVER: self.forward_solver,
            STUDY_RESIDUAL: self.study_residual,
            STUDY_MAP: self.study_map,
            STUDY_INTERNFIELD: self.study_internfield,
            RESULTS: self.results
        }

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import data from a saved object."""
        data = cfg.import_dict(file_name, file_path)
        self.name = data[NAME]
        self.configurations = data[CONFIGURATIONS]
        self.scenarios = data[SCENARIOS]
        self.methods = data[METHODS]
        self.maximum_contrast = data[MAXIMUM_CONTRAST]
        self.maximum_object_size = data[MAXIMUM_OBJECT_SIZE]
        self.maximum_contrast_density = data[MAXIMUM_CONTRAST_DENSITY]
        self.noise = data[NOISE]
        self.map_pattern = data[MAP_PATTERN]
        self.sample_size = data[SAMPLE_SIZE]
        self.synthetization_resolution = data[SYNTHETIZATION_RESOLUTION]
        self.recover_resolution = data[RECOVER_RESOLUTION]
        self.forward_solver = data[FORWARD_SOLVER]
        self.study_internfield = data[STUDY_INTERNFIELD]
        self.study_residual = data[STUDY_RESIDUAL]
        self.study_map = data[STUDY_MAP]
        self.results = data[RESULTS]

    def get_final_value_over_samples(self, group_idx=0, config_idx=0,
                                     method_idx=0, measure=None):
        if measure is None:
            raise error.MissingInputError('Experiments.get_final_value_over_'
                                          + 'samples', 'measure')
        g, c, m = group_idx, config_idx, method_idx
        data = np.zeros(self.sample_size)
        for i in range(self.sample_size):
            if measure == 'zeta_rn':
                data[i] = self.results[g][c][i][m].zeta_rn[-1]
            elif measure == 'zeta_rpad':
                data[i] = self.results[g][c][i][m].zeta_rpad[-1]
            elif measure == 'zeta_epad':
                data[i] = self.results[g][c][i][m].zeta_epad[-1]
            elif measure == 'zeta_ebe':
                data[i] = self.results[g][c][i][m].zeta_ebe[-1]
            elif measure == 'zeta_eoe':
                data[i] = self.results[g][c][i][m].zeta_eoe[-1]
            elif measure == 'zeta_sad':
                data[i] = self.results[g][c][i][m].zeta_sad[-1]
            elif measure == 'zeta_sbe':
                data[i] = self.results[g][c][i][m].zeta_sbe[-1]
            elif measure == 'zeta_soe':
                data[i] = self.results[g][c][i][m].zeta_soe[-1]
            elif measure == 'zeta_tfmpad':
                data[i] = self.results[g][c][i][m].zeta_tfmpad[-1]
            elif measure == 'zeta_tfppad':
                data[i] = self.results[g][c][i][m].zeta_tfppad[-1]
            elif measure == 'zeta_be':
                data[i] = self.results[g][c][i][m].zeta_be[-1]
            else:
                raise error.WrongValueInput('Experiments.get_final_value_over_'
                                            + 'samples', 'measure',
                                            "'zeta_rn'/'zeta_rpad'/"
                                            + "'zeta_epad'/'zeta_ebe'/"
                                            + "'zeta_eoe'/'zeta_sad'/"
                                            + "'zeta_sbe'/'zeta_soe'/'zeta_be'"
                                            + "/'zeta_tfmpad'/'zeta_tfppad'",
                                            measure)
        return data

    def get_measure_set(self, config_idx=0):
        measures = []
        if self.study_residual:
            measures.append('zeta_rn')
            measures.append('zeta_rpad')
        if self.study_map:
            if self.configurations[config_idx].perfect_dielectric:
                if self.scenarios[0][config_idx][0].homogeneous_objects:
                    measures.append('zeta_epad')
                    measures.append('zeta_ebe')
                    measures.append('zeta_eoe')
                else:
                    measures.append('zeta_epad')
            elif self.configurations[config_idx].good_conductor:
                if self.scenarios[0][config_idx][0].homogeneous_objects:
                    measures.append('zeta_sad')
                    measures.append('zeta_sbe')
                    measures.append('zeta_soe')
                else:
                    measures.append('zeta_sad')
            else:
                if self.scenarios[0][config_idx][0].homogeneous_objects:
                    measures.append('zeta_epad')
                    measures.append('zeta_ebe')
                    measures.append('zeta_eoe')
                    measures.append('zeta_sad')
                    measures.append('zeta_sbe')
                    measures.append('zeta_soe')
                else:
                    measures.append('zeta_epad')
                    measures.append('zeta_sad')
            if self.scenarios[0][config_idx][0].homogeneous_objects:
                measures.append('zeta_be')
        if self.study_internfield:
            measures.append('zeta_tfmpad')
            measures.append('zeta_tfppad')
        return measures


def ttest_ind_nonequalvar(y1, y2, alpha=0.05):
    n1, n2 = y1.size, y2.size
    y1h, y2h = np.mean(y1), np.mean(y2)
    S12, S22 = np.sum((y1-y1h)**2)/(n1-1), np.sum((y2-y2h)**2)/(n2-1)
    t0 = (y1h-y2h)/np.sqrt(S12/n1 + S22/n2)
    nu = (S12/n1 + S22/n2)**2/((S12/n1)**2/(n1-1) + (S22/n2)**2/(n2-1))
    ta, tb = sc.stats.t.ppf(alpha/2, nu), sc.stats.t.ppf(1-alpha/2, nu)
    if ta > t0 or tb < t0:
        return False
    else:
        return True


def dunnettest(y0, y):
    """alpha = 0.05"""
    warnings.filterwarnings('ignore', message='Covariance of the parameters '
                            + 'could not be estimated')
    Am1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    F = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                  16, 17, 18, 19, 20, 24, 30, 40, 60, 120, 1e20])
    D = np.array([[2.57, 3.03, 3.29, 3.48, 3.62, 3.73, 3.82, 3.90, 3.97],
                  [2.45, 2.86, 3.10, 3.26, 3.39, 3.49, 3.57, 3.64, 3.71],
                  [2.36, 2.75, 2.97, 3.12, 3.24, 3.33, 3.41, 3.47, 3.53],
                  [2.31, 2.67, 2.88, 3.02, 3.13, 3.22, 3.29, 3.35, 3.41],
                  [2.26, 2.61, 2.81, 2.95, 3.05, 3.14, 3.20, 3.26, 3.32],
                  [2.23, 2.57, 2.76, 2.89, 2.99, 3.07, 3.14, 3.19, 3.24],
                  [2.20, 2.53, 2.72, 2.84, 2.94, 3.02, 3.08, 3.14, 3.19],
                  [2.18, 2.50, 2.68, 2.81, 2.90, 2.98, 3.04, 3.09, 3.14],
                  [2.16, 2.48, 2.65, 2.78, 2.87, 2.94, 3.00, 3.06, 3.10],
                  [2.14, 2.46, 2.63, 2.75, 2.84, 2.91, 2.97, 3.02, 3.07],
                  [2.13, 2.44, 2.61, 2.73, 2.82, 2.89, 2.95, 3.00, 3.04],
                  [2.12, 2.42, 2.59, 2.71, 2.80, 2.87, 2.92, 2.97, 3.02],
                  [2.11, 2.41, 2.58, 2.69, 2.78, 2.85, 2.90, 2.95, 3.00],
                  [2.10, 2.40, 2.56, 2.68, 2.76, 2.83, 2.89, 2.94, 2.98],
                  [2.09, 2.39, 2.55, 2.66, 2.75, 2.81, 2.87, 2.92, 2.96],
                  [2.09, 2.38, 2.54, 2.65, 2.73, 2.80, 2.86, 2.90, 2.95],
                  [2.06, 2.35, 2.51, 2.61, 2.70, 2.76, 2.81, 2.86, 2.90],
                  [2.04, 2.32, 2.47, 2.58, 2.66, 2.72, 2.77, 2.82, 2.86],
                  [2.02, 2.29, 2.44, 2.54, 2.62, 2.68, 2.73, 2.77, 2.81],
                  [2.00, 2.27, 2.41, 2.51, 2.58, 2.64, 2.69, 2.73, 2.77],
                  [1.98, 2.24, 2.38, 2.47, 2.55, 2.60, 2.65, 2.69, 2.73],
                  [1.96, 2.21, 2.35, 2.44, 2.51, 2.57, 2.61, 2.65, 2.69]])

    if type(y) is list:
        a = 1 + len(y)
        N = y0.size
        n = []
        for i in range(len(y)):
            N += y[i].size
            n.append(y[i].size)
        SSE = np.sum((y0-np.mean(y0))**2)
        yh = np.zeros(len(y))
        for i in range(len(y)):
            yh[i] = np.mean(y[i])
            SSE += np.sum((y[i]-yh[i])**2)
    else:
        a = 1 + y.shape[0]
        N = y0.size + y.size
        SSE = np.sum((y0-np.mean(y0))**2)
        yh = np.zeros(y.shape[0])
        n = y.shape[1]*np.ones(y.shape[0])
        for i in range(y.shape[0]):
            yh[i] = np.mean(y[i, :])
            SSE += np.sum((y[i, :]-yh[i])**2)
    MSE = SSE/(N-a)
    f = N-a

    if a-1 < 10:
        popt, _ = curve_fit(fittedcurve, F[:], D[:, a-1],
                            p0=[4.132, -1.204, 1.971],
                            absolute_sigma=False, maxfev=20000)
        d = fittedcurve(f, popt[0], popt[1], popt[2])
    else:
        for i in range(F.size):
            if F-f >= 0:
                break
        popt, _ = curve_fit(fittedcurve, Am1, D[i, :],
                            absolute_sigma=False, maxfev=20000)
        d = fittedcurve(a-1, popt[0], popt[1], popt[2])

    null_hypothesis = []
    y0h = np.mean(y0)
    na = y0.size
    for i in range(a-1):
        if np.abs(yh[i]-y0h) > d*np.sqrt(MSE*(1/n[i]+1/na)):
            null_hypothesis.append(False)
        else:
            null_hypothesis.append(True)
    return null_hypothesis


def fittedcurve(x, a, b, c):
    return a*x**b+c


def data_transformation(data):
    if scipy.stats.shapiro(np.log(data))[1] > .05:
        return np.log(data), 'log'
    elif scipy.stats.shapiro(np.sqrt(data))[1] > .05:
        return np.sqrt(data), 'sqrt'
    else:
        return None


def normalitiyplot(data, axes=None, title=None):
    if axes is None:
        fig = plt.figure()
        axes = rst.get_single_figure_axes(fig)
    else:
        fig = None

    pg.qqplot(data, dist='norm', ax=axes)

    if title is not None:
        axes.set_title(title)
    plt.grid()

    return fig


def confintplot(data, axes=None, xlabel=None, ylabel=None, title=None):
    if type(data) is np.ndarray:
        y = []
        for i in range(data.shape[1]):
            info = stats.weightstats.DescrStatsW(data[:, i])
            cf = info.tconfint_mean()
            y.append((cf[0], info.mean, cf[1]))
    elif type(data) is list:
        y = data.copy()
    else:
        raise error.WrongTypeInput('confintplot', 'data', 'list or ndarray',
                                   str(type(data)))

    if axes is None:
        fig = plt.figure()
        axes = rst.get_single_figure_axes(fig)
    else:
        fig = None

    for i in range(len(y)):
        axes.plot(y[i][::2], [i, i], 'k')
        axes.plot(y[i][0], i, '|k', markersize=20)
        axes.plot(y[i][2], i, '|k', markersize=20)
        axes.plot(y[i][1], i, 'ok')

    plt.grid()
    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        plt.yticks(range(len(y)), ylabel, y=.5)
        axes.set_ylim(ymin=-1, ymax=len(y))
    if title is not None:
        axes.set_title(title)

    return fig


def boxplot(data, axes=None, meanline=False, labels=None, xlabel=None,
            ylabel=None, color='b', legend=None, title=None):
    if axes is None:
        fig = plt.figure()
        axes = rst.get_single_figure_axes(fig)

    bplot = axes.boxplot(data, patch_artist=True, labels=labels)
    for i in range(len(data)):
        bplot['boxes'][i].set_facecolor(color)

    if meanline:
        M = len(data)
        x = np.array([0.5, M+.5])
        means = np.zeros(M)
        for m in range(M):
            means[m] = np.mean(data[m])
        a, b = scipy.stats.linregress(np.arange(1, M+1), means)[:2]
        if legend is not None:
            axes.plot(x, a*x + b, '--', color=color, label=legend)
            plt.legend()
        else:
            axes.plot(x, a*x + b, '--', color=color)

    plt.grid(True)
    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if title is not None:
        axes.set_title(title)

    return axes


def violinplot(data, axes=None, labels=None, xlabel=None, ylabel=None,
               color='b', title=None, show=False, file_name=None, file_path='',
               file_format='eps'):
    plot_opts = {'violin_fc': color,
                 'violin_ec': 'w',
                 'violin_alpha': .2}

    if axes is not None:
        sm.graphics.violinplot(data,
                               ax=axes,
                               labels=labels,
                               plot_opts=plot_opts)            

        if xlabel is not None:
            axes.set_xlabel(xlabel)
        if ylabel is not None:
            axes.set_ylabel(ylabel)
        if title is not None:
            axes.set_title(title)
        axes.grid()

    else:
        sm.graphics.violinplot(data,
                               labels=labels,
                               plot_opts=plot_opts)

        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        plt.grid()

        if show:
            plt.show()
        else:
            if file_name is not None:
                raise error.MissingInputError('boxplot', 'file_name')

            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
            plt.close()


def get_label(measure):
    if measure == 'zeta_rn':
        return rst.LABEL_ZETA_RN
    elif measure == 'zeta_rpad':
        return rst.LABEL_ZETA_RPAD
    elif measure == 'zeta_epad':
        return rst.LABEL_ZETA_EPAD
    elif measure == 'zeta_ebe':
        return rst.LABEL_ZETA_EBE
    elif measure == 'zeta_eoe':
        return rst.LABEL_ZETA_EOE
    elif measure == 'zeta_sad':
        return rst.LABEL_ZETA_SAD
    elif measure == 'zeta_sbe':
        return rst.LABEL_ZETA_SBE
    elif measure == 'zeta_soe':
        return rst.LABEL_ZETA_SOE
    elif measure == 'zeta_tfmpad':
        return rst.LABEL_ZETA_TFMPAD
    elif measure == 'zeta_tfppad':
        return rst.LABEL_ZETA_TFPPAD
    elif measure == 'zeta_be':
        return rst.LABEL_ZETA_BE
    else:
        raise error.WrongValueInput('get_label', 'measure', "'zeta_rn'/"
                                    + "'zeta_rpad'/'zeta_epad'/'zeta_ebe'/"
                                    + "'zeta_eoe'/'zeta_sad'/'zeta_sbe'/"
                                    + "'zeta_soe'/'zeta_be'/'zeta_tfmpad'/"
                                    + "'zeta_tfppad'", measure)


def get_title(measure):
    if measure == 'zeta_rn':
        return 'Residual Norm'
    elif measure == 'zeta_rpad':
        return 'Residual PAD'
    elif measure == 'zeta_epad':
        return 'Rel. Per. PAD'
    elif measure == 'zeta_ebe':
        return 'Rel. Per. Back. PAD'
    elif measure == 'zeta_eoe':
        return 'Rel. Per. Ob. PAD'
    elif measure == 'zeta_sad':
        return 'Con. AD'
    elif measure == 'zeta_sbe':
        return 'Con. Back. AD'
    elif measure == 'zeta_soe':
        return 'Con. Ob. AD'
    elif measure == 'zeta_tfmpad':
        return 'To. Field Mag. PAD'
    elif measure == 'zeta_tfppad':
        return 'To. Field Phase PAD'
    elif measure == 'zeta_be':
        return 'Boundary Error'
    else:
        raise error.WrongValueInput('get_label', 'measure', "'zeta_rn'/"
                                    + "'zeta_rpad'/'zeta_epad'/'zeta_ebe'/"
                                    + "'zeta_eoe'/'zeta_sad'/'zeta_sbe'/"
                                    + "'zeta_soe'/'zeta_be'/'zeta_tfmpad'/"
                                    + "'zeta_tfppad'", measure)


def run_methods(methods, scenario, parallelization=False):
    """Run methods parallely."""
    if parallelization:
        num_cores = multiprocessing.cpu_count()
        output = (Parallel(n_jobs=num_cores)(
            delayed(methods[m].solve)
            (scenario, print_info=False) for m in range(len(methods))
        ))
    results = []
    for m in range(len(methods)):
        if parallelization:
            results.append(output[m])
        else:
            results.append(methods[m].solve(scenario, print_info=False))
    return results


def run_scenarios(method, scenarios, parallelization=False):
    """Run methods parallely."""
    results = []
    if parallelization:
        num_cores = multiprocessing.cpu_count()
        copies = []
        for i in range(len(scenarios)):
            copies.append(cp.deepcopy(method))
        output = (Parallel(n_jobs=num_cores)(
            delayed(copies[i].solve)
            (scenarios[i], print_info=False) for i in range(len(scenarios))
        ))
    for m in range(len(scenarios)):
        if parallelization:
            results.append(output[m])
        else:
            results.append(method.solve(scenarios[i], print_info=False))
    return results


def create_scenario(name, configuration, resolution, map_pattern,
                    maximum_contrast, maximum_contrast_density, noise=None,
                    maximum_object_size=None, compute_residual_error=None,
                    compute_map_error=None, compute_totalfield_error=None):
    """Summarize this method."""
    Lx = configuration.Lx/configuration.lambda_b
    Ly = configuration.Ly/configuration.lambda_b
    epsilon_rb = configuration.epsilon_rb
    sigma_b = configuration.sigma_b
    omega = 2*pi*configuration.f
    homogeneous_objects = False

    if configuration.perfect_dielectric:
        min_sigma = max_sigma = sigma_b
    else:
        min_sigma = 0.
        max_sigma = cfg.get_conductivity(maximum_contrast, omega, epsilon_rb,
                                         sigma_b)

    if configuration.good_conductor:
        min_epsilon_r = max_epsilon_r = epsilon_rb
    else:
        min_epsilon_r = 1.
        max_epsilon_r = cfg.get_relative_permittivity(maximum_contrast,
                                                      epsilon_rb)

    if map_pattern == GEOMETRIC_PATTERN:
        homogeneous_objects = True
        if maximum_object_size is None:
            maximum_object_size = .4*min([Lx, Ly])/2
        dx, dy = Lx/resolution[0], Ly/resolution[1]
        minimum_object_size = 8*max([dx, dy])
        xmin, xmax = cfg.get_bounds(Lx)
        ymin, ymax = cfg.get_bounds(Ly)
        epsilon_r = epsilon_rb*np.ones(resolution)
        sigma = sigma_b*np.ones(resolution)
        chi = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b,
                                   omega)
        while (contrast_density(chi)/np.abs(maximum_contrast) 
               <= .9*maximum_contrast_density):
            radius = minimum_object_size + (maximum_object_size
                                            - minimum_object_size)*rnd.rand()
            epsilon_ro = min_epsilon_r + (max_epsilon_r
                                          - min_epsilon_r)*rnd.rand()
            sigma_o = min_sigma + (max_sigma-min_sigma)*rnd.rand()
            center = [xmin+radius + (xmax-radius-(xmin+radius))*rnd.rand(),
                      ymin+radius + (ymax-radius-(ymin+radius))*rnd.rand()]
            epsilon_r, sigma = draw_random(
                int(np.ceil(15*rnd.rand())), radius, axis_length_x=Lx,
                axis_length_y=Ly, background_relative_permittivity=epsilon_rb,
                background_conductivity=sigma_b,
                object_relative_permittivity=epsilon_ro,
                object_conductivity=sigma_o, center=center,
                relative_permittivity_map=epsilon_r, conductivity_map=sigma
            )
            chi = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b,
                                       omega)
            print(contrast_density(chi)/np.abs(maximum_contrast))

    elif map_pattern == SURFACES_PATTERN:
        if rnd.rand() < .5:
            epsilon_r, sigma = draw_random_waves(
                int(np.ceil(15*rnd.rand())), 10, resolution=resolution,
                rel_permittivity_amplitude=max_epsilon_r-epsilon_rb,
                conductivity_amplitude=max_sigma-sigma_b, axis_length_x=Lx,
                axis_length_y=Ly, background_relative_permittivity=epsilon_rb,
                conductivity_map=sigma_b
            )
        else:
            if maximum_object_size is None:
                maximum_object_size = .4*min([Lx, Ly])/2
            dx, dy = Lx/resolution[0], Ly/resolution[1]
            minimum_object_size = 8*max([dx, dy])
            epsilon_r = epsilon_rb*np.ones(resolution)
            sigma = sigma_b*np.ones(resolution)
            chi = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b,
                                       omega)
            while contrast_density(chi) <= .8*maximum_contrast_density:
                epsilon_r, sigma = draw_random_gaussians(
                    1, maximum_spread=maximum_object_size,
                    minimum_spread=minimum_object_size,
                    rel_permittivity_amplitude=max_epsilon_r,
                    conductivity_amplitude=max_sigma, axis_length_x=Lx,
                    axis_length_y=Ly, background_conductivity=sigma_b,
                    background_relative_permittivity=epsilon_rb,
                    relative_permittivity_map=epsilon_r,
                    conductivity_map=sigma
                )

    scenario = ipt.InputData(name=name,
                             configuration_filename=configuration.name,
                             resolution=resolution,
                             homogeneous_objects=homogeneous_objects,
                             noise=noise)

    if compute_residual_error is not None:
        scenario.compute_residual_error = compute_residual_error
    if compute_map_error is not None:
        scenario.compute_map_error = compute_map_error
    if compute_totalfield_error is not None:
        scenario.compute_totalfield_error = compute_totalfield_error

    if not configuration.good_conductor:
        scenario.epsilon_r = epsilon_r
    if not configuration.perfect_dielectric:
        scenario.sigma = sigma

    return scenario


def contrast_density(contrast_map):
    """Summarize the method."""
    return np.mean(np.abs(contrast_map))


def compute_resolution(wavelength, length_y, length_x,
                       proportion_cell_wavelength):
    """Summarize method."""
    dx = dy = wavelength/proportion_cell_wavelength
    NX = int(np.ceil(length_x/dx))
    NY = int(np.ceil(length_y/dy))
    return NY, NX


def draw_square(side_length, axis_length_x=2., axis_length_y=2.,
                resolution=None, background_relative_permittivity=1.,
                background_conductivity=0., object_relative_permittivity=1.,
                object_conductivity=0., center=[0., 0.],
                relative_permittivity_map=None, conductivity_map=None,
                rotate=0.):
    """Draw a square.

    Parameters
    ----------
        side_length : float
            Length of the side of the square.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_square', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    L = side_length
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) + center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) + center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    epsilon_r[logical_and(logical_and(yp >= -L/2, yp <= L/2),
                          logical_and(xp >= -L/2, xp <= L/2))] = epsilon_ro
    sigma[logical_and(logical_and(yp >= -L/2, yp <= L/2),
                      logical_and(xp >= -L/2, xp <= L/2))] = sigma_o

    return epsilon_r, sigma


def draw_triangle(side_length, axis_length_x=2., axis_length_y=2.,
                  resolution=None, background_relative_permittivity=1.,
                  background_conductivity=0., object_relative_permittivity=1.,
                  object_conductivity=0., center=[0., 0.],
                  relative_permittivity_map=None, conductivity_map=None,
                  rotate=0.):
    """Draw an equilateral triangle.

    Parameters
    ----------
        side_length : float
            Length of the side of the triangle.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_triangle', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    L = side_length
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    epsilon_r[logical_and(logical_and(yp >= -L/2, yp <= 2*xp + L/2),
                          yp <= -2*xp + L/2)] = epsilon_ro
    sigma[logical_and(logical_and(yp >= -L/2, yp <= 2*xp - L/2),
                      yp <= -2*xp + L/2)] = sigma_o

    return epsilon_r, sigma


def draw_6star(side_length, axis_length_x=2., axis_length_y=2.,
               resolution=None, background_relative_permittivity=1.,
               background_conductivity=0., object_relative_permittivity=1.,
               object_conductivity=0., center=[0., 0.],
               relative_permittivity_map=None, conductivity_map=None,
               rotate=0.):
    """Draw a six-pointed star (hexagram).

    Parameters
    ----------
        side_length : float
            Length of the side of each triangle which composes the star.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_star', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    L = side_length
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    epsilon_r[logical_and(logical_and(yp >= -L/4, yp <= 3/2*xp + L/2),
                          yp <= -3/2*xp + L/2)] = epsilon_ro
    sigma[logical_and(logical_and(yp >= -L/4, yp <= 3/2*xp + L/2),
                      yp <= -3/2*xp + L/2)] = sigma_o

    epsilon_r[logical_and(logical_and(yp <= L/4, yp >= 3/2*xp - L/2),
                          yp >= -3/2*xp - L/2)] = epsilon_ro
    sigma[logical_and(logical_and(y <= L/4, yp >= 3/2*xp - L/2),
                      yp >= -3/2*xp-L/2)] = sigma_o

    return epsilon_r, sigma


def draw_ring(inner_radius, outer_radius, axis_length_x=2., axis_length_y=2.,
              resolution=None, background_relative_permittivity=1.,
              background_conductivity=0., object_relative_permittivity=1.,
              object_conductivity=0., center=[0., 0.],
              relative_permittivity_map=None, conductivity_map=None):
    """Draw a ring.

    Parameters
    ----------
        inner_radius, outer_radius : float
            Inner and outer radii of the ring.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_ring', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    ra, rb = inner_radius, outer_radius
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])

    # Set object
    epsilon_r[logical_and(x**2 + y**2 <= rb**2,
                          x**2 + y**2 >= ra**2)] = epsilon_ro
    sigma[logical_and(x**2 + y**2 <= rb**2,
                      x**2 + y**2 >= ra**2)] = sigma_o

    return epsilon_r, sigma


def draw_circle(radius, axis_length_x=2., axis_length_y=2.,
                resolution=None, background_relative_permittivity=1.,
                background_conductivity=0., object_relative_permittivity=1.,
                object_conductivity=0., center=[0., 0.],
                relative_permittivity_map=None, conductivity_map=None):
    """Draw a circle.

    Parameters
    ----------
        radius : float
            Radius of the circle.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_circle', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    r = radius
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])

    # Set object
    epsilon_r[x**2 + y**2 <= r**2] = epsilon_ro
    sigma[x**2 + y**2 <= r**2] = sigma_o

    return epsilon_r, sigma


def draw_ellipse(x_radius, y_radius, axis_length_x=2., axis_length_y=2.,
                 resolution=None, background_relative_permittivity=1.,
                 background_conductivity=0., object_relative_permittivity=1.,
                 object_conductivity=0., center=[0., 0.],
                 relative_permittivity_map=None, conductivity_map=None,
                 rotate=0.):
    """Draw an ellipse.

    Parameters
    ----------
        x_radius, y_radius : float
            Ellipse radii in each axis.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_ellipse', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    a, b = x_radius, y_radius
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    epsilon_r[xp**2/a**2 + yp**2/b**2 <= 1.] = epsilon_ro
    sigma[xp**2/a**2 + yp**2/b**2 <= 1.] = sigma_o

    return epsilon_r, sigma


def draw_cross(height, width, thickness, axis_length_x=2., axis_length_y=2.,
               resolution=None, background_relative_permittivity=1.,
               background_conductivity=0., object_relative_permittivity=1.,
               object_conductivity=0., center=[0., 0.],
               relative_permittivity_map=None, conductivity_map=None,
               rotate=0.):
    """Draw a cross.

    Parameters
    ----------
        height, width, thickness : float
            Parameters of the cross.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_cross', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    horizontal_bar = (
        logical_and(xp >= -width/2,
                    logical_and(xp <= width/2,
                                logical_and(yp >= -thickness/2,
                                            yp <= thickness/2)))
    )
    vertical_bar = (
        logical_and(y >= -height/2,
                    logical_and(y <= height/2,
                                logical_and(x >= -thickness/2,
                                            x <= thickness/2)))
    )
    epsilon_r[np.logical_or(horizontal_bar, vertical_bar)] = epsilon_ro
    sigma[np.logical_or(horizontal_bar, vertical_bar)] = sigma_o

    return epsilon_r, sigma


def draw_line(length, thickness, axis_length_x=2., axis_length_y=2.,
              resolution=None, background_relative_permittivity=1.,
              background_conductivity=0., object_relative_permittivity=1.,
              object_conductivity=0., center=[0., 0.],
              relative_permittivity_map=None, conductivity_map=None,
              rotate=0.):
    """Draw a cross.

    Parameters
    ----------
        length, thickness : float
            Parameters of the line.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_line', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    line = (logical_and(xp >= -length/2,
                        logical_and(xp <= length/2,
                                    logical_and(yp >= -thickness/2,
                                                yp <= thickness/2))))
    epsilon_r[line] = epsilon_ro
    sigma[line] = sigma_o

    return epsilon_r, sigma


def draw_polygon(number_sides, radius, axis_length_x=2., axis_length_y=2.,
                 resolution=None, background_relative_permittivity=1.,
                 background_conductivity=0., object_relative_permittivity=1.,
                 object_conductivity=0., center=[0., 0.],
                 relative_permittivity_map=None, conductivity_map=None,
                 rotate=0.):
    """Draw a polygon with equal sides.

    Parameters
    ----------
        number_sides : int
            Number of sides.

        radius : float
            Radius from the center to each vertex.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_polygon', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    dphi = 2*pi/number_sides
    phi = np.arange(0, number_sides*dphi, dphi)
    xa = radius*np.cos(phi)
    ya = radius*np.sin(phi)
    polygon = np.ones(x.shape, dtype=bool)
    for i in range(number_sides):
        a = -(ya[i]-ya[i-1])
        b = xa[i]-xa[i-1]
        c = (xa[i]-xa[i-1])*ya[i-1] - (ya[i]-ya[i-1])*xa[i-1]
        polygon = logical_and(polygon, a*xp + b*yp >= c)
    epsilon_r[polygon] = epsilon_ro
    sigma[polygon] = sigma_o

    return epsilon_r, sigma


def isleft(x0, y0, x1, y1, x2, y2):
    r"""Check if a point is left, on, right of an infinite line.

    The point to be tested is (x2, y2). The infinite line is defined by
    (x0, y0) -> (x1, y1).

    Parameters
    ----------
        x0, y0 : float
            A point within the infinite line.

        x1, y1 : float
            A point within the infinite line.

        x2, y2 : float
            The point to be tested.

    Returns
    -------
        * float < 0, if it is on the left.
        * float = 0, if it is on the line.
        * float > 0, if it is on the left.

    References
    ----------
    .. [1] Sunday, D 2012, Inclusion of a Point in a Polygon, accessed
       15 July 2020, <http://geomalgorithms.com/a03-_inclusion.html>
    """
    return (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)


def winding_number(x, y, xv, yv):
    r"""Check if a point is within a polygon.

    The method determines if a point is within a polygon through the
    Winding Number Algorithm. If this number is zero, then it means that
    the point is out of the polygon. Otherwise, it is within the
    polygon.

    Parameters
    ----------
        x, y : float
            The point that should be tested.

        xv, yv : :class:`numpy.ndarray`
            A 1-d array with vertex points of the polygon.

    Returns
    -------
        bool

    References
    ----------
    .. [1] Sunday, D 2012, Inclusion of a Point in a Polygon, accessed
       15 July 2020, <http://geomalgorithms.com/a03-_inclusion.html>
    """
    # The first vertex must come after the last one within the array
    if xv[-1] != xv[0] or yv[-1] != yv[0]:
        _xv = np.hstack((xv.flatten(), xv[0]))
        _yv = np.hstack((yv.flatten(), yv[0]))
        n = xv.size
    else:
        _xv = np.copy(xv)
        _yv = np.copy(yv)
        n = xv.size-1

    wn = 0  # the  winding number counter

    # Loop through all edges of the polygon
    for i in range(n):  # edge from V[i] to V[i+1]

        if (_yv[i] <= y):  # start yv <= y
            if (_yv[i+1] > y):  # an upward crossing
                # P left of edge
                if (isleft(_xv[i], _yv[i], _xv[i+1], _yv[i+1], x, y) > 0):
                    wn += 1  # have  a valid up intersect

        else:  # start yv > y (no test needed)
            if (_yv[i+1] <= y):  # a downward crossing
                # P right of edge
                if (isleft(_xv[i], _yv[i], _xv[i+1], _yv[i+1], x, y) < 0):
                    wn -= 1  # have  a valid down intersect
    if wn == 0:
        return False
    else:
        return True


def draw_random(number_sides, maximum_radius, axis_length_x=2.,
                axis_length_y=2., resolution=None,
                background_relative_permittivity=1.,
                background_conductivity=0., object_relative_permittivity=1.,
                object_conductivity=0., center=[0., 0.],
                relative_permittivity_map=None, conductivity_map=None):
    """Draw a random polygon.

    Parameters
    ----------
        number_sides : int
            Number of sides of the polygon.

        maximum_radius : float
            Maximum radius from the origin to each vertex.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_random', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])

    # Create vertices
    # phi = np.sort(2*pi*rnd.rand(number_sides))
    phi = rnd.normal(loc=np.linspace(0, 2*pi, number_sides, endpoint=False),
                     scale=0.5)
    phi[phi >= 2*pi] = phi[phi >= 2*pi] - np.floor(phi[phi >= 2*pi]
                                                   / (2*pi))*2*pi
    phi[phi < 0] = -((np.floor(phi[phi < 0]/(2*pi)))*2*pi - phi[phi < 0])
    phi = np.sort(phi)
    radius = maximum_radius*rnd.rand(number_sides)
    xv = radius*np.cos(phi)
    yv = radius*np.sin(phi)

    # Set object
    for i in range(NX):
        for j in range(NY):
            if winding_number(x[j, i], y[j, i], xv, yv):
                epsilon_r[j, i] = epsilon_ro
                sigma[j, i] = sigma_o

    return epsilon_r, sigma


def draw_rhombus(length, axis_length_x=2., axis_length_y=2., resolution=None,
                 background_relative_permittivity=1.,
                 background_conductivity=0., object_relative_permittivity=1.,
                 object_conductivity=0., center=[0., 0.],
                 relative_permittivity_map=None, conductivity_map=None,
                 rotate=0.):
    """Draw a rhombus.

    Parameters
    ----------
        length : float
            Side length.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default: None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_rhombus', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    a = length/np.sqrt(2)
    rhombus = logical_and(-a*xp - a*yp >= -a**2,
                          logical_and(a*xp - a*yp >= -a**2,
                                      logical_and(a*xp+a*yp >= -a**2,
                                                  -a*xp+a*yp >= -a**2)))
    epsilon_r[rhombus] = epsilon_ro
    sigma[rhombus] = sigma_o

    return epsilon_r, sigma


def draw_trapezoid(upper_length, lower_length, height, axis_length_x=2.,
                   axis_length_y=2., resolution=None,
                   background_relative_permittivity=1.,
                   background_conductivity=0., object_relative_permittivity=1.,
                   object_conductivity=0., center=[0., 0.],
                   relative_permittivity_map=None, conductivity_map=None,
                   rotate=0.):
    """Draw a trapezoid.

    Parameters
    ----------
        upper_length, lower_length, height : float
            Dimensions.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default: None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_trapezoid',
                                      'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    ll, lu, h = lower_length, upper_length, height
    a1, b1, c1 = -h, (lu-ll)/2, -(lu-ll)*h/4 - h*ll/2
    a2, b2, c2 = h, (lu-ll)/2, (lu-ll)*h/4 - h*lu/2
    trapezoid = logical_and(a1*xp + b1*yp >= c1,
                            logical_and(a2*xp + b2*yp >= c2,
                                        logical_and(yp <= height/2,
                                                    yp >= -height/2)))

    epsilon_r[trapezoid] = epsilon_ro
    sigma[trapezoid] = sigma_o

    return epsilon_r, sigma


def draw_parallelogram(length, height, inclination, axis_length_x=2.,
                       axis_length_y=2., resolution=None,
                       background_relative_permittivity=1.,
                       background_conductivity=0.,
                       object_relative_permittivity=1., object_conductivity=0.,
                       center=[0., 0.], relative_permittivity_map=None,
                       conductivity_map=None, rotate=0.):
    """Draw a paralellogram.

    Parameters
    ----------
        length, height : float
            Dimensions.

        inclination : float
            In degrees.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default: None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_parallelogram',
                                      'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    l, h, a = length, height, height/2/np.tan(np.deg2rad(90-inclination))
    parallelogram = logical_and(-h*xp + 2*a*yp >= 2*a*(l/2-a)-h*(l/2-a),
                                logical_and(h*xp-2*a*yp >= h*(a-l/2)-a*h,
                                            logical_and(yp <= height/2,
                                                        yp >= -height/2)))

    epsilon_r[parallelogram] = epsilon_ro
    sigma[parallelogram] = sigma_o

    return epsilon_r, sigma


def draw_5star(radius, axis_length_x=2., axis_length_y=2.,
               resolution=None, background_relative_permittivity=1.,
               background_conductivity=0., object_relative_permittivity=1.,
               object_conductivity=0., center=[0., 0.], rotate=0.,
               relative_permittivity_map=None, conductivity_map=None):
    """Draw a 5-point star.

    Parameters
    ----------
        radius : int
            Length from the center of the star to the main vertices.

        maximum_radius : float
            Maximum radius from the origin to each vertex.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_random', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Create vertices
    delta = 2*pi/5
    phi = np.array([0, 2*delta, 4*delta, delta, 3*delta, 0]) + pi/2 - 2*pi/5
    xv = radius*np.cos(phi)
    yv = radius*np.sin(phi)

    # Set object
    for i in range(NX):
        for j in range(NY):
            if winding_number(xp[j, i], yp[j, i], xv, yv):
                epsilon_r[j, i] = epsilon_ro
                sigma[j, i] = sigma_o

    return epsilon_r, sigma


def draw_4star(radius, axis_length_x=2., axis_length_y=2., resolution=None,
               background_relative_permittivity=1.,
               background_conductivity=0., object_relative_permittivity=1.,
               object_conductivity=0., center=[0., 0.], rotate=0.,
               relative_permittivity_map=None, conductivity_map=None):
    """Draw a 4-point star.

    Parameters
    ----------
        radius : float
            Radius of the vertex from the center of the star.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_relative_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default: None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_4star', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_relative_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    a, b = radius, .5*radius
    rhombus1 = logical_and(-a*xp - b*yp >= -a*b,
                           logical_and(a*xp - b*yp >= -a*b,
                                       logical_and(a*xp+b*yp >= -a*b,
                                                   -a*xp+b*yp >= -a*b)))
    rhombus2 = logical_and(-b*xp - a*yp >= -a*b,
                           logical_and(b*xp - a*yp >= -a*b,
                                       logical_and(b*xp+a*yp >= -a*b,
                                                   -b*xp+a*yp >= -a*b)))
    epsilon_r[np.logical_or(rhombus1, rhombus2)] = epsilon_ro
    sigma[np.logical_or(rhombus1, rhombus2)] = sigma_o

    return epsilon_r, sigma


def draw_wave(number_peaks, rel_permittivity_peak=1., conductivity_peak=0.,
              rel_permittivity_valley=None, conductivity_valley=None,
              resolution=None, number_peaks_y=None, axis_length_x=2.,
              axis_length_y=2., background_relative_permittivity=1.,
              background_conductivity=0., object_relative_permittivity=1.,
              object_conductivity=0., relative_permittivity_map=None,
              conductivity_map=None, wave_bounds_proportion=(1., 1.),
              center=[0., 0.], rotate=0.):
    """Draw waves.

    Parameters
    ----------
        number_peaks : int
            Number of peaks for both direction or for x-axis (if
            `number_peaks_x` is not None).

        number_peaks_y : float, optional
            Number of peaks in y-direction.

        wave_bounds_proportion : 2-tuple
            The wave may be placed only at a rectangular area of the
            image controlled by this parameters. The values should be
            proportional to `axis_length_y` and `axis_length_x`,
            respectively, i.e, the values should be > 0. and < 1. Then,
            you may control center and rotation of the figure.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        rel_permittivity_peak : float, default: 1.0
            Peak value of relative permittivity.

        rel_permittivity_valley : None or float
            Valley value of relative permittivity. If None, then peak
            value is assumed.

        conductivity_peak : float, default: 1.0
            Peak value of conductivity.

        conductivity_valley : None or float
            Valley value of conductivity. If None, then peak value
            is assumed.

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default: None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_wave', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) + center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) + center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Wave area
    ly, lx = wave_bounds_proportion[0]*Ly, wave_bounds_proportion[1]*Lx
    wave = logical_and(xp >= -lx/2,
                       logical_and(xp <= lx/2,
                                   logical_and(yp >= -ly/2, yp <= ly/2)))

    # Wave parameters
    number_peaks_x = number_peaks
    if number_peaks_y is None:
        number_peaks_y = number_peaks
    Kx = 2*number_peaks_x-1
    Ky = 2*number_peaks_y-1

    # Set up valley magnitude
    if (rel_permittivity_peak == background_relative_permittivity
            and rel_permittivity_valley is None):
        rel_permittivity_valley = background_relative_permittivity
    elif rel_permittivity_valley is None:
        rel_permittivity_valley = rel_permittivity_peak
    if (conductivity_peak == background_conductivity
            and conductivity_valley is None):
        conductivity_valley = background_conductivity
    elif conductivity_valley is None:
        conductivity_valley = conductivity_peak

    # Relative permittivity
    epsilon_r[wave] = (np.cos(2*pi/(2*lx/Kx)*xp[wave])
                       * np.cos(2*pi/(2*ly/Ky)*yp[wave]))
    epsilon_r[logical_and(wave, epsilon_r >= 0)] = (
        rel_permittivity_peak*epsilon_r[logical_and(wave, epsilon_r >= 0)]
    )
    epsilon_r[logical_and(wave, epsilon_r < 0)] = (
        rel_permittivity_valley*epsilon_r[logical_and(wave, epsilon_r < 0)]
    )
    epsilon_r[wave] = epsilon_r[wave] + epsilon_rb
    epsilon_r[logical_and(wave, epsilon_r < 1.)] = 1.

    # Conductivity
    sigma[wave] = (np.cos(2*pi/(2*lx/Kx)*xp[wave])
                   * np.cos(2*pi/(2*ly/Ky)*yp[wave]))
    sigma[logical_and(wave, epsilon_r >= 0)] = (
        conductivity_peak*sigma[logical_and(wave, sigma >= 0)]
    )
    sigma[logical_and(wave, sigma < 0)] = (
        conductivity_valley*sigma[logical_and(wave, sigma < 0)]
    )
    sigma[wave] = sigma[wave] + sigma_b
    sigma[logical_and(wave, sigma < 0.)] = 0.

    return epsilon_r, sigma


def draw_random_waves(number_waves, maximum_number_peaks,
                      maximum_number_peaks_y=None, resolution=None,
                      rel_permittivity_amplitude=0., conductivity_amplitude=0.,
                      axis_length_x=2., axis_length_y=2.,
                      background_relative_permittivity=1.,
                      background_conductivity=0.,
                      relative_permittivity_map=None, conductivity_map=None,
                      wave_bounds_proportion=(1., 1.), center=[0., 0.],
                      rotate=0., edge_smoothing=0.03):
    """Draw random waves.

    Parameters
    ----------
        number_waves : int
            Number of wave components.

        maximum_number_peaks : int
            Different wavelengths are considered. The maximum number of
            peaks controls the size of the smallest possible wavelength.

        maximum_number_peaks_y : float, optional
            Maximum number of peaks in y-direction. If None, then it
            will be the same as `maximum_number_peaks`.

        wave_bounds_proportion : 2-tuple
            The wave may be placed only at a rectangular area of the
            image controlled by this parameter. The values should be
            proportional to `axis_length_y` and `axis_length_x`,
            respectively, i.e, the values should be > 0. and < 1. Then,
            you may control center and rotation of the figure.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        rel_permittivity_amplitude : float, default: 1.0
            Maximum amplitude of relative permittivity variation

        conductivity_amplitude : float, default: 1.0
            Maximum amplitude of conductivity variation

        conductivity_valley : None or float
            Valley value of conductivity. If None, then peak value
            is assumed.

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default: None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.

        edge_smoothing : float, default: 0.03
            Percentage of cells at the boundary of the wave area which
            will be smoothed.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_random_waves',
                                      'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) + center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) + center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Wave area
    ly, lx = wave_bounds_proportion[0]*Ly, wave_bounds_proportion[1]*Lx
    wave = logical_and(xp >= -lx/2,
                       logical_and(xp <= lx/2,
                                   logical_and(yp >= -ly/2, yp <= ly/2)))

    # Wave parameters
    max_number_peaks_x = maximum_number_peaks
    if maximum_number_peaks_y is None:
        max_number_peaks_y = maximum_number_peaks
    m = np.round((max_number_peaks_x-1)*rnd.rand(number_waves)) + 1
    n = np.round((max_number_peaks_y-1)*rnd.rand(number_waves)) + 1
    lam_x = lx/m
    lam_y = ly/n
    phi = 2*pi*rnd.rand(2, number_waves)
    peaks = rnd.rand(number_waves)

    # Boundary smoothing
    bd = np.ones(xp.shape)
    nx, ny = np.round(edge_smoothing*NX), np.round(edge_smoothing*NY)
    left_bd = logical_and(xp >= -lx/2, xp <= -lx/2+nx*dx)
    right_bd = logical_and(xp >= lx/2-nx*dx, xp <= lx/2)
    lower_bd = logical_and(yp >= -ly/2, yp <= -ly/2+ny*dy)
    upper_bd = logical_and(yp >= ly/2-ny*dy, yp <= ly/2)
    edge1 = logical_and(left_bd, lower_bd)
    edge2 = logical_and(left_bd, upper_bd)
    edge3 = logical_and(upper_bd, right_bd)
    edge4 = logical_and(right_bd, lower_bd)
    f_left = (2/nx/dx)*(xp+lx/2) - (1/nx**2/dx**2)*(xp + lx/2)**2
    f_right = ((2/nx/dx)*(xp-(lx/2-2*nx*dx))
               - (1/nx**2/dx**2)*(xp-(lx/2-2*nx*dx))**2)
    f_lower = (2/ny/dy)*(yp+ly/2) - (1/ny**2/dy**2)*(yp + ly/2)**2
    f_upper = (((2/ny/dy)*(yp-(ly/2-2*ny*dy))
                - (1/ny**2/dy**2)*(yp-(ly/2-2*nx*dy))**2))
    bd[left_bd] = f_left[left_bd]
    bd[right_bd] = f_right[right_bd]
    bd[lower_bd] = f_lower[lower_bd]
    bd[upper_bd] = f_upper[upper_bd]
    bd[edge1] = f_left[edge1]*f_lower[edge1]
    bd[edge2] = f_left[edge2]*f_upper[edge2]
    bd[edge3] = f_upper[edge3]*f_right[edge3]
    bd[edge4] = f_right[edge4]*f_lower[edge4]
    bd[np.logical_not(wave)] = 1.

    # Relative permittivity
    for i in range(number_waves):
        epsilon_r[wave] = (epsilon_r[wave]
                           + peaks[i]*np.cos(2*pi/(lam_x[i])*xp[wave]
                                             - phi[0, i])
                           * np.cos(2*pi/(lam_y[i])*yp[wave] - phi[1, i]))
    epsilon_r[wave] = (rel_permittivity_amplitude*epsilon_r[wave]
                       / np.amax(epsilon_r[wave]))
    epsilon_r[wave] = epsilon_r[wave] + epsilon_rb
    epsilon_r = epsilon_r*bd
    epsilon_r[logical_and(wave, epsilon_r < 1.)] = 1.

    # Conductivity
    for i in range(number_waves):
        sigma[wave] = (sigma[wave]
                       + peaks[i]*np.cos(2*pi/(lam_x[i])*xp[wave]
                                         - phi[0, i])
                       * np.cos(2*pi/(lam_y[i])*yp[wave] - phi[1, i]))
    sigma[wave] = (conductivity_amplitude*sigma[wave]
                   / np.amax(sigma[wave]))
    sigma[wave] = sigma[wave] + sigma_b
    sigma = sigma*bd
    sigma[logical_and(wave, sigma < 0.)] = 0.

    return epsilon_r, sigma


def draw_random_gaussians(number_distributions, maximum_spread=.8,
                          minimum_spread=.5, distance_from_border=.1,
                          resolution=None, surface_area=(1., 1.),
                          rel_permittivity_amplitude=0.,
                          conductivity_amplitude=0., axis_length_x=2.,
                          axis_length_y=2., background_conductivity=0.,
                          background_relative_permittivity=1.,
                          relative_permittivity_map=None, center=[0., 0.],
                          conductivity_map=None, rotate=0.,
                          edge_smoothing=0.03):
    """Draw random gaussians.

    Parameters
    ----------
        number_distributions : int
            Number of distributions.

        minimum_spread, maximum_spread : float, default: .5 and .8
            Control the spread of the gaussian function, proportional to
            the length of the gaussian area. This means that these
            parameters should be > 0 and < 1. 1 means that :math:`sigma
            = L_x/6`.

        distance_from_border : float, default: .1
            Control the bounds of the center of the distribution. It is
            proportional to the length of the area.

        surface_area : 2-tuple, default: (1., 1.)
            The distribution may be placed only at a rectangular area of
            the image controlled by this parameter. The values should be
            proportional to `axis_length_y` and `axis_length_x`,
            respectively, i.e, the values should be > 0. and < 1. Then,
            you may control center and rotation of the figure.

        wave_bounds_proportion : 2-tuple
            The wave may be placed only at a rectangular area of the
            image controlled by this parameters. The values should be
            proportional to `axis_length_y` and `axis_length_x`,
            respectively, i.e, the values should be > 0. and < 1. Then,
            you may control center and rotation of the figure.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or relative_permittivity_map or
            conductivity_map must be given!*

        background_relative_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        rel_permittivity_amplitude : float, default: 1.0
            Maximum amplitude of relative permittivity variation

        conductivity_amplitude : float, default: 1.0
            Maximum amplitude of conductivity variation

        conductivity_valley : None or float
            Valley value of conductivity. If None, then peak value
            is assumed.

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        relative_permittivity_map : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity_map : :class:`numpy.ndarray`, default: None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.

        edge_smoothing : float, default: 0.03
            Percentage of cells at the boundary of the image area which
            will be smoothed.
    """
    # Check input requirements
    if resolution is None and (relative_permittivity_map is None
                               or conductivity_map is None):
        raise error.MissingInputError('draw_random_gaussians',
                                      'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity_map')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_relative_permittivity
    sigma_b = background_conductivity

    # Set map variables
    if relative_permittivity_map is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = relative_permittivity_map
    if conductivity_map is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity_map

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) + center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) + center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Wave area
    ly, lx = surface_area[0]*Ly, surface_area[1]*Lx
    area = logical_and(xp >= -lx/2,
                       logical_and(xp <= lx/2,
                                   logical_and(yp >= -ly/2, yp <= ly/2)))

    # Boundary smoothing
    bd = np.ones(xp.shape)
    nx, ny = np.round(edge_smoothing*NX), np.round(edge_smoothing*NY)
    left_bd = logical_and(xp >= -lx/2, xp <= -lx/2+nx*dx)
    right_bd = logical_and(xp >= lx/2-nx*dx, xp <= lx/2)
    lower_bd = logical_and(yp >= -ly/2, yp <= -ly/2+ny*dy)
    upper_bd = logical_and(yp >= ly/2-ny*dy, yp <= ly/2)
    edge1 = logical_and(left_bd, lower_bd)
    edge2 = logical_and(left_bd, upper_bd)
    edge3 = logical_and(upper_bd, right_bd)
    edge4 = logical_and(right_bd, lower_bd)
    f_left = (2/nx/dx)*(xp+lx/2) - (1/nx**2/dx**2)*(xp + lx/2)**2
    f_right = ((2/nx/dx)*(xp-(lx/2-2*nx*dx))
               - (1/nx**2/dx**2)*(xp-(lx/2-2*nx*dx))**2)
    f_lower = (2/ny/dy)*(yp+ly/2) - (1/ny**2/dy**2)*(yp + ly/2)**2
    f_upper = (((2/ny/dy)*(yp-(ly/2-2*ny*dy))
                - (1/ny**2/dy**2)*(yp-(ly/2-2*nx*dy))**2))
    bd[left_bd] = f_left[left_bd]
    bd[right_bd] = f_right[right_bd]
    bd[lower_bd] = f_lower[lower_bd]
    bd[upper_bd] = f_upper[upper_bd]
    bd[edge1] = f_left[edge1]*f_lower[edge1]
    bd[edge2] = f_left[edge2]*f_upper[edge2]
    bd[edge3] = f_upper[edge3]*f_right[edge3]
    bd[edge4] = f_right[edge4]*f_lower[edge4]
    bd[np.logical_not(area)] = 1.

    # General parameters
    s = np.zeros((2, number_distributions))
    xmin, xmax = -lx/2+distance_from_border*lx, lx/2-distance_from_border*lx
    ymin, ymax = -ly/2+distance_from_border*ly, ly/2-distance_from_border*ly

    # Relative permittivity
    y0 = ymin + rnd.rand(number_distributions)*(ymax-ymin)
    x0 = xmin + rnd.rand(number_distributions)*(xmax-xmin)
    s[0, :] = (minimum_spread + (maximum_spread-minimum_spread)
               * rnd.rand(number_distributions))*ly/6
    s[1, :] = (minimum_spread + (maximum_spread-minimum_spread)
               * rnd.rand(number_distributions))*lx/6
    phi = 2*pi*rnd.rand(number_distributions)
    A = rnd.rand(number_distributions)
    for i in range(number_distributions):
        sy, sx = s[0, i], s[1, i]
        x = np.cos(phi[i])*xp[area] + np.sin(phi[i])*yp[area]
        y = -np.sin(phi[i])*xp[area] + np.cos(phi[i])*yp[area]
        epsilon_r[area] = epsilon_r[area] + A[i]*np.exp(-((x-x0[i])**2
                                                          / (2*sx**2)
                                                          + (y-y0[i])**2
                                                          / (2*sy**2)))
    epsilon_r[area] = epsilon_r[area] - np.amin(epsilon_r[area])
    epsilon_r[area] = (rel_permittivity_amplitude*epsilon_r[area]
                       / np.amax(epsilon_r[area]))
    epsilon_r = epsilon_r*bd
    epsilon_r[area] = epsilon_r[area] + epsilon_rb

    # Conductivity
    y0 = ymin + rnd.rand(number_distributions)*(ymax-ymin)
    x0 = xmin + rnd.rand(number_distributions)*(xmax-xmin)
    s[0, :] = (minimum_spread + (maximum_spread-minimum_spread)
               * rnd.rand(number_distributions))*ly/6
    s[1, :] = (minimum_spread + (maximum_spread-minimum_spread)
               * rnd.rand(number_distributions))*lx/6
    phi = 2*pi*rnd.rand(number_distributions)
    A = rnd.rand(number_distributions)
    for i in range(number_distributions):
        sy, sx = s[0, i], s[1, i]
        x = np.cos(phi[i])*xp[area] + np.sin(phi[i])*yp[area]
        y = -np.sin(phi[i])*xp[area] + np.cos(phi[i])*yp[area]
        sigma[area] = sigma[area] + A[i]*np.exp(-((x-x0[i])**2/(2*sx**2)
                                                  + (y-y0[i])**2/(2*sy**2)))
    sigma[area] = sigma[area] - np.amin(sigma[area])
    sigma[area] = (conductivity_amplitude*sigma[area]
                   / np.amax(sigma[area]))
    sigma = sigma*bd
    sigma[area] = sigma[area] + sigma_b

    return epsilon_r, sigma