r"""Experiments Module

This module is intended to provide tools to analyse the perfomance of
solvers for microwave imaging problems. According to the definition of
some parameters, simulations may be carried out in order to synthesize
data and there are tools for statistical studies.

This module provides the following class and function for experiments:

    :class:`Experiment`
        A container for joining methods, inputs and configurations for
        statistical analysis of performance.
    :func:`run_methods`
        Run a list of methods for a specific input.
    :func:`run_scenarios`
        Run a list of inputs (scenarios) for a specific method.
    :func:`create_scenario`
        A routine to create random scenarios for experiments.
    :func:`contrast_density`
        Evaluate the contrast density of a given map.
    :func:`compute_resolution`:
        Compute resolution given a specific wavelength.

It also provides the following statistic tools [1]_:

    :func:`factorial_analysis`
        Factorial analysis of samples.
    :func:`ttest_ind_nonequalvar`
        Welch T-test for independent sample with different variances.
    :func:`dunnetttest`
        Dunnett's test for all-to-one comparisons.
    :func:`fittedcurve`
        Standard function for curve fitting for Dunnett's test.
    :func:`data_transformation`
        Try transformation formulas to obtain a normal distribution.
    :func:`normalitiyplot`
        Quantile-Quantile plot for graphic verification of normality
        assumption.
    :func:`homoscedasticityplot`
        Graphic verification of homoscedasticity assumption (equal
        variance among samples).
    :func:`boxplot`
        An improved routine for boxplot.
    :func:`violinplot`
        An improved routine for violin plot.
    :func:`confintplot`
        Confidence interval plot for multiple samples.

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

It also provides some helpful routines:

    :func:`get_label`
        Returns LaTeX symbol of a measure.
    :func:`get_title`
        Returns formal name of a measure.
    :func:`isleft`
        Determine if a point is on the left of a line.
    :func:`winding_number`
        Determine if a point is inside a polygon.

References
----------
.. [1] Montgomery, Douglas C. Design and analysis of experiments.
   John wiley & sons, 2017.
"""

# Standard libraries
import sys
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
RANDOM_POLYGONS_PATTERN = 'random_polygons'
REGULAR_POLYGONS_PATTERN = 'regular_polygons'
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
STUDY_EXECUTIONTIME = 'study_executiontime'
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
            A list with the defined kind of contrast pattern in the
            image.

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

        study_residual : bool
            If `True`, then the residual error will be recorded and
            available for study.

        study_map : bool
            If `True`, then the map (relative permittivity,
            conductivity) error willbe recorded and available for study.

        study_internfield : bool
            If `True`, then the intern total field error will be
            recorded and available for study.

        study_executiontime : bool
            If `True`, then the execution time will be available for
            study.

    Data synthesization
    -------------------
        :func:`define_synthetization_resolution`
            Compute appropriated resolution for sythesized images.

        :func:`define_recover_resolution`
            Compute appropriated resolution for recovered images.

        :func:`randomize_scenarios`
            Generate random scenarios.

        :func:`synthesize_scattered_field`
            Run forward problem for data synthesization.

        :func:`solve_scenarios`
            Run methods for the defined scenarios (samples).

        :func:`run`
            Gather the last 5 methods as a shortcut for a complete
            automatic and random experiment.

    Visualize results
    -----------------
        :func:`fixed_sampleset_plot`
            Compare observations of a sample.

        :func:`fixed_sampleset_violinplot`
            Compare results among methods for a single sample.

        :func:`fixed_measure_violinplot`
            Compare results of multiple methods for multiple samples
            given a measure.

        :func:`evolution_boxplot`
            Compare methods for when varying some parameter.

        :func:`plot_sampleset_results`
            Plot recovered images from a sample.

        :func:`plot_nbest_results`
            Plot N-best images recovered by methods.

        :func:`plot_normality`
            Quantile-Quantile plot for samples to check normality
            assumption.

    Compare algorithms
    ------------------
        :func:`study_single_mean`
            Determine confidence interval of means and compare among
            algorithms.

        :func:`compare_two_methods`
            Paired design between two methods.

        :func:`compare_multiple_methods`
            Compare multiple methods through Analysis of Variance and
            all-to-all and one-to-all comparisons.

        :func:`evolution_boxplot`
            Compare methods for when varying some parameter.

        :func:`factor_study`
            Determine which model parameters are relevant for the
            performance of a single method.

    Helpful methods
    ---------------
        :func:`save`
            Save experiment data.

        :func:`import_data`
            Load saved object.

        :func:`get_final_value_over_samples`
            Return the results of a single sample.

        :func:`get_measure_set`
            Return the set of available measures.
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
        if type(map_pattern) is str:
            if (map_pattern == RANDOM_POLYGONS_PATTERN
                    or map_pattern == REGULAR_POLYGONS_PATTERN
                    or map_pattern == SURFACES_PATTERN):
                self._map_pattern = [map_pattern]
            else:
                raise error.WrongValueInput('Experiment', 'map_pattern',
                                            RANDOM_POLYGONS_PATTERN + ' or '
                                            + SURFACES_PATTERN, map_pattern)
        elif type(map_pattern) is list:
            self._map_pattern = []
            for i in range(len(map_pattern)):
                if (map_pattern[i] == RANDOM_POLYGONS_PATTERN
                        or map_pattern == REGULAR_POLYGONS_PATTERN
                        or map_pattern[i] == SURFACES_PATTERN):
                    self._map_pattern.append(map_pattern[i])
                else:
                    raise error.WrongValueInput('Experiment', 'map_pattern',
                                                RANDOM_POLYGONS_PATTERN
                                                + ' or ' + SURFACES_PATTERN,
                                                map_pattern[i])
        else:
            self._map_pattern = None

    def __init__(self, name=None, maximum_contrast=None,
                 maximum_object_size=None, maximum_contrast_density=None,
                 map_pattern=None, sample_size=30,
                 synthetization_resolution=None, recover_resolution=None,
                 configurations=None, scenarios=None, methods=None,
                 forward_solver=None, noise=None, study_residual=True,
                 study_map=False, study_internfield=False,
                 study_executiontime=False, import_filename=None,
                 import_filepath=''):
        """Create the experiment object.

        The object should be defined with one of the following
        possibilities of combination of parameters (maximum_contrast,
        maximum_object_size, maximum_contrast_density, noise,
        map_pattern): (i) all are single values; (ii) one is list and
        the others are single values; and (iii) all are list of same
        size.

        You may create a new object or import a pre-saved file through
        variables `import_filename` or `import_filepath`.

        Parameters
        ----------
            name : str
                The name of the experiment.

            maximum_contrast : float or complex or list
                The maximum contrast value allowed on the map.

            maximum_object_size : float or list
                The maximum object size allowed on the map.

            maximum_contrast_density : float or list
                Maximum value for contrast (absolute value) per pixel
                normalized by the maximum contrast allowed.

            noise : float or list
                Noise level added to scattered field data.

            map_pattern : {'random_polygons', 'regular_polygons',
                           'surfaces'} or list
                Kind of objects on the image.

            sample_size : int, default: 30
                Number of scenarios per combination of configuration and
                other parameter.

            synthetization_resolution : list of tuple, default: None
                Resolution of synthesized images. The list must be two-
                dimensional: [groups, configurations].

            recover_resolution : list of tuple, default: None
                Resolution of recovered images. The list must be two-
                dimensional: [groups, configurations].

            configurations : :class:`configuration.Configuration` or 
                             list, default: None
                Configuration objects.

            scenarios : list of :class:`inputdata.InputData`,
                        default: None
                Set of scenarios. The list must be three-dimensional:
                [groups, configurations, sample_size].

            methods : :class:`solver.Solver` or list, default: None
                Set of methods.

            forward_solver : :class:`forward.Forward`, default: None
                Forward solver for synthesizing scattered field.

            study_residual : bool, default: True
                Flag to indicate that the residual of the equations will
                be addressed.

            study_map : bool, default: False
                Flag to indicate that the error on the recovered map
                (relative permittivity and conductivity) will be
                addressed.

            study_internfield : bool, default: False
                Flag to indicate that the error on intern total field
                will be addressed.

            study_executiontime : bool, default: False
                Flag to indicate that the execution time will be
                addressed.

            import_filename : str, default: None
                Import an object saved before with the name specified
                by this argument.

            import_filepath : str, default: ''
                Path to file with the object saved.
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
        self.study_executiontime = study_executiontime
        self.results = None

        # Enforcing that all experimentation parameters are of same length
        if (len(self.maximum_contrast) == len(self.maximum_object_size)
                and len(self.maximum_object_size)
                == len(self.maximum_contrast_density)
                and len(self.maximum_contrast_density) == len(self.noise)
                and len(self.noise) == len(self.map_pattern)):
            pass
        elif (len(self.maximum_contrast) > 1
                and len(self.maximum_object_size) == 1
                and len(self.maximum_contrast_density) == 1
                and len(self.noise) == 1
                and len(self.map_pattern) == 1):
            N = len(self.maximum_contrast)
            self.maximum_object_size = N * self.maximum_object_size
            self.maximum_contrast_density = N * self.maximum_contrast_density
            self.noise = N * self.noise
            self.map_pattern = N * self.map_pattern
        elif (len(self.maximum_contrast) == 1
                and len(self.maximum_object_size) > 1
                and len(self.maximum_contrast_density) == 1
                and len(self.noise) == 1
                and len(self.map_pattern) == 1):
            N = len(self.maximum_object_size)
            self.maximum_contrast = N * self.maximum_contrast
            self.maximum_contrast_density = N * self.maximum_contrast_density
            self.noise = N * self.noise
            self.map_pattern = N * self.map_pattern
        elif (len(self.maximum_contrast) == 1
                and len(self.maximum_object_size) == 1
                and len(self.maximum_contrast_density) > 1
                and len(self.noise) == 1
                and len(self.map_pattern) == 1):
            N = len(self.maximum_contrast_density)
            self.maximum_contrast = N*self.maximum_contrast
            self.maximum_object_size = N*self.maximum_object_size
            self.noise = N * self.noise
            self.map_pattern = N * self.map_pattern
        elif (len(self.maximum_contrast) == 1
                and len(self.maximum_object_size) == 1
                and len(self.maximum_contrast_density) == 1
                and len(self.noise) > 1
                and len(self.map_pattern) == 1):
            N = len(self.noise)
            self.maximum_contrast = N*self.maximum_contrast
            self.maximum_object_size = N*self.maximum_object_size
            self.maximum_contrast_density = N * self.maximum_contrast_density
            self.map_pattern = N * self.map_pattern
        elif (len(self.maximum_contrast) == 1
                and len(self.maximum_object_size) == 1
                and len(self.maximum_contrast_density) == 1
                and len(self.noise) == 1
                and len(self.map_pattern) > 1):
            N = len(self.map_pattern)
            self.maximum_contrast = N*self.maximum_contrast
            self.maximum_object_size = N*self.maximum_object_size
            self.maximum_contrast_density = N * self.maximum_contrast_density
            self.noise = N * self.noise
        else:
            raise error.WrongValueInput('Experiment.__init__',
                                        'maximum_contrast and ' +
                                        'maximum_object_size and ' +
                                        'maximum_contrast_density' +
                                        'noise and map_pattern',
                                        'all float/complex or ' +
                                        'one list and float/complex',
                                        'More than one are list')

    def run(self, configurations=None, scenarios=None, methods=None,
            save_per_iteration=False, save_solver_screeninfo=False,
            file_path=''):
        """Run experiment.

        This routine run the experiment without analysing the results.
        This may be accomplished latter with the specific methods.

        The arguments are options in case these parameters were not
        given when building the object.

        Paremeters
        ----------
            configurations : list of :class:`configuration.Configuration`

            scenarios : list of :class:`inputdata.InputData`

            methods : list of :class:`solver.Solver`

            save_per_iteration : bool, default: False
                If `True`, then the data will be saved in each iteration
                when running the forward problem and solving the
                inverse problem. This may be useful for long experiments
                and when the server may turn off unexpectedly.

            save_solver_screeninfo : bool, default: False
                If `True`, then the routine will save the information
                displayed by the methods through a specific .txt file
                and save it with the name of the experiment.

            file_path : str, default: ''
                Path to save the object.
        """
        # Check required attributes
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

        # Screen introduction
        print('Experiment: ' + self.name)

        # Define resolution of the maps for data synthesization
        print('Check resolution for synthesized data...', end=' ')
        if self.synthetization_resolution is None:
            print('defining...', end=' ')
            self.define_synthetization_resolution()
        print('ok!')

        # Define resolution of the recovered maps
        print('Check resolution for data reconstruction...', end=' ')
        if self.recover_resolution is None:
            print('defining...', end=' ')
            self.define_recover_resolution()
        print('ok!')

        # Build scenarios if it has not been given
        print('Check scenarios...', end=' ')
        if self.scenarios is None:
            print('building...', end=' ')
            self.randomize_scenarios(self.synthetization_resolution)
        print('ok!')

        if save_per_iteration:
            self.save(file_path=file_path)

        # Check forward solver for data synthesization
        print('Check forward solver for data synthesization...', end=' ')
        if self.forward_solver is None:
            print('setting MoM-CG-FFT...', end=' ')
            self.forward_solver = mom.MoM_CG_FFT(self.configurations[0])
        print('ok!')

        # Synthesize the scattered field
        self.synthesize_scattered_field(save_per_iteration=save_per_iteration,
                                        file_path=file_path)

        if save_solver_screeninfo:
            screen_object = open(self.name + '.txt', 'w')
        else:
            screen_object = sys.stdout

        # Solving scenarios
        self.solve_scenarios(save_per_iteration=save_per_iteration,
                             file_path=file_path)

        if save_solver_screeninfo:
            screen_object.close()

    def define_synthetization_resolution(self):
        """Set synthetization resolution attribute."""
        # Check required attributes
        if self.maximum_contrast is None:
            raise error.MissingAttributesError('Experiment',
                                               'maximum_contrast')
        if self.configurations is None or len(self.configurations) == 0:
            raise error.MissingAttributesError('Experiment',
                                               'configurations')

        # The attribute will be a list with two dimensions. The first
        # dimension is the groups of samples and the second is the set
        # of configurations.add()
        self.synthetization_resolution = []

        k = 0
        N = len(self.maximum_contrast)*len(self.configurations)
        for i in range(len(self.maximum_contrast)):
            self.synthetization_resolution.append(list())
            for j in range(len(self.configurations)):

                k += 1
                message = 'Resolution %d/' % k + '%d' % N
                print(message, end='\b'*len(message), flush=True)

                # Maximum relative permittivity
                epsilon_rd = cfg.get_relative_permittivity(
                    self.maximum_contrast[i],
                    self.configurations[j].epsilon_rb
                )

                # Maximum conductivity
                sigma_d = cfg.get_conductivity(
                    self.maximum_contrast[i],
                    self.configurations[j].epsilon_rb,
                    2*pi*self.configurations[j].f,
                    self.configurations[j].sigma_b
                )

                # Maximum wavelength
                lam_d = cfg.compute_wavelength(self.configurations[j].f,
                                               epsilon_r=epsilon_rd,
                                               sigma=sigma_d)

                # Computing resolution with standard value
                resolution = compute_resolution(
                    lam_d, self.configurations[j].Ly,
                    self.configurations[j].Lx,
                    STANDARD_SYNTHETIZATION_RESOLUTION
                )
                self.synthetization_resolution[i].append(resolution)

        print('Resolution %d/' % N + '%d' % N, end=' ', flush=True)

    def define_recover_resolution(self):
        """Set recover resolution variable."""
        # Check required attributes
        if self.maximum_contrast is None:
            raise error.MissingAttributesError('Experiment',
                                               'maximum_contrast')
        if self.configurations is None or len(self.configurations) == 0:
            raise error.MissingAttributesError('Experiment',
                                               'configurations')

        # The attribute will be a list with two dimensions. The first
        # dimension is the groups of samples and the second is the set
        # of configurations.add()
        self.recover_resolution = []

        k = 0
        N = len(self.maximum_contrast)*len(self.configurations)
        for i in range(len(self.maximum_contrast)):
            self.recover_resolution.append(list())
            for j in range(len(self.configurations)):

                k += 1
                message = 'Resolution %d/' % k + '%d' % N
                print(message, end='\b'*len(message), flush=True)

                # Maximum relative permittivity
                epsilon_rd = cfg.get_relative_permittivity(
                    self.maximum_contrast[i],
                    self.configurations[j].epsilon_rb
                )

                # Maximum conductivity
                sigma_d = cfg.get_conductivity(
                    self.maximum_contrast[i],
                    self.configurations[j].epsilon_rb,
                    2*pi*self.configurations[j].f,
                    self.configurations[j].sigma_b
                )

                # Maximum wavelength
                lam_d = cfg.compute_wavelength(self.configurations[j].f,
                                               epsilon_r=epsilon_rd,
                                               sigma=sigma_d)

                # Computing resolution with standard value
                resolution = compute_resolution(
                    lam_d, self.configurations[j].Ly,
                    self.configurations[j].Lx,
                    STANDARD_RECOVER_RESOLUTION
                )
                self.recover_resolution[i].append(resolution)

        print('Resolution %d/' % N + '%d' % N, end=' ', flush=True)

    def randomize_scenarios(self, resolution=None):
        """Create random scenarios.

        Parameters
        ----------
            resolution : 2-tuple of int, optional
                Y- and X-axis amount of pixels, respectively.
        """
        # Check required attributes
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

        # The attribute will be a list with 3 dimensions. The first
        # dimension stands for the group of factors (maximum_contrast
        # etc); the second stands for the set of configurations; the
        # third is sample, i.e., each scenario created.
        self.scenarios = []

        n = 0
        N = (len(self.maximum_contrast) * len(self.configurations)
             * self.sample_size)
        for i in range(len(self.maximum_contrast)):
            self.scenarios.append(list())
            for j in range(len(self.configurations)):
                self.scenarios[i].append(list())

                # Print information
                message = 'Scenarios %d/' % n + '%d' % N
                print(message, end='\b'*len(message), flush=True)
                n += self.sample_size

                # Create the sample parallely
                num_cores = multiprocessing.cpu_count()
                output = Parallel(n_jobs=num_cores)(delayed(create_scenario)(
                    'rand' + '%d' % i + '%d' % j + '%d' % k,
                    self.configurations[j], resolution[i][j],
                    self.map_pattern[i], self.maximum_contrast[i],
                    self.maximum_contrast_density[i],
                    maximum_object_size=self.maximum_object_size[i],
                    noise=self.noise[i],
                    compute_residual_error=self.study_residual,
                    compute_map_error=self.study_map,
                    compute_totalfield_error=self.study_internfield
                ) for k in range(self.sample_size))

                # Append scenarios into the list
                for k in range(self.sample_size):
                    new_scenario = output[k]
                    self.scenarios[i][j].append(cp.deepcopy(new_scenario))

        # Print information
        message = 'Scenarios %d/' % n + '%d' % N
        print(message, end=' ', flush=True)

    def synthesize_scattered_field(self, save_per_iteration=False,
                                   file_path=''):
        """Run forward problem to synthesize scattered field.

        Parameters
        ----------
            save_per_iteration : bool, default: False
                If `True`, then the object will be saved after each
                iteration, i.e., for each scenario.

            file_path : str, default: ''
                Path to save the object.
        """
        # Check required attributes
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

        # Number of executions
        N = (len(self.maximum_contrast) * len(self.configurations)
             * self.sample_size)
        n = 0

        for i in range(len(self.maximum_contrast)):
            for j in range(len(self.configurations)):
                self.forward_solver.configuration = self.configurations[j]
                for k in range(self.sample_size):

                    # Print information
                    message = 'Solved %d/' % n + '%d' % N
                    print(message, end='\b'*len(message), flush=True)
                    n += 1

                    # Solve forward problem
                    self.forward_solver.solve(
                        self.scenarios[i][j][k],
                        noise=self.scenarios[i][j][k].noise,
                        SAVE_INTERN_FIELD=SAVE_INTERN_FIELD
                    )

                    if save_per_iteration:
                        self.save(file_path)

        # Print final information
        message = 'Solved %d/' % N + '%d' % N
        print(message, end=' ', flush=True)

    def solve_scenarios(self, parallelization=False, save_per_iteration=False,
                        file_path='', screen_object=sys.stdout):
        """Run methods for each scenario.

        Parameters
        ---------
            parallelization : bool
                If the methods may run in parallel.

            save_per_iteration : bool, default: False
                If `True`, then the object will be saved after each
                iteration.

            file_path : str, default: ''
                Path to save the object.

        screen_object : :class:`_io.TextIOWrapper`, default: sys.stdout
            Output object to print solver information.
        """
        # Check required attributes
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

        # The results list with have 4 dimensions: (i) group of factors;
        # (ii) configuration; (iii) sample; (iv) method.
        self.results = []

        # Number of executions
        N = (len(self.maximum_contrast) * len(self.configurations)
             * len(self.methods) * self.sample_size)
        n = 0

        for i in range(len(self.maximum_contrast)):
            self.results.append(list())
            for j in range(len(self.configurations)):
                self.results[i].append(list())

                # Set the current configuration for each method
                for m in range(len(self.methods)):
                    self.methods[m].configuration = self.configurations[j]

                for k in range(self.sample_size):
                    self.results[i][j].append(list())

                    # Set the recovering resolution
                    self.scenarios[i][j][k].resolution = (
                        self.recover_resolution[i][j]
                    )

                    # Print info
                    message = 'Executions %d/' % n + '%d' % N
                    print(message, end='\b'*len(message), flush=True)
                    n += len(self.methods)

                    if screen_object != sys.stdout:
                        print_info = True
                    else:
                        print_info = False

                    # Run methods
                    self.results[i][j][k] = (
                        run_methods(self.methods, self.scenarios[i][j][k],
                                    parallelization=parallelization,
                                    print_info=print_info,
                                    screen_object=screen_object)
                    )

                    if save_per_iteration:
                        self.save(file_path)

        # Print info
        message = 'Executions %d/' % N + '%d' % N
        print(message, end=' ', flush=True)

    def fixed_sampleset_plot(self, group_idx=0, config_idx=0, method_idx=0,
                             show=False, file_path='', file_format='eps'):
        """Plot observations of a sample over the x-axis.

        This method takes a specified sample and plot each observation
        in a different position of x-axis. In this way, results of each
        scenario may be compared among themselves and methods may be
        compared scenario by scenario from the same sample. The
        comparison will be done through all available measures.

        Parameters
        ----------
            group_idx : int, default: 0
                Factor combination index (maximum contrast, maximum
                object size etc).

            config_idx : int, default: 0
                Configuration index.

            method_idx : int of list of int, default: 0
                Method index.

            show : bool, default: False
                If `True`, the plot is shown. Otherwise, the plot is
                saved with the name `singlesample` plus the indexes of
                configuration and group.

            file_path : str, default: ''
                Path to save the figure.

            file_format : {'eps', 'png', 'pdf', 'svg'}
                Format of the figure to be saved. Only formats supported
                by `matplotlib.pyplot.savefig`.
        """
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')

        # Check the type of the inputs
        if type(group_idx) is not int:
            raise error.WrongTypeInput('Experiment.fixed_sampleset_plot',
                                       'group_idx', 'int', type(group_idx))
        if type(config_idx) is not int:
            raise error.WrongTypeInput('Experiment.fixed_sampleset_plot',
                                       'config_idx', 'int', type(config_idx))
        if type(method_idx) is not int and type(method_idx) is not list:
            raise error.WrongTypeInput('Experiment.fixed_sampleset_plot',
                                       'method_idx', 'int', type(method_idx))

        # Fix the format of the method index as list
        if type(method_idx) is int:
            method_idx = [method_idx]

        # Check the values of the inputs
        if group_idx < 0 or group_idx >= len(self.maximum_contrast):
            raise error.WrongValueInput('Experiment.fixed_sampleset_plot',
                                        'group_idx', '0 to %d'
                                        % len(self.maximum_contrast),
                                        str(group_idx))
        if config_idx < 0 or config_idx >= len(self.configurations):
            raise error.WrongValueInput('Experiment.fixed_sampleset_plot',
                                        'config_idx', '0 to %d'
                                        % len(self.configurations),
                                        str(config_idx))
        if min(method_idx) < 0 or max(method_idx) >= len(self.method_idx):
            raise error.WrongValueInput('Experiment.fixed_sampleset_plot',
                                        'method_idx', '0 to %d'
                                        % len(self.methods),
                                        str(method_idx))

        # Main variables
        g, c = group_idx, config_idx
        y = np.zeros((self.sample_size, len(method_idx)))
        measures = self.get_measure_set(config_idx)
        nplots = len(measures)

        # Image configuration
        nrows = int(np.sqrt(nplots))
        ncols = int(np.ceil(nplots/nrows))
        image_size = (3.*ncols, 3.*nrows)
        figure = plt.figure(figsize=image_size)
        rst.set_subplot_size(figure)

        # Quick access to method names
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        # Plotting results
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

        # Show or save results
        if show:
            plt.show()
        else:
            plt.savefig(file_path + self.name + '_singlesample_%d_' % c + '%d'
                        % g + '.' + file_format, format=file_format)
            plt.close()

    def fixed_sampleset_violinplot(self, group_idx=0, config_idx=0,
                                   method_idx=[0], show=False,
                                   file_path='', file_format='eps'):
        """Violin plot for a single sample and multiple methods.

        This method takes a specified sample and multiple methods and
        compare them considering each measure through violinplot. This
        graphic may be used to compare each result through
        characteristics of its distribution.

        Parameters
        ----------
            group_idx : int, default: 0
                Factor combination index (maximum contrast, maximum
                object size etc).

            config_idx : int, default: 0
                Configuration index.

            method_idx : int of list of int, default: 0
                Method index.

            show : bool, default: False
                If `True`, the plot is shown. Otherwise, the plot is
                saved with the name `violinplot` plus the indexes of
                configuration and group.

            file_path : str, default: ''
                Path to save the figure.

            file_format : {'eps', 'png', 'pdf', 'svg'}
                Format of the figure to be saved. Only formats supported
                by `matplotlib.pyplot.savefig`.
        """
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')

        # Check the type of the inputs
        if type(group_idx) is not int:
            raise error.WrongTypeInput('Experiment.fixed_sampleset_violinplot',
                                       'group_idx', 'int', type(group_idx))
        if type(config_idx) is not int:
            raise error.WrongTypeInput('Experiment.fixed_sampleset_violinplot',
                                       'config_idx', 'int', type(config_idx))
        if type(method_idx) is not int and type(method_idx) is not list:
            raise error.WrongTypeInput('Experiment.fixed_sampleset_violinplot',
                                       'method_idx', 'int', type(method_idx))

        # Fix the format of the method index as list
        if type(method_idx) is int:
            method_idx = [method_idx]

        # Check the values of the inputs
        if group_idx < 0 or group_idx >= len(self.maximum_contrast):
            raise error.WrongValueInput('Experiment.fixed_sampleset_'
                                        + 'violinplot', 'group_idx', '0 to %d'
                                        % len(self.maximum_contrast),
                                        str(group_idx))
        if config_idx < 0 or config_idx >= len(self.configurations):
            raise error.WrongValueInput('Experiment.fixed_sampleset_'
                                        + 'violinplot', 'config_idx', '0 to %d'
                                        % len(self.configurations),
                                        str(config_idx))
        if min(method_idx) < 0 or max(method_idx) >= len(self.method_idx):
            raise error.WrongValueInput('Experiment.fixed_sampleset_'
                                        + 'violinplot',
                                        'method_idx', '0 to %d'
                                        % len(self.methods),
                                        str(method_idx))

        # Main variables
        g, c = group_idx, config_idx
        measures = self.get_measure_set(config_idx)
        nplots = len(measures)

        # Image configuration
        nrows = int(np.sqrt(nplots))
        ncols = int(np.ceil(nplots/nrows))
        image_size = (3.*ncols, 3.*nrows)
        figure = plt.figure(figsize=image_size)
        rst.set_subplot_size(figure)

        # Quick access to method names
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        # Plotting results
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

        # Show or save results
        if show:
            plt.show()
        else:
            plt.savefig(file_path + self.name + '_violinplot_%d' % config_idx
                        + '%d' % group_idx + '.' + file_format,
                        format=file_format)
            plt.close()

    def fixed_measure_violinplot(self, measure, group_idx=[0], config_idx=[0],
                                 method_idx=[0], show=False, file_path='',
                                 file_format='eps'):
        """Violin plot for a single measure and multiple arrangements.

        This method takes a specified measure and compare the results of
        multiple methods in multiple arrangements of factors and
        configurations.

        Parameters
        ----------
            measure : {'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe',
                       'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe',
                       'zeta_tfmpad', 'zeta_tfppad', 'zeta_be',
                       'execution_time'}
                String to indicate which measure.

            group_idx : int or list of int, default: [0]
                Factor combination index (maximum contrast, maximum
                object size etc).

            config_idx : int or list of int, default: [0]
                Configuration index.

            method_idx : int of list of int, default: 0
                Method index.

            show : bool, default: False
                If `True`, the plot is shown. Otherwise, the plot is
                saved with the name of the measure.

            file_path : str, default: ''
                Path to save the figure.

            file_format : {'eps', 'png', 'pdf', 'svg'}
                Format of the figure to be saved. Only formats supported
                by `matplotlib.pyplot.savefig`.
        """
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')

        # Check the type of the inputs
        if type(group_idx) is not int and type(group_idx) is not list:
            raise error.WrongTypeInput('Experiment.fixed_measure_violinplot',
                                       'group_idx', 'int/list of int',
                                       type(group_idx))
        if type(config_idx) is not int and type(config_idx) is not list:
            raise error.WrongTypeInput('Experiment.fixed_measure_violinplot',
                                       'config_idx', 'int', type(config_idx))
        if type(method_idx) is not int and type(method_idx) is not list:
            raise error.WrongTypeInput('Experiment.fixed_measure_violinplot',
                                       'method_idx', 'int', type(method_idx))

        # Fix the format of the method index as list
        if type(method_idx) is int:
            method_idx = [method_idx]
        if type(group_idx) is int:
            group_idx = [group_idx]
        if type(config_idx) is int:
            config_idx = [config_idx]

        # Check the values of the inputs
        if min(group_idx) < 0 or max(group_idx) >= len(self.maximum_contrast):
            raise error.WrongValueInput('Experiment.fixed_measure_violinplot',
                                        'group_idx', '0 to %d'
                                        % len(self.maximum_contrast),
                                        str(group_idx))
        if min(config_idx) < 0 or max(config_idx) >= len(self.configurations):
            raise error.WrongValueInput('Experiment.fixed_measure_violinplot',
                                        'config_idx', '0 to %d'
                                        % len(self.configurations),
                                        str(config_idx))
        if min(method_idx) < 0 or max(method_idx) >= len(self.method_idx):
            raise error.WrongValueInput('Experiment.fixed_measure_violinplot',
                                        'method_idx', '0 to %d'
                                        % len(self.methods),
                                        str(method_idx))
        try:
            get_title(measure)
        except Exception:
            raise error.WrongValueInput('Experiments.fixed_measure_violinplot',
                                        'measure', "{'zeta_rn', 'zeta_rpad',"
                                        + " 'zeta_epad', 'zeta_ebe', "
                                        + "'zeta_eoe', 'zeta_sad', 'zeta_sbe',"
                                        + " 'zeta_soe', 'zeta_tfmpad', "
                                        + "'zeta_tfppad', 'zeta_be', "
                                        + "'execution_time'}", measure)

        # Figure parameters
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

        # Plot graphics
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

        # Save or show the figure
        if show:
            plt.show()
        else:
            plt.savefig(file_path + self.name + '_' + measure + '.'
                        + file_format, format=file_format)
            plt.close()

    def evolution_boxplot(self, group_idx=[0], config_idx=[0],
                          measure=None, method_idx=[0], show=False,
                          file_path='', file_format='eps'):
        """Compare multiples samples to study the behavior of methods.

        This routine intends to plot the behavior of multiple methods
        when a factor variates (or some parameter of configuration). So,
        if you want to investigate if the variation of some parameter
        tends to influence the behavior of an algorithm, then you select
        the indexes of the corresponding configurations and factor
        combinations and the routine will show it.

        If one configuration index is provided, then the results will be
        boxplots will be arranged by factor variation. And vice-versa.
        Furthermore, if both list of indexes have more than one
        elements, than there will be one figure per configuration index,
        i.e., the configuration will be fixed and the factor will vary.

        Parameters
        ----------
            group_idx : int or list of int, default: [0]
                Factor combination index (maximum contrast, maximum
                object size etc).

            config_idx : int or list of int, default: [0]
                Configuration index.

            measure : {'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe',
                       'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe',
                       'zeta_tfmpad', 'zeta_tfppad', 'zeta_be',
                       'execution_time'}
                A string or a list of string to indicate the considered
                measures. If `None`, then all the available ones will be
                considered.

            method_idx : int of list of int, default: 0
                Method index.

            show : bool, default: False
                If `True`, the plot is shown. Otherwise, the plot is
                saved with the name 'evolution' plus the indexes of
                configurations and factor groups.

            file_path : str, default: ''
                Path to save the figure.

            file_format : {'eps', 'png', 'pdf', 'svg'}
                Format of the figure to be saved. Only formats supported
                by `matplotlib.pyplot.savefig`.
        """
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')

        # Check the type of the inputs
        if type(group_idx) is not int and type(group_idx) is not list:
            raise error.WrongTypeInput('Experiment.evolution_boxplot',
                                       'group_idx', 'int/list of int',
                                       type(group_idx))
        if type(config_idx) is not int and type(config_idx) is not list:
            raise error.WrongTypeInput('Experiment.evolution_boxplot',
                                       'config_idx', 'int', type(config_idx))
        if type(method_idx) is not int and type(method_idx) is not list:
            raise error.WrongTypeInput('Experiment.evolution_boxplot',
                                       'method_idx', 'int', type(method_idx))
        if (measure is not None and type(measure) is not str
                and type(measure) is not list):
            raise error.WrongTypeInput('Experiment.evolution_boxplot',
                                       'meausre', 'None/str/list',
                                       type(measure))

        # Fix the format of the method index as list
        if type(method_idx) is int:
            method_idx = [method_idx]
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

        # Check the values of the inputs
        if min(group_idx) < 0 or max(group_idx) >= len(self.maximum_contrast):
            raise error.WrongValueInput('Experiment.evolution_boxplot',
                                        'group_idx', '0 to %d'
                                        % len(self.maximum_contrast),
                                        str(group_idx))
        if min(config_idx) < 0 or max(config_idx) >= len(self.configurations):
            raise error.WrongValueInput('Experiment.evolution_boxplot',
                                        'config_idx', '0 to %d'
                                        % len(self.configurations),
                                        str(config_idx))
        if min(method_idx) < 0 or max(method_idx) >= len(self.method_idx):
            raise error.WrongValueInput('Experiment.evolution_boxplot',
                                        'method_idx', '0 to %d'
                                        % len(self.methods),
                                        str(method_idx))
        if type(measure) is str:
            try:
                get_title(measure)
            except Exception:
                raise error.WrongValueInput(
                    'Experiments.compare_two_methods', 'measure',
                    "{'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe', "
                    + "'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe', "
                    + "'zeta_tfmpad', 'zeta_tfppad', 'zeta_be', "
                    + "'execution_time'}", measure)

        # Quick access to method names
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        # Possible colors of boxes. Are you going to need more than 10?
        colors = ['cornflowerblue', 'indianred', 'seagreen', 'mediumorchid',
                  'chocolate', 'palevioletred', 'teal', 'rosybrown', 'tan',
                  'crimson']

        # Fixed configuration
        if len(group_idx) > 1:

            # Group names
            labels = []
            for j in group_idx:
                labels.append('g%d' % j)

            # One plot per configuration
            for i in config_idx:

                # Different configurations may have different measures
                if none_measure:
                    measure = self.get_measure_set(i)

                # Plot configurations
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

                # For each measure, a graphic
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

                # Show or save figure
                if show:
                    plt.show()
                else:
                    plt.savefig(file_path + self.name + '_evolution_c%d' % i
                                + '.' + file_format, format=file_format)
                    plt.close()

        # Fixed factor group
        else:

            # Configuration names
            labels = []
            for i in config_idx:
                labels.append('c%d' % i)
            j = group_idx[0]

            # We fix the measure set by the first configuration
            if none_measure:
                measure = self.get_measure_set(config_idx[0])

            # Figure configuration
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

            # A graphic per measure
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

            # Show or save the figure
            if show:
                plt.show()
            else:
                plt.savefig(file_path + self.name + '_evolution_c%d' % i
                            + '.' + file_format, format=file_format)
                plt.close()

    def plot_sampleset_results(self, group_idx=[0], config_idx=[0],
                               method_idx=[0], show=False, file_path='',
                               file_format='eps'):
        """Plot the recovered images of a sample set.

        Given a sample (by the indexes of configuration and factor
        group), it plots all the recovered images. If one method index
        is provided, than each figure will have all the recovered maps
        of the sample specified by the configuration and group indexes.
        Otherwise, one figure will be generated containing the benchmark
        map and recovered ones by the methods.

        Parameters
        ----------
            group_idx : int or list of int, default: [0]
                Factor combination index (maximum contrast, maximum
                object size etc).

            config_idx : int or list of int, default: [0]
                Configuration index.

            method_idx : int of list of int, default: 0
                Method index.

            show : bool, default: False
                If `True`, the plot is shown. Otherwise, the plot is
                saved with the name `recoverd_images` plus the indexes of
                configuration, group, and scenario.

            file_path : str, default: ''
                Path to save the figure.

            file_format : {'eps', 'png', 'pdf', 'svg'}
                Format of the figure to be saved. Only formats supported
                by `matplotlib.pyplot.savefig`.
        """
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')

        # Check the type of the inputs
        if type(group_idx) is not int and type(group_idx) is not list:
            raise error.WrongTypeInput('Experiment.plot_sampleset_results',
                                       'group_idx', 'int/list of int',
                                       type(group_idx))
        if type(config_idx) is not int and type(config_idx) is not list:
            raise error.WrongTypeInput('Experiment.plot_sampleset_results',
                                       'config_idx', 'int/list of int',
                                       type(config_idx))
        if type(method_idx) is not int and type(method_idx) is not list:
            raise error.WrongTypeInput('Experiment.plot_sampleset_results',
                                       'method_idx', 'int/list of int',
                                       type(method_idx))

        # Fix the format of the method index as list
        if type(config_idx) is int:
            config_idx = [config_idx]
        if type(group_idx) is int:
            group_idx = [group_idx]
        if type(method_idx) is int:
            method_idx = [method_idx]

        # Check the values of the inputs
        if min(group_idx) < 0 or max(group_idx) >= len(self.maximum_contrast):
            raise error.WrongValueInput('Experiment.plot_sampleset_results',
                                        'group_idx', '0 to %d'
                                        % len(self.maximum_contrast),
                                        str(group_idx))
        if min(config_idx) < 0 or max(config_idx) >= len(self.configurations):
            raise error.WrongValueInput('Experiment.plot_sampleset_results',
                                        'config_idx', '0 to %d'
                                        % len(self.configurations),
                                        str(config_idx))
        if min(method_idx) < 0 or max(method_idx) >= len(self.method_idx):
            raise error.WrongValueInput('Experiment.plot_sampleset_results',
                                        'method_idx', '0 to %d'
                                        % len(self.methods),
                                        str(method_idx))

        # Quick access to method names
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        # Figure parameters
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

                # If there is only one method, each figure will have all
                # recovered images.
                if len(method_idx) == 1:
                    figure = plt.figure(figsize=image_size)
                    rst.set_subplot_size(figure)
                    n = 1

                # One figure for each scenario
                for k in range(self.sample_size):

                    # If there are more than one method, each figure
                    # will have the benchmark and recovered maps by the
                    # methods.
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
                        p += 1

                    # Save or show the image per scenario
                    if len(method_idx) > 1:
                        if show:
                            plt.show()
                        else:
                            plt.savefig(file_path + self.name
                                        + 'recoverd_images' + str(i) + str(j)
                                        + str(k) + '.' + file_format,
                                        format=file_format)
                            plt.close()

                # Save or show the image per sample set
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
        """Plot the N-best recovered maps given a specified measure.

        Given a specific measure, the routine plots the N-best recovered
        images by each combination of configuration-group-method.

        Parameters
        ----------
            n : int
                Size of the set of best images.

            measure : {'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe',
                       'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe',
                       'zeta_tfmpad', 'zeta_tfppad', 'zeta_be',
                       'execution_time'}
                A string to indicate the considered measures. If `None`, then
                all the available ones will be considered.

            group_idx : int or list of int, default: [0]
                Factor combination index (maximum contrast, maximum
                object size etc).

            config_idx : int or list of int, default: [0]
                Configuration index.

            method_idx : int of list of int, default: None
                Method index. If `None`, all methods will be considered.

            show : bool, default: False
                If `True`, the plot is shown. Otherwise, the plot is
                saved with the name `nbest` plus the indexes of
                configuration, group, and method.

            file_path : str, default: ''
                Path to save the figure.

            file_format : {'eps', 'png', 'pdf', 'svg'}
                Format of the figure to be saved. Only formats supported
                by `matplotlib.pyplot.savefig`.
        """
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')

        # Check the type of the inputs
        if type(group_idx) is not int and type(group_idx) is not list:
            raise error.WrongTypeInput('Experiment.plot_nbest_results',
                                       'group_idx', 'int/list of int',
                                       type(group_idx))
        if type(config_idx) is not int and type(config_idx) is not list:
            raise error.WrongTypeInput('Experiment.plot_nbest_results',
                                       'config_idx', 'int/list of int',
                                       type(config_idx))
        if (method_idx is not None and type(method_idx) is not int
                and type(method_idx) is not list):
            raise error.WrongTypeInput('Experiment.plot_nbest_results',
                                       'method_idx', 'None/int/list of int',
                                       type(method_idx))

        # Fix the format of the method index as list
        if type(config_idx) is int:
            config_idx = [config_idx]
        if type(group_idx) is int:
            group_idx = [group_idx]
        if type(method_idx) is int:
            method_idx = [method_idx]
        elif method_idx is None:
            method_idx = [i for i in range(len(self.methods))]

        # Check the values of the inputs
        if min(group_idx) < 0 or max(group_idx) >= len(self.maximum_contrast):
            raise error.WrongValueInput('Experiment.plot_sampleset_results',
                                        'group_idx', '0 to %d'
                                        % len(self.maximum_contrast),
                                        str(group_idx))
        if min(config_idx) < 0 or max(config_idx) >= len(self.configurations):
            raise error.WrongValueInput('Experiment.plot_sampleset_results',
                                        'config_idx', '0 to %d'
                                        % len(self.configurations),
                                        str(config_idx))
        if min(method_idx) < 0 or max(method_idx) >= len(self.method_idx):
            raise error.WrongValueInput('Experiment.plot_sampleset_results',
                                        'method_idx', '0 to %d'
                                        % len(self.methods),
                                        str(method_idx))
        try:
            get_title(measure)
        except Exception:
            raise error.WrongValueInput('Experiments.plot_nbest_results',
                                        'measure', "{'zeta_rn', 'zeta_rpad',"
                                        + " 'zeta_epad', 'zeta_ebe', "
                                        + "'zeta_eoe', 'zeta_sad', 'zeta_sbe',"
                                        + " 'zeta_soe', 'zeta_tfmpad', "
                                        + "'zeta_tfppad', 'zeta_be', "
                                        + "'execution_time'}", measure)

        # Quick access to method names
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        # Figure configuration
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

                    # One figure per each combination of configuration,
                    # group and method.
                    figure = plt.figure(figsize=image_size)
                    rst.set_subplot_size(figure)
                    y = self.get_final_value_over_samples(group_idx=i,
                                                          config_idx=j,
                                                          method_idx=m,
                                                          measure=measure)
                    yi = np.argsort(y)

                    for k in range(nplots):

                        # Plot the contrast function, not the relative
                        # permittivity or conductivity maps.
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

                    # Show or save the figures
                    if show:
                        plt.show()
                    else:
                        plt.savefig(file_path + self.name + 'nbest' + str(i)
                                    + str(j) + str(m) + '.' + file_format,
                                    format=file_format)
                        plt.close()

    def study_single_mean(self, measure=None, group_idx=[0], config_idx=[0],
                          method_idx=[0], show=False, file_path='',
                          file_format='eps', printscreen=False, write=False):
        """Study confidence interval of means.

        Given a combination of measure-configuration-group, the
        confidence interval of means among methods is determined. A
        figure is generated to compare the methods. The results may also
        be printed on the screen or in a .txt file.

        Parameters
        ----------
            measure : {'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe',
                       'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe',
                       'zeta_tfmpad', 'zeta_tfppad', 'zeta_be',
                       'execution_time'}
                A string or a list of string to indicate the considered
                measures. If `None`, then all the available ones will be
                considered.

            group_idx : int or list of int, default: [0]
                Factor combination index (maximum contrast, maximum
                object size etc).

            config_idx : int or list of int, default: [0]
                Configuration index.

            method_idx : int of list of int, default: 0
                Method index.

            show : bool, default: False
                If `True`, the plot is shown. Otherwise, the plot is
                saved with the name 'confint' plus the indexes of
                configurations and factor groups.

            file_path : str, default: ''
                Path to save the figure.

            file_format : {'eps', 'png', 'pdf', 'svg'}
                Format of the figure to be saved. Only formats supported
                by `matplotlib.pyplot.savefig`.

            printscreen : bool, default: False
                If `True`, the results for the confidence intervals will
                be printed on the screen.

            write : bool, default: False
                If `True`, the results for the confidence intervals will
                be printed in a .txt file.
        """
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')

        # Check the type of the inputs
        if type(group_idx) is not int and type(group_idx) is not list:
            raise error.WrongTypeInput('Experiment.study_single_mean',
                                       'group_idx', 'int/list of int',
                                       type(group_idx))
        if type(config_idx) is not int and type(config_idx) is not list:
            raise error.WrongTypeInput('Experiment.study_single_mean',
                                       'config_idx', 'int', type(config_idx))
        if type(method_idx) is not int and type(method_idx) is not list:
            raise error.WrongTypeInput('Experiment.study_single_mean',
                                       'method_idx', 'int', type(method_idx))
        if (measure is not None and type(measure) is not str
                and type(measure) is not list):
            raise error.WrongTypeInput('Experiment.study_single_mean',
                                       'meausre', 'None/str/list',
                                       type(measure))

        # Fix the format of the method index as list
        if type(method_idx) is int:
            method_idx = [method_idx]
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

        # Check the values of the inputs
        if min(group_idx) < 0 or max(group_idx) >= len(self.maximum_contrast):
            raise error.WrongValueInput('Experiment.study_single_mean',
                                        'group_idx', '0 to %d'
                                        % len(self.maximum_contrast),
                                        str(group_idx))
        if min(config_idx) < 0 or max(config_idx) >= len(self.configurations):
            raise error.WrongValueInput('Experiment.study_single_mean',
                                        'config_idx', '0 to %d'
                                        % len(self.configurations),
                                        str(config_idx))
        if min(method_idx) < 0 or max(method_idx) >= len(self.method_idx):
            raise error.WrongValueInput('Experiment.study_single_mean',
                                        'method_idx', '0 to %d'
                                        % len(self.methods),
                                        str(method_idx))
        if type(measure) is str:
            try:
                get_title(measure)
            except Exception:
                raise error.WrongValueInput(
                    'Experiments.study_single_mean', 'measure',
                    "{'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe', "
                    + "'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe', "
                    + "'zeta_tfmpad', 'zeta_tfppad', 'zeta_be', "
                    + "'execution_time'}", measure
                )

        # Quick access to method names
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        # Title of the written results
        if write or printscreen:
            title = 'Confidence Interval of Means - *' + self.name + '*'
            text = ''.join(['*' for _ in range(len(title))]) + '\n'
            text = text + title + '\n' + text

        # Multiple measures
        if measure is None or (type(measure) is list and len(measure) > 1):

            if measure is None:
                none_measure = True
            else:
                none_measure = False

            for i in config_idx:

                # Figure configuration
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

                    #  One figure per combination of configuration and group.
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

                            # Normality test
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

                        axes = figure.add_subplot(nrows, ncols, k)
                        confintplot(y, axes=axes, xlabel=get_label(mea),
                                    ylabel=method_names, title=get_title(mea))
                        k += 1

                    plt.suptitle('c%d' % i + 'g%d' % j)

                    # Plot or show the figure
                    if show:
                        plt.show()
                    else:
                        plt.savefig(file_path + self.name + '_confint_'
                                    + str(i) + str(j) + '.' + file_format,
                                    format=file_format)
                        plt.close()

        # Single measure
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

            # For single combination, one single figure
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

            # For this kind of combination, one single figure with multiple
            # subplots
            elif len(group_idx) == 1 and len(config_idx) > 1:

                # Figure configuration
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

                        # Test normality
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

                # Show or save the figure
                if show:
                    plt.show()
                else:
                    plt.savefig(file_path + self.name + '_confint_' + mea + '_'
                                + 'g%d' % j + '.' + file_format,
                                format=file_format)
                    plt.close()

            # Otherwise, one figure per configuration
            else:

                # Figure configuration
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

                            # Test normality
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

                    # Show or save the figure
                    if show:
                        plt.show()
                    else:
                        plt.savefig(file_path + self.name + '_confint_' + mea
                                    + '_' + 'c%d' % i + '.' + file_format,
                                    format=file_format)
                        plt.close()

        # Print or write results
        if printscreen:
            print(text)
        if write:
            file = open(file_path + self.name + '_confint.txt', 'w')
            file.write(text)
            file.close()

    def plot_normality(self, measure=None, group_idx=[0], config_idx=[0],
                       method_idx=[0], show=False, file_path='',
                       file_format='eps'):
        """Check normality of samples.

        Given a combination of measure-configuration-group-method, the
        sample is graphically compared to a standard normal
        distribution. For single method and multiple measures, each
        figure will have N subplots where N is the number of measures.
        And vice-versa. In case of multiple methods and measures,
        each figure will address a single measure.

        Parameters
        ----------
            measure : {'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe',
                       'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe',
                       'zeta_tfmpad', 'zeta_tfppad', 'zeta_be',
                       'execution_time'}
                A string or a list of string to indicate the considered
                measures. If `None`, then all the available ones will be
                considered.

            group_idx : int or list of int, default: [0]
                Factor combination index (maximum contrast, maximum
                object size etc).

            config_idx : int or list of int, default: [0]
                Configuration index.

            method_idx : int of list of int, default: 0
                Method index.

            show : bool, default: False
                If `True`, the plot is shown. Otherwise, the plot is
                saved with the name 'normality' plus the indexes of
                configurations and factor groups.

            file_path : str, default: ''
                Path to save the figure.

            file_format : {'eps', 'png', 'pdf', 'svg'}
                Format of the figure to be saved. Only formats supported
                by `matplotlib.pyplot.savefig`.
        """
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')

        # Check the type of the inputs
        if type(group_idx) is not int and type(group_idx) is not list:
            raise error.WrongTypeInput('Experiment.plot_normality',
                                       'group_idx', 'int/list of int',
                                       type(group_idx))
        if type(config_idx) is not int and type(config_idx) is not list:
            raise error.WrongTypeInput('Experiment.plot_normality',
                                       'config_idx', 'int', type(config_idx))
        if type(method_idx) is not int and type(method_idx) is not list:
            raise error.WrongTypeInput('Experiment.plot_normality',
                                       'method_idx', 'int', type(method_idx))
        if (measure is not None and type(measure) is not str
                and type(measure) is not list):
            raise error.WrongTypeInput('Experiment.plot_normality',
                                       'meausre', 'None/str/list',
                                       type(measure))

        # Fix the format of the method index as list
        if type(method_idx) is int:
            method_idx = [method_idx]
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

        # Check the values of the inputs
        if min(group_idx) < 0 or max(group_idx) >= len(self.maximum_contrast):
            raise error.WrongValueInput('Experiment.plot_normality',
                                        'group_idx', '0 to %d'
                                        % len(self.maximum_contrast),
                                        str(group_idx))
        if min(config_idx) < 0 or max(config_idx) >= len(self.configurations):
            raise error.WrongValueInput('Experiment.plot_normality',
                                        'config_idx', '0 to %d'
                                        % len(self.configurations),
                                        str(config_idx))
        if min(method_idx) < 0 or max(method_idx) >= len(self.method_idx):
            raise error.WrongValueInput('Experiment.plot_normality',
                                        'method_idx', '0 to %d'
                                        % len(self.methods),
                                        str(method_idx))
        if type(measure) is str:
            try:
                get_title(measure)
            except Exception:
                raise error.WrongValueInput(
                    'Experiments.plot_normality', 'measure',
                    "{'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe', "
                    + "'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe', "
                    + "'zeta_tfmpad', 'zeta_tfppad', 'zeta_be', "
                    + "'execution_time'}", measure)

        # Quick access to method names
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        for i in config_idx:

            if none_measure:
                measure = self.get_measure_set(i)

            for j in group_idx:

                # Single method case
                if len(measure) > 1 and len(method_idx) == 1:

                    # Figure configuration
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

                    # Show or save the figure
                    if show:
                        plt.show()
                    else:
                        plt.savefig(file_path + self.name + '_normality_'
                                    + 'c%d' % i + 'g%d' % j + '_'
                                    + method_names[0] + '.' + file_format,
                                    format=file_format)
                        plt.close()

                # Single measure case
                elif len(measure) == 1 and len(method_idx) > 1:

                    # Figure configuration
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

                    # Show or save the figure
                    if show:
                        plt.show()
                    else:
                        plt.savefig(file_path + self.name + '_normality_'
                                    + 'c%d' % i + 'g%d' % j + '_'
                                    + measure[0] + '.' + file_format,
                                    format=file_format)
                        plt.close()

                # Multiple methods and measures
                else:

                    # Figure configuration
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

                        # Show or save the figure
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
        """Paired comparison between two methods.

        Given a measure, configuration and a factor group, the Paired
        Comparison is performed to compare two algorithms. The results
        will indicate evidences or not for difference in performance
        considering the mean case, i.e., mean study. The results may
        be printed on the screen or recorded in .txt file. The
        significance level is defined as 0.05 and the effect size is
        computed for a 0.8 power level.

        Parameters
        ----------
            measure : {'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe',
                       'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe',
                       'zeta_tfmpad', 'zeta_tfppad', 'zeta_be',
                       'execution_time'}
                A string or a list of string to indicate the considered
                measures. If `None`, then all the available ones will be
                considered.

            group_idx : int or list of int, default: [0]
                Factor combination index (maximum contrast, maximum
                object size etc).

            config_idx : int or list of int, default: [0]
                Configuration index.

            method_idx : int of list of int, default: 0
                Method index.

            file_path : str, default: ''
                Path to save the .txt file.

            printscreen : bool, default: False
                If `True`, the results for each test will be printed on
                the screen.

            write : bool, default: False
                If `True`, the results for each test will be printed in
                a .txt file.

        Returns
        -------
            results : list of str
                3-d list (configuration-group-measure) containing one
                of the three options of strings: '1<2' (first method
                had a better performance), '1>2' (otherwise), '1=2'
                (no difference detected).

            mtd1 : int
                Number of times that the first method had a better
                performance.

            mtd2 : int
                Number of times that the second method had a better
                performance.

            equal : int
                Number of times that no evidence for difference in
                performance was found.
        """
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')

        # Check the type of the inputs
        if type(group_idx) is not int and type(group_idx) is not list:
            raise error.WrongTypeInput('Experiment.compare_two_methods',
                                       'group_idx', 'int/list of int',
                                       type(group_idx))
        if type(config_idx) is not int and type(config_idx) is not list:
            raise error.WrongTypeInput('Experiment.compare_two_methods',
                                       'config_idx', 'int', type(config_idx))
        if type(method_idx) is not int and type(method_idx) is not list:
            raise error.WrongTypeInput('Experiment.compare_two_methods',
                                       'method_idx', 'int', type(method_idx))
        if (measure is not None and type(measure) is not str
                and type(measure) is not list):
            raise error.WrongTypeInput('Experiment.compare_two_methods',
                                       'meausre', 'None/str/list',
                                       type(measure))

        # Fix the format of the method index as list
        if type(method_idx) is int:
            method_idx = [method_idx]
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

        # Check the values of the inputs
        if min(group_idx) < 0 or max(group_idx) >= len(self.maximum_contrast):
            raise error.WrongValueInput('Experiment.compare_two_methods',
                                        'group_idx', '0 to %d'
                                        % len(self.maximum_contrast),
                                        str(group_idx))
        if min(config_idx) < 0 or max(config_idx) >= len(self.configurations):
            raise error.WrongValueInput('Experiment.compare_two_methods',
                                        'config_idx', '0 to %d'
                                        % len(self.configurations),
                                        str(config_idx))
        if min(method_idx) < 0 or max(method_idx) >= len(self.method_idx):
            raise error.WrongValueInput('Experiment.compare_two_methods',
                                        'method_idx', '0 to %d'
                                        % len(self.methods),
                                        str(method_idx))
        if type(measure) is str:
            try:
                get_title(measure)
            except Exception:
                raise error.WrongValueInput(
                    'Experiments.compare_two_methods', 'measure',
                    "{'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe', "
                    + "'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe', "
                    + "'zeta_tfmpad', 'zeta_tfppad', 'zeta_be', "
                    + "'execution_time'}", measure)

        # Quick access to method names
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        # Heading of the results
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

            # Different configurations may have different measures
            if none_measure:
                measure = self.get_measure_set(i)

            # The results are divided in sections by the configuration
            if write or printscreen:
                section = 'Configuration ' + self.configurations[i].name
                aux = ''.join(['=' for _ in range(len(section))])
                text = text + section + '\n' + aux + '\n\n'

            results.append(list())
            for j in group_idx:

                # The results are divided in subsections by the group
                if write or printscreen:
                    subsection = 'Group %d' % j
                    aux = ''.join(['-' for _ in range(len(subsection))])
                    text = text + subsection + '\n' + aux + '\n'

                results[-1].append(list())
                for k in range(len(measure)):

                    # First sample
                    y1 = self.get_final_value_over_samples(
                        group_idx=j, config_idx=i, method_idx=method_idx[0],
                        measure=measure[k]
                    )

                    # Second sample
                    y2 = self.get_final_value_over_samples(
                        group_idx=j, config_idx=i, method_idx=method_idx[1],
                        measure=measure[k]
                    )

                    # Each measure is a topic
                    if write or printscreen:
                        topic = '* ' + measure[k]
                        text = text + topic

                    # If data is normally distributed (required assumption)
                    if scipy.stats.shapiro(y1-y2)[1] > .05:

                        # Paired T-test
                        pvalue, lower, upper = stats.weightstats.ttost_paired(
                            y1, y2, 0, 0
                        )

                        # Effect size calculation
                        delta = stats.power.tt_solve_power(
                            nobs=self.sample_size, alpha=0.05, power=.80
                        ) / np.std(y1-y2)

                        result, text = self._pairedtest_result(pvalue, lower,
                                                               upper,
                                                               method_names,
                                                               delta,
                                                               text + ': ')

                    # If data is not normally distributed, try Log
                    # Transformation
                    elif scipy.stats.shapiro(np.log(y1) - np.log(y2))[1] > .05:

                        # Paired T-test
                        pvalue, lower, upper = stats.weightstats.ttost_paired(
                            np.log(y1), np.log(y2), 0, 0
                        )

                        # Effect size calculation
                        delta = stats.power.tt_solve_power(
                            nobs=self.sample_size, alpha=0.05, power=.80
                        ) / np.std(np.log(y1)-np.log(y2))

                        result, text = self._pairedtest_result(
                            pvalue, lower, upper, method_names, delta,
                            text + ' (Log Transformation): '
                        )

                    # If data is not normally distributed, try Square-
                    # Root Transformation
                    elif scipy.stats.shapiro(np.sqrt(y1)
                                             - np.sqrt(y2))[1] > .05:
                        # Paired T-test
                        pvalue, lower, upper = stats.weightstats.ttost_paired(
                            np.log(y1), np.log(y2), 0, 0
                        )

                        # Effect size calculation
                        delta = stats.power.tt_solve_power(
                            nobs=self.sample_size, alpha=0.05, power=.80
                        ) / np.std(np.sqrt(y1)-np.sqrt(y2))

                        result, text = self._pairedtest_result(
                            pvalue, lower, upper, method_names, delta, text
                            + ' (Square-root Transformation): '
                        )

                    # Paired test for non-normal data
                    else:

                        # Wilcoxon Test (two-sided)
                        pvalue = scipy.stats.wilcoxon(y1, y2)[1]
                        text = text + ' (Wilcoxon-Test): '

                        # Null hypothesis is not rejected
                        if pvalue > .05:
                            text = (text + 'Equality hypothesis not rejected '
                                    '(pvalue: %.2e)' % pvalue + '\n')
                            result = '1=2'

                        # Null hypothesis is rejected
                        else:
                            text = (text + 'Equality hypothesis rejected '
                                    '(pvalue: %.2e)' % pvalue + '\n')

                            # One-sided to identify which method has
                            # the lowest mean.
                            _, lower = scipy.stats.wilcoxon(
                                y1, y2, alternative='less'
                            )
                            _, upper = scipy.stats.wilcoxon(
                                y1, y2, alternative='greater'
                            )

                            # First method has a lower measure
                            if lower < .05:
                                text = (text + '  Better performance of '
                                        + method_names[0]
                                        + ' has been detected (pvalue: %.2e).'
                                        % lower + '\n')
                                result = '1<2'
                            # Second method has a lower measure
                            if upper < .05:
                                text = (text + '  Better performance of '
                                        + method_names[1]
                                        + ' has been detected (pvalue: %.2e).'
                                        % upper + '\n')
                                result = '1>2'

                    results[-1][-1].append(result)

                if write or printscreen:
                    text = text + '\n'

        # Count cases - equal performance, superiority of first method
        # and superiority of second method
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

        # Print or write a file
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
        """Compare multiple method (Analysis of Variance).

        Given a measure, a configuration, and a factor group, multiple
        methods are compared through Analysis of Variance. Firstly,
        it is determined if there is evidence for any difference in
        mean performance. Then, all-to-all or one-to-all comparison
        can be made to identify which methods have different
        performances. The significance level is defined as 0.05 and the
        effect size is computed for a 0.8 power level.

        Parameters
        ----------
            measure : {'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe',
                       'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe',
                       'zeta_tfmpad', 'zeta_tfppad', 'zeta_be',
                       'execution_time'}
                A string or a list of string to indicate the considered
                measures. If `None`, then all the available ones will be
                considered.

            group_idx : int or list of int, default: [0]
                Factor combination index (maximum contrast, maximum
                object size etc).

            config_idx : int or list of int, default: [0]
                Configuration index.

            method_idx : int of list of int, default: 0
                Method index.

            file_path : str, default: ''
                Path to save the .txt file.

            printscreen : bool, default: False
                If `True`, the results for each test will be printed on
                the screen.

            write : bool, default: False
                If `True`, the results for each test will be printed in
                a .txt file.

            all2all : bool, default: True
                If `True`, all methods will be compared to each other
                if any difference is detected.

            one2all : None or int, default: None
                If this argument is an integer, then it will be
                interpreted as the index of the method in the list in
                which all methods will be compared with, i.e., the
                control group. Otherwise, it must be `None`.
        """
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')

        # Check the type of the inputs
        if type(group_idx) is not int and type(group_idx) is not list:
            raise error.WrongTypeInput('Experiment.compare_two_methods',
                                       'group_idx', 'int/list of int',
                                       type(group_idx))
        if type(config_idx) is not int and type(config_idx) is not list:
            raise error.WrongTypeInput('Experiment.compare_two_methods',
                                       'config_idx', 'int', type(config_idx))
        if type(method_idx) is not int and type(method_idx) is not list:
            raise error.WrongTypeInput('Experiment.compare_two_methods',
                                       'method_idx', 'int', type(method_idx))
        if (measure is not None and type(measure) is not str
                and type(measure) is not list):
            raise error.WrongTypeInput('Experiment.compare_two_methods',
                                       'meausre', 'None/str/list',
                                       type(measure))

        # Fix the format of the method index as list
        if type(method_idx) is int:
            method_idx = [method_idx]
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

        # Check the values of the inputs
        if min(group_idx) < 0 or max(group_idx) >= len(self.maximum_contrast):
            raise error.WrongValueInput('Experiment.compare_two_methods',
                                        'group_idx', '0 to %d'
                                        % len(self.maximum_contrast),
                                        str(group_idx))
        if min(config_idx) < 0 or max(config_idx) >= len(self.configurations):
            raise error.WrongValueInput('Experiment.compare_two_methods',
                                        'config_idx', '0 to %d'
                                        % len(self.configurations),
                                        str(config_idx))
        if min(method_idx) < 0 or max(method_idx) >= len(self.method_idx):
            raise error.WrongValueInput('Experiment.compare_two_methods',
                                        'method_idx', '0 to %d'
                                        % len(self.methods),
                                        str(method_idx))
        if type(measure) is str:
            try:
                get_title(measure)
            except Exception:
                raise error.WrongValueInput(
                    'Experiments.compare_two_methods', 'measure',
                    "{'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe', "
                    + "'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe', "
                    + "'zeta_tfmpad', 'zeta_tfppad', 'zeta_be', "
                    + "'execution_time'}", measure)

        # Quick access to method names
        method_names = []
        for m in method_idx:
            method_names.append(self.methods[m].alias)

        # Heading of the file
        title = 'Multile Comparison - *' + self.name + '*'
        subtitle = 'Methods: '
        for i in range(len(method_names)-1):
            subtitle += method_names[i] + ', '
        subtitle += method_names[-1]
        text = ''.join(['*' for _ in range(len(title))])
        text = text + '\n' + title + '\n' + text + '\n\n'
        aux = ''.join(['#' for _ in range(len(subtitle))])
        text = text + subtitle + '\n' + aux + '\n\n'
        text = text + 'Significance level: %.2f\n\n' % 0.05

        for i in config_idx:

            # Different configurations may have different measures
            if none_measure:
                measure = self.get_measure_set(i)

            # The results are divided in sections by the configuration
            section = 'Configuration ' + self.configurations[i].name
            aux = ''.join(['=' for _ in range(len(section))])
            text = text + section + '\n' + aux + '\n\n'

            for j in config_idx:

                # The results are divided in subsections by the group
                subsection = 'Group %d' % j
                aux = ''.join(['-' for _ in range(len(subsection))])
                text = text + subsection + '\n' + aux + '\n'

                k = 0
                for mea in measure:

                    # Each measure is a topic
                    text += '* ' + measure[k]

                    # Gather data
                    data = []
                    for m in method_idx:
                        data.append(self.get_final_value_over_samples(j, i, m,
                                                                      mea))

                    # Compute residuals for normality assumption test
                    residuals = np.zeros((len(method_idx), self.sample_size))
                    for p in range(len(method_idx)):
                        for q in range(self.sample_size):
                            residuals[p, q] = data[p][q]-np.mean(data[p])

                    normal_data = True
                    if scipy.stats.shapiro(residuals.flatten())[1] < .05:

                        # In case of non-normal data, transformation is
                        # tried.
                        output = data_transformation(data, residuals=True)

                        # If no transformation has been succeed, then
                        # a flag will be set for a specific test
                        if output is None:
                            normal_data = False

                        # If any transformation has succeed, then this
                        # information is added
                        else:
                            data = output[0]
                            if output[1] == 'log':
                                text += ' (Log transformation)'
                            elif output[1] == 'sqrt':
                                text += ' (Square-root transformation)'

                    # When the normality assumption is checked
                    if normal_data:

                        # The equal-variance condition
                        # (homoscedasticity) is checked
                        if scipy.stats.fligner(*data)[1] > .05:
                            homoscedasticity = True

                            # One-Way ANOVA
                            output = oneway.anova_oneway(data,
                                                         use_var='equal')

                            # Try to compute the effect-size (errors
                            # may occur)
                            try:
                                delta = (
                                    stats.power.FTestAnovaPower().solve_power(
                                        nobs=self.sample_size, alpha=.05,
                                        power=.8, k_groups=len(method_idx)
                                    ) / np.std(data[0])
                                )
                            except Exception:
                                delta = None

                        # In case that the homoscedasticity cannot be
                        # assumed, then Welch Anova + Satterthwaite-
                        # Welch degrees of freedom is performed
                        else:
                            delta = None
                            homoscedasticity = False
                            text += ' (unequal variances)'
                            output = oneway.anova_oneway(data,
                                                         use_var='unequal')

                        if delta is not None:
                            aux = ', effect-size for 0.8 power: %.3e' % delta
                        else:
                            aux = ''

                        # If any difference in means has NOT been
                        # detected
                        if output.pvalue > .05:
                            text += (': failure in reject the hypothesis of '
                                     + "equal means (p-value: %.3e"
                                     % output.pvalue + aux + ' ).')

                        # If any difference in means has NOT been
                        # detected
                        else:
                            text += (': Equality of means hypothesis rejected'
                                     '(p-value: %.3e' % output.pvalue + aux
                                     + ').')

                            # All-to-all comparisons, in case of
                            # homoscedasticity, is addressed by Tukey's
                            # HSD test
                            if all2all and homoscedasticity:

                                # Adjusting arrays to routine
                                data2 = np.zeros(residuals.shape)
                                groups = []
                                for m in range(len(method_idx)):
                                    data2[m, :] = data[m]
                                    groups += ([method_names[m]]
                                               * self.sample_size)
                                data2 = data2.flatten()

                                # Tukey's HSD Test
                                output = snd.stats.multicomp.MultiComparison(
                                    data2, groups
                                ).tukeyhsd()

                                # Add text
                                text += ('\n  - All-to-all comparison '
                                         + '(Tukey HSD):')

                                # Auxiliar variables to identify pairs
                                # of comparison
                                pair_comparison = []
                                for p in range(len(method_idx)-1):
                                    for q in range(p, len(method_idx)):
                                        pair_comparison.append(
                                            [method_names[p], method_names[q]]
                                        )

                                # Check each comparison pair
                                for p in range(len(output.reject)):

                                    # Names of the methods
                                    text += ('\n    * '
                                             + pair_comparison[p][0] + ' and '
                                             + pair_comparison[p][1] + ': ')

                                    # No evidence for difference
                                    if output.reject[p]:
                                        text += ('Not enough evidence for '
                                                 + 'difference in performance')

                                    # Evidence for difference
                                    else:
                                        text += ('Detected evidence for '
                                                 'difference in performance')

                                    # Add p-value information
                                    text += (' (p-value: %.3e), '
                                             % output.pvalues[p]
                                             + 'Confi. Inter.: (%.2e, '
                                             % output.confint[p][0] + '%.2e).'
                                             % output.confint[p][1])

                            # All-to-all comparisons, in case of
                            # heteroscedasticity, is addressed by
                            # Multiple Welch tests with Bon-Ferroni
                            # correction of the significance level
                            elif all2all and not homoscedasticity:

                                # Add text
                                text += ('\n  - All-to-all comparison '
                                         + '(unequal variances):')

                                # Bon-Ferroni correction
                                a = len(method_idx)
                                alpha = 0.05/(a*(a-1)/2)

                                # Check each pair
                                for p in range(len(method_idx)-1):
                                    for q in range(i, len(method_idx)):

                                        # Separe two samples
                                        y1, y2 = data[p], data[q]

                                        # Welch Test of independent
                                        # samples
                                        H0, _, pvalue, _, cf = (
                                            ttest_ind_nonequalvar(y1, y2,
                                                                  alpha)
                                        )

                                        # Names of the methods
                                        text += ('\n    * ' + method_names[p]
                                                 + ' and ' + method_names[q]
                                                 + ': ')

                                        # No evidence for the rejection
                                        # of the hypothesis of equal
                                        # means
                                        if H0 is True:
                                            text += ('Not enough evidence for'
                                                     + ' difference in '
                                                     + 'performance')

                                        # Detected difference in means
                                        else:
                                            text += ('Detected evidence for'
                                                     + ' difference in '
                                                     + 'performance')

                                        # Add p-value information
                                        text += (' (p-value: %.3e), ' % pvalue
                                                 + 'Confi. Inter. (%.2e, '
                                                 % cf[0] + '%.2e).' % cf[1])

                            # One-to-all test in case of
                            # homoscedasticity is performed by Dunnett's
                            # test
                            if one2all is not None and homoscedasticity:

                                # Find method
                                p = np.argwhere(np.array(method_idx)
                                                == one2all)[0][0]

                                # Add text
                                text += ('\n  - One-to-all comparison '
                                         + "(Dunnet's test) - "
                                         + method_names[p] + ':')

                                # Gather samples
                                y0, y, q = data[p], [], []
                                for m in range(len(method_idx)):
                                    if method_idx[m] != p:
                                        y.append(data[m])
                                        q.append(m)

                                # Dunnett's Test
                                output = dunnetttest(y0, y)

                                # Check each comparison
                                for i in range(len(output)):

                                    # Names of the methods
                                    text += ('\n    * ' + method_names[p]
                                             + ' and ' + method_names[q[i]]
                                             + ': ')

                                    # Not enough evidence against
                                    # performance equality
                                    if output[i]:
                                        text += ('Not enough evidence for '
                                                 + 'difference in'
                                                 + 'performance.')

                                    # Detected evidence for difference
                                    # in performance
                                    else:
                                        text += ('Detected evidence for '
                                                 + 'difference in'
                                                 + 'performance.')

                            # One-to-all test in case of
                            # heteroscedasticity is performed by
                            # multiple Welch test with Bon-Ferroni
                            # correction of significance level
                            elif one2all is not None and not homoscedasticity:

                                # Find method
                                p = np.argwhere(np.array(method_idx)
                                                == one2all)[0][0]

                                # Add text
                                text += ('\n  - One-to-all comparison '
                                         + "(unequal variances) - "
                                         + method_names[p] + ':')

                                # Bon-Ferroni correction
                                a = len(method_idx)
                                alpha = 0.05/(a-1)

                                # Gather samples
                                y0, y, q = data[p], [], []
                                for m in range(len(method_idx)):
                                    if method_idx[m] != p:
                                        y.append(data[m])
                                        q.append(m)

                                # Check each comparison
                                for i in range(a-1):

                                    # Welch test for independent samples
                                    H0, _, pvalue, _, cf = (
                                        ttest_ind_nonequalvar(y0, y[i], alpha)
                                    )

                                    # Names of the methods
                                    text += ('\n    * ' + method_names[p]
                                             + ' and ' + method_names[q[i]]
                                             + ': ')

                                    # Not enough evidence against
                                    # performance equality
                                    if H0 is True:
                                        text += ('Not enough evidence for'
                                                 + ' difference in '
                                                 + 'performance')

                                    # Detected evidence for difference
                                    # in performance
                                    else:
                                        text += ('Detected evidence for'
                                                 + ' difference in '
                                                 + 'performance')

                                    # Add p-value information and
                                    # confidence level for difference
                                    # in means
                                    text += (' (p-value: %.3e), ' % pvalue
                                             + 'Confi. Inter. (%.2e, '
                                             % cf[0] + '%.2e).' % cf[1])

                    # In case of Non-Normal data, The Kruskal-Wallis'
                    # Test is performed
                    else:

                        # Add text
                        text += ' (Non Normal Data): '
                        _, pvalue = scipy.stats.kruskal(*data)

                        # Failure in rejecting the null hypothesis
                        if pvalue > 0.05:
                            text += ('Not enough evidence against difference'
                                     + ' in performance (p-value: %.3e).'
                                     % pvalue)

                        # The null hypothesis is rejected
                        else:
                            text += ('Evidence has been detected for '
                                     + 'difference in performance (p-value: '
                                     + '%.3e).' % pvalue)

                            # For all-to-all comparisons, the
                            # Mann-Whitney Rank Test is performed for
                            # detecting difference in the probability
                            # of superior results
                            if all2all:

                                # Add text
                                text += ('\n  - All-to-all comparison '
                                         + '(Mann-Whitney Rank Test):')

                                # Check each comparison
                                for p in range(len(method_idx)-1):
                                    for q in range(p, len(method_idx)):

                                        # Mann-Whitney Rank Test
                                        _, pvalue = scipy.stats.mannwhitneyu(
                                            data[p], data[q]
                                        )

                                        # Names of the methods
                                        text += ('\n    * ' + method_names[p]
                                                 + ' and ' + method_names[q]
                                                 + ': ')

                                        # Failure in rejecting the null
                                        # hypothesis
                                        if pvalue > 0.05:
                                            text += ('Not enough evidence '
                                                     + 'against the hypothesis'
                                                     + ' of same probability '
                                                     + 'of superiority')

                                        # Null hypothesis has been
                                        # rejected
                                        else:
                                            text += ('Evidence detected for '
                                                     + 'difference in '
                                                     + 'probability of '
                                                     + 'superiorit')

                                        # Add p-value information
                                        text += ' (p-value: %.3e).' % pvalue

                            # For one-to-all comparisons, the
                            # Mann-Whitney Rank Test is performed for
                            # detecting difference in the probability
                            # of superior results
                            if one2all is not None:

                                # Find the control method
                                p = np.argwhere(np.array(method_idx)
                                                == one2all)[0][0]

                                # Add text
                                text += ('\n  - One-to-all comparison '
                                         + '(Mann-Whitney Rank Test) - '
                                         + method_names[p] + ':')

                                # Gather samples
                                y0, y, q = data[p], [], []
                                for m in range(len(method_idx)):
                                    if method_idx[m] != p:
                                        y.append(data[m])
                                        q.append(m)

                                # Check each comparison
                                for i in range(a-1):

                                    # Mann-Whitney Rank Test
                                    _, pvalue = scipy.stats.mannwhitneyu(y0,
                                                                         y[i])

                                    # Names of the methods
                                    text += ('\n    * ' + method_names[p]
                                             + ' and ' + method_names[q[i]]
                                             + ': ')

                                    # Failure in rejecting the null
                                    # hypothesis
                                    if pvalue > 0.05:
                                        text += ('Not enough evidence '
                                                 + 'against the hypothesis'
                                                 + ' of same probability '
                                                 + 'of superiority')

                                    # Null hypothesis has been
                                    # rejected
                                    else:
                                        text += ('Evidence detected for '
                                                 + 'difference in '
                                                 + 'probability of '
                                                 + 'superiorit')
                                    text += ' (p-value: %.3e).' % pvalue

                    text += '\n'
                    k += 1

        # Print or write results
        if printscreen:
            print(text)
        if write:
            file = open(file_path + self.name + '_multiple_comparisons.txt',
                        'w')
            file.write(text)
            file.close()

    def factor_study(self, method_idx, measure=None, group_idx=None,
                     config_idx=None, printscreen=False, write=False,
                     file_path='', show=False, figure_format='eps'):
        """Analyse factor influence on a single method.

        Scenario and configuration characteristics may influence the
        performance of the algorithm. Factorial analysis is a
        statistical tool to determine these influences and possible
        interactions between characteristics.

        Given a method and a set of measures, the routine determines
        the factors and level given the configuration and groups
        indexes, i.e., it automatically identifies the factors that
        should be taken into account on the analysis.

        If more than one configuration index is provided, than it is
        automatically defined as the levels of one factor of the
        factorial analysis. The group indexes should be passed taking
        into account all the levels of the desired factors. The routine
        automatically finds the factors and levels.

        ONLY TWO OR THREE FACTORS ARE ALLOWED IN THIS VERSION! The
        significance level is defined as 0.05.

        Parameters
        ----------
            method_idx : int
                Method index.

            measure : {'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe',
                       'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe',
                       'zeta_tfmpad', 'zeta_tfppad', 'zeta_be',
                       'execution_time'}
                A string or a list of string to indicate the considered
                measures. If `None`, then all the available ones will be
                considered.

            group_idx : int or list of int, default: None
                Factor combination index (maximum contrast, maximum
                object size etc). If `None`, then all the groups are
                considered.

            config_idx : int or list of int, default: None
                Configuration index. If `None`, then the first is
                considered.

            file_path : str, default: ''
                Path to save the .txt file.

            printscreen : bool, default: False
                If `True`, the results for each test will be printed on
                the screen.

            write : bool, default: False
                If `True`, the results for each test will be printed in
                a .txt file.

            show : bool, default: False
                If `True`, then the normality and homoscedasticity plots
                are shown for graphic assumption verification.
                Otherwise, a figure is save with the name
                'factorialanalysis'.

            file_format : {'eps', 'png', 'pdf', 'svg'}
                Format of the figure to be saved. Only formats supported
                by `matplotlib.pyplot.savefig`.
        """
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')

        # Check the type of the inputs
        if (type(group_idx) is not int and type(group_idx) is not list
                and group_idx is not None):
            raise error.WrongTypeInput('Experiment.factor_study',
                                       'group_idx', 'int/list of int'
                                       + '/None', type(group_idx))
        if (type(config_idx) is not int
                and type(config_idx) is not list
                and config_idx is not None):
            raise error.WrongTypeInput('Experiment.factor_study',
                                       'config_idx', 'int/list of int/'
                                       + 'None', type(config_idx))
        if type(method_idx) is not int:
            raise error.WrongTypeInput('Experiment.factor_study',
                                       'method_idx', 'int', type(method_idx))
        if (measure is not None and type(measure) is not str
                and type(measure) is not list):
            raise error.WrongTypeInput('Experiment.factor_study',
                                       'meausre', 'None/str/list',
                                       type(measure))

        # Fix the format of the method index as list
        if type(group_idx) is int:
            group_idx = [group_idx]
        elif group_idx is None:
            group_idx = np.arange(len(self.maximum_contrast))
        if type(config_idx) is int:
            config_idx = [config_idx]
        elif config_idx is None:
            config_idx = np.array([0])
        if type(measure) is str:
            try:
                get_title(measure)
                measure = [measure]
            except Exception:
                raise error.WrongValueInput(
                    'Experiments.plot_normality', 'measure',
                    "{'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe', "
                    + "'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe', "
                    + "'zeta_tfmpad', 'zeta_tfppad', 'zeta_be', "
                    + "'execution_time'}", measure)
            

        # Check the values of the inputs
        if min(group_idx) < 0 or max(group_idx) >= len(self.maximum_contrast):
            raise error.WrongValueInput('Experiment.factor_study',
                                        'group_idx', '0 to %d'
                                        % len(self.maximum_contrast),
                                        str(group_idx))
        if min(config_idx) < 0 or max(config_idx) >= len(self.configurations):
            raise error.WrongValueInput('Experiment.factor_study',
                                        'config_idx', '0 to %d'
                                        % len(self.configurations),
                                        str(config_idx))
        if min(method_idx) < 0 or max(method_idx) >= len(self.method_idx):
            raise error.WrongValueInput('Experiment.factor_study',
                                        'method_idx', '0 to %d'
                                        % len(self.methods),
                                        str(method_idx))

        # Auxiliar variables
        nfactors = 0
        which_factors = []
        levels = []
        levels_idx = []
        nlevels = []

        # Detect configuration levels
        if len(config_idx) > 1:
            nfactors += 1
            which_factors.append('configuration')
            levels.append([self.configurations[i].name for i in config_idx])
            levels_idx.append(config_idx)
            nlevels.append(len(config_idx))

        # Check if maximum_contrast is a factor
        for i in group_idx[:-2]:
            if (self.maximum_contrast[i]
                    != self.maximum_contrast[group_idx[-1]]):
                nfactors += 1
                which_factors.append('maximum_contrast')
                unique, unique_inverse = np.unique(
                    [self.maximum_contrast[j] for j in group_idx],
                    return_inverse=True
                )
                levels.append(unique)
                nlevels.append(unique.size)
                levels_idx.append(unique_inverse)
                break

        # Check if maximum_object_size is a factor
        for i in group_idx[:-2]:
            if (self.maximum_object_size[i]
                    != self.maximum_object_size[group_idx[-1]]):
                nfactors += 1
                which_factors.append('maximum_object_size')
                unique, unique_inverse = np.unique(
                    [self.maximum_object_size[j] for j in group_idx],
                    return_inverse=True
                )
                levels.append(unique)
                nlevels.append(unique.size)
                levels_idx.append(unique_inverse)
                break

        # Check if maximum_contrast_density is a factor
        for i in group_idx[:-2]:
            if (self.maximum_contrast_density[i]
                    != self.maximum_contrast_density[group_idx[-1]]):
                nfactors += 1
                which_factors.append('maximum_contrast_density')
                unique, unique_inverse = np.unique(
                    [self.maximum_contrast_density[j] for j in group_idx],
                    return_inverse=True
                )
                levels.append(unique)
                nlevels.append(unique.size)
                levels_idx.append(unique_inverse)
                break

        # Check if noise is a factor
        for i in group_idx[:-2]:
            if (self.noise[i] != self.noise[group_idx[-1]]):
                nfactors += 1
                which_factors.append('noise')
                unique, unique_inverse = np.unique(
                    [self.noise[j] for j in group_idx],
                    return_inverse=True
                )
                levels.append(unique)
                nlevels.append(unique.size)
                levels_idx.append(unique_inverse)
                break

        # Check if map_pattern is a factor
        for i in group_idx[:-2]:
            if (self.map_pattern[i] != self.map_pattern[group_idx[-1]]):
                nfactors += 1
                which_factors.append('map_pattern')
                unique, unique_inverse = np.unique(
                    [self.map_pattern[j] for j in group_idx],
                    return_inverse=True
                )
                levels.append(unique)
                nlevels.append(unique.size)
                levels_idx.append(unique_inverse)
                break

        # It only supports two or three factors
        if nfactors != 2 and nfactors != 3:
            return None

        # Heading of the results
        title = 'Factor Study - *' + self.name + '*'
        subtitle = 'Method: ' + self.methods[method_idx].alias
        text = ''.join(['*' for _ in range(len(title))])
        text = text + '\n' + title + '\n' + text + '\n\n'
        aux = ''.join(['#' for _ in range(len(subtitle))])
        text = text + subtitle + '\n' + aux + '\n\n'
        text = text + 'Significance level: %.2f\n' % 0.05
        text += 'Factors: '
        for i in range(nfactors):
            text += which_factors[i] + ' (levels: '
            if type(levels[i][0]) is str or type(levels[i][0]) is np.str_:
                for j in range(len(levels[i])-1):
                    text += levels[i][j] + ', '
                text += levels[i][-1] + ')'
            elif type(levels[i][0]) is int:
                for j in range(len(levels[i])-1):
                    text += '%d, ' % levels[i][j]
                text += '%d)' % levels[i][-1]
            elif min([abs(j) for j in levels[i]]) > 1e-2:
                for j in range(len(levels[i])-1):
                    text += '%.2f, ' % levels[i][j]
                text += '%.2f)' % levels[i][-1]
            else:
                for j in range(len(levels[i])-1):
                    text += '%.1e, ' % levels[i][j]
                text += '%.1e)' % levels[i][-1]
            if i != nfactors-1:
                text += ', '
        text += '\n\n'

        # If no measure is specified, then it will be assumed the one
        # valid for the first configuration index provided.
        if measure is None:
            measure = self.get_measure_set(config_idx[0])

        # Run a study for each measure
        for i in range(len(measure)):

            # Each measure is a section
            section = 'Measure: ' + measure[i]
            aux = ''.join(['=' for _ in range(len(section))])
            text = text + section + '\n' + aux + '\n\n'

            # Two-factor analysis
            if nfactors == 2:

                data = np.zeros((nlevels[0], nlevels[1], self.sample_size))

                # First factor
                for m in range(nlevels[0]):

                    # Second factor
                    for n in range(nlevels[1]):

                        # Determine the correpondent sample 
                        if which_factors[0] == 'configuration':
                            p = config_idx[m]
                            q = np.argwhere(levels_idx[1] == n)[0][0]
                            q = group_idx[q]
                        elif which_factors[1] == 'configuration':
                            p = config_idx[n]
                            q = np.argwhere(levels_idx[0] == m)[0][0]
                            q = group_idx[q]
                        else:
                            p = config_idx[0]
                            q = group_idx[
                                np.logical_and(levels_idx[0] == m,
                                               levels_idx[1] == n)
                            ][0]

                        data[m, n, :] = self.get_final_value_over_samples(
                            group_idx=q, config_idx=p, method_idx=method_idx,
                            measure=measure[i]
                        )

                # Run factorial analysis
                output = factorial_analysis(data, group_names=which_factors,
                                            ylabel=get_label(measure[i]))

                # Factor A results
                text += '* ' + which_factors[0] + ' (main effect): '
                if output[0][0]:
                    text += ('failure in rejecting the hypothesis of '
                             + 'equality between levels ')
                else:
                    text += 'detected difference between levels '
                text += '(p-value: %.3e).\n' % output[1][1]

                # Factor B results
                text += '* ' + which_factors[1] + ' (main effect): '
                if output[0][1]:
                    text += ('failure in rejecting the hypothesis of '
                             + 'equality between levels ')
                else:
                    text += 'detected difference between levels '
                text += '(p-value: %.3e).\n' % output[1][1]

                # Interaction AB results
                text += '* Interaction effect: '
                if output[0][2]:
                    text += ('failure in rejecting the hypothesis of '
                             + 'no-iteraction effect between factors ')
                else:
                    text += 'detected iteraction effect between factors '
                text += '(p-value: %.3e).\n' % output[1][2]

                # In case of data transformation...
                if output[-1] is not None:
                    text += '* Data transformation: ' + output[-1] + '\n'

                # Normality assumption check
                text += "* Normality assumption (Shapiro-Wilk's test): "
                if output[2] > 0.05:
                    text += 'not rejected '
                else:
                    text += 'rejected '
                text += ' (p-value: %.3e).\n' % output[2]

                # Homoscedasticity assumption check
                text += "* Homoscedascity assumption (Fligner-Killen's Test): "
                if output[3] > 0.05:
                    text += 'not rejected '
                else:
                    text += 'rejected '
                text += ' (p-value: %.3e).\n\n' % output[3]

                # Show or plot graphic assumptions verification
                if show:
                    plt.show()
                else:
                    plt.savefig(file_path + self.name + '_factorialanalysis_'
                                + measure[i] + '.' + figure_format,
                                format=figure_format)
                    plt.close()

            # Three-factor analysis
            elif nfactors == 3:

                data = np.zeros((nlevels[0], nlevels[1], nlevels[2],
                                 self.sample_size))

                # First factor
                for k in range(nlevels[0]):
                    # Second factor
                    for m in range(nlevels[1]):
                        # Third factor
                        for n in range(nlevels[2]):

                            # Determine the correpondent sample 
                            if which_factors[0] == 'configuration':
                                p = config_idx[k]
                                q = group_idx[
                                    np.logical_and(levels_idx[1] == m,
                                                   levels_idx[2] == n)
                                ][0]

                            elif which_factors[1] == 'configuration':
                                p = config_idx[m]
                                q = group_idx[
                                    np.logical_and(levels_idx[0] == k,
                                                   levels_idx[2] == n)
                                ][0]

                            elif which_factors[2] == 'configuration':
                                p = config_idx[n]
                                q = group_idx[
                                    np.logical_and(levels_idx[0] == k,
                                                   levels_idx[1] == m)
                                ][0]
                            else:
                                p = config_idx[0]
                                q = group_idx[
                                    np.logical_and(
                                        levels_idx[0] == k,
                                        np.logical_and(levels_idx[1] == m,
                                                       levels_idx[2] == n)
                                    )
                                ][0]

                            data[k, m, n, :] = (
                                self.get_final_value_over_samples(
                                    group_idx=q, config_idx=p,
                                    method_idx=method_idx, measure=measure[i]
                                )
                            )

                # Run factorial analysis
                output = factorial_analysis(data, group_names=which_factors,
                                            ylabel=get_label(measure[i]))

                # Factor A results
                text += '* ' + which_factors[0] + ' (main effect): '
                if output[0][0]:
                    text += ('failure in rejecting the hypothesis of '
                             + 'equality between levels ')
                else:
                    text += 'detected difference between levels '
                text += '(p-value: %.3e).\n' % output[1][1]

                # Factor B results
                text += '* ' + which_factors[1] + ' (main effect): '
                if output[0][1]:
                    text += ('failure in rejecting the hypothesis of '
                             + 'equality between levels ')
                else:
                    text += 'detected difference between levels '
                text += '(p-value: %.3e).\n' % output[1][1]

                # Factor C results
                text += '* ' + which_factors[2] + ' (main effect): '
                if output[0][2]:
                    text += ('failure in rejecting the hypothesis of '
                             + 'equality between levels ')
                else:
                    text += 'detected difference between levels '
                text += '(p-value: %.3e).\n' % output[1][2]

                # Iteraction AB results
                text += ('* Interaction effect between ' + which_factors[0]
                         + ' and ' + which_factors[1] + ': ')
                if output[0][3]:
                    text += ('failure in rejecting the hypothesis of '
                             + 'no-iteraction effect between these factors ')
                else:
                    text += 'detected iteraction effect between these factors '
                text += '(p-value: %.3e).\n' % output[1][3]

                # Iteraction AC results
                text += ('* Interaction effect between ' + which_factors[0]
                         + ' and ' + which_factors[2] + ': ')
                if output[0][4]:
                    text += ('failure in rejecting the hypothesis of '
                             + 'no-iteraction effect between these factors ')
                else:
                    text += 'detected iteraction effect between these factors '
                text += '(p-value: %.3e).\n' % output[1][4]

                # Iteraction BC results
                text += ('* Interaction effect between ' + which_factors[1]
                         + ' and ' + which_factors[2] + ': ')
                if output[0][5]:
                    text += ('failure in rejecting the hypothesis of '
                             + 'no-iteraction effect between these factors ')
                else:
                    text += 'detected iteraction effect between these factors '
                text += '(p-value: %.3e).\n' % output[1][5]

                # Iteraction ABC results
                text += '* Interaction effect between among all factors: '
                if output[0][6]:
                    text += ('failure in rejecting the hypothesis of '
                             + 'no-iteraction effect among all factors ')
                else:
                    text += 'detected iteraction effect among all factors '
                text += '(p-value: %.3e).\n' % output[1][6]

                # In case of data transformation...
                if output[-1] is not None:
                    text += '* Data transformation: ' + output[-1] + '\n'

                # Normality assumption check
                text += "* Normality assumption (Shapiro-Wilk's test): "
                if output[2] > 0.05:
                    text += 'not rejected '
                else:
                    text += 'rejected '
                text += ' (p-value: %.3e).\n' % output[2]

                # Homoscedasticity assumption check
                text += "* Homoscedascity assumption (Fligner-Killen's Test): "
                if output[3] > 0.05:
                    text += 'not rejected '
                else:
                    text += 'rejected '
                text += ' (p-value: %.3e).\n\n' % output[3]

                # Show or plot graphic assumptions verification
                if show:
                    plt.show()
                else:
                    plt.savefig(file_path + self.name + '_factorialanalysis_'
                                + measure[i] + '.' + figure_format,
                                format=figure_format)
                    plt.close()

        # Print and write a file
        if printscreen:
            print(text)
        if write:
            file = open(file_path + self.name + '_factorialanalysis.txt', 'w')
            file.write(text)
            file.close()

    def _pairedtest_result(self, pvalue, lower, upper, method_names,
                           effect_size=None, text=None):
        """Auxiliar method for displaying results of Paired T-Test.

        Parameters
        ----------
            pvalue : float

            lower : tuple
                Results of lower side of the one-side test.

            upper : tuple
                Results of upper side of the one-side test.

            method_names : list of str

            effect_size : float

            test : str
                Variable in which the results will be written.

        Returns
        -------
            result : {'1<2', '1=2', '1>2'}

            text : str
        """
        if text is None:
            text = ''

        if effect_size is None:
            aux = ''
        else:
            aux = ', effect size: %.3e' % effect_size

        # Two-sided results
        if pvalue < .05:
            text = (text + 'Difference hypothesis rejected (pvalue: %.2e'
                    % pvalue + aux + ').\n')
            result = '1=2'
        else:
            text = (text + 'No evidence against difference in performance '
                    + '(pvalue: %.2e' % pvalue + aux + ').\n')

            # One-side test
            if lower[2] > .05:
                text = (text + '  No evidence against a better performance of '
                        + method_names[0] + ' (pvalue: %.2e).' % lower[2]
                        + '\n')
                result = '1<2'

            # One-side test
            else:
                text = (text + '  No evidence against a better performance of '
                        + method_names[1] + ' (pvalue: %.2e).' % upper[2]
                        + '\n')
                result = '1>2'
        return result, text

    def __str__(self):
        """Print the object information."""

        # Name
        message = 'Experiment name: ' + self.name

        # Maximum contrast values
        if all(i == self.maximum_contrast[0] for i in self.maximum_contrast):
            message = (message
                       + '\nMaximum Contrast: %.2f'
                       % np.real(self.maximum_contrast[0]) + ' %.2ej'
                       % np.imag(self.maximum_contrast[0]))
        else:
            message = (message + '\nMaximum Contrast: '
                       + str(self.maximum_contrast))

        # Maximum object size values
        if all(i == self.maximum_object_size[0]
               for i in self.maximum_object_size):
            message = (message + '\nMaximum Object Size: %.1f [lambda]'
                       % self.maximum_object_size[0])
        else:
            message = (message + '\nMaximum Object Size: '
                       + str(self.maximum_object_size))

        # Maximum contrast density values
        if all(i == self.maximum_contrast_density[0]
               for i in self.maximum_contrast_density):
            message = (message + '\nMaximum Constrast Density: %.1f'
                       % np.real(self.maximum_contrast_density[0]) + ' + %.2ej'
                       % np.imag(self.maximum_contrast_density[0]))
        else:
            message = (message + '\nMaximum Contrast Density: '
                       + str(self.maximum_contrast_density))

        # Noise values
        if all(i == 0 for i in self.noise):
            message = message + '\nNoise levels: None'
        elif all(i == self.noise[0] for i in self.noise):
            message = message + '\nNoise levels: %.1e' % self.noise[0]
        else:
            message = message + '\nNoise levels: ' + str(self.noise)

        # Map patterns
        if all(i == self.map_pattern[0]
               for i in self.map_pattern):
            message = (message + '\nMap pattern: ' + self.map_pattern[0])
        else:
            message = (message + '\nMap pattern: ' + str(self.map_pattern))

        # Sample size
        if self.sample_size is not None:
            message = message + '\nSample Size: %d' % self.sample_size

        # Considered studies
        message = message + 'Study residual error: ' + str(self.study_residual)
        message = message + 'Study map error: ' + str(self.study_map)
        message = (message + 'Study intern field error: '
                   + str(self.study_internfield))
        message = (message + 'Study execution time: '
                   + str(self.study_executiontime))

        # Configurations list
        if self.configurations is not None and len(self.configurations) > 0:
            message = message + '\nConfiguration names:'
            for i in range(len(self.configurations)-1):
                message = message + ' ' + self.configurations[i].name + ','
            message = message + ' ' + self.configurations[-1].name

        # Methods list
        if self.methods is not None and len(self.methods) > 0:
            message = message + '\nMethods:'
            for i in range(len(self.methods)-1):
                message = message + ' ' + self.methods[i].alias + ','
            message = message + ' ' + self.methods[-1].alias

        # Forward solver for data synthesization
        if self.forward_solver is not None:
            message = message + '\nForward solver: ' + self.forward_solver.name

        # Resolution for synthesized maps
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

        # Resolution for recovered images
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

        # Number of scenarios
        if self.scenarios is not None:
            message = (message + '\nNumber of scenarios: %d'
                       % (len(self.scenarios)*len(self.scenarios[0])
                          * len(self.scenarios[0][0])))
        return message

    def save(self, file_path=''):
        """Save object information."""
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
            STUDY_EXECUTIONTIME: self.study_executiontime,
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
        self.study_executiontime = data[STUDY_EXECUTIONTIME]
        self.results = data[RESULTS]

    def get_final_value_over_samples(self, group_idx=0, config_idx=0,
                                     method_idx=0, measure='zeta_rn'):
        """Return the results of a single sample.

        Given a group, a configuration, a method, and a measure, it
        returns the final value obtained for each scenario in the
        sample. This routine is useful for separing sample results.

        Parameters
        ----------
            group_idx : int, default: 0
                Group index.

            config_idx: int, default: 0
                Configuration index.

            method_idx : int, default: 0
                Method index.

            measure : str, default: 'zeta_rn'
                Measure name.

        Return
        ------
            data : 1-d :class:`numpy.ndarray`
        """
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Experiment', 'results')

        # Check the type of the inputs
        if (type(group_idx) is not int):
            raise error.WrongTypeInput('Experiment.get_final_value_over'
                                       + '_samples', 'group_idx', 'int',
                                       type(group_idx))
        if type(config_idx) is not int:
            raise error.WrongTypeInput('Experiment.get_final_value_over'
                                       '_samples', 'config_idx', 'int',
                                       type(config_idx))
        if type(method_idx) is not int:
            raise error.WrongTypeInput('Experiment.get_final_value_over'
                                       '_samples', 'method_idx', 'int',
                                       type(method_idx))
        if type(measure) is not str:
            raise error.WrongTypeInput('Experiment.get_final_value_over'
                                       '_samples', 'measure', 'str',
                                       type(measure))

        try:
            get_title(measure)
        except Exception:
            raise error.WrongValueInput(
                'Experiments.plot_normality', 'measure',
                "{'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe', "
                + "'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe', "
                + "'zeta_tfmpad', 'zeta_tfppad', 'zeta_be', 'execution_time'}",
                measure
            )
            

        # Check the values of the inputs
        if group_idx < 0 or group_idx >= len(self.maximum_contrast):
            raise error.WrongValueInput('Experiment.get_final_value_'
                                        + 'over_samples', 'group_idx',
                                        '0 to %d'
                                        % len(self.maximum_contrast),
                                        str(group_idx))
        if config_idx < 0 or config_idx >= len(self.configurations):
            raise error.WrongValueInput('Experiment.get_final_value_'
                                        + 'over_samples', 'config_idx',
                                        '0 to %d'
                                        % len(self.configurations),
                                        str(config_idx))
        if method_idx < 0 or method_idx >= len(self.method_idx):
            raise error.WrongValueInput('Experiment.get_final_value_'
                                        + 'over_samples', 'method_idx',
                                        '0 to %d' % len(self.methods),
                                        str(method_idx))
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
            elif measure == 'execution_time':
                data[i] = self.results[g][c][i][m].execution_time
            else:
                raise error.WrongValueInput('Experiments.get_final_value_over_'
                                            + 'samples', 'measure',
                                            "'zeta_rn'/'zeta_rpad'/"
                                            + "'zeta_epad'/'zeta_ebe'/"
                                            + "'zeta_eoe'/'zeta_sad'/"
                                            + "'zeta_sbe'/'zeta_soe'/'zeta_be'"
                                            + "/'zeta_tfmpad'/'zeta_tfppad'/"
                                            + "'execution_time'", measure)
        return data

    def get_measure_set(self, config_idx=0):
        """Return the available measures given a configuration.

        It also takes into account the `study` flags defined on this
        object.

        Parameter
        ---------
            config_idx : int, default: 0
                Configuration index.

        Returns
        -------
            measures : list of str
        """
        measures = []

        # Residual measures
        if self.study_residual:
            measures.append('zeta_rn')
            measures.append('zeta_rpad')

        # Contrast measures
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

        # Intern field measures
        if self.study_internfield:
            measures.append('zeta_tfmpad')
            measures.append('zeta_tfppad')

        # Execution time measure
        if self.study_executiontime:
            measures.append('execution_time')

        return measures


def factorial_analysis(data, alpha=0.05, group_names=None, ylabel=None):
    r"""Perform factorial analysis.

    Given a data set with some amount of factors and levels, the method
    performs the factorial analysis in order to find evidences for
    impact of single factors (main effects) and combination among them
    (interaction effects) [1]_.

    In this current version, it only supports two or three factors and
    balanced data.

    Parameters
    ----------
        data : :class:`numpy.ndarray`
            The data set in array format in which each dimension
            represents a factor and the number of elements represents
            the number of levels of respective factor. The shape must be
            either (a, b, n), for two-factors, or (a, b, c, n), for
            three factors.  *Obs.*: the last dimension is the number of
            samples for each combination of factors-levels.

        alpha : float, default: 0.05
            Significance level.

        group_names : list, default: None
            Factor names for plot purposes.

        ylabel : str
            Y-axis label for plot purposes (meaning of the data).

    Returns
    -------
        null_hypothesis : list
            The list with the results of the null hypothesis of the
            statistic tests. If `True`, means that the test failed to
            reject the null hypothesis; if `False`, means the null
            hypothesis was rejected. For two-factor anaylsis, the each
            element represents the test on the following factors
            `[A, B, AB]`. For three-factor,
            `[A, B, C, AB, AC, BC, ABC]`.

        pvalues : list
            The list with the p-values of each test. The order follows
            the same defined for `null_hypothesis`.

        shapiro_pvalue: float
            The p-value of the Shapiro-Wilk's test for normality of
            residuals assumption. A p-value less than 0.05 means the
            rejection of the assumption.

        fligner_pvalue: float
            The p-value of the Fligner-Killen's test for homoscedascity
            (variance equality) of samples. A p-value less than 0.05
            means the rejection of the assumption.

        fig : :class:`matplotlib.figure.Figure`
            A plot showing the normality and homoscedascity assumption.
            The graphic way to anaylise the assumptions.

        transformation : None or str
            If `None`, no transformation was applied on the data in
            order to fix it for following the assumption. Otherwise,
            it is a string saying the type of transformation. 

    References
    ----------
    .. [1] Montgomery, Douglas C. Design and analysis of experiments.
       John wiley & sons, 2017.
    """
    NF = data.ndim-1

    # Two-Factor Analysis
    if NF == 2:

        # Number of levels and samples
        a, b, n = data.shape

        # Computing residuals and separing samples
        res = np.zeros((a, b, n))
        samples = []
        for i in range(a):
            for j in range(b):
                res[i, j, :] = data[i, j, :] - np.mean(data[i, j, :])
                samples.append(data[i, j, :])

        # Check normality assumption
        if scipy.stats.shapiro(res.flatten())[1] < .05:

            # For Box-Cox transformation, it is required positive data.
            if np.amin(res) <= 0 and np.amin(res) < np.amin(data):
                delta = -np.amin(res) + 1
            elif np.amin(data) <= 0 and np.amin(data) <= np.amin(res):
                delta = -np.amin(data) + 1
            else:
                delta = 0

            # In case of non-normality, the Box-Cox transformation
            # is performed.
            _, lmbda = scipy.stats.boxcox(res.flatten() + delta)
            y = scipy.stats.boxcox(data.flatten() + delta, lmbda=lmbda)
            y = y.reshape((a, b, n))
            transformation = 'boxcox, lambda=%.3e' % lmbda
            res = np.zeros((a, b, n))
            samples = []
            for i in range(a):
                for j in range(b):
                    res[i, j, :] = y[i, j, :] - np.mean(y[i, j, :])
                    samples.append(y[i, j, :])
        else:
            y = np.copy(data)
            transformation = None

        # Save results of assumptions.
        _, shapiro_pvalue = scipy.stats.shapiro(res.flatten())
        _, fligner_pvalue = scipy.stats.fligner(*samples)

        # Plot normality and homoscedascity
        fig = plt.figure(figsize=rst.IMAGE_SIZE_1x2)
        rst.set_subplot_size(fig)
        axes = fig.add_subplot(1, 2, 1)
        normalitiyplot(res.flatten(), axes=axes)
        axes = fig.add_subplot(1, 2, 2)
        homoscedasticityplot(y, axes=axes, title='Homoscedascity',
                             ylabel=ylabel, names=group_names)

        # Means
        yhi = np.sum(y, axis=(1, 2))/(b*n)
        yhj = np.sum(y, axis=(0, 2))/(a*n)
        yhij = np.sum(y, axis=2)/n
        yh = np.sum(y)/(a*b*n)

        # Square sums
        SSA = b*n*np.sum((yhi-yh)**2)
        SSB = a*n*np.sum((yhj-yh)**2)
        SSAB = 0
        SSE = 0
        for i in range(a):
            for j in range(b):
                SSAB += n*(yhij[i, j] - yhi[i] - yhj[j] + yh)**2
                SSE += np.sum((y[i, j, :]-yhij[i, j])**2)

        # Degrees of freedom
        dfA, dfB, dfAB, dfE = a-1, b-1, (a-1)*(b-1), a*b*(n-1)

        # Means of square sums
        MSA = SSA/(a-1)
        MSB = SSB/(b-1)
        MSAB = SSAB/(a-1)/(b-1)
        MSE = SSE/(a*b)/(n-1)

        # Statistics
        F0A, F0B, F0AB = MSA/MSE, MSB/MSE, MSAB/MSE

        # Critical values
        FCA = scipy.stats.f.ppf(1-alpha, dfA, dfE)
        FCB = scipy.stats.f.ppf(1-alpha, dfB, dfE)
        FCAB = scipy.stats.f.ppf(1-alpha, dfAB, dfE)

        # Hypothesis tests
        null_hypothesis = [F0A < FCA, F0B < FCB, F0AB < FCAB]

        # P-value computation
        pvalue_a = 1-scipy.stats.f.cdf(F0A, dfA, dfE)
        pvalue_b = 1-scipy.stats.f.cdf(F0B, dfB, dfE)
        pvalue_ab = 1-scipy.stats.f.cdf(F0AB, dfAB, dfE)

        return (null_hypothesis, [pvalue_a, pvalue_b, pvalue_ab],
                shapiro_pvalue, fligner_pvalue, fig, transformation)

    # Three-factor analysis
    elif NF == 3:

        # Number of levels and samples
        a, b, c, n = data.shape

        # Computing residuals and separing samples
        res = np.zeros((a, b, c, n))
        samples = []
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    res[i, j, k, :] = (data[i, j, k, :]
                                       - np.mean(data[i, j, k, :]))
                    samples.append(data[i, j, k, :])

        # Check normality assumption
        if scipy.stats.shapiro(res.flatten())[1] < .05:

            # For Box-Cox transformation, it is required positive data.
            if np.amin(res) <= 0 and np.amin(res) < np.amin(data):
                delta = -np.amin(res) + 1
            elif np.amin(data) <= 0 and np.amin(data) <= np.amin(res):
                delta = -np.amin(data) + 1
            else:
                delta = 0

            # In case of non-normality, the Box-Cox transformation
            # is performed.
            _, lmbda = scipy.stats.boxcox(res.flatten() + delta)
            y = scipy.stats.boxcox(data.flatten() + delta, lmbda=lmbda)
            y = y.reshape((a, b, c, n))
            transformation = 'boxcox, lambda=%.3e' % lmbda
            res = np.zeros((a, b, c, n))
            samples = []
            for i in range(a):
                for j in range(b):
                    for k in range(c):
                        res[i, j, k, :] = (y[i, j, k, :]
                                           - np.mean(y[i, j, k, :]))
                        samples.append(y[i, j, k, :])

        else:
            y = np.copy(data)
            transformation = None

        # Save results of assumptions.
        _, shapiro_pvalue = scipy.stats.shapiro(res.flatten())
        _, fligner_pvalue = scipy.stats.fligner(*samples)

        # Plot normality and homoscedascity
        fig = plt.figure(figsize=rst.IMAGE_SIZE_1x2)
        rst.set_subplot_size(fig)
        axes = fig.add_subplot(1, 2, 1)
        normalitiyplot(res.flatten(), axes=axes)
        axes = fig.add_subplot(1, 2, 2)
        homoscedasticityplot(y, axes=axes, title='Homocedascity',
                             ylabel=ylabel, names=group_names)

        # Sums
        ydddd = np.sum(y)
        yiddd = np.sum(y, axis=(1, 2, 3))
        ydjdd = np.sum(y, axis=(0, 2, 3))
        yddkd = np.sum(y, axis=(0, 1, 3))
        yijdd = np.sum(y, axis=(2, 3))
        yidkd = np.sum(y, axis=(1, 3))
        ydjkd = np.sum(y, axis=(0, 3))
        yijkd = np.sum(y, axis=3)

        # Square sums
        SST = np.sum(y**2) - ydddd**2/(a*b*c*n)
        SSA = 1/(b*c*n)*np.sum(yiddd**2) - ydddd**2/(a*b*c*n)
        SSB = 1/(a*c*n)*np.sum(ydjdd**2) - ydddd**2/(a*b*c*n)
        SSC = 1/(a*b*n)*np.sum(yddkd**2) - ydddd**2/(a*b*c*n)
        SSAB = 1/(c*n)*np.sum(yijdd**2) - ydddd**2/(a*b*c*n)-SSA-SSB
        SSAC = 1/(b*n)*np.sum(yidkd**2) - ydddd**2/(a*b*c*n)-SSA-SSC
        SSBC = 1/(a*n)*np.sum(ydjkd**2) - ydddd**2/(a*b*c*n)-SSB-SSC
        SSABC = (1/n*np.sum(yijkd**2)-ydddd**2/(a*b*c*n)-SSA-SSB-SSC-SSAB-SSAC
                 - SSBC)
        SSE = SST-SSABC-SSA-SSB-SSC-SSAB-SSAC-SSBC

        # Means of square sums
        MSA = SSA/(a-1)
        MSB = SSB/(b-1)
        MSC = SSC/(c-1)
        MSAB = SSAB/(a-1)/(b-1)
        MSAC = SSAC/(a-1)/(c-1)
        MSBC = SSBC/(b-1)/(c-1)
        MSABC = SSABC/(a-1)/(b-1)/(c-1)
        MSE = SSE/(a*b*c)/(n-1)

        # Statistics
        F0A, F0B, F0C = MSA/MSE, MSB/MSE, MSC/MSE
        F0AB, F0AC, F0BC, F0ABC = MSAB/MSE, MSAC/MSE, MSBC/MSE, MSABC/MSE

        # Degrees of freedom
        dfA, dfB, dfC = a-1, b-1, c-1
        dfAB, dfAC, dfBC = dfA*dfB, dfA*dfC, dfB*dfC
        dfABC, dfE = dfA*dfB*dfC, a*b*c*(n-1)

        # Critical values
        FCA = scipy.stats.f.ppf(1-alpha, dfA, dfE)
        FCB = scipy.stats.f.ppf(1-alpha, dfB, dfE)
        FCC = scipy.stats.f.ppf(1-alpha, dfC, dfE)
        FCAB = scipy.stats.f.ppf(1-alpha, dfAB, dfE)
        FCAC = scipy.stats.f.ppf(1-alpha, dfAC, dfE)
        FCBC = scipy.stats.f.ppf(1-alpha, dfBC, dfE)
        FCABC = scipy.stats.f.ppf(1-alpha, dfABC, dfE)

        # Hypothesis tests
        null_hypothesis = [F0A < FCA, F0B < FCB, F0C < FCC, F0AB < FCAB,
                           F0AC < FCAC, F0BC < FCBC, F0ABC < FCABC]

        # P-value computation
        pvalue_a = 1-scipy.stats.f.cdf(F0A, dfA, dfE)
        pvalue_b = 1-scipy.stats.f.cdf(F0B, dfB, dfE)
        pvalue_c = 1-scipy.stats.f.cdf(F0C, dfC, dfE)
        pvalue_ab = 1-scipy.stats.f.cdf(F0AB, dfAB, dfE)        
        pvalue_ac = 1-scipy.stats.f.cdf(F0AC, dfAC, dfE)
        pvalue_bc = 1-scipy.stats.f.cdf(F0BC, dfBC, dfE)
        pvalue_abc = 1-scipy.stats.f.cdf(F0ABC, dfABC, dfE)

        return (null_hypothesis, [pvalue_a, pvalue_b, pvalue_c, pvalue_ab,
                                  pvalue_ac, pvalue_bc, pvalue_abc],
                shapiro_pvalue, fligner_pvalue, fig, transformation)

    # Future implementations will address more factors.
    else:
        return None


def ttest_ind_nonequalvar(y1, y2, alpha=0.05):
    r"""Perform T-Test on independent samples with non-equal variances.

    Statistic test which compares two independent sample without
    assuming variance equality [1]_. The *two-sided* test is performed.

    Parameters
    ----------
        y1, y2 : :class:`numpy.ndarray`
            1-d arrays representing the samples.

        alpha : float, default: 0.05
            Significance level.

    Returns
    -------
        null_hypothesis : bool
            Result of the null hypothesis test. If `True`, it means that
            the test has failed to reject the null hypothesis. If
            `False`, it means that the null hypothesis has been rejected
            at 1-`alpha` confidence level.

        t0 : float
            T statistic.

        pvalue: float

        nu : float
            Degrees of freedom.

        confint : tuple of 2-float
            Confidence interval (lower and upper bounds) of the true
            mean difference.

    References
    ----------
    .. [1] Montgomery, Douglas C. Design and analysis of experiments.
       John wiley & sons, 2017.
    """
    # Samples sizes
    n1, n2 = y1.size, y2.size

    # Estimated means
    y1h, y2h = np.mean(y1), np.mean(y2)

    # Estimated variances
    S12, S22 = np.sum((y1-y1h)**2)/(n1-1), np.sum((y2-y2h)**2)/(n2-1)

    # T-statistics
    t0 = (y1h-y2h)/np.sqrt(S12/n1 + S22/n2)

    # Degrees of freedom
    nu = (S12/n1 + S22/n2)**2/((S12/n1)**2/(n1-1) + (S22/n2)**2/(n2-1))

    # Critical values
    ta, tb = scipy.stats.t.ppf(alpha/2, nu), scipy.stats.t.ppf(1-alpha/2, nu)

    # Hypothesis test
    null_hypothesis = ta > t0 or tb < t0

    # Confidence level
    confint = (y1h-y2h-ta*np.sqrt(S12/n1 + S22/n2),
               y1h-y2h+tb*np.sqrt(S12/n1 + S22/n2))

    # P-value computation
    pvalue = 2*scipy.stats.t.cdf(-np.abs(t0), nu)

    return null_hypothesis, t0, pvalue, nu, confint


def dunnetttest(y0, y):
    r"""Perform all-to-one comparisons through Dunnett's test.

    The Dunnett's test is a procedure for comparing a set of :math:`a-1`
    treatments against a single one called the control group [1]_. The
    test is a modification of the usual t-test where, in each
    comparison, the null hypothesis is the equality of means. The
    significance level is fixed in 0.05.
    
    Parameters
    ----------
        y0 : :class:`numpy.ndarray`
            Control sample (1-d array).

        y : list or :class:`numpy.ndarray`
            :math:`a-1` treatments to be compared. The argument must be
            either a list of Numpy arrays or a matrix with shape
            (a-1, n).

    Returns
    -------
        null_hypothesis : list of bool
            List of boolean values indicating the result of the null
            hypothesis. If `True`, it means that the test has failed in
            rejecting the null hypothesis. If `False`, then the null
            hypothesis of equality of means for the respective
            comparison has been reject at a 0.05 significance level.

    References
    ----------
    .. [1] Montgomery, Douglas C. Design and analysis of experiments.
       John wiley & sons, 2017.
    """
    # Avoiding insignificant messages for the analysis
    warnings.filterwarnings('ignore', message='Covariance of the parameters '
                            + 'could not be estimated')

    # Columns of the statistic table (a-1 predefined values)
    Am1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Rows of the statistic table (predefined degrees of freedom)
    F = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                  16, 17, 18, 19, 20, 24, 30, 40, 60, 120, 1e20])

    # Critical values for Dunnett's Test for 0.05 significance level
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

    # Compute the sum of square for both input types
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

    # Mean square error and degrees of freedom
    MSE = SSE/(N-a)
    f = N-a

    # If the number of comparisons is equal to one of the columns of the
    # table of critical values, then we check if the number of degrees
    # of freedom is also available. If isn't, we approximate a value
    # by curve fitting procedure with the closest number of degrees of 
    # freedom.
    if a-1 < 10:
        if np.any(F-f == 0):
            j = np.argwhere(F-f == 0)[0][0]
            d = D[j, a-2]
        else:
            popt, _ = curve_fit(fittedcurve, F[:], D[:, a-1],
                                p0=[4.132, -1.204, 1.971],
                                absolute_sigma=False, maxfev=20000)
            d = fittedcurve(f, popt[0], popt[1], popt[2])

    # If the number of comparisons is greater than the available, then
    # we approximate a value through curve fitting.
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

    # Hypothesis test
    for i in range(a-1):
        if np.abs(yh[i]-y0h) > d*np.sqrt(MSE*(1/n[i]+1/na)):
            null_hypothesis.append(False)
        else:
            null_hypothesis.append(True)

    return null_hypothesis


def fittedcurve(x, a, b, c):
    """Standard curve for linear regression in Dunnett's test.

    This routine computes the function :math:`ax^b+c` which is used for
    curve fitting in Dunnett's test. 
    """
    return a*x**b+c


def data_transformation(data, residuals=False):
    """Try data transformation for normal distribution assumptions.

    Currently, it only implements the Log and Square-Root
    transformations. The normality assumption may be tested on the data
    or in the residuals.

    Parameters
    ----------
        data : either :class:`numpy.ndarray` or list
            If `residuals` is `False`, then the argument must be an 1-d
            array with the sample to be tested. Otherwise, it must be
            a list of arrays.

        residuals : bool
            If `True`, the transformation will be tried over the
            residuals of the observations. Otherwise, the transformation
            will be tried over the own sample.

    Returns
    -------
        If the transformation succeeds, then it returns the transformed
        data and a string containing the type of transformation.
        Otherwise, it returns `None`.
    """
    # Try transformation over the data
    if not residuals:

        # Log Transformation
        if scipy.stats.shapiro(np.log(data))[1] > .05:
            return np.log(data), 'log'

        # Square-root transformation
        elif scipy.stats.shapiro(np.sqrt(data))[1] > .05:
            return np.sqrt(data), 'sqrt'

        # If both transformations fail
        else:
            return None

    # Try transformation over the residuals
    else:

        # Compute the number of observations
        N = 0
        for m in range(len(data)):
            N += data[m].size

        # Compute the Log Transformation
        res = np.zeros(N)
        i = 0
        for m in range(len(data)):
            res[i:i+data[m].size] = np.log(data[m])-np.mean(np.log(data[m]))
            i += data[m].size

        # Try Log Transformation
        if scipy.stats.shapiro(res)[1] > .05:
            for m in range(len(data)):
                data[m] = np.log(data[m])
            return data, 'log'

        # Compute Square-Root Transformation
        res = np.zeros(N)
        i = 0
        for m in range(len(data)):
            res[i:i+data[m].size] = np.sqrt(data[m])-np.mean(np.sqrt(data[m]))
            i += data[m].size

        # Try Square-Root Transformation
        if scipy.stats.shapiro(res)[1] > .05:
            for m in range(len(data)):
                data[m] = np.sqrt(data[m])
            return data, 'sqrt'

        # If both transformations fail
        else:
            return None


def normalitiyplot(data, axes=None, title=None):
    """Graphic investigation of normality assumption.

    This routine plots a figure comparing a sample to a standard normal
    distribution for the purpose of investigating the assumption of
    normality. This routine does not show any plot. It only draws the
    graphic.

    Parameters
    ----------
        data : :class:`numpy.ndarray`
            An 1-d array representing the sample.

        axes : :class:`matplotlib.Axes.axes`, default: None
            A specified axes for plotting the graphics. If none is
            provided, then one will be created and returned.

        title : str, default: None
            A possible title to the plot.

    Returns
    -------
        fig : :class:`matplotlib.figure.Figure`

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> y1 = np.random.normal(size=30)
    >>> y2 = np.random.normal(size=60)
    >>> fig = plt.figure()
    >>> axes1 = fig.add_subplot(1, 2, 1)
    >>> normalityplot(y1, axes=axes1, title='Sample 1')
    >>> axes2 = fig.add_subplot(1, 2, 1)
    >>> normalityplot(y2, axes=axes2, title='Sample 2')
    >>> plt.show()
    """
    # If no axes is provided, a figure is created.
    if axes is None:
        fig = plt.figure()
        axes = rst.get_single_figure_axes(fig)
    else:
        fig = None

    # QQ Plot
    pg.qqplot(data, dist='norm', ax=axes)

    if title is not None:
        axes.set_title(title)
    plt.grid()

    return fig


def homoscedasticityplot(data, axes=None, title=None, ylabel=None, names=None):
    """Graphic investigation of homoscedasticity assumption.

    This routine plots a figure comparing variance of samples for the
    purpose of investigating the assumption of homoscedasticity
    (samples with equal variance). Each samples is positioned in the
    x-axis in the correspondent value of its own mean. This routine does
    not show any plot. It only draws the graphic.

    Parameters
    ----------
        data : either :class:`numpy.ndarray` or list
            A 2-d array with the samples in which each row is a single
            sample or a list of 1-d arrays.

        axes : :class:`matplotlib.Axes.axes`, default: None
            A specified axes for plotting the graphics. If none is
            provided, then one will be created and returned.

        title : str, default: None
            A possible title to the plot.

        ylabel : str, default: None
            The label of the y-axis which represent the unit of the
            data.

        names : list of str, default: None
            A list with the name of the samples for legend purpose.

    Returns
    -------
        fig : :class:`matplotlib.figure.Figure`

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> y1 = np.random.normal(loc=2., size=30)
    >>> y2 = np.random.normal(loc=4., size=60)
    >>> homoscedasticityplot([y1, y2], title='Samples',
                             names=['Sample 1', 'Sample 2'])
    >>> plt.show()
    """
    if axes is None:
        fig = plt.figure()
        axes = rst.get_single_figure_axes(fig)
    else:
        fig = None

    if type(data) is list:
        for i in range(len(data)):
            if names is None:
                axes.plot(np.mean(data[i])*np.ones(data[i].size),
                          data[i]-np.mean(data[i]), 'o')
            else:
                axes.plot(np.mean(data[i])*np.ones(data[i].size),
                          data[i]-np.mean(data[i]), 'o', label=names[i])

    else:
        for i in range(data.shape[0]):
            if names is None:
                axes.plot(np.mean(data[i, :])*np.ones(data.shape[1]),
                          data[i, :]-np.mean(data[i, :]), 'o')
            else:
                axes.plot(np.mean(data[i, :])*np.ones(data.shape[1]),
                          data[i, :]-np.mean(data[i, :]), 'o', label=names[i])

    plt.grid()
    axes.set_xlabel('Means')
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if title is not None:
        axes.set_title(title)
    if names is not None:
        plt.legend()

    if title is not None:
        axes.set_title(title)
    plt.grid()

    return axes


def confintplot(data, axes=None, xlabel=None, ylabel=None, title=None):
    """Plot the confidence interval of means.

    This routine plots a figure comparing the confidence interval of
    means among samples. The confidence intervals are computed at a 
    0.95 confidence level. This routine does not show any plot. It only
    draws the graphic.

    Parameters
    ----------
        data : either :class:`numpy.ndarray` or list
            A 2-d array with the samples in which each *column* is a
            single sample or a list of 1-d arrays.

        axes : :class:`matplotlib.Axes.axes`, default: None
            A specified axes for plotting the graphics. If none is
            provided, then one will be created and returned.

        title : str, default: None
            A possible title to the plot.

        xlabel : str, default: None
            The label of the x-axis which represent the unit of the
            data.

        ylabel : list of str, default: None
            A list with the name of the samples.

    Returns
    -------
        fig : :class:`matplotlib.figure.Figure`

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> y1 = np.random.normal(loc=2., size=30)
    >>> y2 = np.random.normal(loc=4., size=60)
    >>> y3 = np.random.normal(loc=6., size=10)
    >>> confintplot([y1, y2, y3], title='Samples',
                    ylabel=['Sample 1', 'Sample 2', 'Sample 3'])
    >>> plt.show()
    """
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
    """Improved boxplot routine.

    This routine does not show any plot. It only draws the graphic.

    Parameters
    ----------
        data : list of :class:`numpy.ndarray`
            A list of 1-d arrays meaning the samples.

        axes : :class:`matplotlib.Axes.axes`, default: None
            A specified axes for plotting the graphics. If none is
            provided, then one will be created and returned.

        meanline : bool, default: False
            Draws a line through linear regression of the means among
            the samples.

        labels : list of str, default: None
            Names of the samples.

        xlabel : str, default: None

        ylabel : list of str, default: None

        color : str, default: 'b'
            Color of boxes. Check some `here <https://matplotlib.org/
            3.1.1/gallery/color/named_colors.html>`_

        legend : str, default: None
            Label for meanline.

        title : str, default: None
            A possible title to the plot.

    Returns
    -------
        fig : :class:`matplotlib.figure.Figure`

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> y1 = np.random.normal(loc=2., size=30)
    >>> y2 = np.random.normal(loc=4., size=60)
    >>> y3 = np.random.normal(loc=6., size=10)
    >>> boxplot([y1, y2, y3], title='Samples',
                labels=['Sample 1', 'Sample 2', 'Sample 3'],
                xlabel='Samples', ylabel='Unit', color='tab:blue',
                meanline=True, legend='Progression')
    >>> plt.show()
    """
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
    """Improved violinplot routine.

    *Obs*: if no axes is provided, then a figure will be created and
    showed or saved.

    Parameters
    ----------
        data : list of :class:`numpy.ndarray`
            A list of 1-d arrays meaning the samples.

        axes : :class:`matplotlib.Axes.axes`, default: None
            A specified axes for plotting the graphics. If none is
            provided, then one will be created and showed or saved.

        labels : list of str, default: None
            Names of the samples.

        xlabel : str, default: None

        ylabel : list of str, default: None

        color : str, default: 'b'
            Color of boxes. Check some `here <https://matplotlib.org/3.1.1/gallery/color/named_colors.html>`_

        title : str, default: None
            A possible title to the plot.

        show : bool
            If `True`, then the figure is shown. Otherwise, the figure
            is saved.

        file_name : str
            File name when saving the figure.

        file_path : str
            Path to the saved figure.

        file_format : {'eps', 'png', 'pdf', 'svg'}
            Format of the saved figure.

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> y1 = np.random.normal(loc=2., size=30)
    >>> y2 = np.random.normal(loc=4., size=60)
    >>> y3 = np.random.normal(loc=6., size=10)
    >>> violinplot([y1, y2, y3], title='Samples',
                   labels=['Sample 1', 'Sample 2', 'Sample 3'],
                   xlabel='Samples', ylabel='Unit', color='tab:blue',
                   show=True)
    """
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
    """Quick function for returning the LaTeX label of a measure.

    Parameters
    ----------
        measure : {'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe',
                   'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe',
                   'zeta_tfmpad', 'zeta_tfppad', 'zeta_be',
                   'execution_time'}

    Returns
    -------
        str : the label of the measure
    """
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
    elif measure == 'execution_time':
        return rst.LABEL_EXECUTION_TIME
    else:
        raise error.WrongValueInput('get_label', 'measure', "'zeta_rn'/"
                                    + "'zeta_rpad'/'zeta_epad'/'zeta_ebe'/"
                                    + "'zeta_eoe'/'zeta_sad'/'zeta_sbe'/"
                                    + "'zeta_soe'/'zeta_be'/'zeta_tfmpad'/"
                                    + "'zeta_tfppad'/'execution_time'",
                                    measure)


def get_title(measure):
    """Quick function for returning the formal name of a measure.

    Parameters
    ----------
        measure : {'zeta_rn', 'zeta_rpad', 'zeta_epad', 'zeta_ebe',
                   'zeta_eoe', 'zeta_sad', 'zeta_sbe', 'zeta_soe',
                   'zeta_tfmpad', 'zeta_tfppad', 'zeta_be',
                   'execution_time'}

    Returns
    -------
        str : the name of the measure
    """
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
    elif measure == 'execution_time':
        return 'Execution Time'
    else:
        raise error.WrongValueInput('get_label', 'measure', "'zeta_rn'/"
                                    + "'zeta_rpad'/'zeta_epad'/'zeta_ebe'/"
                                    + "'zeta_eoe'/'zeta_sad'/'zeta_sbe'/"
                                    + "'zeta_soe'/'zeta_be'/'zeta_tfmpad'/"
                                    + "'zeta_tfppad'/'execution_time'",
                                    measure)


def run_methods(methods, scenario, parallelization=False,
                print_info=False, screen_object=sys.stdout):
    """Run methods for a single scenario.

    Parameters
    ----------
        methods : list of :class:`solver.Solver`
            Methods objects.

        scenario : :class:`inputdata.InputData`

        parallelization : bool, default: False
            If `True`, the methods will run in parallel.

        print_info : bool, default: False
            If 'True', then the solver will be able to print
            information.

        screen_object : :class:`_io.TextIOWrapper`, default: sys.stdout
            Output object to print solver information.

    Returns
    -------
        results : list of 'results.Results'
    """
    # Parallel Execution
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
        # Run single method
        else:
            results.append(methods[m].solve(scenario, print_info=print_info,
                                            print_file=screen_object))
    return results


def run_scenarios(method, scenarios, parallelization=False):
    """Run multiple inputs for a single method.

    Parameters
    ----------
        method : :class:`solver.Solver`
            Method object.

        scenario : list of :class:`inputdata.InputData`
            Inputs objects.

        parallelization : bool
            If `True`, the inputs will run in parallel.

    Returns
    -------
        results : list of 'results.Results'
    """
    results = []

    # Run in parallel
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
        # Run sequentially
        else:
            results.append(method.solve(scenarios[i], print_info=False))
    return results


def create_scenario(name, configuration, resolution, map_pattern,
                    maximum_contrast, maximum_contrast_density, noise=None,
                    maximum_object_size=None, compute_residual_error=None,
                    compute_map_error=None, compute_totalfield_error=None):
    """Create a single input case.

    Parameters
    ----------
        name : str
            The name of the case.

        configuration : :class:`configuration.Configuration`

        resolution : 2-tuple of int
            Y-X resolution (number of pixels) of the scenario image.

        map_pattern : {'random_polygons', 'regular_polygons', 'surfaces'}
            Pattern of dielectric information on the image.

        maximum_contrast : complex
            Upper bound of contrast value in the image.

        maximum_contrast_density : float
            Maximum value for the mean contrast per pixel normalized by
            the maximum contrast value. For the case of homogeneous
            objects, it controls the quantity of objects in the image.
            When dealing with surfaces, this information is considered
            for gaussian random functions.

        noise : float, default: None
            Noise level that will be added into the scattered field.

        maximum_object_size : float, default: .4*min([Lx, Ly])/2
            Maximum radius size of homogeneous objects in the image.

        compute_residual_error : bool, default: None
            A flag to save residual error when running the input.

        compute_map_error : bool, default: None
            A flag to save map error when running the input.

        compute_totalfield_error : bool, default: None
            A flag to save total field error when running the input.

    Returns
    -------
        :class:`inputdata.InputData`

    """
    # Basic parameters of the model
    Lx = configuration.Lx/configuration.lambda_b
    Ly = configuration.Ly/configuration.lambda_b
    epsilon_rb = configuration.epsilon_rb
    sigma_b = configuration.sigma_b
    omega = 2*pi*configuration.f
    homogeneous_objects = False

    # Determining bounds of the conductivity values
    if configuration.perfect_dielectric:
        min_sigma = max_sigma = sigma_b
    else:
        min_sigma = 0.
        max_sigma = cfg.get_conductivity(maximum_contrast, omega, epsilon_rb,
                                         sigma_b)

    # Determining bounds of the relative permittivity values
    if configuration.good_conductor:
        min_epsilon_r = max_epsilon_r = epsilon_rb
    else:
        min_epsilon_r = 1.
        max_epsilon_r = cfg.get_relative_permittivity(maximum_contrast,
                                                      epsilon_rb)

    # Polygons with random number of edges
    if map_pattern == RANDOM_POLYGONS_PATTERN:

        # In this case, there are only homogeneous objects
        homogeneous_objects = True

        # Defining the maximum object size if it is not defined in the
        # argument.
        if maximum_object_size is None:
            maximum_object_size = .4*min([Lx, Ly])/2

        # Parameters of the image
        dx, dy = Lx/resolution[0], Ly/resolution[1]
        minimum_object_size = 8*max([dx, dy])
        xmin, xmax = cfg.get_bounds(Lx)
        ymin, ymax = cfg.get_bounds(Ly)

        # Initial map
        epsilon_r = epsilon_rb*np.ones(resolution)
        sigma = sigma_b*np.ones(resolution)
        chi = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b,
                                   omega)

        # Add objects until the density is satisfied
        while (contrast_density(chi)/np.abs(maximum_contrast) 
               <= .9*maximum_contrast_density):

            # Determine the maximum radius of the edges of the polygon (random)
            radius = minimum_object_size + (maximum_object_size
                                            - minimum_object_size)*rnd.rand()

            # Determine randomly the relative permittivity of the object
            epsilon_ro = min_epsilon_r + (max_epsilon_r
                                          - min_epsilon_r)*rnd.rand()

            # Determine randomly the conductivity of the object
            sigma_o = min_sigma + (max_sigma-min_sigma)*rnd.rand()

            # Determine randomly the position of the object
            center = [xmin+radius + (xmax-radius-(xmin+radius))*rnd.rand(),
                      ymin+radius + (ymax-radius-(ymin+radius))*rnd.rand()]

            # Draw the polygon over the current image (random choice of the
            # number of edges, max: 15)
            epsilon_r, sigma = draw_random(
                int(np.ceil(15*rnd.rand())), radius, axis_length_x=Lx,
                axis_length_y=Ly, background_relative_permittivity=epsilon_rb,
                background_conductivity=sigma_b,
                object_relative_permittivity=epsilon_ro,
                object_conductivity=sigma_o, center=center,
                relative_permittivity_map=epsilon_r, conductivity_map=sigma
            )

            # Compute contrast function
            chi = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b,
                                       omega)

    # Traditional geometric shapes pattern
    elif map_pattern == REGULAR_POLYGONS_PATTERN:

        # In this case, there are only homogeneous objects
        homogeneous_objects = True

        # Defining the maximum object size if it is not defined in the
        # argument.
        if maximum_object_size is None:
            maximum_object_size = .4*min([Lx, Ly])/2

        # Parameters of the image
        dx, dy = Lx/resolution[0], Ly/resolution[1]
        minimum_object_size = 8*max([dx, dy])
        xmin, xmax = cfg.get_bounds(Lx)
        ymin, ymax = cfg.get_bounds(Ly)

        # Initial map
        epsilon_r = epsilon_rb*np.ones(resolution)
        sigma = sigma_b*np.ones(resolution)
        chi = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b,
                                   omega)

        # Add objects until the density is satisfied
        while (contrast_density(chi)/np.abs(maximum_contrast)
               <= .9*maximum_contrast_density):

            # Determine randomly the maximum radius/length of the shape
            radius = minimum_object_size + (maximum_object_size
                                            - minimum_object_size)*rnd.rand()

            # Determine randomly the relative permittivity of the object
            epsilon_ro = min_epsilon_r + (max_epsilon_r
                                          - min_epsilon_r)*rnd.rand()

            # Determine randomly the conductivity of the object
            sigma_o = min_sigma + (max_sigma-min_sigma)*rnd.rand()

            # Determine randomly the position of the object
            center = [xmin+radius + (xmax-radius-(xmin+radius))*rnd.rand(),
                      ymin+radius + (ymax-radius-(ymin+radius))*rnd.rand()]

            # Choose randomly one of the 14 geometric shapes available
            shape = rnd.randint(14)

            # Square
            if shape == 0:
                epsilon_r, sigma = draw_square(
                    2*radius, axis_length_x=Lx, axis_length_y=Ly,
                    background_relative_permittivity=epsilon_rb,
                    rotate=rnd.rand()*180,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # Triangle
            elif shape == 1:
                epsilon_r, sigma = draw_triangle(
                    2*radius, axis_length_x=Lx, axis_length_y=Ly,
                    background_relative_permittivity=epsilon_rb,
                    rotate=rnd.rand()*180,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # Circle
            elif shape == 2:
                epsilon_r, sigma = draw_circle(
                    radius, axis_length_x=Lx, axis_length_y=Ly,
                    background_relative_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # Ring
            elif shape == 3:
                epsilon_r, sigma = draw_ring(
                    rnd.rand()*radius, radius, axis_length_x=Lx,
                    axis_length_y=Ly,
                    background_relative_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # Ellipse
            elif shape == 4:
                epsilon_r, sigma = draw_ellipse(
                    rnd.rand()*radius, radius, axis_length_x=Lx,
                    axis_length_y=Ly, rotate=rnd.rand()*180,
                    background_relative_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # 4-point star
            elif shape == 5:
                epsilon_r, sigma = draw_4star(
                    radius, axis_length_x=Lx, axis_length_y=Ly,
                    background_relative_permittivity=epsilon_rb,
                    rotate=rnd.rand()*180,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # 5-point star
            elif shape == 6:
                epsilon_r, sigma = draw_5star(
                    radius, axis_length_x=Lx, axis_length_y=Ly,
                    background_relative_permittivity=epsilon_rb,
                    rotate=rnd.rand()*180,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # 6-point star
            elif shape == 7:
                epsilon_r, sigma = draw_6star(
                    radius, axis_length_x=Lx, axis_length_y=Ly,
                    background_relative_permittivity=epsilon_rb,
                    rotate=rnd.rand()*180,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # Rhombus
            elif shape == 8:
                epsilon_r, sigma = draw_rhombus(
                    2*radius, axis_length_x=Lx, axis_length_y=Ly,
                    background_relative_permittivity=epsilon_rb,
                    rotate=rnd.rand()*180,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # Trapezoid
            elif shape == 9:
                epsilon_r, sigma = draw_trapezoid(
                    rnd.rand()*radius, radius, radius, axis_length_x=Lx,
                    axis_length_y=Ly, rotate=rnd.rand()*360,
                    background_relative_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # Parallelogram
            elif shape == 10:
                epsilon_r, sigma = draw_parallelogram(
                    radius, (.5+.5*rnd.rand())*radius, 30 + 30*rnd.rand(),
                    axis_length_x=Lx, axis_length_y=Ly, rotate=rnd.rand()*360,
                    background_relative_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # Regular polygon (pentagon, hexagon, ...)
            elif shape == 11:
                epsilon_r, sigma = draw_polygon(
                    5+rnd.randint(6), radius,
                    axis_length_x=Lx, axis_length_y=Ly, rotate=rnd.rand()*180,
                    background_relative_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # Cross
            elif shape == 12:
                epsilon_r, sigma = draw_cross(
                    radius, (0.5 + 0.5*rnd.rand())*radius, .1*radius,
                    axis_length_x=Lx, axis_length_y=Ly, rotate=rnd.rand()*180,
                    background_relative_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # Line
            elif shape == 13:
                epsilon_r, sigma = draw_line(
                    radius, .1*radius, axis_length_x=Lx, axis_length_y=Ly,
                    background_relative_permittivity=epsilon_rb,
                    background_conductivity=sigma_b, rotate=rnd.rand()*180,
                    object_relative_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    relative_permittivity_map=epsilon_r, conductivity_map=sigma
                )

            # Compute contrast function
            chi = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b,
                                       omega)

    # Random surfaces (waves of gaussian)
    elif map_pattern == SURFACES_PATTERN:

        # Randomly decide between waves or gaussian functions
        if rnd.rand() < .5:
            epsilon_r, sigma = draw_random_waves(
                int(np.ceil(15*rnd.rand())), 10, resolution=resolution,
                rel_permittivity_amplitude=max_epsilon_r-epsilon_rb,
                conductivity_amplitude=max_sigma-sigma_b, axis_length_x=Lx,
                axis_length_y=Ly, background_relative_permittivity=epsilon_rb,
                conductivity_map=sigma_b
            )

        else:

            # When setting gaussian functions, the maximum object size
            # is used as a measure of the variance
            if maximum_object_size is None:
                maximum_object_size = .4*min([Lx, Ly])/2

            # Image parameters
            dx, dy = Lx/resolution[0], Ly/resolution[1]
            minimum_object_size = 8*max([dx, dy])
            epsilon_r = epsilon_rb*np.ones(resolution)
            sigma = sigma_b*np.ones(resolution)
            chi = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b,
                                       omega)

            # Add gaussian functions until the density criterion is
            # satisfied
            while contrast_density(chi) <= .9*maximum_contrast_density:
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

    # Build input object
    scenario = ipt.InputData(name=name,
                             configuration_filename=configuration.name,
                             resolution=resolution,
                             homogeneous_objects=homogeneous_objects,
                             noise=noise)

    # Set flags
    if compute_residual_error is not None:
        scenario.compute_residual_error = compute_residual_error
    if compute_map_error is not None:
        scenario.compute_map_error = compute_map_error
    if compute_totalfield_error is not None:
        scenario.compute_totalfield_error = compute_totalfield_error

    # Set maps
    if not configuration.good_conductor:
        scenario.epsilon_r = epsilon_r
    if not configuration.perfect_dielectric:
        scenario.sigma = sigma

    return scenario


def contrast_density(contrast_map):
    """Compute the contrast density of a map.

    The contrast density is defined as the mean of the absolute value
    per pixel.

    Parameters
    ----------
        contrast_map : :class:`numpy.ndarray`
            2-d array.
    """
    return np.mean(np.abs(contrast_map))


def compute_resolution(wavelength, length_y, length_x,
                       proportion_cell_wavelength):
    """Determine a reasonable resolution.

    Compute a reasonable resolution for an image given the wavelength,
    the size of the image and proportion cell (pixel) per wavelength.

    Parameters
    ----------
        wavelength : float

        length_x, length_y : float

        proportion_cell_wavelength : int

    Returns
    -------
        NY, NX : int
    """
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
