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
import copy as cp
import numpy as np
from numpy import random as rnd
from numpy import pi, logical_and
import time as tm
from joblib import Parallel, delayed
import multiprocessing

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
        elif type(value) is None:
            self._noise = [0.]

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

    def __init__(self, name, maximum_contrast, maximum_object_size,
                 maximum_contrast_density, map_pattern, sample_size=None,
                 synthetization_resolution=None, recover_resolution=None,
                 configurations=None, scenarios=None, methods=None,
                 forward_solver=None, noise=None, study_residual=True,
                 study_map=False, study_internfield=False):
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
                        SAVE_INTERN_FIELD=False
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

    def analyse_residual(self):
        if self.results is None:
            raise error.MissingAttributesError('Experiment','results')
        

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
        if all(i == self.maximum_object_size[0] for i in self.maximum_object_size):
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
            (scenarios[i],print_info=False) for i in range(len(scenarios))
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
        while contrast_density(chi) <= .8*maximum_contrast_density:
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
