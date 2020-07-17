"""Give a title to the module.

Explain the module.
"""

# Standard libraries
import numpy as np
from numpy import random as rnd

# Developed libraries
import library_v2.error as error
import library_v2.configuration as cfg
import library_v2.inputdata as ipt
import library_v2.solver as slv
import library_v2.results as rst

# Constants
STANDARD_SYNTHETIZATION_RESOLUTION = 30
STANDARD_RECOVER_RESOLUTION = 20


class Experiment:
    """Give a title to the class."""

    name = str
    maximum_contrast = complex
    maximum_object_size = float
    maximum_average_contrast = float
    sample_size = int
    synthetization_resolution = (int, int)
    recover_resolution = (int, int)
    configurations = list
    scenarios = list
    methods = list
    results = list

    def __init__(self, name, maximum_contrast, maximum_object_size,
                 maximum_average_contrast, sample_size=None,
                 synthetization_resolution=None, recover_resolution=None,
                 configurations=None, scenarios=None, methods=None):
        """Summarize the method."""
        self.name = name
        self.maximum_contrast = maximum_contrast
        self.maximum_average_contrast = maximum_object_size
        self.sample_size = sample_size
        self.synthetization_resolution = synthetization_resolution
        self.recover_resolution = recover_resolution
        self.configurations = configurations
        self.scenarios = scenarios
        self.methods = methods
        self.results = None

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

        if self.synthetization_resolution is None:
            if isinstance(self.configurations, list):
                self.synthetization_resolution = list()
                for i in range(len(self.configurations)):
                    resolution = compute_resolution(
                        self.configurations[i].lambda_b,
                        self.configurations[i].Ly, self.configurations[i].Lx,
                        STANDARD_SYNTHETIZATION_RESOLUTION
                    )
                    self.synthetization_resolution.append(resolution)
            else:
                self.synthetization_resolution = compute_resolution(
                    self.configurations[i].lambda_b,
                    self.configurations[i].Ly, self.configurations[i].Lx,
                    STANDARD_SYNTHETIZATION_RESOLUTION
                )

        if self.recover_resolution is None:
            if self.scenarios is not None:
                if isinstance(self.scenarios, list):
                    if self.scenarios[0].resolution is not None:
                        self.recover_resolution = self.scenarios[0].resolution
                else:
                    if self.scenarios.resolution is not None:
                        self.recover_resolution = self.scenarios.resolution

            if self.recover_resolution is None:         
                if isinstance(self.configurations, list):
                    self.recover_resolution = list()
                    for i in range(len(self.configurations)):
                        resolution = compute_resolution(
                            self.configurations[i].lambda_b,
                            self.configurations[i].Ly,
                            self.configurations[i].Lx,
                            STANDARD_RECOVER_RESOLUTION
                        )
                        self.recover_resolution.append(resolution)
                else:
                    self.recover_resolution = compute_resolution(
                        self.configurations[i].lambda_b,
                        self.configurations[i].Ly, self.configurations[i].Lx,
                        STANDARD_RECOVER_RESOLUTION
                    )


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
    epsilon_r[np.logical_and(np.logical_and(yp >= -L/2, yp <= L/2),
                             np.logical_and(xp >= -L/2,
                                            xp <= L/2))] = epsilon_ro
    sigma[np.logical_and(np.logical_and(yp >= -L/2, yp <= L/2),
                         np.logical_and(xp >= -L/2, xp <= L/2))] = sigma_o

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
    epsilon_r[np.logical_and(np.logical_and(yp >= -L/2, yp <= 2*xp + L/2),
                             yp <= -2*xp + L/2)] = epsilon_ro
    sigma[np.logical_and(np.logical_and(yp >= -L/2, yp <= 2*xp - L/2),
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
    epsilon_r[np.logical_and(np.logical_and(yp >= -L/4, yp <= 3/2*xp + L/2),
                             yp <= -3/2*xp + L/2)] = epsilon_ro
    sigma[np.logical_and(np.logical_and(yp >= -L/4, yp <= 3/2*xp + L/2),
                         yp <= -3/2*xp + L/2)] = sigma_o

    epsilon_r[np.logical_and(np.logical_and(yp <= L/4, yp >= 3/2*xp - L/2),
                             yp >= -3/2*xp - L/2)] = epsilon_ro
    sigma[np.logical_and(np.logical_and(y <= L/4, yp >= 3/2*xp - L/2),
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
    epsilon_r[np.logical_and(x**2 + y**2 <= rb**2,
                             x**2 + y**2 >= ra**2)] = epsilon_ro
    sigma[np.logical_and(x**2 + y**2 <= rb**2,
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
        np.logical_and(xp >= -width/2,
                       np.logical_and(xp <= width/2,
                                      np.logical_and(yp >= -thickness/2,
                                                     yp <= thickness/2)))
    )
    vertical_bar = (
        np.logical_and(y >= -height/2,
                       np.logical_and(y <= height/2,
                                      np.logical_and(x >= -thickness/2,
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
    line = (np.logical_and(xp >= -length/2,
                           np.logical_and(xp <= length/2,
                                          np.logical_and(yp >= -thickness/2,
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
    dphi = 2*np.pi/number_sides
    phi = np.arange(0, number_sides*dphi, dphi)
    xp = radius*np.cos(phi)
    yp = radius*np.sin(phi)
    polygon = np.ones(x.shape, dtype=bool)
    for i in range(number_sides):
        a = -(yp[i]-yp[i-1])
        b = xp[i]-xp[i-1]
        c = (xp[i]-xp[i-1])*yp[i-1] - (yp[i]-yp[i-1])*xp[i-1]
        polygon = np.logical_and(polygon, a*xp + b*yp >= c)
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
    # phi = np.sort(2*np.pi*rnd.rand(number_sides))
    phi = rnd.normal(loc=np.linspace(0, 2*np.pi, number_sides, endpoint=False),
                     scale=0.5)
    phi[phi >= 2*np.pi] = phi[phi >= 2*np.pi] - np.floor(phi[phi >= 2*np.pi]
                                                   / (2*np.pi))*2*np.pi
    phi[phi < 0] = -((np.floor(phi[phi < 0]/(2*np.pi)))*2*np.pi - phi[phi < 0])
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
    rhombus = np.logical_and(-a*xp - a*yp >= -a**2,
                             np.logical_and(a*xp - a*yp >= -a**2,
                                            np.logical_and(a*xp+a*yp >= -a**2,
                                                           -a*xp+a*yp >= -a**2)
                                            ))
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
        raise error.MissingInputError('draw_trapezoid', 'resolution or relative'
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
    trapezoid = np.logical_and(a1*xp + b1*yp >= c1,
                               np.logical_and(a2*xp + b2*yp >= c2,
                                              np.logical_and(yp <= height/2,
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
    l, h, a = length, height, height/2/np.tan(np.deg2rad(90-angle)) 
    parallelogram = np.logical_and(-h*xp + 2*a*yp >= 2*a*(l/2-a)-h*(l/2-a),
                                   np.logical_and(h*xp-2*a*yp >= h*(a-l/2)-a*h,
                                                  np.logical_and(yp <= height/2,
                                                                 yp >= 
                                                                 -height/2)))

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
    rhombus1 = np.logical_and(-a*xp - b*yp >= -a*b,
                              np.logical_and(a*xp - b*yp >= -a*b,
                                             np.logical_and(a*xp+b*yp >= -a*b,
                                                            -a*xp+b*yp >= -a*b)
                                             ))
    rhombus2 = np.logical_and(-b*xp - a*yp >= -a*b,
                              np.logical_and(b*xp - a*yp >= -a*b,
                                             np.logical_and(b*xp+a*yp >= -a*b,
                                                            -b*xp+a*yp >= -a*b)
                                             ))
    epsilon_r[np.logical_or(rhombus1, rhombus2)] = epsilon_ro
    sigma[np.logical_or(rhombus1, rhombus2)] = sigma_o

    return epsilon_r, sigma
