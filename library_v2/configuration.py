import numpy as np
import scipy.constants as ct

# Developed libraries
import error


class Configuration:
    """Problem configuration class.
    
    Attributes:
        NM -- number of measurements
        NS -- number of sources
        Ro -- radius of observation (S-domain) [m]
        Lx -- size of image domain (D-domain) in x-axis [m]
        epsilon_rb -- background relative permittivity
        sigma_b -- background conductivity [S/m]
        frequency -- linear frequency of operation [Hz]
        lambdab -- background wavelength [m]
        kb -- background wavenumber [1/m]
        perfect_dielectric -- flag for assuming perfect dielectric 
            objects
        good_conductor -- flag for assuming good conductor objects
    """
    
    NM, NS = int(), int()
    Ro, Lx, Ly = float(), float(), float()
    epsilon_rb, sigma_b = float(), float()
    frequency, lambdab, kb = float(), float(), float()
    perfect_dielectric, good_conductor = bool(), bool()
    
    def __init__(self,number_measurements=10, number_sources=10,
                 observation_radius=None, frequency=None, wavelength=None,
                 background_permittivity=1., background_conductivity=.0,
                 image_size=[1.,1.], wavelength_unit=True,
                 perfect_dielectric=True, good_conductor=False):
        """Build the configuration object. 
        
        Obs.: either frequency or wavelength must be given.
        
        Keyword arguments:
            number_measurements -- receivers in S-domain (default 10)
            number_sources -- sources in S-domain (default 10)
            observation_radius -- radius for circular array of sources
                and receivers at S-domain [m]
                (default 1.1*sqrt(2)*max([Lx,Ly]))
            frequency -- linear frequency of operation [Hz]
            wavelength -- background wavelength [m]
            background_permittivity -- Relative permittivity (default 1.0)
            background_conductivity -- [S/m] (default 0.0)
            image_size -- a tuple with the side sizes of image domain
                (D-domain). It may be given in meters or in wavelength
                proportion (default (1.,1.))
            wavelength_unit -- a flag to indicate if image_size is given
                in wavelength or not (default True)
            perfect_dielectric -- a flag to indicate the assumption of 
                only perfect dielectric objects (default True)
            good_conductor -- a flag to indicate the assumption of 
                only good conductors objects (default True)
        """
        
        if wavelength is None and frequency is None:
            raise error.InputError('frequency=None,wavelength=None',
                                   'ERROR:CONFIGURATION:CONFIGURATION:'
                                   + '__INIT__: Either frequency or '
                                   + 'wavelenth must be given!')
        elif wavelength is not None and frequency is not None:
            raise error.InputError('frequency=None,wavelength=None',
                                   'ERROR:CONFIGURATION:CONFIGURATION:'
                                   + '__INIT__: Either frequency or '
                                   + 'wavelenth must be none!')
        
        self.NM = number_measurements
        self.NS = number_sources
        self.Ro = observation_radius
        self.epsilon_rb = background_permittivity
        self.sigma_b = background_conductivity
        self.perfect_dielectric = perfect_dielectric
        self.good_conductor = good_conductor
        
        if frequency is not None:
            self.frequency = frequency
            self.wavelength = compute_wavelength(frequency,self.epsilon_rb)
        else:
            self.lambdab = wavelength
            self.frequency = compute_frequency(self.lambdab,self.epsilon_rb)
            
        self.kb = compute_wavenumber_real(self.frequency,self.epsilon_rb)
        
        if wavelength_unit:
            self.Lx = image_size[0]*self.lambdab
            self.Ly = image_size[1]*self.lambdab
            
def compute_wavelength(frequency, epsilon_r=1., mu_r=1.):
    """Compute wavelength [m]."""
    return 1/np.sqrt(epsilon_r*ct.epsilon_0*mu_r*ct.mu_0)/frequency

def compute_frequency(wavelength, epsilon_r=1., mu_r=1.):
    """Compute frequency [Hz]."""
    return 1/np.sqrt(epsilon_r*ct.epsilon_0*mu_r*ct.mu_0)/wavelength

def compute_wavenumber_real(frequency, epsilon_r=1., mu_r=1.):
    """ Compute real part of wavenumber."""
    return 2*np.pi*np.sqrt(epsilon_r*ct.epsilon_0*mu_r*ct.mu_0)*frequency