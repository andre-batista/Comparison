import numpy as np
import copy as cp
import numpy.linalg as lag
import pickle
import sys
from numba import jit
import scipy.constants as ct
import scipy.interpolate as interp
import matplotlib as mpl
# mpl.use('Agg') # Avoiding error when using ssh protocol
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod 

import model as md
import domain as dm
import bim

class ConjugatedGradient(bim.BIM):
    
    delta = float()
    regularization_method_name = 'CONJUGATED GRADIENT'
    
    def set_regularization_parameter(self, newparameter):
        self.delta = newparameter

          
    def regularization_method(self, es, noconductivity=False, 
                              nopermittivity=False):
        
        K = bim.get_operator_matrix(self.model.Et,self.model.domain.M,
                                self.model.domain.L,self.model.GS,
                                self.model.Et.shape[0])
        y = self.get_y(es)
        x0 = np.zeros(self.model.Et.shape[0],dtype=complex)
        x = cg_method(K,y,x0,self.delta)
        
        if nopermittivity:
            epsilon_r = self.model.epsilon_rb*np.ones((self.model.Nx,
                                                       self.model.Ny))
        else:
            epsilon_r = np.reshape(self.model.epsilon_rb*(np.real(x)+1),
                                   (self.model.Nx,self.model.Ny))
            epsilon_r[epsilon_r<1] = 1
        
        if noconductivity:
            sigma = self.model.sigma_b*np.ones(epsilon_r.shape)
        else:
            sigma = np.reshape(self.model.sigma_b 
                               - np.imag(x)*self.model.omega*ct.epsilon_0,
                               (self.model.Nx,self.model.Ny))
            sigma[sigma<0] = 0
        
        return epsilon_r, sigma
     
    def solve_linear(self, es, et=None, nopermittivity=False, 
                     noconductivity=False,delta=None):

        if delta is not None:
            self.delta = delta

        if et is None:
            self.model.Et = self.model.Ei
        else:
            self.model.Et = et
        
        return self.regularization_method(es,nopermittivity=nopermittivity,
                                          noconductivity=noconductivity)
            
    def _BIM__update_regularization_parameter(self,es,residual):
        pass

    def _BIM__save_results(self, epsilon_r, sigma, residual, file_name, 
                       file_path=''):
        
        data = {
            'relative_permittivity_map':epsilon_r,
            'conductivity_map':sigma,
            'residual_convergence':residual,
            'delta':self.delta,
            'number_iterations':self.N_ITER
        }

        with open(file_path + file_name,'wb') as datafile:
            pickle.dump(data,datafile)
            
    def _BIM__initialize_variables(self):
        
        if self.delta < 0:
            self.delta = 1e-2

@jit(nopython=True)
def cg_method(K,y,x0,delta):
    
    p = -K.conj().T@y
    x = np.copy(x0)
    
    
    while True:
        
        kp = K@p
        tm = np.vdot(K@x-y,kp)/lag.norm(kp)**2
        x_last = np.copy(x)
        x = x - tm*p
        
        if  K.conj().T@(K@x-y) > delta:
            break
        
        gamma = (lag.norm(K.conj().T@(K@x-y))**2
                 /lag.norm(K.conj().T@(K@x_last-y))**2)
        p = K.conj().T@(K@x-y)+gamma*p
        
    return x