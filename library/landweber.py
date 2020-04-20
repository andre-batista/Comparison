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

class Landweber(bim.BIM):
    
    a = float()
    M = int()
    regularization_method_name = 'LANDWEBER'
    
    def set_regularization_parameter(self, newparameter, 
                                     parametername='m', Et=None):
        
        if parametername == 'a':
            if Et is None:    
                K = bim.get_operator_matrix(self.model.Ei,
                                            self.model.domain.M,
                                            self.model.domain.L,
                                            self.model.GS,
                                            self.model.Ei.shape[0])
            else:
                K = bim.get_operator_matrix(Et,self.model.domain.M,
                                            self.model.domain.L,
                                            self.model.GS,
                                            Et.shape[0])
            self.a = newparameter/lag.norm(K)**2
            if self.M < 1:
                self.M = 100
            
        else:
            self.M = newparameter
            if self.a <= 0:
                K = bim.get_operator_matrix(self.model.Ei,
                                            self.model.domain.M,
                                            self.model.domain.L,
                                            self.model.GS,
                                            self.model.Ei.shape[0])
                self.a = newparameter/lag.norm(K)**2
            
    def set_a(self,newparameter,Et=None):
            if Et is None:    
                K = bim.get_operator_matrix(self.model.Ei,
                                            self.model.domain.M,
                                            self.model.domain.L,
                                            self.model.GS,
                                            self.model.Ei.shape[0])
            else:
                K = bim.get_operator_matrix(Et,self.model.domain.M,
                                            self.model.domain.L,
                                            self.model.GS,
                                            Et.shape[0])
            self.a = newparameter/lag.norm(K)**2
          
    def regularization_method(self, es, noconductivity=False, 
                              nopermittivity=False):
        
        K = bim.get_operator_matrix(self.model.Et,self.model.domain.M,
                                self.model.domain.L,self.model.GS,
                                self.model.Et.shape[0])
        y = self.get_y(es)
        x0 = np.zeros(self.model.Et.shape[0],dtype=complex)
        x = landweber_iteration(K,y,self.a,x0,self.M)
        
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
                     noconductivity=False,a=None,M=None):

        if a is not None:
            self.set_a(a,Et=et)
        if M is not None:
            self.M = M

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
            'a':self.a,
            'M':self.M,
            'number_iterations':self.N_ITER
        }

        with open(file_path + file_name,'wb') as datafile:
            pickle.dump(data,datafile)
            
    def _BIM__initialize_variables(self):
        
        if self.a <= 0:
            self.a = self.set_a(.5)
        if self.M <= 0:
            self.M = 50
        
        self.execution_info = 'a = %.3e, ' %self.a + 'M = %d' %self.M

@jit(nopython=True)
def landweber_iteration(K,y,a,x0,M):
    x = np.copy(x0)
    d = lag.norm(y-K@x)
    d_last = 2*d
    it = 0
    while it < M and (d_last-d)/d_last > 0.01:
        x = x + a*K.T.conj()@(y-K@x)
        d_last = d
        d = lag.norm(y-K@x)
        it += 1
    return x