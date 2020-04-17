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

class Tikhonov(bim.BIM):
    
    alpha = float()
    alpha_choice_method = 'lavarello'
    regularization_method_name = 'TIKHONOV'
    
    def set_regularization_parameter(self, new_parameter, method=None,
                                     delta=None):
        
        if method is None:
            self.alpha = self.__lavarello_choice()
            self.alpha0 = self.alpha
        elif method == 'kirsch':
            self.alpha_choice_method = method
            self.alpha = self.__kirsch_choice(new_parameter,delta)
        else:
            self.alpha_choice_method = 'fixed'
            self.alpha = new_parameter
          
    def regularization_method(self, es, noconductivity=False, 
                              nopermittivity=False):
        
        K = bim.get_operator_matrix(self.model.Et,self.model.domain.M,
                                self.model.domain.L,self.model.GS,
                                self.model.Et.shape[0])
        y = self.get_y(es)
        x = solve_tikhonov_regularization(K,y,self.alpha)
        
        it = 0
        while self.Ja[it] != 0:
            it+=1
        
        self.Ja[it] = self.compute_tikhonov_functional(K=K,y=y,x=x)
        self.iteration_info = ' - Ja(X) = %.3e' %self.Ja[it]
        
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
     
    def solve_linear(self, es, et=None, alpha=None, nopermittivity=False, 
                     noconductivity=False):

        if alpha is not None:
            self.alpha = alpha
        if et is None:
            self.model.Et = self.model.Ei
        else:
            self.model.Et = et
        
        return self.regularization_method(es,nopermittivity=nopermittivity,
                                          noconductivity=noconductivity)
            
    def _BIM__update_regularization_parameter(self,es,residual):
        
        if self.alpha_choice_method == 'lavarello':
            self.iteration_info = (self.iteration_info 
                                   + ' - alpha = %.3e' %self.alpha)
            self.alpha = self.__lavarello_update(es,residual,self.alpha0)
            
    def __lavarello_choice(self):
        K = bim.get_operator_matrix(self.model.Ei,self.model.domain.M,
                                self.model.domain.L,self.model.GS,
                                self.model.Ei.shape[0])
        _, S, _ = lag.svd(K)
        return S[0]**2
    
    def __lavarello_update(self,es,residual,alpha0):
        RRE = residual/lag.norm(es.reshape(-1))
        if 0.5 < RRE:
            return alpha0/2
        elif .25 < RRE and RRE <= .5:
            return alpha0/20
        elif RRE <= .25:
            return alpha0/200

    def __kirsch_choice(self,es,delta):
        K = bim.get_operator_matrix(self.model.Ei,self.model.domain.M,
                                self.model.domain.L,self.model.GS,
                                self.model.Ei.shape[0])
        
        return delta*lag.norm(K)**2/(lag.norm(es.reshape(-1))-delta)

    def _BIM__save_results(self, epsilon_r, sigma, residual, file_name, 
                       file_path=''):
        
        data = {
            'relative_permittivity_map':epsilon_r,
            'conductivity_map':sigma,
            'tikhonov_functional_convergence':self.Ja,
            'residual_convergence':residual,
            'alpha':self.alpha,
            'number_iterations':self.N_ITER
        }

        with open(file_path + file_name,'wb') as datafile:
            pickle.dump(data,datafile)
            
    def _BIM__initialize_variables(self):
        
        if self.alpha <= 0:
            self.set_regularization_parameter(None)
        
        self.Ja = np.zeros(self.N_ITER)
        self.execution_info = ('Alpha = %.3e' %self.alpha + ' (' 
                               + self.alpha_choice_method + ')')

    def compute_tikhonov_functional(self,K=None,x=None,y=None,et=None,
                                    es=None,epsilon_r=None,sigma=None):
        if y is None and es is None:
            print('COMPUTE_TIKHONOV_FUNCTIONAL ERROR: Either K-x-y or et-es '
                  + 'must be given!')
            sys.exit()
            
        elif y is None:
            if et is None:
                print('COMPUTE_TIKHONOV_FUNCTIONAL ERROR: et is missing!')
                sys.exit()
            if epsilon_r is None:
                print('COMPUTE_TIKHONOV_FUNCTIONAL ERROR: epsilon_r is missing!')
                sys.exit()
            if sigma is None:
                print('COMPUTE_TIKHONOV_FUNCTIONAL ERROR: sigma is missing!')
                sys.exit()
            
            K = bim.get_operator_matrix(et,self.model.domain.M,
                                        self.model.domain.L,self.model.GS,
                                        et.shape[0])
            x = self.compute_contrast_function(epsilon_r,sigma).reshape(-1)
            y = self.get_y(es)
            
        elif es is None:
            if x is None:
                print('COMPUTE_TIKHONOV_FUNCTIONAL ERROR: x is missing!')
                sys.exit()
            if y is None:
                print('COMPUTE_TIKHONOV_FUNCTIONAL ERROR: y is missing!')
                sys.exit()            
        
        return lag.norm(K@x-y)**2 + self.alpha*lag.norm(x)**2

@jit(nopython=True)
def solve_tikhonov_regularization(K,y,alpha):
    x = lag.solve(K.conj().T@K+alpha*np.eye(K.shape[1]),K.conj().T@y)
    return x

@jit(nopython=True)
def rqi(A,x=None,k=8):
    I = np.eye(A.shape[0])
    if x is None:
        x = np.random.rand(I)-0.5
    for j in range(k):
        u = x/lag.norm(x) # normalize
        lam = np.dot(u,np.dot(A,u)) # Rayleigh quotient
        x = lag.solve(A-lam*I,u) # inverse power iteration
    u = x/lag.norm(x)
    lam = np.dot(u,np.dot(A,u))
    return lam,x/lag.norm(x,np.inf)