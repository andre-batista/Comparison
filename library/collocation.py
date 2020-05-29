import numpy as np
import copy as cp
import numpy.linalg as lag
import pickle
import sys
from numba import jit
from scipy.special import hankel2
import scipy.constants as ct
import scipy.interpolate as interp
import matplotlib as mpl
# mpl.use('Agg') # Avoiding error when using ssh protocol
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod 

import model as md
import domain as dm
import bim

mub = ct.mu_0
BASIS_BILINEAR = 'bilinear'
BASIS_MININUM_NORM = 'mininum_norm'
STANDARD_BASISF = BASIS_BILINEAR

class Collocation(bim.BIM):
    
    regularization_method_name = 'COLLOCATION'
    basis_function = ''
    xmn, ymn = np.array([]), np.array([])
    xpq, ypq = np.array([]), np.array([])
    R = np.array([])
    M, N, P, Q = int(), int(), int(), int()
    
    def set_basis_function(self,function_name,P=None,Q=None):
        
        self.M, self.N = self.model.domain.M, self.model.domain.L
        
        if P is None:
            self.P = self.M
        else:
            self.P = P
            
        if Q is None:
            self.Q = self.N
        else:
            self.Q = Q
            
        self.xmn, self.ymn = (
            np.reshape(np.tile(self.model.xm.reshape((-1,1)),(1,self.N))
                       ,(-1)),
            np.reshape(np.tile(self.model.ym.reshape((-1,1)),(1,self.N))
                       ,(-1))
        )
        
        self.xpq, self.ypq = build_collocation_mesh(self.model.Nx,
                                                    self.model.Ny,
                                                    self.model.dx,
                                                    self.model.dy,
                                                    self.P,self.Q)
        
        self.u, self.v = self.model.domain.x.reshape(-1), self.model.domain.y.reshape(-1)
        self.du, self.dv = self.model.dx, self.model.dy
        
        self.R = np.zeros((self.M*self.N,self.u.size))
        for i in range(self.M*self.N):
            self.R[i,:] = np.sqrt((self.xmn[i]-self.u)**2
                                  +(self.ymn[i]-self.v)**2)
            
        if function_name == BASIS_BILINEAR:
            self.basis_function = function_name
            self.fij = bilinear_basisf(self.u.reshape((self.model.Ny,
                                                       self.model.Nx)),
                                       self.v.reshape((self.model.Ny,
                                                       self.model.Nx)),
                                       self.xpq.reshape((self.Q,self.P)),
                                       self.ypq.reshape((self.Q,self.P)))
        
        elif function_name == BASIS_MININUM_NORM:
            self.basis_function = function_name
    
    def set_regularization_parameter(self, newparameter=None):
        
        if isinstance(newparameter,str):
            self.set_basis_function(newparameter)
        elif isinstance(newparameter,tuple):
            self.set_basis_function(self.basis_function,P=newparameter[0],
                                    Q=newparameter[1])
        
    def regularization_method(self, es, noconductivity=False, 
                              nopermittivity=False):
        
        K = self.get_operator(self.model.Et)
        
        if self.basis_function == BASIS_MININUM_NORM:
            self.fij = self.__minimum_norm_basisf(self.model.Et)
        
        A = computeA(self.M,self.N,self.P,self.Q,self.model.Nx,
                     self.model.Ny,K,self.fij,self.du,self.dv)
        beta = np.copy(es.reshape(-1))
        
        if self.basis_function == BASIS_BILINEAR:
            self.boundary_condition(A,beta)
            
        alpha = lag.lstsq(A,beta,rcond=8e-3)[0]
        
        J,I = self.model.Ny, self.model.Nx
        fa = np.zeros((J,I),dtype=complex)
        for i in range(I):
            for j in range(J):        
                fa[j,i] = np.sum(alpha*self.fij[:,j*I+i])
        
        if nopermittivity:
            epsilon_r = self.model.epsilon_rb*np.ones((self.model.Nx,
                                                       self.model.Ny))
        else:
            epsilon_r = (np.imag(fa)/ct.epsilon_0/self.model.omega 
                         + self.model.epsilon_rb)
            epsilon_r[epsilon_r<1] = 1
        
        if noconductivity:
            sigma = self.model.sigma_b*np.ones(epsilon_r.shape)
        else:
            sigma = np.real(fa) + self.model.sigma_b
            sigma[sigma<0] = 0
        
        return epsilon_r, sigma
     
    def solve_linear(self, es, et=None, nopermittivity=False, 
                     noconductivity=False):

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
            'P': self.P,
            'Q': self.Q,
            'number_iterations':self.N_ITER
        }

        with open(file_path + file_name,'wb') as datafile:
            pickle.dump(data,datafile)
            
    def _BIM__initialize_variables(self):
        
        if self.basis_function == '':
            self.set_basis_function(STANDARD_BASISF)
            
        self.execution_info = 'Basis function = ' + self.basis_function
        self.execution_info = (self.execution_info 
                               + '\nPxQ = %dx' %self.P + '%d' %self.Q)
        
    def get_operator(self,et):
        K = np.zeros((self.M*self.N,self.model.Nx*self.model.Ny),
                     dtype=complex)
        l = 0
        for i in range(self.M*self.N):
            K[i,:] = (1j*self.model.omega*mub*et[:,l]*1j/4
                      * hankel2(0,self.model.kb*self.R[i,:]))
            if l == self.model.domain.L-1:
                l = 0
            else:
                l += 1
                
        return K

    def boundary_condition(self,A,beta):
        A = np.vstack((A,np.zeros((2*self.P+2*self.Q,self.P*self.Q),
                                  dtype=complex)))
        beta = np.hstack((beta,np.zeros(2*self.P+2*self.Q,dtype=complex)))
        i = self.M*self.N
        for q in range(self.Q):
            p = 0
            A[i,q*self.P+p] = 1
            beta[i] = 0
            i += 1
            p = self.P-1
            A[i,q*self.P+p] = 1
            beta[i] = 0
            i += 1
        for p in range(self.P):
            q = 0
            A[i,q*self.P+p] = 1
            beta[i] = 0
            i += 1
            q = self.P-1
            A[i,q*self.P+p] = 1
            beta[i] = 0
            i += 1

    def __minimum_norm_basisf(self,et):
        
        N = self.u.size
        Kpq = np.zeros((self.P*self.Q,N),dtype=complex)
        l = 0
        for i in range(self.P*self.Q):
            R = np.sqrt((self.xpq[i]-self.u)**2+(self.ypq[i]-self.v)**2)   
            Kpq[i,:] = 1j*self.model.omega*mub*et[:,l]*1j/4*hankel2(0,self.model.kb*R)
            if l == self.model.domain.L-1:
                l = 0
            else:
                l += 1
                
        return Kpq

def build_collocation_mesh(I,J,dx,dy,p,q):
    x_min, x_max = md.get_bounds(I*dx)
    y_min, y_max = md.get_bounds(J*dy)
    xpq, ypq = np.meshgrid(np.linspace(x_min,x_max,p),
                           np.linspace(y_min,y_max,q))
    xpq, ypq = xpq.reshape(-1), ypq.reshape(-1)
    return xpq, ypq

def bilinear_basisf (u,v,x,y):
    """ Evaluate the triangular function. Given the collocation points
    in x and y, the function returns the evaluation of the triangular
    function in points specificated by the variables u and v. Each of
    the four variables must be 2D (meshgrid format).
    """
    Q, P = u.shape
    N, M  = x.shape
    f = np.zeros((x.size,u.size))
    
    for p in range(P):
        for q in range(Q):
        
            m = np.argwhere(u[q,p] >= x[0,:])[-1][0]
            n = np.argwhere(v[q,p] >= y[:,0])[-1][0]               
        
            if m+1 < M and n+1 < N:
                eta = 2*(u[q,p]-x[n,m])/(x[n,m+1]-x[n,m]) - 1
                qsi = 2*(v[q,p]-y[n,m])/(y[n+1,m]-y[n,m]) - 1
            
                f[n*M+m,q*P+p] = .25*(1-qsi)*(1-eta) # 1
                f[(n+1)*M+m,q*P+p] = .25*(1+qsi)*(1-eta) # 2
                f[(n+1)*M+m+1,q*P+p] = .25*(1+qsi)*(1+eta) # 3
                f[n*M+m+1,q*P+p] = .25*(1-qsi)*(1+eta) # 4
                
                
            
            elif m+1 < M and n == N-1:
                eta = 2*(u[q,p]-x[n,m])/(x[n,m+1]-x[n,m]) - 1
                # qsi = -1
            
                f[n*M+m,q*P+p] = .25*2*(1-eta) # 1
                f[n*M+m+1,q*P+p] = .25*2*(1+eta) # 4
            
            elif m == M-1 and n+1 < N:
                # eta = -1
                qsi = 2*(v[q,p]-y[n,m])/(y[n+1,m]-y[n,m]) - 1
            
                f[n*M+m,q*P+p] = .25*(1-qsi)*2 # 1
                f[(n+1)*M+m,q*P+p] = .25*(1+qsi)*2 # 2
            
            elif m == M-1 and n == N-1:
                # qsi = -1
                # eta = -1
            
                f[n*M+m,q*P+p] = 1. # 1
                
    return f

@jit(nopython=True)
def computeA(M,N,P,Q,I,J,K,fij,du,dv):
    A = 1j*np.zeros((M*N,P*Q))
    for i in range(M*N):
        for j in range(P*Q):
            A[i,j] = np.trapz(np.trapz(K[i,:].reshape((J,I))
                                       *fij[j,:].reshape((J,I)),
                                       dx=du),dx=dv)
    return A

