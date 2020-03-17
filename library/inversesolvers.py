import numpy as np
import copy as cp
import numpy.linalg as lag
from numba import jit
import scipy.constants as ct

import experiment as exp
import model as md
import domain as dm

class Solver:
    
    model = md.Model(dm.Domain(0,0,0,0,0))
    M, L, N, F = int(), int(), int(), int()
    dx, dy = float(), float()
    ei = np.zeros((N,L,F),dtype=complex)
    es = np.zeros((M,L,F),dtype=complex)
    gs = np.zeros((M,N,F),dtype=complex)
    
    def __init__(self,model,datafile_name,datafile_path=''):
        
        data_dict = load_data_file(file_name,file_path)
        self.__load_variables(data_dict)
        self.model = cp.deepcopy(model)
        
    def __load_variables(self,data_dict):
        
        self.M = data_dict['number_measurements']
        self.L = data_dict['number_sources']
        self.dx, self.dy = data_dict['dx'], data_dict['dy']
        self.ei = data_dict['incident_field']
        self.es = data_dict['scattered_field']
        self.gs = data_dict['green_function_s']
        
        if es.ndim == 3:
            self.F = es.shape[2]
        else:
            self.F = 1
        
    def solve(self):
        pass

class BIM_Tikhonov(Solver):
    
    N_ITER = 10
    alpha = 1e-13
    
    def set_number_iterations(self,number_iterations):
        self.N_ITER = number_iterations
    
    def set_regularization_parameter(self,alpha):
        self.alpha = alpha
    
    def solve(self):
            
        for it in range(self.N_ITER):
            
            
    @jit(nopython=True)
    def __tikhonov_regularization(self,et,es):
        
        K = np.zeros((self.M*self.L,self.N),dtype=complex)
        row = 0
        for m in range(self.M):
            for l in range(self.L):
                K[row,:] = gs[:,f].reshape(-1)*et[:,l].reshape(-1)
                row += 1
        y = np.reshape(es,(-1))
        x = lag.solve(K.conj().T@K+self.alpha*np.eye(K.shape[0]),K.conj().T@y)
        return x
    
    def __get_dielectric_constants(self,x):
        
        epsilon_r = np.real(x)+self.model.epsilon_rb
        sigma = self.model.sigma_b - np.imag(x)*self.model.omega*ct.epsilon_0
        
        
def load_data_file(self,file_name,file_path=''):
    
    with open(file_path+file_name,'rb') as datafile:
        data = pickle.load(datafile)
    return data