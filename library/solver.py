import numpy as np
import copy as cp
import pickle
from abc import ABC, abstractmethod 

import model as md
import domain as dm

class Solver(ABC):
    
    model = md.Model(dm.Domain(0,0,0,0,0))
    M, L, N, F = int(), int(), int(), int()
    dx, dy = float(), float()
    ei = np.zeros((N,L,F),dtype=complex)
    es = np.zeros((M,L,F),dtype=complex)
    gs = np.zeros((M,N,F),dtype=complex)
    
    def __init__(self,model=None,model_path=''):
        
        if model is None:
            pass
        else:
            self.set_model(model,model_path)
        
    def __load_model_file(self,file_name,file_path=''):
        
        with open(file_path + file_name,'rb') as datafile:
            data = pickle.load(datafile)
        return data
    
    def __load_model_variables(self,data_dict):
        
        self.model = md.Model(dm.Domain(data_dict['Lx'],data_dict['Ly'],
                                        data_dict['radius_observation'],
                                        data_dict['number_sources'],
                                        data_dict['number_measurements']),
                              model_name=data_dict['model_name'],
                              frequencies=data_dict['frequency'],
                              Nx=data_dict['Nx'],Ny=data_dict['Ny'],
                              incident_field_magnitude=
                              data_dict['incident_field_magnitude'],
                              epsilon_r_background=
                              data_dict['relative_permittivity_background'],
                              sigma_background=
                              data_dict['conductivity_background'])
        
        if isinstance(data_dict['frequency'],float):
            self.F = 1
        else:
            self.F = data_dict['frequency'].size       
            
    def __load_Model_object(self,model):
        
        self.model = cp.deepcopy(model)
        
        if isinstance(self.model.f,float):
            self.F = 1
        else:
            self.F = self.model.f.size
               
    def set_model(self,model,model_path=''):
        
        if isinstance(model,str):
            data_dict = self.__load_model_file(model,model_path)
            self.__load_model_variables(data_dict)
        
        elif isinstance(model,dict):
            self.__load_model_variables(data_dict)

        elif isinstance(model,md.Model):
            self.__load_Model_object(model)

    @abstractmethod        
    def solve(self,es):
        pass
