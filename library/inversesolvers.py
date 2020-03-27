import numpy as np
import copy as cp
import numpy.linalg as lag
import pickle
import sys
from numba import jit
import scipy.constants as ct
import matplotlib as mpl
# mpl.use('Agg') # Avoiding error when using ssh protocol
import matplotlib.pyplot as plt

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
        
    def solve(self,es):
        pass

class BIM_Tikhonov(Solver):
    
    N_ITER = 10
    alpha = 1e-13
    
    def set_number_iterations(self,number_iterations):
        self.N_ITER = number_iterations
    
    def set_regularization_parameter(self,alpha):
        self.alpha = alpha
    
    def solve(self,es,Nx=None,Ny=None,model=None,model_path='',
              number_iterations=None,alpha=None,experiment_name=None,
              save_results=False,plot_results=False,file_path='',
              file_format='eps'):
            
        if model is not None:
            self.set_model(model,model_path)
            
        if Nx is not None or Ny is not None:
            if Nx is None:
                Nx = self.model.Nx
            if Ny is None:
                Ny = self.model.Ny
            self.model.change_discretization(Nx,Ny)  
            
        if number_iterations is not None:
            self.N_ITER = number_iterations
        
        if alpha is not None:
            self.alpha = alpha
            
        print("==============================================================")
        print("BORN ITERATIVE METHOD - TIKHONOV REGULARIZATION")
        if experiment_name is not None:
            print("Experiment name: " + experiment_name)        
            
        self.model.Et = self.model.Ei
        Ja = np.zeros(self.N_ITER)
        residual = np.zeros(self.N_ITER)
        
        # epsilon_r, sigma = initialsolution1(es,self.model.Et,self.model.GS,self.model.domain.M,self.model.domain.L,self.model.Nx*self.model.Ny,self.model.epsilon_rb,self.model.sigma_b,2*np.pi*self.model.f)
        # epsilon_r = epsilon_r.reshape((self.model.Nx,self.model.Ny))
        # sigma = sigma.reshape((self.model.Nx,self.model.Ny))
        
        for it in range(self.N_ITER):

            # self.alpha = self.alpha*1e-2
            epsilon_r, sigma = self.tikhonov_regularization(es)
            self.model.solve(epsilon_r=epsilon_r,sigma=sigma,maximum_iterations=1000)

                        
            Ja[it] = self.__compute_tikhonov_functional(et=self.model.Et,es=es,
                                                        epsilon_r=epsilon_r,
                                                        sigma=sigma)
            residual[it] = self.compute_error(self.model.Et,es,
                                              epsilon_r=epsilon_r,sigma=sigma)
            print("Iteration %d " %(it+1) + "- Ja(X) = %.3e " %Ja[it] 
                  + "- Residual: %.3e" %residual[it])
            
        if save_results:
            if experiment_name is None:
                self.__save_results(epsilon_r,sigma,Ja,residual,'results',
                                    file_path)
            else:
                self.__save_results(epsilon_r,sigma,Ja,residual,
                                    experiment_name+'_results',file_path)
        
        if plot_results:
            if save_results:
                self.plot_results(file_name=experiment_name,file_path=file_path,
                                  file_format=file_format,epsilon_r=epsilon_r,
                                  sigma=sigma,Ja=Ja,residual=residual)
            else:
                self.plot_results(file_path=file_path,file_format=file_format,
                                  epsilon_r=epsilon_r,sigma=sigma,Ja=Ja,
                                  residual=residual)
                                 
        return epsilon_r, sigma, Ja, residual

    def load_results(self,file_name,file_path=''):
        
        with open(file_path+file_name,'rb') as datafile:
            data = pickle.load(datafile)

        epsilon_r = data['relative_permittivity_map']
        sigma = data['conductivity_map']
        Ja = data['tikhonov_functional_convergence']
        alpha = data['alpha']
        number_iterations = data['number_iterations']
        
        return epsilon_r, sigma, Ja, alpha, number_iterations
    
    def plot_results(self,file_name=None,file_path='',file_format='eps',
                     epsilon_r=None,sigma=None,Ja=None,residual=None,
                     title=True):
        
        if epsilon_r is None and sigma is None and Ja is None and residual is None:
            epsilon_r, sigma, Ja, residual, _ = self.load_results(file_name,
                                                                  file_path)
        
        xmin = self.model.domain.x[0,0]/self.model.lambda_b
        xmax = self.model.domain.x[0,-1]/self.model.lambda_b
        ymin = self.model.domain.y[0,0]/self.model.lambda_b
        ymax = self.model.domain.y[-1,0]/self.model.lambda_b
        
        if epsilon_r is not None:
            plt.imshow(epsilon_r,extent=[xmin,xmax,ymin,ymax])
            plt.xlabel(r'x [$\lambda_b$]')
            plt.ylabel(r'y [$\lambda_b$]')
            cbar = plt.colorbar()
            cbar.set_label(r'$\epsilon_r$')
            if title is True:
                plt.title('Relative Permittivity Map')
            elif isinstance(title,str):
                plt.title(title)
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_path+file_name+'_res_epsr' + '.' + file_format,
                            format=file_format)
                plt.close()
                
        if sigma is not None:
            plt.imshow(sigma,extent=[xmin,xmax,ymin,ymax])
            plt.xlabel(r'x [$\lambda_b$]')
            plt.ylabel(r'y [$\lambda_b$]')
            cbar = plt.colorbar()
            cbar.set_label(r'$\sigma$ [S/m]')
            if title is True:
                plt.title('Conductivity Map')
            elif isinstance(title,str):
                plt.title(title)
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_path+file_name+'_res_sig' + '.' + file_format,
                            format=file_format)
                plt.close()
                
        if Ja is not None and residual is not None:
            
            fig = plt.figure()
            
            ax1 = fig.add_subplot(1,2,1)
            ax1.plot(np.arange(Ja.size)+1,Ja,'--*')
            ax1.set_xlabel('Iterations')
            ax1.set_ylabel(r'$J_{\alpha}(\chi)$')
            ax1.grid()
            if title is True:
                ax1.set_title('Tikhonov Functional Convergence')
            elif isinstance(title,str):
                ax1.set_title(title + ' - Functional')
                
            ax2 = fig.add_subplot(1,2,2)
            ax2.plot(np.arange(residual.size)+1,residual,'--*')
            ax2.set_xlabel('Iterations')
            ax2.set_ylabel(r'$||y-Kx||^2$')
            ax2.grid()
            if title is True:
                ax2.set_title('Residual Convergence')
            elif isinstance(title,str):
                ax2.set_title(title + ' - Residual')
                
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_path+file_name+'_res_conv' + '.' + file_format,
                            format=file_format)
                plt.close()
                
        if Ja is not None:
            plt.plot(np.arange(Ja.size)+1,Ja,'--*')
            plt.xlabel('Iterations')
            plt.ylabel(r'$J_{\alpha}(\chi)$')
            plt.grid()
            if title is True:
                plt.title('Tikhonov Functional Convergence')
            elif isinstance(title,str):
                plt.title(title)
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_path+file_name+'_res_ja' + '.' + file_format,
                            format=file_format)
                plt.close()
                
        if residual is not None:
            plt.plot(np.arange(residual.size)+1,residual,'--*')
            plt.xlabel('Iterations')
            plt.ylabel(r'$||y-Kx||^2$')
            plt.grid()
            if title is True:
                plt.title('Residual Convergence')
            elif isinstance(title,str):
                plt.title(title)
                
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_path+file_name+'_res_residual' + '.' 
                            + file_format,format=file_format)
                plt.close()
        
    def __save_results(self,epsilon_r,sigma,Ja,residual,file_name,file_path=''):
        
        data = {
            'relative_permittivity_map':epsilon_r,
            'conductivity_map':sigma,
            'tikhonov_functional_convergence':Ja,
            'residual_convergence':residual,
            'alpha':self.alpha,
            'number_iterations':self.N_ITER
        }

        with open(file_path + file_name,'wb') as datafile:
            pickle.dump(data,datafile)

    def tikhonov_regularization(self,es):
        K = get_operator_matrix(self.model.Et,self.model.domain.M,
                                self.model.domain.L,self.model.GS,
                                self.model.Et.shape[0])
        y = self.__get_y(es)
        x = solve_tikhonov_regularization(K,y,self.alpha)
        epsilon_r = np.reshape(self.model.epsilon_rb*(np.real(x)+1),(self.model.Nx,
                                                                 self.model.Ny))
        sigma = np.reshape(self.model.sigma_b - np.imag(x)*self.model.omega
                           *ct.epsilon_0,(self.model.Nx,self.model.Ny))
        # sigma = self.model.sigma_b*np.ones(epsilon_r.shape)
        epsilon_r[epsilon_r<1] = 1
        sigma[sigma<0] = 0
        
        return epsilon_r, sigma
        
    def compute_contrast_function(self,epsilon_r,sigma):
        X = ((epsilon_r-1j*sigma/self.model.omega/ct.epsilon_0)
             /(self.model.epsilon_rb-1j*self.model.sigma_b/self.model.omega
               /ct.epsilon_0)-1)
        return X
    
    def compute_error(self,et,es,x=None,epsilon_r=None,sigma=None,K=None):
        if epsilon_r is None and x is None:
            print('COMPUTE_ERROR ERROR: Either x or epsilon_r-sigma must be'
                  + ' given!')
            sys.exit()
        
        if K is None:
            K = get_operator_matrix(et,self.model.domain.M,
                                    self.model.domain.L,self.model.GS,
                                    et.shape[0])
            
        if x is None:
            x = self.compute_contrast_function(epsilon_r,sigma)
            
        y = self.__get_y(es)
        x = x.reshape(-1)
        return np.sum(np.abs((y-K@x)/y))/x.size
    
    def compute_norm_residual(self,et,es,x=None,epsilon_r=None,sigma=None,K=None):
        
        if epsilon_r is None and x is None:
            print('COMPUTE_ERROR ERROR: Either x or epsilon_r-sigma must be'
                  + ' given!')
            sys.exit()
        
        if K is None:
            K = get_operator_matrix(et,self.model.domain.M,
                                    self.model.domain.L,self.model.GS,
                                    et.shape[0])
            
        if x is None:
            x = self.compute_contrast_function(epsilon_r,sigma)
            
        y = self.__get_y(es)
        x = x.reshape(-1)
        return lag.norm(y-K@x)
    
    def __get_y(self,es):
        return np.reshape(es,(-1))
    
    def __compute_tikhonov_functional(self,K=None,x=None,y=None,et=None,es=None,
                                      epsilon_r=None,sigma=None):
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
            
            K = get_operator_matrix(et,self.model.domain.M,
                                    self.model.domain.L,self.model.GS,
                                    et.shape[0])
            x = self.compute_contrast_function(epsilon_r,sigma).reshape(-1)
            y = self.__get_y(es)
            
        elif es is None:
            if x is None:
                print('COMPUTE_TIKHONOV_FUNCTIONAL ERROR: x is missing!')
                sys.exit()
            if y is None:
                print('COMPUTE_TIKHONOV_FUNCTIONAL ERROR: y is missing!')
                sys.exit()            
        
        return lag.norm(K@x-y)**2 + self.alpha*lag.norm(x)**2
        
    def compute_map_error(self,epsilon_original=None,epsilon_recovered=None,
                          sigma_original=None,sigma_recovered=None):

        if epsilon_original is not None and sigma_original is not None:
            zeta_e = (np.sqrt(np.sum(np.abs((epsilon_original-epsilon_recovered)
                                           /epsilon_original)**2))
                      /epsilon_original.size*100)
            zeta_s = (np.sqrt(np.sum(np.abs(sigma_original-sigma_recovered)**2))
                      /epsilon_original.size)
            return zeta_e, zeta_s
            
        elif epsilon_original is not None:
            zeta_e = (np.sqrt(np.sum(np.abs((epsilon_original-epsilon_recovered)
                                           /epsilon_original)**2))
                      /epsilon_original.size*100)
            return zeta_e
        
        elif sigma_original is not None:
            zeta_s = (np.sqrt(np.sum(np.abs(sigma_original-sigma_recovered)**2))
                      /epsilon_original.size)
            return zeta_s

    def solve_linear(self,es,et=None,alpha=None):
        
        if alpha is not None:
            self.alpha = alpha
        if et is None:
            self.model.Et = self.model.Ei
        else:
            self.model.Et = et
        
        return self.tikhonov_regularization(es)
        
@jit(nopython=True)
def get_operator_matrix(et,M,L,GS,N):
        
    K = 1j*np.ones((M*L,N))
    row = 0
    for m in range(M):
        for l in range(L):
            K[row,:] = GS[m,:]*et[:,l]
            row += 1
    return K
    
@jit(nopython=True)
def solve_tikhonov_regularization(K,y,alpha):
    x = lag.solve(K.conj().T@K+alpha*np.eye(K.shape[1]),K.conj().T@y)
    return x

def initialsolution1(es,et,GS,M,L,N,epsilon_rb,sigma_b,omega):
    
    K = get_operator_matrix(et,M,L,GS,N)
    y = np.reshape(es,(-1))
    x = K.conj().T@y
    
    epsilon_r = np.real(x)+epsilon_rb
    sigma = sigma_b - np.imag(x)*omega*ct.epsilon_0
    epsilon_r[epsilon_r<1] = 1
    sigma[sigma<0] = 0
    sigma[:] = 0.0
        
    return epsilon_r, sigma