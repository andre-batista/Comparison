import numpy as np
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
import solver as slv

class BIM(slv.Solver):
    
    N_ITER = 10
    regularization_method_name = ''
    execution_info = 'regularization_method_name'
    iteration_info = ''
    
    def set_number_iterations(self,number_iterations):
        self.N_ITER = number_iterations

    @abstractmethod
    def set_regularization_parameter(self,new_parameter):
        pass
    
    def solve(self,es,Nx=None,Ny=None,model=None,model_path='',
              number_iterations=None,experiment_name=None,
              save_results=False,plot_results=False,file_path='',
              file_format='eps',noconductivity=False,nopermittivity=False,
              initial_field=None,epsilon_r_goal=None,sigma_goal=None,
              print_info=True,delta=None,regularizer=None):
            
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
        
        if regularizer is not None:
            self.set_regularization_parameter(regularizer)
            
        self.__initialize_variables()
            
        if print_info:
            print("==============================================================")
            print("BORN ITERATIVE METHOD - " 
                  + self.regularization_method_name
                  + ' REGULARIZATION')
            
            if experiment_name is not None:
                print("Experiment name: " + experiment_name)
                
            print(self.execution_info)
            
        if initial_field is None:
            self.model.Et = self.model.Ei
        else:
            self.model.Et = initial_field
        
        residual = np.zeros(self.N_ITER)
        
        if epsilon_r_goal is not None:
            zeta_e = np.zeros(self.N_ITER)
        
        if sigma_goal is not None:
            zeta_s = np.zeros(self.N_ITER)
                
        for it in range(self.N_ITER):

            epsilon_r, sigma = self.regularization_method(
                es, noconductivity=noconductivity,
                nopermittivity=nopermittivity
            )
            
            self.model.solve(epsilon_r=epsilon_r,sigma=sigma,
                             maximum_iterations=1000)

            residual[it] = self.compute_norm_residual(self.model.Et,es,
                                                      epsilon_r=epsilon_r,
                                                      sigma=sigma)
            
            if epsilon_r_goal is not None and sigma_goal is not None:
                
                zeta_e[it], zeta_s[it] = self.compute_map_error(
                    epsilon_original=epsilon_r_goal,
                    epsilon_recovered=epsilon_r,
                    sigma_original=sigma_goal,
                    sigma_recovered=sigma
                )
            
            elif epsilon_r_goal is not None:
                
                zeta_e[it] = self.compute_map_error(
                    epsilon_original=epsilon_r_goal,
                    epsilon_recovered=epsilon_r
                )
                
            elif sigma_goal is not None:
                
                zeta_s[it] = self.compute_map_error(
                    sigma_original=sigma_goal,
                    sigma_recovered=sigma
                )
            
            iteration_message = (
                "Iteration %d " %(it+1) 
                + "- Residual: %.3e" %residual[it]
            )
            
            if epsilon_r_goal is not None:
                iteration_message = (iteration_message 
                                     + ' - zeta_e = %.2f %%' %zeta_e[it])
                                
            if sigma_goal is not None:
                iteration_message = (iteration_message 
                                     + ' - zeta_s = %.3e [S/m]' %zeta_s[it])

            self.__update_regularization_parameter(es,residual[it])
            iteration_message = (iteration_message + self.iteration_info)
            
            if print_info:
                print(iteration_message)
                
            
            
        if save_results:
            if experiment_name is None:
                self.__save_results(epsilon_r,sigma,residual,'results',
                                    file_path)
            else:
                self.__save_results(epsilon_r,sigma,residual,
                                    experiment_name+'_results',file_path)
        
        if plot_results:
            
            if nopermittivity:
                aux_epsr = None
                aux_sig = sigma
            elif noconductivity:
                aux_sig = None
                aux_epsr = epsilon_r
            else:
                aux_epsr = epsilon_r
                aux_sig = sigma
            
            if save_results:
                if epsilon_r_goal is not None and sigma_goal is not None:
                    self.plot_results(file_name=experiment_name,
                                      file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,
                                      residual=residual,
                                      zeta_e=zeta_e,
                                      zeta_s=zeta_s)
                elif epsilon_r_goal is not None:
                    self.plot_results(file_name=experiment_name,
                                      file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,
                                      residual=residual,
                                      zeta_e=zeta_e)
                    
                elif sigma_goal is not None:
                    self.plot_results(file_name=experiment_name,
                                      file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,
                                      residual=residual,
                                      zeta_s=zeta_s)
                
                else:
                    self.plot_results(file_name=experiment_name,
                                      file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,
                                      residual=residual)
            else:
                if epsilon_r_goal is not None and sigma_goal is not None:
                    self.plot_results(file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,
                                      residual=residual,
                                      zeta_e=zeta_e,
                                      zeta_s=zeta_s)
                elif epsilon_r_goal is not None:
                    self.plot_results(file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,
                                      residual=residual,
                                      zeta_e=zeta_e)
                    
                elif sigma_goal is not None:
                    self.plot_results(file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,
                                      residual=residual,
                                      zeta_s=zeta_s)
                
                else:
                    self.plot_results(file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,
                                      residual=residual)
                                 
        if epsilon_r_goal is not None and sigma_goal is not None:
            return epsilon_r, sigma, residual, zeta_e, zeta_s
        
        elif epsilon_r_goal is not None:
            return epsilon_r, sigma, residual, zeta_e
        
        elif sigma_goal is not None:
            return epsilon_r, sigma, residual, zeta_s
        
        else:
            return epsilon_r, sigma, residual

    def load_results(self,file_name,file_path=''):
        with open(file_path+file_name,'rb') as datafile:
            data = pickle.load(datafile)

        epsilon_r = data['relative_permittivity_map']
        sigma = data['conductivity_map']
        number_iterations = data['number_iterations']
        residual = data['residual_convergence']
        
        return epsilon_r, sigma, residual, number_iterations
    
    def plot_results(self,file_name=None,file_path='',file_format='eps',
                     epsilon_r=None,sigma=None,residual=None,
                     title=True,zeta_e=None,zeta_s=None):
        
        if epsilon_r is None and sigma is None and residual is None:
            epsilon_r, sigma, residual, _ = self.load_results(file_name,
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
                plt.savefig(file_path+file_name+'_res_epsr' + '.' 
                            + file_format, format=file_format)
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
                
        if zeta_e is not None and zeta_s is not None:
            
            fig = plt.figure()
            
            ax1 = fig.add_subplot(1,2,1)
            ax1.plot(np.arange(zeta_e.size)+1,zeta_e,'--*')
            ax1.set_xlabel('Iterations')
            ax1.set_ylabel(r'$\zeta_\epsilon$ [\%]')
            ax1.grid()
            if title is True:
                ax1.set_title('Relative Permittivity Convergence')
            elif isinstance(title,str):
                ax1.set_title(title + r' - $\zeta_\epsilon$')
                
            ax2 = fig.add_subplot(1,2,2)
            ax2.plot(np.arange(zeta_s.size)+1,zeta_s,'--*')
            ax2.set_xlabel('Iterations')
            ax2.set_ylabel(r'$\zeta_\sigma$ [S/m]')
            ax2.grid()
            if title is True:
                ax2.set_title('Conductivity Convergence')
            elif isinstance(title,str):
                ax2.set_title(title + r' - $\zeta_\sigma$')
                
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_path+file_name+'_res_mapconv' + '.' 
                            + file_format,format=file_format)
                plt.close()
                
        elif zeta_e is not None:
            plt.plot(np.arange(zeta_e.size)+1,zeta_e,'--*')
            plt.xlabel('Iterations')
            plt.ylabel(r'$\zeta_\epsilon$ [\%]')
            plt.grid()
            if title is True:
                plt.title('Relative Permittivity Convergence')
            elif isinstance(title,str):
                plt.title(title + r' - $\zeta_\epsilon$')
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_path+file_name+'_res_zeta_e' + '.' 
                            + file_format, format=file_format)
                plt.close()
                
        elif zeta_s is not None:
            plt.plot(np.arange(zeta_s.size)+1,zeta_s,'--*')
            plt.xlabel('Iterations')
            plt.ylabel(r'$\zeta_\sigma$ [S/m]')
            plt.grid()
            if title is True:
                plt.title('Conductivity Convergence')
            elif isinstance(title,str):
                plt.title(title +  r' - $\zeta_\sigma$')
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_path+file_name+'_res_zeta_s' + '.' 
                            + file_format, format=file_format)
                plt.close()
    
    @abstractmethod    
    def __save_results(self,epsilon_r,sigma,Ja,residual,file_name,
                       file_path=''):
        pass

    @abstractmethod
    def regularization_method(self,es,noconductivity=False,
                                nopermittivity=False):
        pass
        
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
            
        y = self.get_y(es)
        x = x.reshape(-1)
        return np.sum(np.abs((y-K@x)/y))/y.size
    
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
            
        y = self.get_y(es)
        x = x.reshape(-1)
        return lag.norm(y-K@x)
    
    def get_y(self,es):
        return np.reshape(es,(-1)) 
        
    def compute_map_error(self,epsilon_original=None,epsilon_recovered=None,
                          sigma_original=None,sigma_recovered=None):
        
        if (epsilon_original is not None 
            and (epsilon_original.shape[0] != epsilon_recovered.shape[0]
                 or epsilon_original.shape[1] != epsilon_recovered.shape[1])):

            Lx, Ly = self.model.domain.Lx, self.model.domain.Ly
            xmin, xmax = dm.get_bounds(Lx)
            ymin, ymax = dm.get_bounds(Ly)

            Nxo, Nyo = epsilon_original.shape[0], epsilon_original.shape[1]
            xo, yo = dm.get_domain_coordinates(Nxo/Lx,Nyo/Ly,xmin,xmax,
                                               ymin,ymax)

            Nxr, Nyr = epsilon_recovered.shape[0], epsilon_recovered.shape[1]
            xr, yr = dm.get_domain_coordinates(Nxr/Lx,Nyr/Ly,xmin,xmax,
                                               ymin,ymax)

            fe = interp.interp2d(xr,yr,epsilon_recovered)
            epsilon_recovered = fe(xo,yo)
            
        if (sigma_original is not None 
            and (sigma_original.shape[0] != sigma_recovered.shape[0]
                 or sigma_original.shape[1] != sigma_recovered.shape[1])):

            Lx, Ly = self.model.domain.Lx, self.model.domain.Ly
            xmin, xmax = dm.get_bounds(Lx)
            ymin, ymax = dm.get_bounds(Ly)

            Nxo, Nyo = sigma_original.shape[0], sigma_original.shape[1]
            xo, yo = dm.get_domain_coordinates(Nxo/Lx,Nyo/Ly,xmin,xmax,
                                               ymin,ymax)

            Nxr, Nyr = sigma_recovered.shape[0], sigma_recovered.shape[1]
            xr, yr = dm.get_domain_coordinates(Nxr/Lx,Nyr/Ly,xmin,xmax,
                                               ymin,ymax)

            fe = interp.interp2d(xr,yr,sigma_recovered)
            sigma_recovered = fe(xo,yo)

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

    def compute_norm_x(self,epsilon_r=None,sigma=None):
        
        if epsilon_r is not None and sigma is None:
            sigma = self.model.sigma_b*np.ones(epsilon_r.shape)
        elif epsilon_r is None and sigma is not None:
            epsilon_r = self.model.epsilon_r*np.ones(sigma.shape)
        elif epsilon_r is None and sigma is None:
            print("COMPUTE NORM X ERROR: one input must be given!")
        
        x = self.compute_contrast_function(epsilon_r,sigma)
        return lag.norm(x.reshape(-1))

    @abstractmethod
    def solve_linear(self,es,et=None,alpha=None,nopermittivity=False,
                     noconductivity=False):
        pass
    
    @abstractmethod
    def __update_regularization_parameter(self,es=None,residual=None):
        pass

    def get_intern_field(self):
        return np.copy(self.model.Et) 
    
    @abstractmethod
    def __initialize_variables(self):
        pass

@jit(nopython=True)
def get_operator_matrix(et,M,L,GS,N):
        
    K = 1j*np.ones((M*L,N))
    row = 0
    for m in range(M):
        for l in range(L):
            K[row,:] = GS[m,:]*et[:,l]
            row += 1
    return K
    
def initialsolution(es,et,GS,M,L,N,epsilon_rb,sigma_b,omega):
    
    K = get_operator_matrix(et,M,L,GS,N)
    y = np.reshape(es,(-1))
    x = K.conj().T@y
    
    epsilon_r = np.real(x)+epsilon_rb
    sigma = sigma_b - np.imag(x)*omega*ct.epsilon_0
    epsilon_r[epsilon_r<1] = 1
    sigma[sigma<0] = 0
    sigma[:] = 0.0
        
    return epsilon_r, sigma