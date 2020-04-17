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
    alpha = float()
    
    def set_number_iterations(self,number_iterations):
        self.N_ITER = number_iterations
    
    def set_regularization_parameter(self,alpha):
        self.alpha = alpha
    
    def solve(self,es,Nx=None,Ny=None,model=None,model_path='',
              number_iterations=None,alpha=None,experiment_name=None,
              save_results=False,plot_results=False,file_path='',
              file_format='eps',noconductivity=False,nopermittivity=False,
              initial_field=None,epsilon_r_goal=None,
              sigma_goal=None,print_info=True,lavarello_alpha=False,
              kirsch_alpha=False,delta=None):
            
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
            
        if lavarello_alpha is True:
            self.alpha = self.__lavarello_alpha()
            alpha0 = self.alpha
            
        if kirsch_alpha:
            if delta is None:
                print("SOLVE ERROR: If you choose Kirsch's alpha," + 
                      ' then you must provide and delta error!')
                sys.exit()
            self.alpha = self.__kirsch_choice(es,delta)
        
        if self.alpha is None:
            self.alpha = self.__lavarello_alpha()
            alpha0 = self.alpha
    
            
        if print_info:
            print("==============================================================")
            print("BORN ITERATIVE METHOD - TIKHONOV REGULARIZATION")
            if experiment_name is not None:
                print("Experiment name: " + experiment_name)
            if lavarello_alpha:
                print('Alpha (Lavarello): %.3e' %self.alpha)
            elif kirsch_alpha:
                print('Alpha (Kirsch): %.3e' %self.alpha)
            else:
                print('Alpha: %.3e' %self.alpha)
            
        if initial_field is None:
            self.model.Et = self.model.Ei
        else:
            self.model.Et = initial_field
        
        Ja = np.zeros(self.N_ITER)
        residual = np.zeros(self.N_ITER)
        
        if epsilon_r_goal is not None:
            zeta_e = np.zeros(self.N_ITER)
        
        if sigma_goal is not None:
            zeta_s = np.zeros(self.N_ITER)
                
        for it in range(self.N_ITER):

            epsilon_r, sigma = self.tikhonov_regularization(
                es, noconductivity=noconductivity,
                nopermittivity=nopermittivity
            )
            
            self.model.solve(epsilon_r=epsilon_r,sigma=sigma,
                             maximum_iterations=1000)

            Ja[it] = self.__compute_tikhonov_functional(et=self.model.Et,
                                                        es=es,
                                                        epsilon_r=epsilon_r,
                                                        sigma=sigma)
            
            residual[it] = self.compute_error(self.model.Et,es,
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
                + "- Ja(X) = %.3e " %Ja[it] 
                + "- Residual: %.3e" %residual[it]
            )
            
            if epsilon_r_goal is not None:
                iteration_message = (iteration_message 
                                     + ' - zeta_e = %.2f %%' %zeta_e[it])
                                
            if sigma_goal is not None:
                iteration_message = (iteration_message 
                                     + ' - zeta_s = %.3e [S/m]' %zeta_s[it])

            if lavarello_alpha:
                iteration_message = (iteration_message 
                                     + ' - Alpha: %.3e' %self.alpha)
                self.alpha = self.__lavarello_update(es,residual[it],
                                                     alpha0)
            
            if print_info:
                print(iteration_message)
            
        if save_results:
            if experiment_name is None:
                self.__save_results(epsilon_r,sigma,Ja,residual,'results',
                                    file_path)
            else:
                self.__save_results(epsilon_r,sigma,Ja,residual,
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
                                      sigma=aux_sig,Ja=Ja,
                                      residual=residual,
                                      zeta_e=zeta_e,
                                      zeta_s=zeta_s)
                elif epsilon_r_goal is not None:
                    self.plot_results(file_name=experiment_name,
                                      file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,Ja=Ja,
                                      residual=residual,
                                      zeta_e=zeta_e)
                    
                elif sigma_goal is not None:
                    self.plot_results(file_name=experiment_name,
                                      file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,Ja=Ja,
                                      residual=residual,
                                      zeta_s=zeta_s)
                
                else:
                    self.plot_results(file_name=experiment_name,
                                      file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,Ja=Ja,
                                      residual=residual)
            else:
                if epsilon_r_goal is not None and sigma_goal is not None:
                    self.plot_results(file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,Ja=Ja,
                                      residual=residual,
                                      zeta_e=zeta_e,
                                      zeta_s=zeta_s)
                elif epsilon_r_goal is not None:
                    self.plot_results(file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,Ja=Ja,
                                      residual=residual,
                                      zeta_e=zeta_e)
                    
                elif sigma_goal is not None:
                    self.plot_results(file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,Ja=Ja,
                                      residual=residual,
                                      zeta_s=zeta_s)
                
                else:
                    self.plot_results(file_path=file_path,
                                      file_format=file_format,
                                      epsilon_r=aux_epsr,
                                      sigma=aux_sig,Ja=Ja,
                                      residual=residual)
                                 
        if epsilon_r_goal is not None and sigma_goal is not None:
            return epsilon_r, sigma, Ja, residual, zeta_e, zeta_s
        
        elif epsilon_r_goal is not None:
            return epsilon_r, sigma, Ja, residual, zeta_e
        
        elif sigma_goal is not None:
            return epsilon_r, sigma, Ja, residual, zeta_s
        
        else:
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
                     title=True,zeta_e=None,zeta_s=None):
        
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
                
        elif Ja is not None:
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
                
        elif residual is not None:
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

    def tikhonov_regularization(self,es,noconductivity=False,
                                nopermittivity=False):
        
        K = get_operator_matrix(self.model.Et,self.model.domain.M,
                                self.model.domain.L,self.model.GS,
                                self.model.Et.shape[0])
        y = self.__get_y(es)
        x = solve_tikhonov_regularization(K,y,self.alpha)
        
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

    def solve_linear(self,es,et=None,alpha=None,nopermittivity=False,
                     noconductivity=False):
        
        if alpha is not None:
            self.alpha = alpha
        if et is None:
            self.model.Et = self.model.Ei
        else:
            self.model.Et = et
        
        return self.tikhonov_regularization(es,
                                            nopermittivity=nopermittivity,
                                            noconductivity=noconductivity)

    def get_intern_field(self):
        return np.copy(self.model.Et)
 
    def __lavarello_alpha(self):
        K = get_operator_matrix(self.model.Ei,self.model.domain.M,
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
        K = get_operator_matrix(self.model.Ei,self.model.domain.M,
                                self.model.domain.L,self.model.GS,
                                self.model.Ei.shape[0])
        
        return delta*lag.norm(K)**2/(lag.norm(es.reshape(-1))-delta)

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