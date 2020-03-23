'''
  Library  : Method of Moments embedded in a class for electromagnetic foward
             problem.  
  Author   : Jose Olger Vargas and Andre Costa Batista (translation to Python
             and object orientation).
  Date     : OCTOBER 2019, Universidade Federal de Minas Gerais  
  Function : To compute the electric field for scattering
             by an arbitrary shape object with Method of Moments (MoM) 
             and Conjugate Gradient- Fast Fourier Trasnsform (CG-FFT).
'''

# Libraries
import sys
import time
import pickle
import numpy as np
from numpy import linalg as LA
from numpy import fft
import scipy.constants as ct
import scipy.special as spc
import domain as dm
from joblib import Parallel, delayed
import multiprocessing
import matplotlib as mpl
# mpl.use('Agg') # Avoiding error when using ssh protocol
import matplotlib.pyplot as plt

MEMORY_LIMIT = 16e9

class Model:
    """ MODEL: 
        Class to represent a microwave imaging problem with a solver for 
        sythesize data.  
    
    Constructor arguments:
    -- domain: an object of class Domain
    -- model_name: string
    -- frequencies: either a single or a numpy array of frequencies for 
       measurements [Hz]
    -- incident_field_magnitude: magnitude of time-harmonic incident wave [V/m]
    -- epsilon_r_background: relative permittivity of background medium. 
       Default: 1.0
    -- sigma_background: conductivity of background medium [S/m]. Default: 0.0 
       [S/m]
    -- maximum_iterations: maximum number of iterations allowed for solver
       algorithm. Default: 100
    -- tolerance: maximum error allowed for stop criterion of the solver. 
       Default: 1e-6
    """
    
    domain = dm.Domain(.0,.0,.0,0,0) # Imaging domain object
    model_name = ''
    f = np.array([]) # Frequencies of measuremnt [Hz]
    E0 = float() # Magnitude of incident wave
    max_it, TOL = int(), float() # Maximum number of iterations and error
    # tolerance
    
    def __init__(self,domain,model_name=None,frequencies=None, Nx=None, Ny=None,
                 incident_field_magnitude=1.,epsilon_r_background=1.,
                 sigma_background=.0,maximum_iterations=100, tolerance=1e-6):
        
        self.domain = domain
        self.E0 = incident_field_magnitude
        self.epsilon_rb = epsilon_r_background
        self.sigma_b = sigma_background
        
        if model_name is not None:
            self.model_name = model_name
            
        self.max_it = maximum_iterations
        self.TOL = tolerance
        
        if Nx is not None and Ny is not None and frequencies is not None:
            self.f = frequencies
            self.__compute_model_parameters(Ny,Ny)
        else:
            if Nx is not None:
                self.Nx = Nx
            else:
                self.Nx = None
                
            if Ny is not None:
                self.Ny = Ny
            else:
                self.Ny = None
                
            if frequencies is not None:
                self.f = frequencies
            else:
                self.f = None
            
    def solve(self,epsilon_r=None,sigma=None,frequencies=None,
              maximum_iterations=None, tolerance=None,simulation_name=None,
              save=False,file_path='',COMPUTE_INTERN_FIELD=True,
              PRINT_INFO=False):
        """ Solve the scattered field for a given dielectric map. Either 
        epsilon_r or sigma or both matrices must be given. You may or not give
        a new single or a set of frequencies. If you haven't given yet, you
        must do it now. You may change the maximum number of iterations and
        tolerance error."""
        
        if PRINT_INFO:
            print('==========================================================')
            print('Method of Moments (MoM)\nConjugate Gradient-Fast Fourier Trasnsform (CG-FFT)')
        if PRINT_INFO and simulation_name is not None:
            print('Model name: ' + simulation_name)
        elif PRINT_INFO and self.model_name is not None:
            print('Model name: ' + self.model_name)

        if frequencies is None and self.f is None:
            print('SOLVE ERROR: Either a frequency or a set of frequencies' +
                  ' must be given!')
            sys.exit()
        elif frequencies is not None:
            self.f = frequencies

        # Check inputs
        if epsilon_r is None and sigma is None:
            print('SOLVE ERROR: Either epsilon_r or sigma or both must be given!')
            sys.exit()
        elif sigma is None:
            Nx, Ny = epsilon_r.shape
            sigma = self.sigma_b*np.ones(epsilon_r.shape)
        elif epsilon_r is None:
            Nx, Ny = sigma.shape
            epsilon_r = self.epsilon_rb*np.ones(sigma.shape)
        else:
            Nx, Ny = epsilon_r.shape

        if ((self.Nx is None and self.Ny is None) or (Nx != self.Nx 
                                                      or Ny != self.Ny) 
            or frequencies is not None):
            self.__compute_model_parameters(Nx,Ny)
            
        if isinstance(self.f,float):
            MONO_FREQUENCY = True
        else:
            MONO_FREQUENCY = False
            
        if maximum_iterations is not None:
            self.max_it = maximum_iterations
        
        if tolerance is not None:
            self.TOL = tolerance
            
        # Copying variables
        x, y = self.domain.x, self.domain.y
        xmin, ymin, xmax, ymax = self.xmin, self.ymin, self.xmax, self.ymax
        dx, dy = self.dx, self.dy
        L, M, N = self.domain.L, self.domain.M, Nx*Ny
        omega = self.omega
        kb = self.kb
        if not isinstance(self.f,float):
            F = self.f.size  
        
        # Some approximation parameters
        deltasn = self.dx*self.dy # area of the cell
        an = np.sqrt(deltasn/np.pi) # radius of the equivalent circle
        Xr = get_contrast_map(epsilon_r,sigma,self.epsilon_rb,self.sigma_b,
                              omega)
        
        # Using circular convolution
        [xe,ye] = np.meshgrid(np.arange(xmin-(Nx/2-1)*dx,xmax+Nx/2*dx,dx),
                              np.arange(ymin-(Ny/2-1)*dy,ymax+Ny/2*dy,dy)) # extended domain (2N-1)x(2N-1)
        
        Rmn = np.sqrt(xe**2 + ye**2) # distance between the cells
        Z = self.__get_extended_matrix(Rmn,kb,an,Nx,Ny)

        
        if MONO_FREQUENCY:
            b = np.tile(Xr.reshape((-1,1)),(1,L))*self.Ei

        else:
            b = np.zeros((N,L,F),dtype=complex)
            for f in range(F):
                b[:,:,f] = (np.tile(Xr[:,:,f].reshape((-1,1)),(1,L))
                            *self.Ei[:,:,f])

        if MONO_FREQUENCY:
            tic = time.time()
            J,niter,error = self.__CG_FFT(Z,b,Nx,Ny,L,Xr,self.max_it,self.TOL,
                                          PRINT_INFO)
            time_cg_fft=time.time()-tic
            if PRINT_INFO:
                print('Execution time: %.2f' %time_cg_fft + ' [sec]')
            
        else:
            J = np.zeros((N,L,F),dtype=complex)
            niter = np.zeros(F)
            error = np.zeros((self.max_it,F))
            num_cores = multiprocessing.cpu_count()
                
            results = (Parallel(n_jobs=num_cores)(delayed(self.__CG_FFT)
                                                  (np.squeeze(Z[:,:,f]),
                                                   np.squeeze(b[:,:,f]),
                                                   Nx,Ny,self.domain.L,
                                                   np.squeeze(Xr[:,:,f]),
                                                   self.max_it,self.TOL,False) 
                                                  for f in range(F)))
            
            for f in range(F):
                J[:,:,f] = results[f][0]
                niter[f] = results[f][1]
                error[:,f] = results[f][2]
                print('Frequency: %.3f ' %(self.f[f]/1e9) + '[GHz] - ' 
                      + 'Number of iterations: %d - ' %(niter[f]+1)
                      + 'Error: %.3e' %error[int(niter[f]),f])
        
        if MONO_FREQUENCY:
            ## Scattered Field
            Esc_z = self.GS@J  # Ns x Ni
        
        else:
            
            Esc_z = np.zeros((M,L,F),dtype=complex)          
            for f in range(F):
                Esc_z[:,:,f] = self.GS[:,:,f]@J[:,:,f]
                
        if COMPUTE_INTERN_FIELD:
            
            if MONO_FREQUENCY:
                GD = get_greenfunction(x.reshape(-1),y.reshape(-1),x,y,kb)
                Et = GD@J
                
            else:
                Et = np.zeros((N,L,F),dtype=complex)
                if 8*(N)**2*F < MEMORY_LIMIT:
                    GD = get_greenfunction(x.reshape(-1),y.reshape(-1),x,y,kb)
                    for f in range(F):
                        Et[:,:,f] = GD[:,:,f]@J[:,:,f]
                else:
                    for f in range(F):
                        GD = get_greenfunction(x.reshape(-1),y.reshape(-1),
                                               x,y,kb[f])
                        Et[:,:,f] = GD@J[:,:,f]
                       
            Et = Et + self.Ei
            
            if MONO_FREQUENCY:
                self.Et = np.copy(Et)
            else:
                self.Et = np.copy(Et)
            
        if save:
            if COMPUTE_INTERN_FIELD:
                if simulation_name is not None:
                    self.__save_simulation_data(Esc_z,epsilon_r,sigma,
                                                simulation_name,file_path,Et=Et)
                else:
                    self.__save_simulation_data(Esc_z,epsilon_r,sigma,
                                                self.model_name,file_path,Et=Et)
            else:
                if simulation_name is not None:
                    self.__save_simulation_data(Esc_z,epsilon_r,sigma,
                                                simulation_name,file_path)
                else:
                    self.__save_simulation_data(Esc_z,epsilon_r,sigma,
                                                self.model_name,file_path)

        
        return Esc_z
        
    def get_incident_field(self,Nx,Ny,frequencies):
        """ Compute the incident field for a given mesh."""
            
        phi = get_angles(self.domain.L)
        xmin, xmax = get_bounds(self.domain.Lx)
        ymin, ymax = get_bounds(self.domain.Ly)
        dx, dy = self.domain.Lx/Nx, self.domain.Ly/Ny
        x, y = get_domain_coordinates(dx,dy,xmin,xmax,ymin,ymax)
        kb = get_wavenumber(frequencies,epsilon_r=self.epsilon_rb)

        if isinstance(kb,float):
            Ei = self.E0*np.exp(-1j*kb*(x.reshape((-1,1))
                                        @ np.cos(phi.reshape((1,-1))) 
                                        + y.reshape((-1,1))
                                        @ np.sin(phi.reshape((1,-1)))))
        else:
            Ei = np.zeros((Nx*Ny,self.domain.L,kb.size),dtype=complex)
            for f in range(kb.size):
                Ei[:,:,f] = self.E0*np.exp(-1j*kb[f]*(
                    x.reshape((-1,1))@np.cos(phi.reshape((1,-1))) 
                    + y.reshape((-1,1))@np.sin(phi.reshape((1,-1)))
                ))

        return Ei

    def __get_extended_matrix(self,Rmn,kb,an,Nx,Ny):
        """ Return the extended matrix of Method of Moments"""

        if isinstance(kb,float):

            # Matrix elements for off-diagonal entries
            Zmn = ((1j*np.pi*kb*an)/2)*spc.jv(1,kb*an)*spc.hankel2(0,kb*Rmn) # m=/n
            # Matrix elements for diagonal entries 
            Zmn[Nx-1,Ny-1]= ((1j*np.pi*kb*an)/2)*spc.hankel2(1,kb*an)+1 # m==n

            # Extended matrix (2N-1)x(2N-1) 
            Z = np.zeros((2*Nx-1,2*Ny-1),dtype=complex)
            Z[:Nx,:Ny] = Zmn[Nx-1:2*Nx-1,Ny-1:2*Ny-1]
            Z[Nx:2*Nx-1,Ny:2*Ny-1] = Zmn[:Nx-1,:Ny-1]
            Z[:Nx,Ny:2*Ny-1] = Zmn[Nx-1:2*Nx-1,:Ny-1]
            Z[Nx:2*Nx-1,:Ny] = Zmn[:Nx-1,Ny-1:2*Ny-1]
            
        else:
            
            Z = np.zeros((2*Nx-1,2*Ny-1,kb.size),dtype=complex)
            
            for f in range(kb.size):
                
                # Matrix elements for off-diagonal entries
                Zmn = (((1j*np.pi*kb[f]*an)/2)*spc.jv(1,kb[f]*an)
                       * spc.hankel2(0,kb[f]*Rmn)) # m=/n
                # Matrix elements for diagonal entries 
                Zmn[Nx-1,Ny-1]= ((1j*np.pi*kb[f]*an)/2)*spc.hankel2(1,kb[f]*an)+1 # m==n
                
                Z[:Nx,:Ny,f] = Zmn[Nx-1:2*Nx-1,Ny-1:2*Ny-1]
                Z[Nx:2*Nx-1,Ny:2*Ny-1,f] = Zmn[:Nx-1,:Ny-1]
                Z[:Nx,Ny:2*Ny-1,f] = Zmn[Nx-1:2*Nx-1,:Ny-1]
                Z[Nx:2*Nx-1,:Ny,f] = Zmn[:Nx-1,Ny-1:2*Ny-1]
    
        return Z
        
    def __CG_FFT(self,Z,b,Nx,Ny,Ni,Xr,max_it,TOL,PRINT_CONVERGENCE):
        '''
                Congugate-Gradient Method (CGM)

            inputs:
            * Z:       extended matrix     (2N-1)x(2N-1)
            * b:       excitation source    N^2 x Ni
            * N:       DOI size             1x1
            * Ni:      number of incidences 1x1
            * Xr:      contrast function    NxN
            * max_it:  number of iterations (integer number)
            * TOL:     error tolerance      

            output:  
            * J:       current density N^2xNi
        
        '''

        Jo = np.zeros((Nx*Ny,Ni),dtype=complex) # initial guess
        ro = self.__fft_A(Jo,Z,Nx,Ny,Ni,Xr)-b # ro = A.Jo - b;
        go = self.__fft_AH(ro,Z,Nx,Ny,Ni,Xr) # Complex conjugate AH
        po = -go
        error_res = np.zeros(max_it)

        for n in range(max_it):
    
            alpha = -1*(np.sum(np.conj(self.__fft_A(po,Z,Nx,Ny,Ni,Xr))
                              *(self.__fft_A(Jo,Z,Nx,Ny,Ni,Xr)-b),axis=0)
                        / LA.norm(np.reshape(self.__fft_A(po,Z,Nx,Ny,Ni,Xr),
                                             (Nx*Ny*Ni,1),order='F'),
                                  ord='fro')**2) # 1 x Ni
                                                  
            J = Jo + np.tile(alpha,(Nx*Ny,1))*po 
            r = self.__fft_A(J,Z,Nx,Ny,Ni,Xr)-b
            g = self.__fft_AH(r,Z,Nx,Ny,Ni,Xr) 
    
            error = LA.norm(r)/LA.norm(b) # error tolerance
            error_res[n] = error
            
            if PRINT_CONVERGENCE:
                print('Iteration %d ' %(n+1) + ' - Error: %.3e' %error)
            
            if error < TOL: # stopping criteria
                break
    
            beta = np.sum(np.conj(g)*(g-go),axis=0)/np.sum(np.abs(go)**2,axis=0) 
            p    = -g + np.tile(beta,(Nx*Ny,1))*po 
        
            po = p 
            Jo = J 
            go = g 

        return J,n,error_res

    def __fft_A(self,J,Z,Nx,Ny,Ni,Xr):
        """ Compute Matrix-vector product by using two-dimensional FFT."""

        J = np.reshape(J,(Nx,Ny,Ni))
        Z = np.tile(Z[:,:,np.newaxis],(1,1,Ni))
        e = fft.ifft2(fft.fft2(Z,axes=(0,1))
                      * fft.fft2(J,axes=(0,1),s=(2*Nx-1,2*Ny-1)),axes=(0,1))
        e = e[:Nx,:Ny,:]
        e = np.reshape(e,(Nx*Ny,Ni))
        e = np.reshape(J,(Nx*Ny,Ni)) + np.tile(Xr.reshape((-1,1)),(1,Ni))*e

        return e

    def __fft_AH(self,J,Z,Nx,Ny,Ni,Xr):
        """ Compute Matrix-vector product by using two-dimensional FFT*
            complex conjugate operator."""

        J = np.reshape(J,(Nx,Ny,Ni))
        Z = np.tile(Z[:,:,np.newaxis],(1,1,Ni))
        e = fft.ifft2(fft.fft2(np.conj(Z),axes=(0,1))
                      *fft.fft2(J,axes=(0,1),s=(2*Nx-1,2*Ny-1)),axes=(0,1))
        e = e[:Nx,:Ny,:]
        e = np.reshape(e,(Nx*Ny,Ni))
        e = (np.reshape(J,(Nx*Ny,Ni)) 
             + np.conj(np.tile(Xr.reshape((-1,1)),(1,Ni)))*e)

        return e

    def __compute_model_parameters(self,Nx,Ny):
        
        self.Nx, self.Ny = Nx, Ny
        self.xmin, self.xmax = get_bounds(self.domain.Lx)
        self.ymin, self.ymax = get_bounds(self.domain.Ly)
        self.dx, self.dy = self.domain.Lx/Nx, self.domain.Ly/Ny
        x, y = get_domain_coordinates(self.dx,self.dy,self.xmin,self.xmax,
                                      self.ymin,self.ymax)
        self.domain.set_coordinates(x,y)
        self.Ei = self.get_incident_field(Nx,Ny,self.f)
        self.lambda_b = get_wavelength(self.f,epsilon_r=self.epsilon_rb)
        self.kb = get_wavenumber(self.f,epsilon_r=self.epsilon_rb)
        self.xm, self.ym = get_coordinates(self.domain.R_obs,self.domain.M)
        self.GS = get_greenfunction(self.xm,self.ym,x,y,self.kb)
        self.omega = 2*np.pi*self.f

    def save_model_configuration(self,file_name,file_path='',Nx=None,Ny=None,
                                 frequencies=None):
        
        if frequencies is None and self.f is None:
            print("SAVE_MODEL_CONFIGURATION ERROR: none frequency hasn't been given!")
            sys.exit()
        if Nx is None and Ny is None and (self.Nx is None or self.Ny is None):
            print('SAVE_MODEL_CONFIGURATION ERROR: image size must be given!')
            sys.exit()
        
        UPDATE_FLAG = False
        if frequencies is not None:
            self.f = frequencies
            UPDATE_FLAG = True
        if Nx is not None and Ny is not None:
            UPDATE_FLAG = True
        elif Nx is None and Ny is not None:
            Nx = self.Nx
            UPDATE_FLAG = True
        elif Nx is not None and Ny is None:
            Ny = self.Ny
            UPDATE_FLAG = True
        
        if UPDATE_FLAG:
            self.__compute_model_parameters(Nx,Ny)
                
        data = {
            'model_name':self.model_name,
            'Lx':self.domain.Lx, 'Ly':self.domain.Ly,
            'radius_observation':self.domain.R_obs,
            'number_measurements':self.domain.M,
            'number_sources':self.domain.L,
            'dx':self.dx,
            'dy':self.dy,
            'x':self.domain.x, 'y':self.domain.y,
            'incident_field':self.Ei,
            'green_function_s':self.GS,
            'wavelength':self.lambda_b,
            'wavenumber':self.kb,
            'angular_frequency':self.omega,
            'relative_permittivity_background':self.epsilon_rb,
            'conductivity_background':self.sigma_b,
            'measurement_coordinates_x':self.xm,
            'measurement_coordinates_y':self.ym,
            'frequency':self.f,
            'Nx':self.Nx, 'Ny':self.Ny,
            'incident_field_magnitude':self.E0
        }
        
        with open(file_path + file_name,'wb') as datafile:
            pickle.dump(data,datafile)
        
    def __save_simulation_data(self,Es,epsilon_r,sigma,filename,filepath,Et=None):
        """ Save simulation data in a pickle file."""
        
        data = {
            'scattered_field':Es,
            'total_field':Et,
            'relative_permittivity_map':epsilon_r,
            'conductivity_map':sigma,
            'maximum_number_iterations':self.max_it,
            'error_tolerance':self.TOL
        }
        
        with open(filepath + filename,'wb') as datafile:
            pickle.dump(data,datafile)

    def plot_total_field(self,file_name=None,file_path='',file_format='png',
                         frequency_index=None,source_index=None):
        """ Plot the fields from last simulation."""
        
        if isinstance(self.f,float):
            et = np.reshape(self.Et,(self.Nx,self.Ny,self.domain.L))
        else:
            et = np.reshape(self.Et,(self.Nx,self.Ny,self.domain.L,self.f.size))
        
        xmin, xmax = get_bounds(self.domain.Lx)
        ymin, ymax = get_bounds(self.domain.Ly)
        lambda_b = get_wavelength(self.f,epsilon_r=self.epsilon_rb)
        
        xmin, xmax = xmin/lambda_b, xmax/lambda_b
        ymin, ymax = ymin/lambda_b, ymax/lambda_b
      
        if frequency_index is not None and source_index is not None:
            
            plt.imshow(np.abs(et[:,:,source_index,frequency_index]),
                       extent=[xmin[frequency_index],xmax[frequency_index],
                               ymin[frequency_index],ymax[frequency_index]])
            plt.title('Intern field, Source = %d' %(source_index+1) 
                      + ', Frequency = %.3f' %(self.f[frequency_index]/1e9) 
                      + ' [GHz]')
            
            plt.xlabel(r'x [$\lambda_b$]')
            plt.ylabel(r'y [$\lambda_b$]')
            cbar = plt.colorbar()
            cbar.set_label(r'$|E_z^t|$ [V/m]')
             
        elif source_index is not None and isinstance(self.f,float):
            
            plt.imshow(np.abs(et[:,:,source_index]),
                       extent=[xmin,xmax,ymin,ymax])
            plt.title('Intern field, Source = %d' %(source_index+1) 
                       + ', Frequency = %.3f' %(self.f/1e9) + ' [GHz]')
            
            plt.xlabel(r'x [$\lambda_b$]')
            plt.ylabel(r'y [$\lambda_b$]')
            cbar = plt.colorbar()
            cbar.set_label(r'$|E_z^t|$ [V/m]')
            
        elif source_index is None and isinstance(self.f,float):
            
            nrow = np.floor(np.sqrt(self.domain.L)).astype(int)
            ncol = np.ceil(self.domain.L/nrow).astype(int)
            fig = plt.figure(figsize=(int(ncol*3),nrow*3))
            for ifig in range(self.domain.L):
                ax = fig.add_subplot(nrow, ncol, ifig+1)
                fig.subplots_adjust(left=.125, bottom=.1, right=.9, top=.9, 
                                    wspace=.9, hspace=.2)
                temp = ax.imshow(np.abs(et[:,:,ifig]),extent=[xmin,xmax,
                                                              ymin,ymax])
                ax.set_xlabel(r'x [$\lambda_b$]')
                ax.set_ylabel(r'y [$\lambda_b$]')
                cbar = plt.colorbar(ax=ax,mappable=temp,fraction=0.046,pad=0.04)
                cbar.set_label(r'$|E_z^t|$ [V/m]')
                ax.set_title('Intern field, Source = %d' %(ifig+1) 
                             + ', \nFrequency = %.3f' %(self.f/1e9) + ' [GHz]')
                
        elif source_index is None and frequency_index is None:
            
            fig = []
            nrow = np.floor(np.sqrt(self.domain.L)).astype(int)
            ncol = np.ceil(self.domain.L/nrow).astype(int)
            for f in range(self.f.size):
                fig.append(plt.figure(figsize=(int(ncol*3),nrow*3)))
                for ifig in range(self.domain.L):
                    ax = fig[-1].add_subplot(nrow, ncol, ifig+1)
                    fig[-1].subplots_adjust(left=.125, bottom=.1, right=.9, 
                                            top=.9, wspace=.9, hspace=.2)
                    temp = ax.imshow(np.abs(et[:,:,ifig,f]),extent=
                                     [xmin[f],xmax[f],ymin[f],ymax[f]])
                    ax.set_xlabel(r'x [$\lambda_b$]')
                    ax.set_ylabel(r'y [$\lambda_b$]')
                    cbar = plt.colorbar(ax=ax,mappable=temp,fraction=0.046,
                                        pad=0.04)
                    cbar.set_label(r'$|E_z^t|$ [V/m]')
                    ax.set_title('Intern field, Source = %d' %(ifig+1) 
                                 + ', \nFrequency = %.3f' %(self.f[f]/1e9) 
                                 + ' [GHz]')
        
        else:
            
            nrow = np.floor(np.sqrt(self.domain.L)).astype(int)
            ncol = np.ceil(self.domain.L/nrow).astype(int)
            fig = plt.figure(figsize=(int(ncol*3),nrow*3))
            for ifig in range(self.domain.L):
                ax = fig.add_subplot(nrow, ncol, ifig+1)
                temp = ax.imshow(np.abs(et[:,:,ifig,frequency_index]),
                                 extent=[xmin[frequency_index],
                                         xmax[frequency_index],
                                         ymin[frequency_index],
                                         ymax[frequency_index]])
                fig.subplots_adjust(left=.125, bottom=.1, right=.9, 
                                    top=.9, wspace=.9, hspace=.2)
                ax.set_xlabel(r'x [$\lambda_b$]')
                ax.set_ylabel(r'y [$\lambda_b$]')
                cbar = plt.colorbar(ax=ax,mappable=temp,fraction=0.046,pad=0.04)
                cbar.set_label(r'$|E_z^t|$ [V/m]')
                ax.set_title('Intern field, Source = %d' %(ifig+1) 
                             + ', \nFrequency = %.3f' %(self.f[frequency_index]
                                                      /1e9) 
                             + ' [GHz]')
        
        if ((frequency_index is not None and source_index is not None)
            or (source_index is not None and isinstance(self.f,float))):
            
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_path + file_name + '.' + file_format,
                            format=file_format)
                plt.close()
        
        elif isinstance(fig,list):
            if file_name is None:
                for i in range(len(fig)):
                    plt.show(fig[i])
            else:
                for i in range(len(fig)):
                    fig[i].savefig(file_path + file_name +'_%d' %i + '.' 
                                   + file_format, format=file_format)
                plt.close()
        
        else:
            
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_path + file_name + '.' + file_format,
                            format=file_format)
                plt.close()

    def draw_setup(self,epsr=None,sig=None,file_name=None,file_path='',
                   file_format='png'):
        """ Draw domain, sources and probes."""
        
        if epsr is None and sig is None:
            Nx, Ny = 100, 100
        elif epsr is not None:
            Nx, Ny = epsr.shape
        else:
            Nx, Ny = sig.shape
        
        dx, dy = self.domain.Lx/Nx, self.domain.Ly/Ny
        min_radius = np.sqrt((self.domain.Lx/2)**2+(self.domain.Ly/2)**2)
        
        if epsr is None:
            epsr = self.epsilon_rb*np.ones((Nx,Ny))
            
        if sig is None:
            sig = self.sigma_b*np.ones((Nx,Ny))
            
        if self.domain.R_obs > min_radius:
            xmin,xmax = -1.05*self.domain.R_obs, 1.05*self.domain.R_obs
            ymin,ymax = -1.05*self.domain.R_obs, 1.05*self.domain.R_obs
        else:
            xmin, xmax = -self.domain.Lx/2, self.domain.Lx/2
            ymin, ymax = -self.domain.Ly/2, self.domain.Ly/2
            
        xm, ym = get_coordinates(self.domain.R_obs,self.domain.M)
        xl, yl = get_coordinates(self.domain.R_obs,self.domain.L)
        x, y = get_domain_coordinates(dx,dy,xmin,xmax,ymin,ymax)    
        
        epsilon_r = self.epsilon_rb*np.ones(x.shape)
        sigma = self.sigma_b*np.ones(x.shape)
        
        epsilon_r[np.ix_(np.logical_and(x[0,:] > -self.domain.Lx/2,
                                        x[0,:] < self.domain.Lx/2),
                         np.logical_and(y[:,0] > -self.domain.Ly/2,
                                        y[:,0] < self.domain.Ly/2))] = epsr
        
        sigma[np.ix_(np.logical_and(x[0,:] > -self.domain.Lx/2,
                                    x[0,:] < self.domain.Lx/2),
                     np.logical_and(y[:,0] > -self.domain.Ly/2,
                                    y[:,0] < self.domain.Ly/2))] = sig
        
        fig = plt.figure(figsize=(10,4))
        fig.subplots_adjust(left=.125, bottom=.1, right=.9, top=.9, wspace=.5, 
                            hspace=.2)
        
        ax = fig.add_subplot(1,2,1)
        
        if isinstance(self.f,float):
            lambda_b = get_wavelength(self.f,epsilon_r=self.epsilon_rb)
            
            im1 = ax.imshow(epsilon_r,extent=[xmin/lambda_b,xmax/lambda_b,
                                              ymin/lambda_b,ymax/lambda_b])
            
            ax.plot(np.array([-self.domain.Lx/2/lambda_b,-self.domain.Lx/2/lambda_b,
                              self.domain.Lx/2/lambda_b,self.domain.Lx/2/lambda_b,
                              -self.domain.Lx/2/lambda_b]),
                    np.array([-self.domain.Ly/2/lambda_b,self.domain.Ly/2/lambda_b,
                              self.domain.Ly/2/lambda_b,-self.domain.Ly/2/lambda_b,
                              -self.domain.Ly/2/lambda_b]),'k--')
            
            lg_m, = ax.plot(xm/lambda_b,ym/lambda_b,'ro',label='Probe')
            lg_l, = ax.plot(xl/lambda_b,yl/lambda_b,'go',label='Source')
            
            ax.set_xlabel(r'x [$\lambda_b$]')
            ax.set_ylabel(r'y [$\lambda_b$]')
            
        else:
            im1 = ax.imshow(epsilon_r,extent=[xmin,xmax,ymin,ymax])
            
            ax.plot(np.array([-self.domain.Lx/2,-self.domain.Lx/2,
                              self.domain.Lx/2,self.domain.Lx/2,
                              -self.domain.Lx/2]),
                    np.array([-self.domain.Ly/2,self.domain.Ly/2,
                              self.domain.Ly/2,-self.domain.Ly/2,
                              -self.domain.Ly/2]),'k--')

            lg_m, = ax.plot(xm,ym,'ro',label='Probe')
            lg_l, = ax.plot(xl,yl,'go',label='Source')
            
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            
        plt.legend(handles=[lg_m,lg_l],loc='upper right')
        cbar = fig.colorbar(im1,fraction=0.046,pad=0.04)
        cbar.set_label(r'$\epsilon_r$')
        ax.set_title('Relative Permittivity')

        ax = fig.add_subplot(1,2,2)

        if isinstance(self.f,float):
            lambda_b = get_wavelength(self.f,epsilon_r=self.epsilon_rb)
            
            im1 = ax.imshow(sigma,extent=[xmin/lambda_b,xmax/lambda_b,
                                          ymin/lambda_b,ymax/lambda_b])
            
            ax.plot(np.array([-self.domain.Lx/2/lambda_b,-self.domain.Lx/2/lambda_b,
                              self.domain.Lx/2/lambda_b,self.domain.Lx/2/lambda_b,
                              -self.domain.Lx/2/lambda_b]),
                    np.array([-self.domain.Ly/2/lambda_b,self.domain.Ly/2/lambda_b,
                              self.domain.Ly/2/lambda_b,-self.domain.Ly/2/lambda_b,
                              -self.domain.Ly/2/lambda_b]),'k--')
            
            lg_m, = ax.plot(xm/lambda_b,ym/lambda_b,'ro',label='Probe')
            lg_l, = ax.plot(xl/lambda_b,yl/lambda_b,'go',label='Source')
            
            ax.set_xlabel(r'x [$\lambda_b$]')
            ax.set_ylabel(r'y [$\lambda_b$]')
            
        else:
            im1 = ax.imshow(sigma,extent=[xmin,xmax,ymin,ymax])
            
            ax.plot(np.array([-self.domain.Lx/2,-self.domain.Lx/2,
                              self.domain.Lx/2,self.domain.Lx/2,
                              -self.domain.Lx/2]),
                    np.array([-self.domain.Ly/2,self.domain.Ly/2,
                              self.domain.Ly/2,-self.domain.Ly/2,
                              -self.domain.Ly/2]),'k--')

            lg_m, = ax.plot(xm,ym,'ro',label='Probe')
            lg_l, = ax.plot(xl,yl,'go',label='Source')
            
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            
        cbar = fig.colorbar(im1,fraction=0.046,pad=0.04)
        cbar.set_label(r'$\sigma$ [S/m]')
        ax.set_title('Conductivity')
        plt.legend(handles=[lg_m,lg_l],loc='upper right')
        
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
            plt.close()

def get_angles(n_samples):
    """ Compute angles [rad] in a circular array of points equaly spaced."""
    return np.arange(0,2*np.pi,2*np.pi/n_samples)

def get_coordinates(radius,n_samples):
    """ Compute coordinates of points in a circular array equaly spaced."""
    phi = get_angles(n_samples)
    return radius*np.cos(phi), radius*np.sin(phi)

def get_bounds(length):
    """ Compute the standard bound coordinates."""
    return -length/2, length/2

def get_domain_coordinates(dx,dy,xmin,xmax,ymin,ymax):
    """ Return the meshgrid of the image domain."""
    return np.meshgrid(np.arange(xmin+.5*dx,xmax+.5*dx,dx),
                       np.arange(ymin+.5*dy,ymax+.5*dy,dy))
    
def get_wavelength(frequencies,epsilon_r=1.,mu_r=1.):
    """ Compute wavelength [m]."""
    return 1/np.sqrt(epsilon_r*ct.epsilon_0*mu_r*ct.mu_0)/frequencies

def get_wavenumber(frequencies,epsilon_r=1.,mu_r=1.,sigma=.0):
    """ Compute the wavenumber."""
    return 2*np.pi*np.sqrt(epsilon_r*ct.epsilon_0*mu_r*ct.mu_0)*frequencies

def get_contrast_map(epsilon_r,sigma,epsilon_rb,sigma_b,omega):
    """ Compute the contrast function for a given image represented by the
    relative permittivity and conductivity."""
    
    if isinstance(omega,float):
        return ((epsilon_r - 1j*sigma/omega/ct.epsilon_0)
                /(epsilon_rb - 1j*sigma_b/omega/ct.epsilon_0) - 1)
    
    else:
        Xr = np.zeros((epsilon_r.shape[0],epsilon_r.shape[1],omega.size),
                      dtype=complex)
        for f in range(omega.size):
            Xr[:,:,f] = ((epsilon_r - 1j*sigma/omega[f]/ct.epsilon_0)
                         /(epsilon_rb - 1j*sigma_b/omega[f]/ct.epsilon_0) 
                         - 1)
        return Xr
    
def get_greenfunction(xm,ym,x,y,kb):
    """ Compute the Green function."""

    Nx, Ny = x.shape
    M = xm.size
    dx, dy = x[0,1]-x[0,0], y[1,0]-y[0,0]
    an = np.sqrt(dx*dy/np.pi) # radius of the equivalent circle
    
    if isinstance(kb,float):
        MONO_FREQUENCY = True
    else:
        MONO_FREQUENCY = False
        
    xg = np.tile(xm.reshape((-1,1)),(1,Nx*Ny))
    yg = np.tile(ym.reshape((-1,1)),(1,Nx*Ny))
    R = np.sqrt((xg-np.tile(np.reshape(x,(Nx*Ny,1)).T,(M,1)))**2 
                + (yg-np.tile(np.reshape(y,(Nx*Ny,1)).T,(M,1)))**2)
    
    if MONO_FREQUENCY: 
        G = (-1j*kb*np.pi*an/2*spc.jv(1,kb*an) * spc.hankel2(0,kb*R))
        G[R==0] = 1j/2*(np.pi*kb*an*spc.hankel2(1,kb*an)-2j)
    
    else:
        G = np.zeros((M,Nx*Ny,kb.size),dtype=complex)
        for f in range(kb.size):
            aux = (-1j*kb[f]*np.pi*an/2*spc.jv(1,kb[f]*an)*spc.hankel2(0,kb[f]*R))
            aux[R==0] = 1j/2*(np.pi*kb[f]*an*spc.hankel2(1,kb[f]*an)-2j)
            G[:,:,f] = aux
    
    return G

def load_scattered_field(file_name,file_path=''):
    with open(file_path + file_name,'rb') as datafile:
        data = pickle.load(datafile)
    return data['scattered_field']

def load_total_field(file_name,file_path=''):
    with open(file_path + file_name,'rb') as datafile:
        data = pickle.load(datafile)
    return data['total_field']