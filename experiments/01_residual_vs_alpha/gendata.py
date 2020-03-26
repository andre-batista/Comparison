import sys
import numpy as np

sys.path.insert(1,'..\..\library')
import analytical as ana

proportion = .5
frequency = 1e9
Nsources = 10
Nsamples = 10
epsilon_rd = 2.
Nx, Ny = 50, 50
file_name = 'ana_1GHz_e2_d0'
file_path = ".\data\\"

ana.get_data(proportion=proportion,
             frequency=frequency,
             Nsources=Nsources,
             Nsamples=Nsamples,
             epsilon_rd=epsilon_rd,
             Nx=Nx,Ny=Ny,
             file_name=file_name,
             file_path=file_path)