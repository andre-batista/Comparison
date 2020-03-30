import sys
import numpy as np

sys.path.insert(1,'..\..\library')
import analytical as ana

proportion = .5
frequency = 1e9
Nsources = 15
Nsamples = 15
epsilon_rd = 2.
file_name = 'ana_1GHz_e2_d0'
file_path = ".\data\\"
delta = 1e-1

ana.get_data(proportion=proportion,
             frequency=frequency,
             Nsources=Nsources,
             Nsamples=Nsamples,
             epsilon_rd=epsilon_rd,
             file_name=file_name,
             file_path=file_path,
             delta=delta)