import numpy as np
import math
from numpy import *
import matplotlib.pyplot as mpl
import h5py
import sys
import os
from matplotlib.pyplot import *
import matplotlib as mpll
import Parameters
import Models
import Operation
import SC_channel
import Observables
import Mag_channel

#########################

symmetry_layer = True    ### also do layer-symmetry on chic0, chi0
symmetry_C4 = True       ### also do C4-symmetry on chic0, chi0
phSymmetry = False
symmetry_wn  = True
symmetry_transpose = True
# symmetrize G4

########################

isLattice = True
# use chiM = chi0M

########################

write_data_file = True
Num = 100               ##  max = NwG4*Nc*nOrb*nOrb
# generate leading and leadingspm file

########################

build_sym_kernel=False
# symmetry on pm

########################

Ts = [0.2,0.175,0.15,0.135,0.125,0.11,0.1,0.09,0.08,0.06]
#Ts = [0.08]
model = 'trilayer'

for iT in range(len(Ts)):
    T = Ts[iT]

    file_tp = '../T='+str(T)+'/dca_tp.hdf5'
    #file_tp = './T='+str(T)+'/dca_tp_mag_qpi0'+'.hdf5' #PH channel

    if (os.path.exists(file_tp)):
        print ("\n =================================\n")
        print ("T =", T)
        ######################################################  read parameters
        params_model = Models.readin_model(model, file_tp)
        params_data  = Parameters.readData(model, file_tp)
      
        if params_data['vertex_channel'] == 'PARTICLE_PARTICLE_UP_DOWN':
            ######################################################  build chic0
            chic0,chic0M = Observables.calcChi0Cluster(model,symmetry_layer,symmetry_C4,symmetry_transpose,params_data)

            ######################################################  build Gamma
            G4r,Gamma,GammaM = Observables.calcGammaIrr(model,chic0M,symmetry_layer,symmetry_C4,phSymmetry,symmetry_wn,symmetry_transpose,params_data)

            ######################################################  build chi0
            nkfine = 100  # coarse-grained number of k' for chi0
            chi0,chi0D,chi0D2,chi0XS,chi0XS2,chi0DXY,chi0DXY2,chi0M = Observables.buildChi0Lattice(model,symmetry_layer,symmetry_C4,symmetry_transpose,params_data,params_model,nkfine)

            ######################################################  build Kernel matrix
            pm = SC_channel.buildKernelMatrix(model,params_data,chic0M,chi0M,Gamma,GammaM,isLattice,build_sym_kernel)

            ######################################################  solve kernel matrix
            SC_channel.calcKernelEigenValues(model,pm,params_data,T,Num,write_data_file)

            ###################################################### solve lattice sc susceptibility
            #GammaRed = SC_channel.calcReducibleLatticeVertex(params_data,pm,GammaM)
            #SC_channel.calcLatticeSCSus(params_data,model,GammaRed,chi0,chi0D,chi0D2,chi0XS,chi0XS2,chi0DXY,chi0DXY2,T)   
            #SC_channel.calcLatticeSCSus_int(params_data,model,GammaRed,chi0,chi0D,chi0D2,chi0XS,chi0XS2,chi0DXY,chi0DXY2,T) #solve interacting part     

        elif params_data['vertex_channel'] == 'PARTICLE_HOLE_MAGNETIC':
            ###################################################### calculate cluster mag susceptibility
            Mag_channel.calcClusterSus(params_data,T)




