import numpy as np
import math
from numpy import *
import matplotlib.pyplot as mpl
import h5py
import sys
import os
from matplotlib.pyplot import *
import matplotlib as mpll
import Operation

# read basic parameters from the data and the cluster one and two particle Green's function
def readData(model,f):

    vertex_channels = ["PARTICLE_PARTICLE_UP_DOWN",          \
                       "PARTICLE_HOLE_CHARGE",               \
                       "PARTICLE_HOLE_MAGNETIC",             \
                       "PARTICLE_HOLE_LONGITUDINAL_UP_UP",   \
                       "PARTICLE_HOLE_LONGITUDINAL_UP_DOWN", \
                       "PARTICLE_HOLE_TRANSVERSE"]

    if isinstance(f, str):
        f = h5py.File(f, 'r')
        close_file = True
    else:
        close_file = False
    params = {}
    
    params['cluster'] = array(f["domains"]["CLUSTER"]["REAL_SPACE"]["super-basis"]["data"])
    print("Cluster vectors:", params['cluster'], '\n')
    params['iwm'] = array(f['parameters']['four-point']['frequency-transfer'])[0]  # transferred frequency in units of 2*pi*temp
    print("Transferred frequency iwm = ", params['iwm'], '\n')
    params['qchannel'] = array(f['parameters']['four-point']['momentum-transfer'])
    print("Transferred momentum q = ", params['qchannel'], '\n')
    params['invT'] = array(f['parameters']['physics']['beta'])[0]
    print("Inverse temperature = ", params['invT'], '\n')
    params['temp'] = 1.0 / params['invT']

    params['fill'] = array(f['parameters']['physics']['density'])[0]
    print("desired filling = ", params['fill'] , '\n')
    params['dens']  = array(f['DCA-loop-functions']['density']['data'])
    print("actual filling:", params['dens'] , '\n')
    params['nk']  = array(f['DCA-loop-functions']['n_k']['data'])
    params['sign']  = array(f['DCA-loop-functions']['sign']['data'])
    print("sign:", params['sign'] , '\n')
    params['orbital']  = array(f['DCA-loop-functions']['orbital-occupancies']['data'])
    print("orbital.shape:", params['orbital'].shape , '\n')
    # ('orbital.shape:', (1, 2, 3))

    ### generally extract the filling in orbitals
    # norbs = self.orbital.shape[2]
    # for ii in range(self.orbital.shape[0]):
    #    print('\n params['sigmaarray')

    #    for iorb in range(norbs):
    #        print("orbital ", iorb, " filling = ", self.orbital[ii, 0, iorb] \
    #              + self.orbital[ii, 1, iorb], '\n')

    #if self.model == 'square' or self.model == 'square_txty':
    #    print("layer/orbital occupancy:", self.orbital[0], '\n')
    #    print("filling =", self.orbital[0, 0, 0] + self.orbital[0, 1, 0], '\n')
    #elif self.model == 'bilayer' or self.model == 'dsmodel':
    #    print("layer/orbital occupancy:", self.orbital[self.orbital.shape[0] - 1], '\n')
    #    print("layer/orbital 1 filling =",
    #          self.orbital[self.orbital.shape[0] - 1, 0, 0] + self.orbital[self.orbital.shape[0] - 1, 1, 0], '\n')
    #    print("layer/orbital 2 filling =",
    #          self.orbital[self.orbital.shape[0] - 1, 0, 1] + self.orbital[self.orbital.shape[0] - 1, 1, 1], '\n')
    if model == 'trilayer':# or self.model == 'sdsmodel' or self.model == 'ddpmodel' or self.model == 'Emery':
        print("orbital occupancy:", params['orbital'][params['orbital'].shape[0] - 1], '\n')
        print("layer/orbital 1 filling =",
              params['orbital'][params['orbital'].shape[0] - 1, 0, 0] + params['orbital'][params['orbital'].shape[0] - 1, 1, 0], '\n')
        print("layer/orbital 2 filling =",
              params['orbital'][params['orbital'].shape[0] - 1, 0, 1] + params['orbital'][params['orbital'].shape[0] - 1, 1, 1], '\n')
        print("layer/orbital 3 filling =",
              params['orbital'][params['orbital'].shape[0] - 1, 0, 2] + params['orbital'][params['orbital'].shape[0] - 1, 1, 2], '\n')

    elif model == 'Eg_bilayer' or model == 'La3Ni2O7':
        print("layer/orbital occupancy:", params['orbital'][params['orbital'].shape[0] - 1], '\n')
        print("layer/orbital 1 filling =",
              params['orbital'][params['orbital'].shape[0] - 1, 0, 0] + params['orbital'][params['orbital'].shape[0] - 1, 1, 0], '\n')
        print("layer/orbital 2 filling =",
              params['orbital'][params['orbital'].shape[0] - 1, 0, 1] + params['orbital'][params['orbital'].shape[0] - 1, 1, 1], '\n')
        print("layer/orbital 3 filling =",
              params['orbital'][params['orbital'].shape[0] - 1, 0, 2] + params['orbital'][params['orbital'].shape[0] - 1, 1, 2], '\n')
        print("layer/orbital 4 filling =",
              params['orbital'][params['orbital'].shape[0] - 1, 0, 3] + params['orbital'][params['orbital'].shape[0] - 1, 1, 3], '\n')

    params['sigmaarray'] = array(f['DCA-loop-functions']['L2_Sigma_difference']['data'])
    print("L2_Sigma_difference =", params['sigmaarray'], '\n')
    for ver in vertex_channels:
        if 'G4_' + ver in f['functions'].keys():
            params['vertex_channel'] = ver
            print("Vertex channel = ", params['vertex_channel'], '\n')
            params['GG'] = array(f['functions']['G4_' + params['vertex_channel']]['data'])
            print("original G4.shape=", params['GG'].shape, '\n')

    params['G4Re'] = array(f['functions']['G4_' + params['vertex_channel']]['data'])[0, 0, :, :, :, :, :, :, :, :, 0]
    params['G4Im'] = array(f['functions']['G4_' + params['vertex_channel']]['data'])[0, 0, :, :, :, :, :, :, :, :, 1]
    params['G4'] =  params['G4Re'] + 1j *  params['G4Im']
    # print ("Extracted G4.shape=", self.G4.shape ,'\n')

    # Now read the cluster Green's function, only need spin up component
    params['GRe'] = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:, :, 0, :, 0, :, 0]
    params['GIm'] = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:, :, 0, :, 0, :, 1]
    params['Green'] = params['GRe'] + 1j * params['GIm']
    # print ("Extracted G.shape=", self.Green.shape,'\n')

    params['GRet'] = array(f['functions']['cluster_greens_function_G_k_t']['data'])[:, :, 0, :, 0, :, 0]
    params['GImt'] = array(f['functions']['cluster_greens_function_G_k_t']['data'])[:, :, 0, :, 0, :, 1]
    params['Greenkt'] = params['GRet'] + 1j * params['GImt']
    params['Greenrt'] = array(f['functions']['cluster_greens_function_G_r_t']['data'])[:, :, 0, :, 0, :]

    # Now read the self-energy
    params['s'] = np.array(f['functions']['Self_Energy']['data'])
    # print ("sigma.shape=", s.shape,'\n')
    params['sigmaoriginal'] = params['s'][:, :, 0, :, 0, :, 0] + 1j * params['s'][:, :, 0, :, 0, :, 1]

    # Now load frequency data
    params['wn'] = np.array(f['domains']['frequency-domain']['elements'])
    params['wnSet'] = np.array(f['domains']['vertex-frequency-domain (COMPACT)']['elements'])
    print('G4\'s iwn = ', params['wnSet'][24], '\n')

    # Now read the K-vectors
    params['Kvecs'] = array(f['domains']['CLUSTER']['MOMENTUM_SPACE']['elements']['data'])
    print("K-vectors: ", params['Kvecs'], '\n')

    # Now read other Hubbard parameters
    params['mu'] = np.array(f['DCA-loop-functions']['chemical-potential']['data'])[0]
    params['nOrb'] = params['Green'].shape[2]
    params['NwG4'] = params['G4'].shape[0]
    params['Nc'] = params['Green'].shape[1]
    params['NwG'] = params['Green'].shape[0]
    params['NtG'] = params['Greenkt'].shape[0]
    params['nt'] = params['Nc'] * params['NwG4'] * params['nOrb'] * params['nOrb']

    print('nw:', params['wnSet'].shape[0])
    print("NwG4: ", params['NwG4'])
    print("NtG: ", params['NtG'])
    print("NwG : ", params['NwG'])
    print("Nc  : ", params['Nc'])
    print("nOrb: ", params['nOrb'])

    params['NwTP'] = 2 * np.array(f['parameters']['domains']['imaginary-frequency']['four-point-fermionic-frequencies'])[0]
    params['iQ'] = Operation.K_2_iK(params,params['qchannel'][0], params['qchannel'][1])
    print ("Index of transferred momentum: ", params['iQ'],'\n')

    # indices of G4 and G at iwn=2*pi*T
    params['iwG40'] = params['NwG4'] / 2
    params['iwG0']  = params['NwG'] / 2

    f.close()
    # Now remove vacuum part of charge G4
    if (params['vertex_channel']=="PARTICLE_HOLE_CHARGE"):
        if (params['qchannel'][0] == 0) & (params['qchannel'][1] == 0):
            for ik1 in range(params['Nc']):
                for ik2 in range(params['Nc']):
                    for iw1 in range(params['NwG4']):
                        for iw2 in range(params['NwG4']):
                            for l1 in range(params['nOrb']):
                                for l2 in range(params['nOrb']):
                                    for l3 in range(params['nOrb']):
                                        for l4 in range(params['nOrb']):
                                            iw1Green = iw1 - self.iwG40 + self.iwG0
                                            iw2Green = iw2 - self.iwG40 + self.iwG0
                                            params['G4'][iw1,ik1,iw2,ik2,l1,l2,l3,l4] -= 2.0 * params['Green'][iw1Green,ik1,l1,l2] \
                                                                                        * params['Green'][iw2Green,ik2,l4,l3]
    return params 
