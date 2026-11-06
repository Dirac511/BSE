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

#########################################################
def calcClusterSus(paramsd,Tval):
    '''
    For non-SC channels
    '''
    print(" ")
    print("Calculate cluster susceptibility via G4 summation:")
    Nc = paramsd['Nc'];nOrb = paramsd['nOrb']
    G4 = paramsd['G4']
    chi_c0 = np.zeros((nOrb, nOrb), dtype='complex')
    G4k = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')

    for l1 in range(nOrb):
        for l2 in range(nOrb):
            chi_c0[l1, l2] = sum(G4[:, :, :, :, l1, l1, l2, l2])
            chi_c0[l1, l2] /= Nc * paramsd['invT']
            # print(l1,l3,': ',chi_c0[l1,l3])

    for l1 in range(nOrb):
        for l2 in range(nOrb):
            for l3 in range(nOrb):
                for l4 in range(nOrb):
                    G4k[l1, l2, l3, l4] = sum(G4[:, :, :, :, l1, l2, l3, l4])

    a = np.array([1 / sqrt(2), 0, 1 / sqrt(2), 0])  # Bonding   x
    b = np.array([0, 1 / sqrt(2), 0, 1 / sqrt(2)])  # Bonding   z
    c = np.array([1 / sqrt(2), 0, -1 / sqrt(2), 0])  # antiBonding  x
    d = np.array([0, 1 / sqrt(2), 0, -1 / sqrt(2)])  # AntiBonding  z

    basis = [a] + [b] + [c] + [d]
    # print(basis)
    M1, M2, M3, M4, Mat = [], [], [], [], []
    for i in range(0, len(basis)):
        for j in range(0, len(basis)):
            mij = np.outer(basis[i], basis[j])
            if i == 0:
                M1.append(mij)
            if i == 1:
                M2.append(mij)
            if i == 2:
                M3.append(mij)
            if i == 3:
                M4.append(mij)
    Matrix = np.array([M1, M2, M3, M4], dtype=object)

    m1 = M1[0] + M1[1] + M2[0] + M2[1]
    m2 = M3[2] + M3[3] + M4[2] + M4[3]

    Mat = np.array([m1, m2], dtype=object)

    G4K = np.zeros((2, 2, 2), dtype='complex')
    for l1 in range(nOrb):
        for l2 in range(nOrb):
            for l3 in range(nOrb):
                for l4 in range(nOrb):
                    for iqz in range(0,2):
                        G4K[0, 0, iqz] += (G4k[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[0][0][l3][l4] + \
                                           G4k[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[1][1][l3][l4] + \
                                           G4k[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[0][0][l3][l4] + \
                                           G4k[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[1][1][l3][l4]) * Mat[iqz][l1][l3]

                        G4K[0, 1, iqz] += (G4k[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[2][2][l3][l4] + \
                                           G4k[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[3][3][l3][l4] + \
                                           G4k[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[2][2][l3][l4] + \
                                           G4k[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[3][3][l3][l4]) * Mat[iqz][l1][l3]

                        G4K[1, 0, iqz] += (G4k[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[0][0][l3][l4] + \
                                           G4k[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[1][1][l3][l4] + \
                                           G4k[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[0][0][l3][l4] + \
                                           G4k[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[1][1][l3][l4]) * Mat[iqz][l1][l3]

                        G4K[1, 1, iqz] += (G4k[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[2][2][l3][l4] + \
                                           G4k[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[3][3][l3][l4] + \
                                           G4k[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[2][2][l3][l4] + \
                                           G4k[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[3][3][l3][l4]) * Mat[iqz][l1][l3]

    print("cluster susceptibility, Total qz=0", ": T = ", Tval, ' ',real(sum(G4K[:, :, 0]) / (Nc * paramsd['invT'])))
    print("cluster susceptibility, Total qz=pi", ": T = ", Tval, ' ',real(sum(G4K[:, :, 1]) / (Nc * paramsd['invT'])))
    print("cluster susceptibility, intra dx2y2 qz=0", ": T = ", Tval, ' ', real(chi_c0[0, 0]))
    print("cluster susceptibility, intra dz2 qz=0", ": T = ", Tval, ' ', real(chi_c0[1, 1]))
    print("cluster susceptibility, intra dx-z qz=0", ": T = ", Tval, ' ', real(chi_c0[0, 1]))
    print("cluster susceptibility, inter dx2y2 qz=0", ": T = ", Tval, ' ', real(chi_c0[0, 2]))
    print("cluster susceptibility, inter dz2 qz=0", ": T = ", Tval, ' ', real(chi_c0[1, 3]))
    print("cluster susceptibility, inter dx-z qz=0", ": T = ", Tval, ' ', real(chi_c0[1, 2]))

    print('\n','Have solved cluster magnetic susceptibility!')
