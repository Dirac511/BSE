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

###########################################################
def reorder_G4(paramsd,G4):
    print ("reorder G4")
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb'];nt = paramsd['nt']
    G4r = np.zeros((NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb),dtype='complex')
    '''
    for ik1 in range(Nc):
        for ik2 in range(Nc):
            for iw1 in range(NwG4):
                for iw2 in range(NwG4):
                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                    if paramsd['vertex_channel'] in ("PARTICLE_HOLE_MAGNETIC","PARTICLE_HOLE_CHARGE"):
                                        c1= G4[iw1,ik1,iw2,ik2,l1,l2,l3,l4]
                                        G4r[iw1,ik1,l1,l3,iw2,ik2,l4,l2]  = c1
                                    elif paramsd['vertex_channel'] =="PARTICLE_PARTICLE_UP_DOWN":
                                        c1 = G4[iw2,ik2,iw1,ik1,l4,l3,l2,l1]
                                        G4r[iw1,ik1,l1,l2,iw2,ik2,l3,l4] = c1
    '''
    if paramsd['vertex_channel'] in ("PARTICLE_HOLE_MAGNETIC", "PARTICLE_HOLE_CHARGE"):
        G4r = np.einsum('wkmpabcd -> wkacmpdc', G4)
    elif paramsd['vertex_channel'] == "PARTICLE_PARTICLE_UP_DOWN":
        G4r = np.einsum('wkmpabcd -> mpdcwkba', G4)

    return G4r

##########################################################
def apply_symmetry_in_wn(paramsd,G4r):
    # for G4[w1,K1,w2,K2]
    # apply symmetry G4(K,wn,K',wn') = G4*(K,-wn,K',-wn')
    # see 2-particle symmetries.pdf's final equation
    # similar to apply_transpose_symmetry
    # Be careful for multi-orbital case, where k=(K,iwn,b) with b index included
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    G4r_sym_wn = np.zeros((NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb), dtype='complex')
    '''
    for l1 in range(nOrb):
        for l2 in range(nOrb):
            for l3 in range(nOrb):
                for l4 in range(nOrb):
                    for iw1 in range(NwG4):
                        for iw2 in range(NwG4):
                            for iK1 in range(Nc):
                                for iK2 in range(Nc):
                                    imw1 = NwG4 - 1 - iw1
                                    imw2 = NwG4 - 1 - iw2
                                    tmp1 = G4r[iw1, iK1, l1, l2, iw2, iK2, l3, l4]
                                    tmp2 = G4r[imw1, iK1, l1, l2, imw2, iK2, l3, l4]
                                    G4r_sym_wn[iw1, iK1, l1, l2, iw2, iK2, l3, l4] = 0.5 * (tmp1 + conj(tmp2))
                                    G4r_sym_wn[imw1, iK1, l1, l2, imw2, iK2, l3, l4] = 0.5 * (conj(tmp1) + tmp2)
    '''
    for iw1 in range(NwG4):
        for iw2 in range(NwG4):
            imw1 = NwG4 - 1 - iw1
            imw2 = NwG4 - 1 - iw2
            tmp1 = G4r[iw1, :, :, :, iw2, :, :, :]
            tmp2 = G4r[imw1, :, :, :, imw2, :, :, :]
            G4r_sym_wn[iw1, :, :, :, iw2, :, :, :] = 0.5 * (tmp1 + conj(tmp2))
            G4r_sym_wn[imw1, :, :, :, imw2, :, :, :] = 0.5 * (conj(tmp1) + tmp2)

    return G4r_sym_wn

##############################################################
'''
def apply_transpose_symmetry(paramsd,G4M):
    # Apply symmetry Gamma(K,K') = Gamma(K',K)
    # Be careful for multi-orbital case, where k=(K,iwn,b) with b index included
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    GP = 0.5 * (G4M + G4M.transpose())
    #GP = 0.5 * (G4M + G4M.conj().T)
    G4r = GP.reshape(NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb)
    G4M = GP
    return G4r,G4M

'''
def apply_transpose_symmetry(paramsd, A):  ##
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb'];nt = paramsd['nt']
    At = np.zeros((NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb), dtype='complex')
    for ik1 in range(Nc):
        for ik2 in range(Nc):
            for iw1 in range(NwG4):
                for iw2 in range(NwG4):
                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                    At[iw1,ik1,l1,l2,iw2,ik2,l3,l4] = 0.5*(A[iw1,ik1,l1,l2,iw2,ik2,l3,l4]+A[iw2,ik2,l3,l4,iw1,ik1,l1,l2])
    AtM = At.reshape(nt, nt)
    return At,AtM

'''
def apply_transpose_symmetry(paramsd, A):  ##
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb'];nt = paramsd['nt']
    At = np.zeros((NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb), dtype='complex')
    for ik1 in range(Nc):
        for ik2 in range(Nc):
            for iw1 in range(NwG4):
                for iw2 in range(NwG4):
                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                    #print(X,L)
                                    a = A[iw1,ik1,l1,l2,iw2,ik2,l3,l4].imag
                                    b = A[iw2,ik2,l3,l4,iw1,ik1,l1,l2].imag
                                    # print(X,L)
                                    if a*b >= 0:
                                        At[iw1,ik1,l1,l2,iw2,ik2,l3,l4] = 0.5*(A[iw1,ik1,l1,l2,iw2,ik2,l3,l4]+A[iw2,ik2,l3,l4,iw1,ik1,l1,l2])
                                    elif a*b < 0:
                                        real = 0.5*(A[iw1,ik1,l1,l2,iw2,ik2,l3,l4].real+A[iw2,ik2,l3,l4,iw1,ik1,l1,l2].real)
                                        imag = 0.5*(A[iw1,ik1,l1,l2,iw2,ik2,l3,l4].imag-A[iw2,ik2,l3,l4,iw1,ik1,l1,l2].imag)
                                        At[iw1,ik1,l1,l2,iw2,ik2,l3,l4] = real + 1.j*imag
    AtM = At.reshape(nt, nt)
    return At,AtM
'''

def check_transpose_symmetry(paramsd, A):  ##symmtrize the G4r on layer
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    for ik1 in range(Nc):
        for ik2 in range(Nc):
            for iw1 in range(31,32):
                for iw2 in range(31,32):
                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                    X = [l1,l2,l3,l4]
                                    a = A[iw1,ik1,l1,l2,iw2,ik2,l3,l4]
                                    b = A[iw2,ik2,l3,l4,iw1,ik1,l1,l2]
                                    # print(X,L)
                                    if (abs(a) - abs(b)) >= 1.e-10:
                                        print(iw1,iw2,ik1,ik2,X)

###############################################################
def apply_ph_symmetry_pp(paramsd):
    # G4pp(k,wn,k',wn') = G4pp(k+Q,wn,k'+Q,wn'), with Q=(pi,pi)
    # 2022.2.21:
    # From Maier's notes on symmetries of G4
    # the simplied notation G4(K,K') = G4(K+Q,K'+Q) means that
    # At (q,iv)=0, G4(-K,K,K',-K') = G4(-K-Q,K+Q,K'+Q,-K'-Q)
    #
    # so that PARTICLE_PARTICLE_UP_DOWN:
    #
    #       -K,l1          -K',l4         -K-Q,l1       -K'-Q,l4
    #     -----<-------------<------    -----<-------------<------
    #             |     |                         |     |
    #             |  G4 |            =            |  G4 |
    #     -----<-------------<------    -----<-------------<------
    #        K,l3           K',l2          K+Q,l3        K'+Q,l2
    #
    # This is not necessarily correct for all Q
    # only when ph_symmetry is satisfied? (probably at half-filling, Q=(pi,pi)
    # is nesting wavevector so that here use Q=(pi,pi))
    nOrb = paramsd['nOrb'];Nc = paramsd['G4'].shape[1];nwn = paramsd['G4'].shape[0]
    G4_phsymmetry = np.zeros((NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb), dtype='complex')
    iKDiff, iKSum = setupMomentumTables(paramsd)
    iKpipi,iKpi0 = determine_specialK(paramsd)
    for l1 in range(nOrb):
        for l2 in range(nOrb):
            for l3 in range(nOrb):
                for l4 in range(nOrb):
                    for iw1 in range(nwn):
                        for iw2 in range(nwn):
                            for iK1 in range(Nc):
                                iK1q = iKSum[iK1, iKPiPi]
                                for iK2 in range(Nc):
                                    iK2q = iKSum[iK2, iKPiPi]
                                    tmp1 = paramsd['G4'][iw1, iK1, iw2, iK2, l1, l2, l3, l4]
                                    tmp2 = paramsd['G4'][iw1, iK1q, iw2, iK2q, l1, l2, l3, l4]
                                    G4_phsymmetry = [iw1, iK1, iw2, iK2, l1, l2, l3, l4] = 0.5 * (tmp1 + tmp2)
                                    G4_phsymmetry = [iw1, iK1q, iw2, iK2q, l1, l2, l3, l4] = 0.5 * (tmp1 + tmp2)
    return G4_phsymmetry

#####################################################################
'''
def apply_symmetry_in_orb(paramsd, A):  ##symmtrize the G4r on layer
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    At = np.zeros((NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb), dtype='complex')
    for l1 in range(nOrb):
        for l2 in range(nOrb):
             for l3 in range(nOrb):
                 for l4 in range(nOrb):
                     At[:,:,l1,l2,:,:,l3,l4] = 0.5*(A[:,:,l1,l2,:,:,l3,l4]+A[:,:,l2,l1,:,:,l4,l3])
    return At

'''
def apply_symmetry_in_orb(paramsd, A):  ##symmtrize the G4r on layer
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    At = np.zeros((NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb), dtype='complex')
    for ik1 in range(Nc):
        for ik2 in range(Nc):
            for iw1 in range(NwG4):
                for iw2 in range(NwG4):
                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                    a = A[iw1,ik1,l1,l2,iw2,ik2,l3,l4].imag
                                    b = A[iw1,ik1,l2,l1,iw2,ik2,l4,l3].imag
                                    # print(X,L)
                                    if a*b >= 0:
                                        At[iw1,ik1,l1,l2,iw2,ik2,l3,l4] = 0.5*(A[iw1,ik1,l1,l2,iw2,ik2,l3,l4]+A[iw1,ik1,l2,l1,iw2,ik2,l4,l3])
                                    elif a*b < 0:
                                        real = 0.5*(A[iw1,ik1,l1,l2,iw2,ik2,l3,l4].real+A[iw1,ik1,l2,l1,iw2,ik2,l4,l3].real)
                                        imag = 0.5*(A[iw1,ik1,l1,l2,iw2,ik2,l3,l4].imag-A[iw1,ik1,l2,l1,iw2,ik2,l4,l3].imag)
                                        At[iw1,ik1,l1,l2,iw2,ik2,l3,l4] = real + 1.j*imag
                    # print(X,L)
    return At


def check_orb_symmetry(paramsd, A):  ##symmtrize the G4r on layer
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    for ik1 in range(Nc):
        for ik2 in range(Nc):
            for iw1 in range(31,32):
                for iw2 in range(31,32):
                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                    X = [l1,l2,l3,l4]
                                    #print(X,L)
                                    a = A[iw1,ik1,l1,l2,iw2,ik2,l3,l4]
                                    b = A[iw1,ik1,l2,l1,iw2,ik2,l4,l3]
                                    # print(X,L)
                                    if (abs(a) - abs(b))>= 1.e-10:
                                        print(iw1,iw2,ik1,ik2,X)

############################################################################
'''
def apply_symmetry_in_layer(model, paramsd, A):  ##symmtrize the G4r on layer
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    At = np.zeros((NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb), dtype='complex')
    for ik1 in range(Nc):
        for ik2 in range(Nc):
            for iw1 in range(NwG4):
                for iw2 in range(NwG4):
                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                    X = [l1, l2, l3, l4]
                                    L = trans_layer(model, X)
                                    #print(X,L)
                                    a = A[iw1,ik1,X[0],X[1],iw2,ik2,X[2],X[3]].imag
                                    b = A[iw1,ik1,L[0],L[1],iw2,ik2,L[2],L[3]].imag
                                    # print(X,L)
                                    if a*b >= 0:
                                        At[iw1,ik1,l1,l2,iw2,ik2,l3,l4] = 0.5*(A[iw1,ik1,X[0],X[1],iw2,ik2,X[2],X[3]]+A[iw1,ik1,L[0],L[1],iw2,ik2,L[2],L[3]])
                                    elif a*b < 0:
                                        real = 0.5*(A[iw1,ik1,X[0],X[1],iw2,ik2,X[2],X[3]].real+A[iw1,ik1,L[0],L[1],iw2,ik2,L[2],L[3]].real)
                                        imag = 0.5*(A[iw1,ik1,X[0],X[1],iw2,ik2,X[2],X[3]].imag-A[iw1,ik1,L[0],L[1],iw2,ik2,L[2],L[3]].imag)
                                        At[iw1,ik1,l1,l2,iw2,ik2,l3,l4] = real + 1.j*imag
    
    return At
'''
def apply_symmetry_in_layer(model, paramsd, A):
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    At = np.zeros((NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb), dtype='complex')
    for l1 in range(nOrb):
        for l2 in range(nOrb):
            for l3 in range(nOrb):
                for l4 in range(nOrb):
                    X = [l1, l2, l3, l4]
                    L = trans_layer(model, X)
                    At[:,:,l1,l2,:,:,l3,l4] = 0.5*(A[:,:,X[0],X[1],:,:,X[2],X[3]]+A[:,:,L[0],L[1],:,:,L[2],L[3]])

    return At

def check_layer_symmetry(model,paramsd, A):  ##symmtrize the G4r on layer
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    for ik1 in range(Nc):
        for ik2 in range(Nc):
            for iw1 in range(2):
                for iw2 in range(2):
                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                    X = [l1, l2, l3, l4]
                                    L = trans_layer(model, X)
                                    #print(X,L)
                                    a = A[iw1,ik1,X[0],X[1],iw2,ik2,X[2],X[3]]
                                    b = A[iw1,ik1,L[0],L[1],iw2,ik2,L[2],L[3]]
                                    # print(X,L)
                                    if (abs(a) - abs(b))>= 1.e-10:
                                        print(iw1,iw2,ik1,ik2,X,L)

###############################################################################
def apply_symmetry_in_C4(model,paramsd, A):  ##symmtrize the G4r with K
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    if model == 'La3Ni2O7':
        Sign = [-1, 1, -1, 1] # dx2y2,dz2,dx2y2,dz2
        At = np.zeros((NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb), dtype='complex')
        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        for i in range(Nc):
                            for j in range(Nc):
                                X = [i,j]
                                L = trans_K(X,Nc)
                                S = Sign[l1]*Sign[l2]*Sign[l3]*Sign[l4]
                                if S == -1 and X != L:
                                    At[:,i,l1,l2,:,j,l3,l4] = 0.5*(A[:,X[0],l1,l2,:,X[1],l3,l4] - A[:,L[0],l1,l2,:,L[1],l3,l4])
                                elif S == -1 and X == L:
                                    At[:,i,l1,l2,:,j,l3,l4] = 0.0
                                elif S == 1:
                                    At[:,i,l1,l2,:,j,l3,l4] = 0.5*(A[:,X[0],l1,l2,:,X[1],l3,l4] + A[:,L[0],l1,l2,:,L[1],l3,l4])
    else:
        At = np.zeros((NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb), dtype='complex')
        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        for i in range(Nc):
                            for j in range(Nc):
                                X = [i,j]
                                L = trans_K(X,Nc)
                                At[:,i,l1,l2,:,j,l3,l4] = 0.5*(A[:,X[0],l1,l2,:,X[1],l3,l4] + A[:,L[0],l1,l2,:,L[1],l3,l4])

    return At

def check_C4_symmetry(paramsd, A):  ##symmtrize the G4r on layer
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    for ik1 in range(Nc):
        for ik2 in range(Nc):
            for iw1 in range(2):
                for iw2 in range(2):
                    for l1 in range(2):
                        for l2 in range(2):
                            for l3 in range(2):
                                for l4 in range(2):
                                    X = [ik1,ik2]
                                    L = trans_K(X,Nc)
                                    a = A[iw1,X[0],l1,l2,iw2,X[1],l3,l4]
                                    b = A[iw1,L[0],l1,l2,iw2,L[1],l3,l4]
                                    if (abs(a) - abs(b)) >= 1.e-10:
                                        print(iw1,iw2,X,L)

######################################################################
def trans_layer(model, X):  ##do symmetry on layer
    L = []
    if model == 'bilayer':
        for i in range(len(X)):
            x = str(X[i])
            l = x.replace('0', 'temp').replace('1', '0').replace('temp', '1')
            l1 = int(l)
            L.append(l1)

    if model == 'trilayer':
        for i in range(len(X)):
            x = str(X[i])
            l = x.replace('0', 'temp').replace('2', '0').replace('temp', '2')
            l1 = int(l)
            L.append(l1)

    if model == 'fivelayer':
        for i in range(len(X)):
            x = str(X[i])
            if x == '0' or x == '4':
                l = x.replace('0', 'temp').replace('4', '0').replace('temp', '4')
            if x == '1' or x == '3':
                l = x.replace('1', 'temp').replace('3', '1').replace('temp', '3')
            l1 = int(l)
            L.append(l1)

    if model == 'La3Ni2O7':
        for i in range(len(X)):
            x = str(X[i])
            if x == '0' or x == '2':
                l = x.replace('0','temp').replace('2','0').replace('temp','2')
            if x == '1' or x == '3':
                l = x.replace('1','temp').replace('3','1').replace('temp','3')
            l1 = int(l)
            L.append(l1)      
    return L

############################################################################
def trans_K(X,Nc):  ##do symmetry on (0,pi) and (pi,0)
    L = []
    if Nc == 4:
        for i in range(len(X)):
            x = str(X[i])
            l = x.replace('1', 'temp').replace('2', '1').replace('temp', '2')
            l1 = int(l)
            L.append(l1)
    if Nc == 8:
        for i in range(len(X)):
            x = str(X[i])
            l = x.replace('1', 'temp').replace('4', '1').replace('temp', '4')
            l1 = int(l)
            L.append(l1)
    return L

#############################################################
def K_2_iK(paramsd, Kx, Ky):
    delta = 1.0e-4
    # First map (Kx,Ky) into [0...2pi,0...2pi] region where Kvecs are defined
    if Kx < -delta: Kx += 2 * pi
    if Ky < -delta: Ky += 2 * pi
    if Kx > 2. * pi - delta: Kx -= 2 * pi
    if Ky > 2. * pi - delta: Ky -= 2 * pi
    # Now find index of Kvec = (Kx,Ky)
    for iK in range(0, paramsd['Nc']):
        if (abs(float(paramsd['Kvecs'][iK, 0] - Kx)) < delta) & (abs(float(paramsd['Kvecs'][iK, 1] - Ky)) < delta): return iK
    print("No Kvec found!!!")

##########################################################
def setupMomentumTables(paramsd):
    # build tables for K+K' and K-K'
    Nc = paramsd['Nc'];Kvecs = paramsd['Kvecs']
    iKDiff = zeros((Nc,Nc),dtype='int')
    iKSum  = zeros((Nc,Nc),dtype='int')
    for iK1 in range(Nc):
        Kx1 = Kvecs[iK1,0]; Ky1 = Kvecs[iK1,1]
        for iK2 in range(0,Nc):
            Kx2 = Kvecs[iK2,0]; Ky2 = Kvecs[iK2,1]
            iKS = K_2_iK(paramsd,Kx1+Kx2,Ky1+Ky2)
            iKD = K_2_iK(paramsd,Kx1-Kx2,Ky1-Ky2)
            iKDiff[iK1,iK2] = iKD
            iKSum[iK1,iK2]  = iKS
    return iKDiff,iKSum

#########################################################
def determine_specialK(paramsd):
    iKPiPi = 0
    iKPi0 = 0
    Nc = paramsd['Nc']
    for iK in range(Nc):
        kx = abs(paramsd['Kvecs'][iK, 0] - np.pi)
        ky = abs(paramsd['Kvecs'][iK, 1] - np.pi)
        kx2 = abs(paramsd['Kvecs'][iK, 0])
        ky2 = abs(paramsd['Kvecs'][iK, 1])
        if kx >= 2 * np.pi: kx -= 2. * pi
        if ky >= 2 * np.pi: ky -= 2. * pi
        if ky2 >= 2 * np.pi: ky -= 2. * pi
        if kx ** 2 + ky ** 2 <= 1.0e-5:
            iKPiPi = iK
        if kx ** 2 + ky2 ** 2 <= 1.0e-5:
            iKPi0 = iK
        if kx2 ** 2 + ky ** 2 <= 1.0e-5:
            iK0Pi = iK
    return iKpipi,iKpi0

################################################################################
def trans_basis(model,paramsd,Tval,evecs,Num,write_data_file):  ### transform the orbital-basis eigenvector into band basis
    # for trilayer
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb'];nt = paramsd['nt']
    targetVec = np.zeros((NwG4, Nc, nOrb, nOrb, nt), dtype='complex')

    if model == 'trilayer':
        tmat = np.zeros((3, 3), dtype=np.float64)
        tmat[0, 0] = 0.5;tmat[0, 1] = -0.5 * sqrt(2);tmat[0, 2] = 0.5  # Bond
        tmat[1, 0] = 0.5 * sqrt(2);tmat[1, 1] = 0;tmat[1, 2] = -0.5 * sqrt(2)  # Non
        tmat[2, 0] = 0.5;tmat[2, 1] = 0.5 * sqrt(2);tmat[2, 2] = 0.5  # Anti

    elif model == 'bilayer':
        tmat = np.zeros((2, 2), dtype=np.float64)
        tmat[0, 0] = 0.5 * sqrt(2);tmat[0, 1] = 0.5 * sqrt(2)  # Bond
        tmat[1, 0] = 0.5 * sqrt(2);tmat[1, 1] = -0.5 * sqrt(2)  # Anti

    elif model == 'La3Ni2O7':
        tmat = np.zeros((4, 4), dtype=np.float64)
        tmat[0, 0] = 0.5 * sqrt(2) ;tmat[0, 1] = 0             ;tmat[0, 2] = 0.5 * sqrt(2)  ;tmat[0, 3] = 0  # Bondx
        tmat[1, 0] = 0             ;tmat[1, 1] = 0.5 * sqrt(2) ;tmat[1, 2] = 0              ;tmat[1, 3] = 0.5 * sqrt(2)  # Bondz
        tmat[2, 0] = 0.5 * sqrt(2) ;tmat[2, 1] = 0             ;tmat[2, 2] = -0.5 * sqrt(2) ;tmat[2, 3] = 0  # Antix
        tmat[3, 0] = 0             ;tmat[3, 1] = 0.5 * sqrt(2) ;tmat[3, 2] = 0              ;tmat[3, 3] = -0.5 * sqrt(2)  # Antiz

    tmatinv = np.linalg.inv(tmat)

    for ia in range(0, Num):
        for iwn in range(NwG4):
            for iNc in range(Nc):
                targetVec[iwn, iNc, :, :, ia] = np.matmul(tmatinv,np.matmul(evecs[iwn, iNc, :, :, ia], tmat)).astype('complex')  # refer

    if write_data_file:
        fname = 'leadingspm_Evec_vs_K_T' + str(Tval) + '.txt'
        fnamew = 'leadingspm_wn_Evec_vs_K_T' + str(Tval) + '.txt'
        if os.path.isfile(fname):
            os.remove(fname)

        for ilam in range(0, Num):
            for iNc in range(Nc):
                kx = paramsd['Kvecs'][iNc, 0];ky = paramsd['Kvecs'][iNc, 1]
                kxs = np.full((1, 1), kx)
                kys = np.full((1, 1), ky)
                Kxs = np.full((NwG4, 1), kx)
                Kys = np.full((NwG4, 1), ky)
                ## wirte the data of band-basis eigenvector at lowest frequency
                if model == 'trilayer':
                    write_data_6cols(fname, kxs, kys, paramsd['wnSet'][int(NwG4 / 2)],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 0, 0, ilam],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 1, 1, ilam],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 2, 2, ilam])
                elif model == 'bilayer':
                    write_data_5cols(fname, kxs, kys, paramsd['wnSet'][int(NwG4 / 2)],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 0, 0, ilam],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 1, 1, ilam])
                elif model == 'La3Ni2O7':
                    write_data_11cols(fname, kxs, kys, paramsd['wnSet'][int(NwG4 / 2)],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 0, 0, ilam],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 2, 2, ilam],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 1, 1, ilam],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 3, 3, ilam],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 0, 2, ilam],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 2, 0, ilam],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 1, 3, ilam],
                                          targetVec[int(NwG4 / 2):int(NwG4 / 2) + 1, iNc, 3, 1, ilam])
                    write_data_w_11cols(fnamew, Kxs, Kys, paramsd['wnSet'],
                                          targetVec[:, iNc, 0, 0, ilam],
                                          targetVec[:, iNc, 2, 2, ilam],
                                          targetVec[:, iNc, 1, 1, ilam],
                                          targetVec[:, iNc, 3, 3, ilam],
                                          targetVec[:, iNc, 0, 2, ilam],
                                          targetVec[:, iNc, 2, 0, ilam],
                                          targetVec[:, iNc, 1, 3, ilam],
                                          targetVec[:, iNc, 3, 1, ilam])

#########################################################################################

def write_data_w_6cols(fname, xs, ys, zs, ws, fs, gs):
    f = open(fname, 'a', 1)
    for i in range(len(xs)):
        f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(xs[i]), float(ys[i]), float(zs[i]),float(ws[i].real), float(fs[i].real),float(gs[i].real)))

def write_data_w_11cols(fname, xs, ys, zs, ws, fs, gs, ks, ts, hs, ms, ns):
    f = open(fname, 'a', 1)
    for i in range(len(xs)):
        f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(xs[i]), float(ys[i]), float(zs[i]),float(ws[i].real), float(fs[i].real),float(gs[i].real),float(ks[i].real), float(ts[i].real), float(hs[i].real),float(ms[i].real), float(ns[i].real)))

def write_data_2cols(fname, xs, ys):
    f = open(fname, 'w', 1)
    for i in range(len(xs)):
        f.write('{:.6e}\t{:.6e}\n'.format(float(xs[i]), float(ys[i])))

def write_data_3cols(fname, xs, ys, zs):
    f = open(fname, 'w', 1)
    for i in range(len(xs)):
        f.write('{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(xs[i]), float(ys[i]), float(zs[i])))

def write_data_4cols(fname, xs, ys, zs, ws):
    f = open(fname, 'a', 1)
    for i in range(len(xs)):
        f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(xs), float(ys), float(zs), float(ws.real)))

def write_data_5cols(fname, xs, ys, zs, ws, gs):
    f = open(fname, 'a', 1)
    for i in range(len(xs)):
        f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(xs), float(ys), float(zs), float(ws.real),float(gs.real)))

def write_data_6cols(fname, xs, ys, zs, ws, fs, gs):
    f = open(fname, 'a', 1)
    for i in range(len(xs)):
        f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(xs), float(ys), float(zs),float(ws.real), float(fs.real),float(gs.real)))

def write_data_11cols(fname, xs, ys, zs, ws, fs, gs, ks, ts, hs, ms, ns):
    f = open(fname, 'a', 1)
    for i in range(len(xs)):
        f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(xs), float(ys), float(zs),float(ws.real), float(fs.real),float(gs.real),float(ks.real), float(ts.real), float(hs.real),float(ms.real), float(ns.real)))

def write_data_7cols(fname, xs, ys, zs, ws, fs, gs, ks):
    f = open(fname, 'a', 1)
    for i in range(len(xs)):
        f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(xs), float(ys), float(zs),float(ws.real), float(fs.real),float(gs.real),float(ks.real)))
