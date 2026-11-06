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
import Models
#from builtins import max, min

################################################################################
def calcChi0Cluster(model,symmetry_layer,symmetry_C4,symmetry_transpose,paramsd):
    print("Now calculating chi0 on cluster chic0 = Gc*Gc, chi^0_c of Eq.(22) in PRB 64, 195130(2001)")
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];NwG = paramsd['NwG'];nOrb = paramsd['nOrb'];nt = paramsd['nt']
    Green = paramsd['Green']
    iKDiff,iKSum = Operation.setupMomentumTables(paramsd)

    chic0 = zeros((NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb), dtype='complex')

    if (paramsd['vertex_channel'] == "PARTICLE_PARTICLE_UP_DOWN"):
        for iw in range(0, NwG4):
            for ik in range(Nc):
                for l1 in range(nOrb):
                    for l2 in range(nOrb):
                        for l3 in range(nOrb):
                            for l4 in range(nOrb):
                                iw1 = int(iw - NwG4 / 2 + NwG / 2)
                                ikPlusQ = int(iKSum[iKDiff[0, ik], paramsd['iQ']])  # -k+Q
                                minusiwPlusiwm = int(min([max([NwG - iw1 - 1 + paramsd['iwm'], 0]), NwG - 1]))  # -iwn + iwm
                                # if (l1==l2==l3==l4==0) & (iw==0): print("ik=",ik," ikPlusQ=",ikPlusQ)

                                # change sign below if one band is Cu
                                # Because of the special dipersion relation of d-p model
                                # for Q=0, e_-k = -e_k
                                if model == 'threeband':
                                    if (l2 != l4 and l2 == 0) or (l2 != l4 and l4 == 0):
                                        c1 = -Green[iw1, ik, l3, l1] * Green[minusiwPlusiwm, ikPlusQ, l4, l2]
                                    else:
                                        c1 = Green[iw1, ik, l3, l1] * Green[minusiwPlusiwm, ikPlusQ, l4, l2]
                                else:
                                    c1 = Green[iw1, ik, l3, l1] * Green[minusiwPlusiwm, ikPlusQ, l4, l2]

                                chic0[iw, ik, l1, l2, iw, ik, l3, l4] = c1
    else:
        for iw in range(NwG4):
            for ik in range(Nc):
                for l1 in range(nOrb):
                    for l2 in range(nOrb):
                        for l3 in range(nOrb):
                            for l4 in range(nOrb):
                                iw1 = int(iw - NwG4 / 2 + NwG / 2)
                                ikPlusQ = int(iKSum[ik, paramsd['iQ']])  # k+Q
                                iwPlusiwm = int(min(max(iw1 + paramsd['iwm'], 0), NwG - 1))  # iwn+iwm
                                # print("iw1,ik,iwPlusiwm,ikPlusQ",iw1,ik,iwPlusiwm,ikPlusQ)
                                c1 = -Green[iw1, ik, l1, l3] * Green[iwPlusiwm, ikPlusQ, l4, l2]
                                chic0[iw, ik, l1, l2, iw, ik, l3, l4] = c1

    #for l1 in range(nOrb):
    #    for l2 in range(nOrb):
    #        for l3 in range(nOrb):
    #            for l4 in range(nOrb):
    #                print(l1,l2,l3,l4,':',chic0[31,1,l1,l2,31,1,l3,l4])

    chic0 = Operation.apply_symmetry_in_orb(paramsd,chic0)
    Operation.check_orb_symmetry(paramsd,chic0)
    if symmetry_layer and symmetry_C4:
        chic0t1 = Operation.apply_symmetry_in_layer(model,paramsd,chic0);print('symmetry layer and C4 chic0')
        Operation.check_layer_symmetry(model,paramsd,chic0t1)

        chic0t2 = Operation.apply_symmetry_in_C4(model,paramsd,chic0t1)
        Operation.check_C4_symmetry(paramsd,chic0t2)
 
        chic0M = chic0t2.reshape(nt, nt)  ##apply layer and C4 symmetry
    elif symmetry_layer and symmetry_C4 == False:
        chic0t1 = Operation.apply_symmetry_in_layer(model,paramsd,chic0);print('symmetry layer chic0')
        Operation.check_layer_symmetry(model,paramsd,chic0t1)

        chic0M = chic0t1.reshape(nt, nt)  ##apply layer symmetry
    elif symmetry_layer == False and symmetry_C4:
        chic0t1 = Operation.apply_symmetry_in_C4(model,paramsd,chic0);print('symmetry C4 chic0')
        Operation.check_C4_symmetry(paramsd,chic0t1)

        chic0M = chic0t1.reshape(nt, nt)  ##apply C4 symmetry
    elif symmetry_layer == False and symmetry_C4 == False:
        chic0M = chic0.reshape(nt, nt) 

    #if symmetry_transpose and paramsd['vertex_channel'] in ("PARTICLE_PARTICLE_SUPERCONDUCTING", "PARTICLE_PARTICLE_UP_DOWN"):
    #    chic0,chic0M = Operation.apply_transpose_symmetry(paramsd,chic0M)

    #for l1 in range(nOrb):
    #    for l2 in range(nOrb):
    #        for l3 in range(nOrb):
    #            for l4 in range(nOrb):
    #                print('\n',l1,l2,l3,l4,':',chic0t1[31,1,l1,l2,31,1,l3,l4])

    #print(chic0t3[0,1,2,1,0,1,1,1],chic0t3[0,1,0,3,0,1,3,3],chic0t3[0,2,2,1,0,2,1,1],chic0t3[0,2,0,3,0,2,3,3])
    #print(chic0t1[0,1,0,0,0,1,2,2],chic0t1[0,1,2,2,0,1,0,0])
    print("calClusterchi0 over!",'\n')
    return chic0,chic0M 

#########################################################################################
def buildChi0Lattice(model,symmetry_layer,symmetry_C4,symmetry_transpose,paramsd,paramsm,nkfine):
    print("Now calculating coarsed-grained chi0, Eq.(18) in PRB 64, 195130(2001)")

    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];NwG = paramsd['NwG'];nOrb = paramsd['nOrb'];nt = paramsd['nt']
    Green = paramsd['Green']
    iKDiff, iKSum = Operation.setupMomentumTables(paramsd)
    
    # Cluster K-grid
    Kset = paramsd['Kvecs'].copy()  # copy() since Kset will be modified

    # Fine mesh
    klin = np.arange(-pi, pi, 2 * pi / nkfine)
    kx, ky = np.meshgrid(klin, klin)
    kset = np.column_stack((kx.flatten(), ky.flatten()))

    kPatch = []

    # shift to 1. BZ
    Nc = Kset.shape[0]
    for iK in range(Nc):
        if Kset[iK, 0] > np.pi: Kset[iK, 0] -= 2 * np.pi
        if Kset[iK, 1] > np.pi: Kset[iK, 1] -= 2 * np.pi

    # Determine k-points in patch
    for k in kset:
        distance0 = k[0] ** 2 + k[1] ** 2
        newPoint = True
        for K in Kset:
            distanceKx = k[0] - K[0];
            distanceKy = k[1] - K[1]
            if distanceKx >= pi: distanceKx -= 2 * pi
            if distanceKy >= pi: distanceKy -= 2 * pi
            if distanceKx <= -pi: distanceKx += 2 * pi
            if distanceKy <= -pi: distanceKy += 2 * pi
            distanceK = distanceKx ** 2 + distanceKy ** 2
            if distanceK < distance0:
                newPoint = False
                break
        if newPoint: kPatch.append(k.tolist())

    kPatch = np.array(kPatch)
    # Load frequency domain
    wnSet = paramsd['wnSet']

    # Load parameters: t,mu
    # t1 = self.t1
    # t2 = self.t2
    mu = paramsd['mu']

    sigma = np.zeros((NwG, Nc, nOrb, nOrb), dtype='complex')
    sigma = paramsd['sigmaoriginal']
    sigmanegk = np.zeros((NwG, Nc, nOrb, nOrb), dtype='complex')

    # for calculation of G2inv below
    # change sign below if one band is Cu
    # Because of the special dipersion relation: e_-k = -e_k for d-p hopping

    for l2 in range(nOrb):
        for l4 in range(nOrb):
            sigmanegk[:, :, l2, l4] = sigma[:, :, l2, l4]  ##not Emery

    if model == 'Emery':
        for l2 in range(nOrb):
            for l4 in range(nOrb):
                if (l2 != l4 and l2 == 0) or (l2 != l4 and l4 == 0):
                    sigmanegk[:, :, l2, l4] = -sigma[:, :, l2, l4]
                else:
                    sigmanegk[:, :, l2, l4] = sigma[:, :, l2, l4]

    # Now coarse-grain G*G to build chi0(K) = Nc/N sum_k Gc(K+k')Gc(-K-k')
    nw = wnSet.shape[0];
    nk = Kset.shape[0];
    chi0 = np.zeros((nw, nk, nOrb, nOrb, nw, nk, nOrb, nOrb), dtype='complex')
    chi0D = np.zeros((nw, nk, nOrb, nOrb, nw, nk, nOrb, nOrb), dtype='complex')
    chi0D2 = np.zeros((nw, nk, nOrb, nOrb, nw, nk, nOrb, nOrb), dtype='complex')
    chi0XS = np.zeros((nw, nk, nOrb, nOrb, nw, nk, nOrb, nOrb), dtype='complex')
    chi0XS2 = np.zeros((nw, nk, nOrb, nOrb, nw, nk, nOrb, nOrb), dtype='complex')
    chi0DXY = np.zeros((nw, nk, nOrb, nOrb, nw, nk, nOrb, nOrb), dtype='complex')
    chi0DXY2 = np.zeros((nw, nk, nOrb, nOrb, nw, nk, nOrb, nOrb), dtype='complex')
    gkdNorm = 0.0
    # self.cG    = np.zeros((nw,nk,nOrb,nOrb),dtype='complex')
    # self.cG0   = np.zeros((nw,nk,nOrb,nOrb),dtype='complex')
    for iwn, wn in enumerate(wnSet):  # reduced tp frequencies !!
        # print("iwn = ",iwn)
        iwG = int(iwn - paramsd['iwG40'] + paramsd['iwG0'])
        for iK, K in enumerate(Kset):
            c0 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
            c1 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
            c2 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
            c3 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
            c4 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
            c5 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
            c6 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
            c7 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
            # cG = np.zeros((nOrb,nOrb),dtype='complex')
            # cG0 = np.zeros((nOrb,nOrb),dtype='complex')
            for k in kPatch:
                kx = K[0] + k[0];ky = K[1] + k[1]
                if (K[0] < -0.01):
                    kx += 2 * np.pi
                if (K[1] < -0.01):
                    ky += 2 * np.pi
                ek = Models.dispersion(model, paramsd, paramsm, kx, ky)
                gkd = cos(kx) - cos(ky)
                gkxs = cos(kx) + cos(ky)
                gkdxy = sin(kx) * sin(ky)
                # G0inv = (1j*wn+self.mu-self.U1*(self.dens[-1]/4.-0.5))*np.identity(nOrb) - ek
                # G0 = linalg.inv(G0inv)
                Qx = paramsd['qchannel'][0];Qy = paramsd['qchannel'][1]
                if (paramsd['vertex_channel'] == "PARTICLE_PARTICLE_UP_DOWN"):
                    emkpq = Models.dispersion(model, paramsd, paramsm, -kx + Qx, -ky + Qy)
                    iKQ = iKSum[iKDiff[0, iK], paramsd['iQ']]
                    # emkpq = self.dispersion(Kset[iKQ,0], Kset[iKQ,1])
                    # print("iK=",iK,"iKQ=",iKQ,"iKQx=",Kset[iKQ,0],"iKQy=",Kset[iKQ,1])
                    minusiwPlusiwm = min([max([NwG - iwG - 1 + paramsd['iwm'], 0]), NwG - 1])  # -iwn + iwm
                    # minusiwPlusiwm = int(min(max(NwG-iw1-1 + self.iwm,0),NwG-1))
                    G1inv = (1j * wn + mu) * np.identity(nOrb) - ek - sigma[iwG, iK, :, :]
                    G2inv = (-1j * wn + mu) * np.identity(nOrb) - emkpq - sigmanegk[minusiwPlusiwm, iK, :, :]
                    G1 = linalg.inv(G1inv);G2 = linalg.inv(G2inv)

                else:
                    ekpq = Models.dispersion(model, paramsd, paramsm, kx + Qx, ky + Qy)
                    iKQ = int(iKSum[iK, paramsd['iQ']])
                    iwPlusiwm = int(min([max([iwG + paramsd['iwm'], 0]), NwG - 1]))  # iwn+iwm

                    G1inv = (1j * wn + mu) * np.identity(nOrb) - ek - sigma[iwG, iK, :, :]
                    G2inv = (1j * wn + mu) * np.identity(nOrb) - ekpq - sigmanegk[iwPlusiwm, iKQ, :, :]
                    G1 = linalg.inv(G1inv);G2 = -linalg.inv(G2inv)

                for l1 in range(nOrb):
                    for l2 in range(nOrb):
                        for l3 in range(nOrb):
                            for l4 in range(nOrb):
                                c0[l1, l2, l3, l4] = G1[l3, l1] * G2[l4, l2]

                c1[:, :, :, :] += c0[:, :, :, :]
                c2[:, :, :, :] += c0[:, :, :, :] * gkd
                c3[:, :, :, :] += c0[:, :, :, :] * gkd ** 2
                c4[:, :, :, :] += c0[:, :, :, :] * gkxs
                c5[:, :, :, :] += c0[:, :, :, :] * gkxs ** 2
                c6[:, :, :, :] += c0[:, :, :, :] * gkdxy
                c7[:, :, :, :] += c0[:, :, :, :] * gkdxy**2
                if (iwn == 0): gkdNorm += gkd ** 2

            chi0[iwn, iK, :, :, iwn, iK, :, :] = c1[:, :, :, :] / kPatch.shape[0]
            chi0D[iwn, iK, :, :, iwn, iK, :, :] = c2[:, :, :, :] / kPatch.shape[0]
            chi0D2[iwn, iK, :, :, iwn, iK, :, :] = c3[:, :, :, :] / kPatch.shape[0]
            chi0XS[iwn, iK, :, :, iwn, iK, :, :] = c4[:, :, :, :] / kPatch.shape[0]
            chi0XS2[iwn, iK, :, :, iwn, iK, :, :] = c5[:, :, :, :] / kPatch.shape[0]
            chi0DXY[iwn, iK, :, :,iwn, iK, :, :]  = c6[:, :, :, :] / kPatch.shape[0]
            chi0DXY2[iwn, iK, :, :,iwn, iK, :, :]  = c7[:, :, :, :] / kPatch.shape[0]

    #for l1 in range(nOrb):
    #    for l2 in range(nOrb):
    #        for l3 in range(nOrb):
    #            for l4 in range(nOrb):
    #                print(l1,l2,l3,l4,':',chi0[31,1,l1,l2,31,1,l3,l4])

    chi0 = Operation.apply_symmetry_in_orb(paramsd,chi0)
    Operation.check_orb_symmetry(paramsd,chi0)
    if symmetry_layer and symmetry_C4:
        chi0t1 = Operation.apply_symmetry_in_layer(model, paramsd, chi0);print('symmetry layer and C4 chi0')
        Operation.check_layer_symmetry(model,paramsd,chi0t1)

        chi0t2 = Operation.apply_symmetry_in_C4(model,paramsd,chi0t1)
        Operation.check_C4_symmetry(paramsd,chi0t2)

        chi0M = chi0t2.reshape(nt, nt)  ##apply layer and C4 symmetry
    elif symmetry_layer and symmetry_C4 == False:
        chi0t1 = Operation.apply_symmetry_in_layer(model, paramsd, chi0);print('symmetry layer chi0')
        Operation.check_layer_symmetry(model,paramsd,chi0t1)

        chi0M = chi0t1.reshape(nt, nt)  ##apply layer symmetry
    elif symmetry_layer == False and symmetry_C4:
        chi0t1 = Operation.apply_symmetry_in_C4(model,paramsd, chi0);print('symmetry C4 chi0')
        Operation.check_C4_symmetry(paramsd,chi0t1)

        chi0M = chi0t1.reshape(nt, nt)  ##apply C4 symmetry
    elif symmetry_layer == False and symmetry_C4 == False:
        chi0M = chi0.reshape(nt, nt)

    #if symmetry_transpose and paramsd['vertex_channel'] in ("PARTICLE_PARTICLE_SUPERCONDUCTING", "PARTICLE_PARTICLE_UP_DOWN"):
    #    chi0,chi0M = Operation.apply_transpose_symmetry(paramsd,chi0M)

    print("calLatticechi0 over!",'\n')
    return chi0,chi0D,chi0D2,chi0XS,chi0XS2,chi0DXY,chi0DXY2,chi0M 
            # self.cG[iwn,iK,:,:] = cG[:,:]/kPatch.shape[0]
            # self.cG0[iwn,iK,:,:] = cG0[:,:]/kPatch.shape[0]

######################################################################################################
def calcGammaIrr(model,chic0M,symmetry_layer,symmetry_C4,phSymmetry,symmetry_wn,symmetry_transpose,paramsd):
    '''
    On Page 7 of PRB 64, 195130(2001): "We see that again it is the irreducible quantity, i.e., the vertex
    function, for which cluster and lattice correspond."
    '''
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb'];nt = paramsd['nt']
    Gamma = zeros((NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb), dtype='complex')
    print("Now calculating the irr. Gamma on cluster, which is used as approximation of that on lattice")

    if phSymmetry:
       G4 = Operation.apply_ph_symmetry_pp(paramsd)
    else:
       G4 = paramsd['G4']

    G4r = Operation.reorder_G4(paramsd,G4)
    G4M = G4r.reshape(nt, nt)
    #for l1 in range(nOrb):
    #    for l2 in range(nOrb):
    #        for l3 in range(nOrb):
    #            for l4 in range(nOrb):
    #                print(l1,l2,l3,l4,':',G4r[31,1,l1,l2,31,1,l3,l4])

    if symmetry_wn:
        G4r = Operation.apply_symmetry_in_wn(paramsd,G4r)

    G4r = Operation.apply_symmetry_in_orb(paramsd,G4r)

    Operation.check_orb_symmetry(paramsd,G4r) 
    
    if symmetry_layer and symmetry_C4:
        G4r = Operation.apply_symmetry_in_layer(model,paramsd,G4r);print('symmetry layer and C4 G4r')
        Operation.check_layer_symmetry(model,paramsd,G4r)

        G4r = Operation.apply_symmetry_in_C4(model,paramsd,G4r)
        Operation.check_C4_symmetry(paramsd,G4r)

        G4M = G4r.reshape(nt, nt)
    elif symmetry_layer and symmetry_C4 == False:
        G4r = Operation.apply_symmetry_in_layer(model,paramsd,G4r);print('symmetry layer G4r')
        Operation.check_layer_symmetry(model,paramsd,G4r)

        G4M = G4r.reshape(nt, nt)
    elif symmetry_layer == False and symmetry_C4:
        G4r = Operation.apply_symmetry_in_C4(model,paramsd,G4r);print('symmetry C4 G4r')
        Operation.check_C4_symmetry(paramsd,G4r)

        G4M = G4r.reshape(nt, nt)
    elif symmetry_layer == False and symmetry_C4 == False:
        G4M = G4r.reshape(nt, nt)

    if symmetry_transpose and paramsd['vertex_channel'] in ("PARTICLE_PARTICLE_SUPERCONDUCTING", "PARTICLE_PARTICLE_UP_DOWN"):
        G4r,G4M = Operation.apply_transpose_symmetry(paramsd,G4r)
        Operation.check_transpose_symmetry(paramsd,G4r)

    #G4rt1 = Operation.apply_symmetry_in_layer(model,paramsd,G4r)
    G4M = linalg.inv(G4M)
    chic0Mt = linalg.inv(chic0M)
    GammaM = chic0Mt - G4M

    # self.GammaM *= float(Nc)*self.invT*float(self.nOrb)
    GammaM *= float(Nc) * paramsd['invT']
    Gamma = GammaM.reshape(NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb)

    #print(G4rt2[0,1,0,0,0,1,0,0],G4rt2[0,2,0,0,0,2,0,0])
    #print(G4rt[0,1,0,1,0,1,0,1],G4rt[0,2,0,1,0,2,0,1])
    #print(Gamma[0,1,0,0,0,1,0,0],Gamma[0,2,0,0,0,2,0,0])
    print("buildGamma over!",'\n')
    return G4r,Gamma,GammaM

'''
def calcGamma_d(self):
        
        See MJ's paper on extended Hubbard model in 2018 PRB
        
        print("Now calculating frequency dependence of the d-wave-projected pairing interaction", '\n')

        Nc = self.Nc;NwG4 = self.NwG4;nOrb = self.nOrb
        NwG4_2 = int(NwG4 / 2)

        Gamma = -self.Gamma  # neg sign is for convention of Gamma
        gkd = cos(self.Kvecs[:, 0]) - cos(self.Kvecs[:, 1])
        Gd = zeros((NwG4_2, nOrb))
        norm = 0.

        for iK1 in range(Nc):
            norm += gkd[iK1] * gkd[iK1]

        for io in range(nOrb):
            for iw1 in range(NwG4_2):
                for iK1 in range(Nc):
                    for iK2 in range(Nc):
                        Gd[iw1, io] += gkd[iK1] * real(Gamma[iw1 + NwG4_2, iK1, io, io, NwG4_2, iK2, io, io]) * gkd[iK2]
        if norm != 0:
            Gd /= norm
'''

