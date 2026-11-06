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

###########################################################################################
def buildKernelMatrix(model,paramsd,chic0M,chi0M,Gamma,GammaM,isLattice,build_sym_kernel):
    print("Build kernel matrix GammaIrrCluster(=GammaIrrLattice) * coarse_grained chi0")
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    # Build kernel matrix Gamma*chi0

    if isLattice:
        chiM = chi0M
    else:
        chiM = chic0M

    pm = np.dot(GammaM, chiM)

    # self.pm *= 1.0/(self.invT*float(self.Nc)*float(self.nOrb))
    pm *= 1.0 / (paramsd['invT'] * float(Nc))

    if paramsd['vertex_channel'] in ("PARTICLE_PARTICLE_SUPERCONDUCTING", \
                               "PARTICLE_PARTICLE_UP_DOWN", \
                               "PARTICLE_PARTICLE_SINGLET") and build_sym_kernel == True:
        # 1. one way to symmetrize the pairing kernel (note that abs might be wrong, but otherwise NaN error)
        pm2 = np.dot(sqrt(abs(real(chiM))), np.dot(real(GammaM), sqrt(abs(real(chiM)))))
        pm2 *= 1.0 / (paramsd['invT'] * float(Nc))

        # 2. another way to symmetrize the pairing kernel
        # see PRB 103, 144514 (2021) Eq.(8)
        wtemp, vtemp = linalg.eig(chiM)
        wttemp = abs(wtemp - 1)
        ileadtemp = argsort(wttemp)
        lambdastemp = wtemp[ileadtemp]
        evecstemp = vtemp[:, ileadtemp]

        Lambdatemp = sqrt(np.diag(lambdastemp))
        chiMasqrt = np.dot(evecstemp, np.dot(Lambdatemp, linalg.inv(evecstemp)))

        pm3 = np.dot(chiMasqrt, np.dot(GammaM, chiMasqrt))
        pm3 *= 1.0 / (paramsd['invT'] * float(Nc))

        # 3. Build symmetric kernel matrix M = 0.5*(Gamma(wn,wn')*chi0(wn')+Gamma(wn,-wn')*chi0(-wn')
        GammaTemp = Gamma.reshape(NwG4, Nc * nOrb * nOrb, NwG4, Nc * nOrb * nOrb)
        Gamma2 = np.zeros_like(GammaTemp)
        for iw1 in range(NwG4):
            for iw2 in range(NwG4):
                Gamma2[iw1, :, iw2, :] = GammaTemp[iw1, :, NwG4 - iw2 - 1]
        Gamma2 = Gamma2.reshape(NwG4 * Nc * nOrb * nOrb, NwG4 * Nc * nOrb * nOrb)
        pm4 = 0.5 * (np.dot(GammaM, chiM) + np.dot(Gamma2, conj(chiM)))
        # self.pm *= 1.0/(self.invT*float(self.Nc)*float(self.nOrb))
        pm4 *= 1.0 / (paramsd['invT'] * float(Nc))
    print("buildKernelMatrix over!",'\n')
    return pm

#####################################################################################
def calcKernelEigenValues(model,pm,paramsd,Tval,Num,write_data_file):
    print("calKernelMatrix begin!")
    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];NwG = paramsd['NwG'];nOrb = paramsd['nOrb'];nt = paramsd['nt']
    w, v = linalg.eig(pm)
    wt = abs(w-1)
    ilead = argsort(wt)
    lambdas = w[ilead]
    evecs = v[:, ilead]
    evecs = evecs.reshape(NwG4, Nc, nOrb, nOrb, nt)
    if model == 'bilayer' or model == 'trilayer' or model == 'La3Ni2O7':
        Operation.trans_basis(model,paramsd,Tval,evecs,Num,write_data_file)

    print('\n', "Leading 16 eigenvalues of BSE (no symmetrization)", Tval, lambdas[0:Num])

    if write_data_file:
        fname = 'leading_Evec_vs_K_T' + str(Tval) + '.txt'
        fnamelam = 'lambda_vs_K_T' + str(Tval) + '.txt'
        fnamew = 'leading_wn_Evec_vs_K_T' + str(Tval) + '.txt'
        if os.path.isfile(fname):
            os.remove(fname)
        ilams = [];lambdass = [] 
        for ilam in range(0,nt):
            ilams.append(ilam);lambdass.append(lambdas[ilam])
        Operation.write_data_2cols(fnamelam,ilams,lambdass)

        for io1 in range(nOrb):
            for io2 in range(nOrb):
                o1s = np.full((1, 1), io1)
                o2s = np.full((1, 1), io2)
                O1s = np.full((NwG4, 1), io1)
                O2s = np.full((NwG4, 1), io2)
                # print(o1s,o2s)
                # print first two leading Evec to include cases
                for ilam in range(0, Num):
                    for iNc in range(Nc):
                        kx = paramsd['Kvecs'][iNc, 0];ky = paramsd['Kvecs'][iNc, 1]
                        kxs = np.full((1, 1), kx)
                        kys = np.full((1, 1), ky)
                        Kxs = np.full((NwG4, 1), kx)
                        Kys = np.full((NwG4, 1), ky)
                        # print(kxs,kys)
                        ## wirte the data of orbital-basis eigenvector at lowest frequency
                        Operation.write_data_6cols(fname,o1s,o2s,kxs,kys,paramsd['wnSet'][int(NwG4/2)],evecs[int(NwG4/2):int(NwG4/2)+1,iNc,io1,io2,ilam])
                        ## wirte the data of orbital-basis eigenvector for all frequency 
                        Operation.write_data_w_6cols(fnamew,O1s,O2s,Kxs,Kys,paramsd['wnSet'],evecs[:,iNc,io1,io2,ilam])
    print('\n',"calKernelMatrix over!",'\n')

def calcReducibleLatticeVertex(paramsd,pm,GammaM):
    # see Eq.(22-30) in PRB 64, 195130(2001), GammaRed is that bar(T2)
    nt = paramsd['nt'];Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']
    pminv = np.linalg.inv(np.identity(nt) - pm)
    # self.pminv = np.linalg.inv(np.identity(nt)+pm)
    GammaRed = dot(pminv, GammaM)
    GammaRed = GammaRed.reshape(NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb)
    print('Have built reducible Gamma!')
    return GammaRed

def calcLatticeSCSus(paramsd,model,GammaRed,chi0,chi0D,chi0D2,chi0XS,chi0XS2,chi0DXY,chi0DXY2,Tval):
    '''
    For PARTICLE_PARTICLE_UP_DOWN channel
    Calculate lattice d-wave susceptibility via Eq.(22-30) in PRB 64, 195130(2001)
    '''
    # Calculate from gd*G4*gd = gd*GG*gd + gd*GG*GammaRed*GG*gd
    # GammaRed = self.GammaRed.reshape(self.NwG4*self.Nc,self.NwG4*self.Nc)
    # GammaRed is bar(T2) in Eq.(22-30) in PRB 64, 195130(2001)
    # chi0D2 and chi0D is Eq.(27-28) in PRB 64, 195130(2001)
    
    print("==============================================================================")
    print("Calculate lattice susceptibility with Eq.(22-30) in PRB 64,195130(2001)", '\n')

    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']

    csumd1 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
    csumdxy1 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
    csum1xs = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
    #ccsum1 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
    csum1spm = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')

    csumd = np.einsum('wkabwkef,wkefmngh,mnghmncd -> abcd', chi0D, GammaRed, chi0D)

    csumdxy = np.einsum('wkabwkef,wkefmngh,mnghmncd -> abcd', chi0DXY, GammaRed, chi0DXY)

    csumxs = np.einsum('wkabwkef,wkefmngh,mnghmncd -> abcd', chi0XS, GammaRed, chi0XS)

    #ccsum = np.einsum('k,wkabwkef,wkefmngh,mnghmncd,n -> abcd', gkd, chi0, GammaRed, chi0, gkd)

    csumspm = np.einsum('wkabwkef,wkefmngh,mnghmncd -> abcd', chi0, GammaRed, chi0)

    csumd[:, :, :, :] /= (Nc * paramsd['invT']) ** 2
    csumdxy[:, :, :, :] /= (Nc * paramsd['invT']) ** 2
    csumxs[:, :, :, :] /= (Nc * paramsd['invT']) ** 2
    #ccsum[:, :, :, :] /= (self.Nc * paramsd['invT']) ** 2
    csumspm[:, :, :, :] /= (Nc * paramsd['invT']) ** 2

    chi0D2sim = np.zeros((NwG4, Nc, nOrb, nOrb, nOrb, nOrb), dtype='complex')
    chi0DXY2sim = np.zeros((NwG4, Nc, nOrb, nOrb, nOrb, nOrb), dtype='complex')
    chi0XS2sim = np.zeros((NwG4, Nc, nOrb, nOrb, nOrb, nOrb), dtype='complex')
    chi0sim = np.zeros((NwG4, Nc, nOrb, nOrb, nOrb, nOrb), dtype='complex')
    for iw in range(NwG4):
        for iK in range(Nc):
            chi0D2sim[iw, iK, :, :, :, :] = chi0D2[iw, iK, :, :, iw, iK, :, :]
            chi0DXY2sim[iw, iK, :, :, :, :] = chi0DXY2[iw, iK, :, :, iw, iK, :, :]
            chi0XS2sim[iw, iK, :, :, :, :] = chi0XS2[iw, iK, :, :, iw, iK, :, :]
            chi0sim[iw, iK, :, :, :, :] = chi0[iw, iK, :, :, iw, iK, :, :]
    for l1 in range(nOrb):
        for l2 in range(nOrb):
            for l3 in range(nOrb):
                for l4 in range(nOrb):
                    csumd1[l1, l2, l3, l4] = sum(chi0D2sim[:, :, l1, l2, l3, l4]) / (Nc * paramsd['invT'])
                    csumdxy1[l1, l2, l3, l4] = sum(chi0DXY2sim[:, :, l1, l2, l3, l4]) / (Nc * paramsd['invT'])
                    csum1xs[l1, l2, l3, l4] = sum(chi0XS2sim[:, :, l1, l2, l3, l4]) / (Nc * paramsd['invT'])
                    #ccsum1[l1, l2, l3, l4] = sum(np.dot(chi0sim[:, :, l1, l2, l3, l4], gkd ** 2)) / (self.Nc * self.invT)
                    csum1spm[l1, l2, l3, l4] = sum(chi0sim[:, :, l1, l2, l3, l4]) / (Nc * paramsd['invT'])

    csumd[:, :, :, :] += csumd1[:, :, :, :]
    csumdxy[:, :, :, :] += csumdxy1[:, :, :, :]
    csumxs[:, :, :, :] += csum1xs[:, :, :, :]
    #ccsum[:, :, :, :] += ccsum1[:, :, :, :]
    csumspm[:, :, :, :] += csum1spm[:, :, :, :]

    Pd = 0;Pdxy = 0
    for l1 in range(nOrb):
        for l2 in range(nOrb):
            Pd += csumd[l1, l1, l2, l2] / 4.
            Pdxy += csumdxy[l1, l1, l2, l2]
            print(l1,l2,':',csumd[l1, l1, l2, l2] / 4.)

    if model == 'bilayer':
        Pspm = 0.25 * (csumspm[0, 0, 0, 0] + csumspm[1, 1, 1, 1] + csumspm[0, 0, 1, 1] + csumspm[1, 1, 0, 0])
        print("Total spm SC susceptibility: ", Tval, ' ', real(Pspm))

    if model == 'La3Ni2O7':
        ##for spm pair-field susceptibility with the Cluster form-factors
        a = np.array([1 / sqrt(2), 0, 1 / sqrt(2), 0])  # Bonding   x
        b = np.array([0, 1 / sqrt(2), 0, 1 / sqrt(2)])  # Bonding   z
        c = np.array([1 / sqrt(2), 0, -1 / sqrt(2), 0])  # antiBonding  x
        d = np.array([0, 1 / sqrt(2), 0, -1 / sqrt(2)])  # AntiBonding  z

        basis = [a] + [b] + [c] + [d]
        # print(basis)
        M1, M2, M3, M4 = [], [], [], []
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

        Pspm = 0;Pspmx = 0;Pspmz = 0
        csumspmK = np.zeros((2, 2), dtype='complex')
        csumspmKx = np.zeros((2, 2), dtype='complex')
        csumspmKz = np.zeros((2, 2), dtype='complex')
        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        csumspmK[0, 0] += csumspm[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[0][0][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[1][1][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[1][1][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[0][0][l3][l4]

                        csumspmK[0, 1] += csumspm[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[2][2][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[3][3][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[2][2][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[3][3][l3][l4]

                        csumspmK[1, 0] += csumspm[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[0][0][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[1][1][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[0][0][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[1][1][l3][l4]

                        csumspmK[1, 1] += csumspm[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[2][2][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[3][3][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[3][3][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[2][2][l3][l4]

                        csumspmKx[0, 0] += csumspm[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[0][0][l3][l4]
                        csumspmKx[0, 1] += csumspm[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[2][2][l3][l4]
                        csumspmKx[1, 0] += csumspm[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[0][0][l3][l4]
                        csumspmKx[1, 1] += csumspm[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[2][2][l3][l4]

                        csumspmKz[0, 0] += csumspm[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[1][1][l3][l4]
                        csumspmKz[0, 1] += csumspm[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[3][3][l3][l4]
                        csumspmKz[1, 0] += csumspm[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[1][1][l3][l4]
                        csumspmKz[1, 1] += csumspm[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[3][3][l3][l4]
                        
        for i in range(2):
            for j in range(2):
                Pspm += 0.25 * csumspmK[i, j] * cos(i * np.pi) * cos(j * np.pi)
                Pspmx += 0.25 * csumspmKx[i, j] * cos(i * np.pi) * cos(j * np.pi)
                Pspmz += 0.25 * csumspmKz[i, j] * cos(i * np.pi) * cos(j * np.pi)
        Pspm1 = Pspm - Pspmx - Pspmz
 
    print('Total ', 'spm-wave pair-feild susceptibility lattice:    ', Tval, ' ', Pspm)
    print('dx2y2 ', 'spm-wave pair-feild susceptibility lattice:    ', Tval, ' ', Pspmx)
    print('dz2 ', 'spm-wave pair-feild susceptibility lattice:    ', Tval, ' ', Pspmz)
    print('inter-orb ', 'spm-wave pair-feild susceptibility lattice:    ', Tval, ' ', Pspm1)
    print('\n')
    print('Total', 'd-wave pair-feild susceptibility : ', Tval, ' ', Pd)
    print('Total', 'dxy-wave pair-feild susceptibility : ', Tval, ' ', Pdxy)

    print('\n','Have solved SC susceptibility!')

def calcLatticeSCSus_int(paramsd,model,GammaRed,chi0,chi0D,chi0D2,chi0XS,chi0XS2,chi0DXY,chi0DXY2,Tval):
    '''
    For PARTICLE_PARTICLE_UP_DOWN channel
    Calculate lattice d-wave susceptibility via Eq.(22-30) in PRB 64, 195130(2001)
    '''
    # Calculate from gd*G4*gd = gd*GG*gd + gd*GG*GammaRed*GG*gd
    # GammaRed = self.GammaRed.reshape(self.NwG4*self.Nc,self.NwG4*self.Nc)
    # GammaRed is bar(T2) in Eq.(22-30) in PRB 64, 195130(2001)
    # chi0D2 and chi0D is Eq.(27-28) in PRB 64, 195130(2001)

    print("==============================================================================")
    print("Calculate interacting part of lattice susceptibility with Eq.(22-30) in PRB 64,195130(2001)", '\n')

    Nc = paramsd['Nc'];NwG4 = paramsd['NwG4'];nOrb = paramsd['nOrb']

    csumd1 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
    csumdxy1 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
    csum1xs = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
    #ccsum1 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
    csum1spm = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')

    csumd = np.einsum('wkabwkef,wkefmngh,mnghmncd -> abcd', chi0D, GammaRed, chi0D)

    csumdxy = np.einsum('wkabwkef,wkefmngh,mnghmncd -> abcd', chi0DXY, GammaRed, chi0DXY)

    csumxs = np.einsum('wkabwkef,wkefmngh,mnghmncd -> abcd', chi0XS, GammaRed, chi0XS)

    #ccsum = np.einsum('k,wkabwkef,wkefmngh,mnghmncd,n -> abcd', gkd, chi0, GammaRed, chi0, gkd)

    csumspm = np.einsum('wkabwkef,wkefmngh,mnghmncd -> abcd', chi0, GammaRed, chi0)

    csumd[:, :, :, :] /= (Nc * paramsd['invT']) ** 2
    csumdxy[:, :, :, :] /= (Nc * paramsd['invT']) ** 2
    csumxs[:, :, :, :] /= (Nc * paramsd['invT']) ** 2
    #ccsum[:, :, :, :] /= (self.Nc * paramsd['invT']) ** 2
    csumspm[:, :, :, :] /= (Nc * paramsd['invT']) ** 2

    Pd = 0;Pdxy = 0
    for l1 in range(nOrb):
        for l2 in range(nOrb):
            Pd += csumd[l1, l1, l2, l2] / 4.
            Pdxy += csumdxy[l1, l1, l2, l2]

    if model == 'bilayer':
        Pspm = 0.25 * (csumspm[0, 0, 0, 0] + csumspm[1, 1, 1, 1] + csumspm[0, 0, 1, 1] + csumspm[1, 1, 0, 0])
        print("Total spm SC susceptibility: ", Tval, ' ', real(Pspm))

    if model == 'La3Ni2O7':
        ##for spm pair-field susceptibility with the Cluster form-factors
        a = np.array([1 / sqrt(2), 0, 1 / sqrt(2), 0])  # Bonding   x
        b = np.array([0, 1 / sqrt(2), 0, 1 / sqrt(2)])  # Bonding   z
        c = np.array([1 / sqrt(2), 0, -1 / sqrt(2), 0])  # antiBonding  x
        d = np.array([0, 1 / sqrt(2), 0, -1 / sqrt(2)])  # AntiBonding  z

        basis = [a] + [b] + [c] + [d]
        # print(basis)
        M1, M2, M3, M4 = [], [], [], []
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

        Pspm = 0;Pspmx = 0;Pspmz = 0
        csumspmK = np.zeros((2, 2), dtype='complex')
        csumspmKx = np.zeros((2, 2), dtype='complex')
        csumspmKz = np.zeros((2, 2), dtype='complex')
        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        csumspmK[0, 0] += csumspm[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[0][0][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[1][1][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[1][1][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[0][0][l3][l4]

                        csumspmK[0, 1] += csumspm[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[2][2][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[3][3][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[2][2][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[3][3][l3][l4]

                        csumspmK[1, 0] += csumspm[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[0][0][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[1][1][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[0][0][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[1][1][l3][l4]

                        csumspmK[1, 1] += csumspm[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[2][2][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[3][3][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[3][3][l3][l4] + \
                                          csumspm[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[2][2][l3][l4]

                        csumspmKx[0, 0] += csumspm[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[0][0][l3][l4]
                        csumspmKx[0, 1] += csumspm[l1, l2, l3, l4] * Matrix[0][0][l1][l2] * Matrix[2][2][l3][l4]
                        csumspmKx[1, 0] += csumspm[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[0][0][l3][l4]
                        csumspmKx[1, 1] += csumspm[l1, l2, l3, l4] * Matrix[2][2][l1][l2] * Matrix[2][2][l3][l4]

                        csumspmKz[0, 0] += csumspm[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[1][1][l3][l4]
                        csumspmKz[0, 1] += csumspm[l1, l2, l3, l4] * Matrix[1][1][l1][l2] * Matrix[3][3][l3][l4]
                        csumspmKz[1, 0] += csumspm[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[1][1][l3][l4]
                        csumspmKz[1, 1] += csumspm[l1, l2, l3, l4] * Matrix[3][3][l1][l2] * Matrix[3][3][l3][l4]

        for i in range(2):
            for j in range(2):
                Pspm += 0.25 * csumspmK[i, j] * cos(i * np.pi) * cos(j * np.pi)
                Pspmx += 0.25 * csumspmKx[i, j] * cos(i * np.pi) * cos(j * np.pi)
                Pspmz += 0.25 * csumspmKz[i, j] * cos(i * np.pi) * cos(j * np.pi)
        Pspm1 = Pspm - Pspmx - Pspmz

    print('Total ', 'interacting spm-wave pair-feild susceptibility lattice:    ', Tval, ' ', real(Pspm))
    print('dx2y2 ', 'interacting spm-wave pair-feild susceptibility lattice:    ', Tval, ' ', real(Pspmx))
    print('dz2 ', 'interacting spm-wave pair-feild susceptibility lattice:    ', Tval, ' ', real(Pspmz))
    print('inter-orb ', 'interacting spm-wave pair-feild susceptibility lattice:    ', Tval, ' ', real(Pspm1))

    print('\n','Have solved interacting SC susceptibility!')


