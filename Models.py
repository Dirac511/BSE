import numpy as np
import math
from numpy import *
import matplotlib.pyplot as mpl
import h5py
import sys
import os
from matplotlib.pyplot import *
import matplotlib as mpll

def readin_model(model, f):

    if isinstance(f, str):
        f = h5py.File(f, 'r')
        close_file = True
    else:
        close_file = False
    params = {}
    try:
        if model == 'bilayer':
            params['e1'] = array(f['parameters']['bilayer-Hubbard-model']['e1'])[0]
            print("e1 = ", params['e1'])
            params['e2'] = array(f['parameters']['bilayer-Hubbard-model']['e2'])[0]
            print("e2 = ", params['e2'])
            params['U1']= array(f['parameters']['bilayer-Hubbard-model']['U1'])[0]
            print("U1 = ", params['U1'])
            params['U2'] = array(f['parameters']['bilayer-Hubbard-model']['U2'])[0]
            print("U2 = ", params['U2'])
            params['t'] = np.array(f['parameters']['bilayer-Hubbard-model']['t1'])[0]
            print("t = ", params['t'])
            params['tp'] = array(f['parameters']['bilayer-Hubbard-model']['t1-prime'])[0]
            print("t-prime = ", params['tp'])
            params['tperp'] = array(f['parameters']['bilayer-Hubbard-model']['t-perp'])[0]
            print("tperp = ", params['tperp'])

        if model == 'trilayer':
            params['e1'] = array(f['parameters']['trilayer-Hubbard-model']['e1'])[0]
            print("e1 = ", params['e1'])
            params['e2'] = array(f['parameters']['trilayer-Hubbard-model']['e2'])[0]
            print("e2 = ", params['e2'])
            params['e3'] = array(f['parameters']['trilayer-Hubbard-model']['e3'])[0]
            print("e3 = ", params['e3'])
            params['U1']= array(f['parameters']['trilayer-Hubbard-model']['U1'])[0]
            print("U1 = ", params['U1'])
            params['U2'] = array(f['parameters']['trilayer-Hubbard-model']['U2'])[0]
            print("U2 = ", params['U2'])
            params['U3'] = array(f['parameters']['trilayer-Hubbard-model']['U3'])[0]
            print("U3 = ", params['U3'])
            params['tp'] = array(f['parameters']['trilayer-Hubbard-model']['t1-prime'])[0]
            print("t-prime = ", params['tp'])
            params['t'] = np.array(f['parameters']['trilayer-Hubbard-model']['t1'])[0]
            print("t = ", params['t'])
            params['tperp12'] = array(f['parameters']['trilayer-Hubbard-model']['t-perp12'])[0]
            print("tperp12 = ", params['tperp12'])
            params['tperpp12'] = array(f['parameters']['trilayer-Hubbard-model']['t-perp-prime12'])[0]
            print("tperp_p12 = ", params['tperpp12'])
            params['tperp23'] = array(f['parameters']['trilayer-Hubbard-model']['t-perp23'])[0]
            print("tperp23 = ", params['tperp23'])
            params['tperpp23'] = array(f['parameters']['trilayer-Hubbard-model']['t-perp-prime23'])[0]
            print("tperp_p23 = ", params['tperpp23'])
            params['coefp'] = array(f['parameters']['trilayer-Hubbard-model']['coefp'])[0]
            print("coefp = ", params['coefp'])
            params['coefm'] = array(f['parameters']['trilayer-Hubbard-model']['coefm'])[0]
            print("coefm = ", params['coefm'])

        if model == 'fivelayer':
            params['e1'] = array(f['parameters']['fivelayer-Hubbard-model']['e1'])[0]
            print("e1 = ", params['e1'])
            params['e2'] = array(f['parameters']['fivelayer-Hubbard-model']['e2'])[0]
            print("e2 = ", params['e2'])
            params['e3'] = array(f['parameters']['fivelayer-Hubbard-model']['e3'])[0]
            print("e3 = ", params['e3'])
            params['e4'] = array(f['parameters']['fivelayer-Hubbard-model']['e4'])[0]
            print("e4 = ", params['e4'])
            params['e5'] = array(f['parameters']['fivelayer-Hubbard-model']['e5'])[0]
            print("e5 = ", params['e5'])

            params['U1'] = array(f['parameters']['fivelayer-Hubbard-model']['U1'])[0]
            print("U1 = ", params['U1'])
            params['U2'] = array(f['parameters']['fivelayer-Hubbard-model']['U2'])[0]
            print("U2 = ", params['U2'])
            params['U3'] = array(f['parameters']['fivelayer-Hubbard-model']['U3'])[0]
            print("U3 = ", params['U3'])
            params['U4'] = array(f['parameters']['fivelayer-Hubbard-model']['U4'])[0]
            print("U4 = ", params['U4'])
            params['U5'] = array(f['parameters']['fivelayer-Hubbard-model']['U5'])[0]
            print("U5 = ", params['U5'])

            params['t1'] = array(f['parameters']['fivelayer-Hubbard-model']['t1'])[0]
            print("t1 = ", params['t1'])
            params['t2'] = array(f['parameters']['fivelayer-Hubbard-model']['t2'])[0]
            print("t2 = ", params['t2'])
            params['t3'] = array(f['parameters']['fivelayer-Hubbard-model']['t3'])[0]
            print("t3 = ", params['t3'])
            params['t4'] = array(f['parameters']['fivelayer-Hubbard-model']['t4'])[0]
            print("t4 = ", params['t4'])
            params['t5'] = array(f['parameters']['fivelayer-Hubbard-model']['t5'])[0]
            print("t5 = ", params['t5'])

            params['t1p'] = array(f['parameters']['fivelayer-Hubbard-model']['t1-prime'])[0]
            print("t1-prime = ", params['t1p'])
            params['t2p'] = array(f['parameters']['fivelayer-Hubbard-model']['t2-prime'])[0]
            print("t2-prime = ", params['t2p'])
            params['t3p'] = array(f['parameters']['fivelayer-Hubbard-model']['t3-prime'])[0]
            print("t3-prime = ", params['t3p'])
            params['t4p'] = array(f['parameters']['fivelayer-Hubbard-model']['t4-prime'])[0]
            print("t4-prime = ", params['t4p'])
            params['t5p'] = array(f['parameters']['fivelayer-Hubbard-model']['t5-prime'])[0]
            print("t5-prime = ", params['t5p'])

            params['tperp'] = array(f['parameters']['fivelayer-Hubbard-model']['t-perp'])[0]
            print("tperp = ", params['tperp'])
            params['tperpp'] = array(f['parameters']['fivelayer-Hubbard-model']['t-perp-prime'])[0]
            print("tperp_p = ", params['tperpp'])

            params['V'] = array(f['parameters']['fivelayer-Hubbard-model']['V'])[0]
            print("V = ", params['V'])
            params['Vp'] = array(f['parameters']['fivelayer-Hubbard-model']['V-prime'])[0]
            print("Vp = ", params['Vp'])

        if model == 'La3Ni2O7':
            params['ex'] = array(f['parameters']['La3Ni2O7-model']['ex'])[0]
            print("ex = ", params['ex'])
            params['ez'] = array(f['parameters']['La3Ni2O7-model']['ez'])[0]
            print("ez = ", params['ez'])
            params['eb'] = array(f['parameters']['La3Ni2O7-model']['eb'])[0]
            print("eb = ", params['eb'])
            params['et'] = array(f['parameters']['La3Ni2O7-model']['et'])[0]
            print("et = ", params['et'])

            params['t11x'] = array(f['parameters']['La3Ni2O7-model']['t11x'])[0]
            print("t11x = ", params['t11x'])
            params['t11xy'] = array(f['parameters']['La3Ni2O7-model']['t11xy'])[0]
            print("t11xy = ", params['t11xy'])
            params['t11xx'] = array(f['parameters']['La3Ni2O7-model']['t11xx'])[0]
            print("t11xx = ", params['t11xx'])

            params['t22x'] = array(f['parameters']['La3Ni2O7-model']['t22x'])[0]
            print("t22x = ", params['t22x'])
            params['t22xy'] = array(f['parameters']['La3Ni2O7-model']['t22xy'])[0]
            print("t22xy = ", params['t22xy'])
            params['t22xx'] = array(f['parameters']['La3Ni2O7-model']['t22xx'])[0]
            print("t22xx = ", params['t22xx'])

            params['t12x'] = array(f['parameters']['La3Ni2O7-model']['t12x'])[0]
            print("t12x = ", params['t12x'])
            params['t12xx'] = array(f['parameters']['La3Ni2O7-model']['t12xx'])[0]
            print("t12xx = ", params['t12xx'])

            params['s110'] = array(f['parameters']['La3Ni2O7-model']['s110'])[0]
            print("s110 = ", params['s110'])
            params['s11x'] = array(f['parameters']['La3Ni2O7-model']['s11x'])[0]
            print("s11x = ", params['s11x'])
            params['s11xy'] = array(f['parameters']['La3Ni2O7-model']['s11xy'])[0]
            print("s11xy = ", params['s11xy'])
            params['s11xx'] = array(f['parameters']['La3Ni2O7-model']['s11xx'])[0]
            print("s11xx = ", params['s11xx'])

            params['s220'] = array(f['parameters']['La3Ni2O7-model']['s220'])[0]
            print("s220 = ", params['s220'])
            params['s22x'] = array(f['parameters']['La3Ni2O7-model']['s22x'])[0]
            print("s22x = ", params['s22x'])
            params['s22xy'] = array(f['parameters']['La3Ni2O7-model']['s22xy'])[0]
            print("s22xy = ", params['s22xy'])
            params['s22xx'] = array(f['parameters']['La3Ni2O7-model']['s22xx'])[0]
            print("s22xx = ", params['s22xx'])

            params['s12x'] = array(f['parameters']['La3Ni2O7-model']['s12x'])[0]
            print("s12x = ", params['s12x'])
            params['s12xx'] = array(f['parameters']['La3Ni2O7-model']['s12xx'])[0]
            print("s12xx = ", params['s12xx'])
    except (KeyError, AttributeError):
        print('No this model:',model)

    if close_file:
        f.close()

    return params

def dispersion(model, paramsd, paramsm, kx, ky):
    try:
        if model == 'bilayer':
            ek = np.zeros((paramsd['nOrb'], paramsd['nOrb']), dtype='complex')
            r11 = paramsm['e1'] - 2. * paramsm['t'] * (cos(kx) + cos(ky)) - 4.0 * paramsm['tp'] * cos(kx) * cos(ky)
            r22 = paramsm['e2'] - 2. * paramsm['t'] * (cos(kx) + cos(ky)) - 4.0 * paramsm['tp'] * cos(kx) * cos(ky)
            r12 = -paramsm['tperp']

            ek[0, 0] = r11
            ek[1, 1] = r22
            ek[0, 1] = r12
            ek[1, 0] = r12

        if model == 'trilayer':
            ek = np.zeros((paramsd['nOrb'], paramsd['nOrb']), dtype='complex')
            r11 = paramsm['e1'] - 2. * paramsm['t'] * (cos(kx) + cos(ky)) - 4.0 * paramsm['tp'] * cos(kx) * cos(ky)
            r22 = paramsm['e2'] - 2. * paramsm['t'] * (cos(kx) + cos(ky)) - 4.0 * paramsm['tp'] * cos(kx) * cos(ky)
            r33 = paramsm['e3'] - 2. * paramsm['t'] * (cos(kx) + cos(ky)) - 4.0 * paramsm['tp'] * cos(kx) * cos(ky)
            r12 = -paramsm['tperp12'] - 2. * paramsm['coefp']  * paramsm['tperpp12'] * (cos(kx) + cos(ky)) - \
                  2. * paramsm['coefm'] * paramsm['tperpp12'] * (cos(kx) - cos(ky))
            r23 = -paramsm['tperp23'] - 2. * paramsm['coefp']  * paramsm['tperpp23'] * (cos(kx) + cos(ky)) - \
                  2. * paramsm['coefm'] * paramsm['tperpp23'] * (cos(kx) - cos(ky))

            ek[0, 0] = r11
            ek[1, 1] = r22
            ek[2, 2] = r33
            ek[0, 1] = r12
            ek[1, 2] = r23
            ek[1, 0] = r12
            ek[2, 1] = r23

        if model == 'fivelayer':
            ek = np.zeros((paramsd['nOrb'], paramsd['nOrb']), dtype='complex')

            def intra_layer(e, t, tp, kx, ky):
                return e - 2. * t * (cos(kx) + cos(ky)) - 4.0 * tp * cos(kx) * cos(ky)

            # Diagonal (intra-layer) terms
            ek[0, 0] = intra_layer(paramsm['e1'], paramsm['t1'], paramsm['t1p'], kx, ky)
            ek[1, 1] = intra_layer(paramsm['e2'], paramsm['t2'], paramsm['t2p'], kx, ky)
            ek[2, 2] = intra_layer(paramsm['e3'], paramsm['t3'], paramsm['t3p'], kx, ky)
            ek[3, 3] = intra_layer(paramsm['e4'], paramsm['t4'], paramsm['t4p'], kx, ky)
            ek[4, 4] = intra_layer(paramsm['e5'], paramsm['t5'], paramsm['t5p'], kx, ky)

            t_ij = -paramsm['tperp'] - 2. * paramsm['tperpp'] * (cos(kx) - cos(ky))

            ek[0, 1] = t_ij
            ek[1, 0] = t_ij
            ek[1, 2] = t_ij
            ek[2, 1] = t_ij
            ek[2, 3] = t_ij
            ek[3, 2] = t_ij
            ek[3, 4] = t_ij
            ek[4, 3] = t_ij

        if model=='La3Ni2O7':
           ek = np.zeros((paramsd['nOrb'], paramsd['nOrb']), dtype='complex')
           Hx0  = paramsm['ex'] + paramsm['eb'] + 2.*paramsm['t11x']*(cos(kx)+cos(ky)) + 4.*paramsm['t11xy']*cos(kx)*cos(ky) + 2.*paramsm['t11xx']*(cos(2*kx)+cos(2*ky))
           Hz1  = paramsm['ez'] + paramsm['eb'] + 2.*paramsm['t22x']*(cos(kx)+cos(ky)) + 4.*paramsm['t22xy']*cos(kx)*cos(ky) + 2.*paramsm['t22xx']*(cos(2*kx)+cos(2*ky))
           Hx2  = paramsm['ex'] + paramsm['et'] + 2.*paramsm['t11x']*(cos(kx)+cos(ky)) + 4.*paramsm['t11xy']*cos(kx)*cos(ky) + 2.*paramsm['t11xx']*(cos(2*kx)+cos(2*ky))
           Hz3  = paramsm['ez'] + paramsm['et'] + 2.*paramsm['t22x']*(cos(kx)+cos(ky)) + 4.*paramsm['t22xy']*cos(kx)*cos(ky) + 2.*paramsm['t22xx']*(cos(2*kx)+cos(2*ky))
           V   = 2.*paramsm['t12x']*(cos(kx)-cos(ky)) + 2.*paramsm['t12xx']*(cos(2*kx)-cos(2*ky))

           Hxp = paramsm['s110'] + 2.*paramsm['s11x']*(cos(kx)+cos(ky)) + 4. * paramsm['s11xy']*cos(kx)*cos(ky) + 2.*paramsm['s11xx']*(cos(2*kx)+cos(2*ky))
           Hzp = paramsm['s220'] + 2.*paramsm['s22x']*(cos(kx)+cos(ky)) + 4. * paramsm['s22xy']*cos(kx)*cos(ky) + 2.*paramsm['s22xx']*(cos(2*kx)+cos(2*ky))
           Vp  = 2.*paramsm['s12x']*(cos(kx)-cos(ky)) + 2.*paramsm['s12xx']*(cos(2.*kx)-cos(2.*ky))

           ek[0,0] = Hx0 ; ek[0,1] = V   ; ek[0,2] = Hxp ; ek[0,3] = Vp
           ek[1,0] = V   ; ek[1,1] = Hz1 ; ek[1,2] = Vp  ; ek[1,3] = Hzp
           ek[2,0] = Hxp ; ek[2,1] = Vp  ; ek[2,2] = Hx2 ; ek[2,3] = V
           ek[3,0] = Vp  ; ek[3,1] = Hzp ; ek[3,2] = V   ; ek[3,3] = Hz3
    except (KeyError, AttributeError):
        print('No this model dispersion:',model)

    return ek
