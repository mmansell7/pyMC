#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:48:44 2021

@author: MattMansell
"""

import pandas as pd
import numpy as np

mustar    = 1.0 # mu/(kB*T)
Hestar    = 5.0 # sigmaArAr
T         = 135 # K
seed      = 1

rng       = np.random.default_rng(seed)
kB        = 1.380649E-23 # J/K (Boltzmann constant)
mu        = mustar * kB * T
sigmaArAr = 0.3405 # nm
rcArAr    = 5.0*sigmaArAr # nm
epsArAr   = 119.8 * kB # J
sigmaCC   = 0.340 # nm
rcCC      = 5.0*sigmaCC # nm
epsCC     = 28.0 * kB # J
kC        = 18.1E-18  # J/m**2 (C-C spring constant)
sigmaArC  = (sigmaArAr + sigmaCC) / 2.0 # nm
rcArC     = 5.0*sigmaArC # nm
epsArC    = np.sqrt(epsArAr*epsCC) # J

Lxwall = 3.408  # nm
Lywall = 6.8866 # nm
He     = Hestar * sigmaArAr  # nm
d002e  = {87.3:0.3395, 135:0.3405, 300:0.3438}[T] # nm
lz = He + 5.0*d002e
box = np.array([[-Lxwall/2.0,Lxwall/2.0],
                [-3.0*Lywall/2.0,3.0*Lywall/2.0],
                [-lz/2.0,lz/2.0]])

x0 = np.array([0.0,Lywall,0.0]) # Origin for present unit cell of the lattice
d1 = 0.142
d4 = d1*np.sin(np.pi/3.0)
lc = []  # Lattice coordinates
Czs = box[2,1]
Czs = np.array([-Czs,-Czs+d002e,-Czs+2.0*d002e,Czs-2.0*d002e,Czs-d002e,Czs])
for k in range(0,6):
    x0[0] = box[0,0]
    while x0[0] < box[0,1]-0.001:
        x0[1] = -0.5*Lywall
        while x0[1] < 0.5*Lywall-0.001:
            lc.append(x0+np.array([0.0   , d4,Czs[k]]))
            lc.append(x0+np.array([0.5*d1,0.0,Czs[k]]))
            lc.append(x0+np.array([1.5*d1,0.0,Czs[k]]))
            lc.append(x0+np.array([2.0*d1, d4,Czs[k]]))
            x0[1] += 2.0*d4
        x0[0] += 3.0*d1
lc = np.array(lc)

xC = lc.copy()
NC = xC.shape[0]
xAr = rng.random((1000,3))
xAr[:,0] = box[0,0] + (box[0,1]-box[0,0])*xAr[:,0]
xAr[:,1] = box[1,0] + (box[1,1]-box[1,0])*xAr[:,1]
xAr[:,2] = -0.1 + 0.2*xAr[:,2]
NAr = xAr.shape[0]
x = np.concatenate((xC,xAr,np.empty((5000,3))))
x[-5000:] = np.nan
im = np.zeros(x.shape,dtype='int')
tags  = np.arange(0,x.shape[0],dtype='int')
tags[-5000:] = -1
types = np.empty(x.shape[0],dtype='<U8')
types[           :  int(NC/6)]   = 'C0'
types[  int(NC/6):2*int(NC/6)]   = 'C1'
types[2*int(NC/6):3*int(NC/6)]   = 'C2'
types[3*int(NC/6):4*int(NC/6)]   = 'C3'
types[4*int(NC/6):5*int(NC/6)]   = 'C4'
types[5*int(NC/6):  int(NC)  ]   = 'C5'
types[int(NC):int(NC)+int(NAr)]  = 'Ar'
types[int(NC)+int(NAr):]         = ''

def wrap(x,im,bbox):
    while x[0] < bbox[0,0]:
        x[0] += (bbox[0,1]-bbox[0,0])
        im[0] += 1
    while x[0] >= bbox[0,1]:
        x[0] -= (bbox[0,1]-bbox[0,0])
        im[0] -= 1
    return x

def dist(x1,x2,bbox):
    ds = x2.copy() - x1
    l  = (bbox[:,1]-bbox[:,0]).flatten()
    while ds[0] < -l[0]/2.0:
        ds[0] += l[0]
    while ds[0] >= l[0]/2.0:
        ds[0] -=  l[0]
    dd = np.sqrt(np.dot(ds,ds))
    return dd,ds

def is_outside_box(xx,bbox):
    if ((xx[0] < bbox[0,0]) or (xx[0] >= bbox[0,1])
    or (xx[1] < bbox[1,0]) or (xx[1] >= bbox[1,1])
    or (xx[2] < bbox[2,0]) or (xx[2] >= bbox[2,1])):
        return True
    return False

def energyCexternal(i):
    tt = types[i]
    if tt in ['C0','C5']:
        return 0.0
    if is_outside_box(x[i],box):
        return np.infty
    
    _,dv = dist(lc[i],x[i],box)
    dd = np.sqrt(np.dot(dv[:2],dv[:2]))
    return 0.5*kC*dd**2

def energyArexternal(i):
    if is_outside_box(x[i],box):
        return np.infty
    return 0.0

def energyCC(i,j):
    dd,_ = dist(x[i],x[j],box)
    if dd >= rcCC:
        return 0.0
    dd   = (sigmaCC/dd)**2
    dd6  = dd**3
    dd12 = dd6*dd6
    return 4.0*epsCC*(dd12-dd6)

def energyArAr(i,j):
    dd,_ = dist(x[i],x[j],box)
    if dd >= rcArAr:
        return 0.0
    dd   = (sigmaArAr/dd)**2
    dd6  = dd**3
    dd12 = dd6*dd6
    return 4.0*epsArAr*(dd12-dd6)

def energyArC(i,j):
    dd,_ = dist(x[i],x[j],box)
    if dd >= rcArC:
        return 0.0
    dd   = (sigmaArC/dd)**2
    dd6  = dd**3
    dd12 = dd6*dd6
    return 4.0*epsArC*(dd12-dd6)

def energy():
    E = 0.0
    for i in range(0,x.shape[0]):
        if i % 100 == 0: print(i)
        tti = types[i]
        if tti in ['C0','C1','C2','C3','C4','C5']:
            E += energyCexternal(i)
            for j in range(i+1,x.shape[0]):
                ttj = types[j]
                if ttj in ['C0','C1','C2','C3','C4','C5']:
                    if (ttj != tti):
                        E += energyCC(i,j)
                    else:
                        pass
                elif ttj == 'Ar':
                    E += energyArC(i,j)
                elif not ttj:
                    break
                else:
                    raise ValueError('A Unknown type {} for particle at index {}'.format(ttj,j))
        elif tti == 'Ar':
            E += energyArexternal(i)
            for j in range(i+1,x.shape[0]):
                ttj = types[j]
                if ttj in ['C0','C1','C2','C3','C4','C5']:
                    E += energyArC(i,j)
                elif ttj == 'Ar':
                    E += energyArAr(i,j)
                elif not ttj:
                    break
                else:
                    raise ValueError('B Unknown type {} for particle at index {}'.format(ttj,j))
        elif not tti:
            break
        else:
            raise ValueError('C Unknown type {} for particle at index {}'.format(tti,i))
    return E


        




