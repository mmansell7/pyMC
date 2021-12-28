#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 20:50:06 2021

@author: MattMansell
"""

import numpy as np
import geometry


class CartesianBox(geometry.Geometry):
    
    def __init__(self,x,bcs):
        if not x.ndim == 2:
            raise ValueError('x must be a two-dimensional array. ' +
                             'Example: x = [[0,0,0],[1,1,1]] indicates a ' +
                             'box with corners at [0,0,0] and [1,1,1].')
        self.x    = x
        self.L    = self.x[1] - self.x[0]
        if not np.all(self.L>0):
            raise ValueError('High corner of the box must have greater' +
                             'coordinate in each dimension than low corner.')
        self.ndim = self.L.shape[0]
        
        if not np.all(np.isin(bcs,['fixed','periodic'])):
            raise ValueError('bcs must be either \'fixed\' or \'periodic\'')
        if not bcs.ndim == 1:
            raise ValueError('Boundary conditions (bc) must be a one-dimensional array')
        if not bcs.shape[0] == self.ndim:
            raise ValueError('Shape of boundary conditions must make sense ' +
                             'given shape of x.')
        self.bcs = bcs
        
        return
    
    def distance(self,r1,r2):
        dr = r2-r1
        for d in range(0,self.ndim):
            bc = self.bcs[d]
            if bc == 'fixed':
                continue
            elif bc == 'periodic':
                while dr[d] < -0.5*self.L[d]:
                    dr[d] += self.L[d]
                while dr[d] >= 0.5*self.L[d]:
                    dr[d] -= self.L[d]
        dr2 = np.dot(dr,dr)
        return np.sqrt(dr2),dr2,dr
    
    def wrap_position(self,r1,im1):
        r2 = r1.copy()
        im2 = im1.copy()
        for d in range(0,self.ndim):
            bc = self.bcs[d]
            if bc == 'periodic':
                while r2[d] < self.x[0,d]:
                    r2[d] += self.L[d]
                    im2[d] -= 1
                while r2[d] >= self.x[1,d]:
                    r2[d] -= self.L[d]
                    im2[d] += 1
        return r2,im2
    


