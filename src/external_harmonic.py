#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:10:18 2021

@author: MattMansell
"""

import numpy as np
import external

class ExternalHarmonic(external.External):
    
    def __init__(self,mc,k,r0):
        if 'distance' not in dir(mc.geom):
            raise ValueError('geom must implement a distance function')
        super().__init__(mc)
        if k <= 0.0:
            raise ValueError('k must be greater than zero.')
        self.k = k
        if not (r0.shape[0] == self.geom.ndim):
            raise ValueError('r0 must have the same number of dimensions as geom.')
        self.r0  = r0
        
    def phi(self,r):
        r0 = np.where(np.isnan(self.r0),r,self.r0)
        d,d2,dvec = self.geom.distance(r0,r)
        en = 0.5*self.k*d2
        f  = -self.k * dvec
        
        return en,f
    
    