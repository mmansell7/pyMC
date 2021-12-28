#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:10:18 2021

@author: MattMansell
"""

import numpy as np
import pair

class PairLJCut(pair.Pair):
    
    def __init__(self,epsilon,sigma,rc):
        if epsilon < 0.0:
            raise ValueError('Epsilon must be greater than zero.')
        self.epsilon = epsilon
        if sigma < 0.0:
            raise ValueError('Sigma must be greater than zero.')
        self.sigma   = sigma
        self.sigma2  = np.square(self.sigma)
        if rc < 0.0:
            raise ValueError('Cutoff distance be greater than zero.')
        self.rc  = rc
        self.rc2 = np.square(self.rc)
    
    def phi(self,rsq):
        if rsq >= self.rc2:
            return 0.0,0.0
        rsqinv = self.sigma2/rsq
        r6    = (np.power(rsqinv,3))
        r12   = np.square(r6)
        en    = 4.0*self.epsilon*(r12-r6)
        f     = 24.0*self.epsilon/rsq*(2.0*r12-r6) # This is f/r, NOT f
        return en,f
    
    