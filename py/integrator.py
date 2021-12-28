#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:13:33 2021

@author: MattMansell
"""

import numpy as np


class Integrator():
    
    
    def __init__(self,at):
        self.at = at
        self.stepnum = 0
        pass
    
    
    def step(self):
        self.stepnum += 1



class IntegratorMC_mu_geom_T_1():
    
    def __init__(self,at,mu,geom,T,max_move):
        self.mu = mu
        self.geom = geom
        self.T = T
        self.max_move = max_move
        
        self.N = self.at.n
        self.P = np.nan
        self.en = np.nan
        # super().__init__(at)
        return
    
    
    def attempt_translation(ind=None,dx=None):
        
        if ind is None:
            ind = np.random.randint(0,self.at.n)
        
        if dx is None:
            dx = (np.random.uniform(size=self.geom.ndim)-0.5)*2.0*self.max_move
        
        atype = self.at.atype[ind]
        
        x0 = self.at.x[ind]
        l0 = self.at.neighbors[0].calc_one_atom(ind)
        en0 = self.at.external_type[atype].phi(x0)
        for j in l0:
            _,dsq,_ = self.geom.distance(x0,self.at.x[j])
            en0 += self.at.pair_type[(atype,self.at.atype[j])].phi(dsq)
        
        x1 = x0 + dx
        self.at.translate_atom(ind,)
        l0 = self.at.neighbors[0].calc_one_atom(ind)
        
        
        
        xnew = self.at.x[ind] + dx
        
    
    

