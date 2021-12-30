#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:13:33 2021

@author: MattMansell
"""

import numpy as np


class Integrator():
    
    def __init__(self,at,seed):
        self.rng = np.random.default_rng(seed=seed)
        self.at = at
        self.neigh = None
        self.stepnum = 0
        pass
    
    
    def step(self):
        raise NotImplementedError('Child classes of Integrator must ' +
                    'implement')



class IntegratorMC_mu_geom_T_1(Integrator):
    
    def __init__(self,at,mu,geom,T,max_move,seed):
        super().__init__(at,seed)
        self.mu = mu
        self.geom = geom
        self.T = T
        self.max_move = max_move
        
        self.N = self.at.n
        self.P = np.nan
        self.en = np.nan
        # super().__init__(at)
        return
    
    
    
    def attempt_translation(self,ind=None,dx=None):
        
        if ind is None:
            ind = self.rng.randint(0,self.at.n)
        
        if dx is None:
            dx = (self.rng.uniform(size=self.geom.ndim)-0.5)*2.0*self.max_move
        
        itype = self.at.atype[ind]
        ext = self.at.external_type[itype]
        
        x0 = self.at.x[ind]
        im0 = self.at.im[ind]
        x1,im1 = self.geom.wrap_position(x0 + dx,im0)
        
        en_external0,f_ext0 = ext.phi(x0)
        en_external1,f_ext1 = ext.phi(x1)
        en_pair0 = np.zeros((self.neigh.nn[ind]),dtype=self.at.en_pair.dtype)
        en_pair1 = en_pair0.copy()
        f_pair0 = np.zeros((self.neigh.nn[ind],self.at.f.shape[1]),dtype=self.at.f.dtype)
        f_pair1 = f_pair0.copy()
        
        for k,j in enumerate(self.neigh.l[ind,:self.neigh.nn[ind]]):
            jtype = self.at.atype[j]
            pair = self.at.pair_type[(itype,jtype)]
            
            d0,dsq0,dr0 = self.geom.distance(x0,self.at.x[j])
            d1,dsq1,dr1 = self.geom.distance(x1,self.at.x[j])
            
            en,f = pair.phi(dsq0)
            en_pair0[k] = en
            f_pair0[k]  = -dr0*f
            
            en,f = pair.phi(dsq1)
            en_pair1[k] = en
            f_pair1[k]  = -dr1*f
            
        total_energy_change = ( en_external1 + np.sum(en_pair1) -
                                en_external0 - np.sum(en_pair0) )
        
        if total_energy_change <= 0.0:
            accept = True
        else:
            bf = np.exp(-total_energy_change/(self.kB*self.T)) # Boltzmann factor
            test_random = self.rng.uniform()
            if bf > test_random:
                accept = True
            else:
                accept = False
        
        if accept == False:
            self.last_accept = False
        elif accept == True:
            self.last_accept = True
            self.at.x[ind] = x1
            self.at.im[ind] = im1
            self.at.en_external[ind] = en_external1
            self.at.en_pair[ind] = np.sum(en_pair1)
            self.at.f[ind] = f_ext1 + np.sum(f_pair1,axis=0)
            
            for k,j in enumerate(self.neigh.l[ind,:self.neigh.nn[ind]]):
                self.at.en_pair[j] += en_pair1[k] - en_pair0[k]
                self.at.f[j]       -= f_pair1[k] - f_pair0[k]
            
            self.force.en_external += en_external1 - en_external0
            self.force.en_pair += 1.0/2.0*(np.sum(en_pair1) - np.sum(en_pair0))
            self.force.en_total = self.force.en_external + self.force.en_pair
            
        else:
            raise ValueError('Unrecognized value of \'accept\'.')
            
            
            
            
            
            
            
            
            
            
            
            
            
            