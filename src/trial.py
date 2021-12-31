#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 21:24:09 2021

@author: mattmansell
"""

import numpy as np

class Trial():
    
    def __init__(self,mc):
        self.mc = mc
        self.grat = None
        
    def execute(self):
        raise NotImplementedError('Child classes of Trial must override ' +
                    'method execute.')
    
    
class Trial_Translation(Trial):
    
    def __init__(self,mc,max_move,itype):
        super().__init__(mc)
        self.max_move = max_move
        self.itype = itype
        
    @property
    def itype(self):
        return self.__itype
    
    @itype.setter
    def itype(self,val):
        if val not in self.mc.at.atype:
            raise ValueError('There must be at least one atom of type itype.')
        self.__itype = val
        return
    
    @property
    def ext(self):
        return self.mc.at.external_type[self.itype]
    
    @property
    def rng(self):
        return self.mc.rng
    
    @property
    def geom(self):
        return self.mc.geom
    
    @property
    def neigh(self):
        return self.mc.neigh
    
    @property
    def at(self):
        return self.mc.at
    
    @property
    def force(self):
        return self.mc.force
    
    @property
    def kB(self):
        return self.mc.kB
    
    @property
    def T(self):
        return self.grat.T
    
    def execute(self,ind=None,dx=None):
        
        if ind is None:
            itype_inds = np.argwhere(self.at.atype==self.itype).flatten()
            ind = self.rng.choice(itype_inds)
        else:
            if self.at.atype[ind] != self.itype:
                raise ValueError(('Atom type at ind {} is inconsistent with ' +
                                 'type associated with this instance of ' +
                                 'Trial_Translation ({}).').format(
                                     self.at.atype[ind],self.itype))
        if dx is None:
            dx = (self.rng.uniform(size=self.geom.ndim)-0.5)*2.0*self.max_move
        
        x0 = self.at.x[ind]
        im0 = self.at.im[ind]
        x1,im1 = self.geom.wrap_position(x0 + dx,im0)
        
        en_external0,f_ext0 = self.ext.phi(x0)
        en_external1,f_ext1 = self.ext.phi(x1)
        en_pair0 = np.zeros((self.neigh.nn[ind]),dtype=self.at.en_pair.dtype)
        en_pair1 = en_pair0.copy()
        f_pair0 = np.zeros((self.neigh.nn[ind],self.at.f.shape[1]),dtype=self.at.f.dtype)
        f_pair1 = f_pair0.copy()
        
        for k,j in enumerate(self.neigh.l[ind,:self.neigh.nn[ind]]):
            jtype = self.at.atype[j]
            pair = self.at.pair_type[(self.itype,jtype)]
            
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
            self.grat.last_accept = False
        elif accept == True:
            self.grat.last_accept = True
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
            
        return self.last_accept
    

def Trial_Exchange(Trial):
    