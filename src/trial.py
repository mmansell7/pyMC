#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 21:24:09 2021

@author: mattmansell
"""

import numpy as np

import pointers

class Trial(pointers.Pointers):
    
    def __init__(self,mc):
        super().__init__(mc)
        
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
        # It would be useful to verify atom type is valid, but right now, this
        #   lines below prevent the creation of trials and integrators.
        # if val not in self.mc.at.atype:
        #     raise ValueError('There must be at least one atom of type itype.')
        self.__itype = val
        return
    
    @property
    def ext(self):
        return self.at.external_type[self.itype]
    
    @property
    def rng(self):
        return self.grat.rng
    
    @property
    def kB(self):
        return self.mc.kB
    
    @property
    def T(self):
        return self.grat.T
    
    def execute(self,ind=None,dx=None):
        
        if ind is None:
            # Select a random particle to move
            itype_inds = np.argwhere(self.at.atype==self.itype).flatten()
            ind = self.rng.choice(itype_inds)
        else:
            if self.at.atype[ind] != self.itype:
                raise ValueError(('Atom type at ind {} is inconsistent with ' +
                                 'type associated with this instance of ' +
                                 'Trial_Translation ({}).').format(
                                     self.at.atype[ind],self.itype))
        if dx is None:
            # Select a random displacement
            dx = (self.rng.uniform(size=self.geom.ndim)-0.5)*2.0*self.max_move
        
        x0 = self.at.x[ind]
        im0 = self.at.im[ind]
        x1,im1 = self.geom.wrap_position(x0 + dx,im0)
        
        en_change = self.ext.phi(x1)[0] - self.ext.phi(x0)[0]
        print('1: {}'.format(en_change))
        for k,j in enumerate(self.neigh.l[ind,:self.neigh.nn[ind]]):
            jtype = self.at.atype[j]
            pair = self.at.pair_type[(self.itype,jtype)]
            
            d0,dsq0,dr0 = self.geom.distance(x0,self.at.x[j])
            d1,dsq1,dr1 = self.geom.distance(x1,self.at.x[j])
            
            en_change += pair.phi(dsq1)[0] - pair.phi(dsq0)[0]
            
        bf = np.exp(-en_change/(self.kB*self.T)) # Boltzmann factor
        test_random = np.nan
        
        if en_change <= 0.0:
            accept = True
        else:
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
            
            self.force.en += en_change
            
        else:
            raise ValueError('Unrecognized value of \'accept\'.')
            
        return bf,test_random
    

class Trial_Exchange(Trial):
    
    def __init__(self,mc,itype):
        super().__init__(mc)
        self.itype = itype
        
    @property
    def itype(self):
        return self.__itype
    
    @itype.setter
    def itype(self,val):
        # It would be useful to verify atom type is valid, but right now, this
        #   lines below prevent the creation of trials and integrators.
        # if val not in self.mc.at.atype:
        #     raise ValueError('There must be at least one atom of type itype.')
        self.__itype = val
        return
    
    @property
    def ext(self):
        return self.at.external_type[self.itype]
    
    @property
    def rng(self):
        return self.grat.rng
    
    @property
    def kB(self):
        return self.mc.kB
    
    @property
    def T(self):
        return self.grat.T
    
    @property
    def beta(self):
        return 1.0/(self.kB*self.T)
    
    @property
    def mu(self):
        return self.grat.mu[self.itype]
    
    @property
    def Lambda(self):
        return np.sqrt(self.h**2/(2.0*np.pi*self.m**self.kB*self.T))
    
    @property
    def zz(self):
        return np.exp(self.beta*self.mu) / np.power(self.Lambda,self.geom.ndim)
    
    def execute(self,insert=None,ind=None,x=None):
        '''
        Three choices --
        1. insert = None
            Insertion-vs-deletion selected randomly
            If deletion, particle index selected randomly
            If insertion, x selected randomly
        2. insert = True and x = None
            Insertion at random location
        3. insert = True and x not None
            Insertion at x
        4. insert = False and ind = None
            Delete a randomly selected particle index
        5. insert = False and ind not None
            Delete particle at index ind
            
        '''
        if insert is None:
            # Select insertion or deletion randomly
            insert = self.rng.choice([True,False])
        
        if insert:
            fact = 1.0
            bf = self.zz*self.geom.vol / (self.at.n+1)
            ind = self.at.n
            if x is None:
                x = (self.geom.x[0] + 
                      (self.rng.uniform(size=self.geom.ndim)-0.5)*self.geom.L)
        else: # deletion
            fact = -1.0
            bf = self.at.n / (self.zz*self.geom.vol)
            if ind is None:
                if self.at.n < 1:
                    bf = 1.0
                    test_random = np.nan
                    self.grat.last_accept = False
                    return bf,test_random
                else:
                    # randomly choose a particle index to delete
                    itype_inds = np.argwhere(self.at.atype==self.itype).flatten()
                    ind = self.rng.choice(itype_inds)
            x = self.at.x[ind]
        
        en_change = self.ext.phi(x)[0]
        for j in self.neigh.l[ind]:
            jtype = self.at.atype[j]
            pair = self.at.pair_type[(self.itype,jtype)]
            d,dsq,dr = self.geom.distance(x,self.at.x[j])
            en_change += pair.phi(dsq)[0]
        
        en_change = fact*en_change
        bf = bf * np.exp(-self.beta*en_change) # Boltzmann factor
        test_random = np.nan
        
        if en_change <= 0.0:
            accept = True
        else:
            test_random = self.rng.uniform()
            if bf > test_random:
                accept = True
            else:
                accept = False
        
        if accept:
            self.grat.last_accept = False
        else:
            self.grat.last_accept = True
            self.at.x[ind] = x
            self.at.im[ind] = np.zeros((self.at.im.shape[1]),dtype=int)
            self.force.en += en_change
                
        return bf,test_random
            
    