#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 17:06:20 2021

@author: mattmansell
"""

import numpy as np

import atom_listener

class Force():
    '''
    A class to calculate forces, energies, etc.
    
    Attributes
    ----------
    last_update : 
        Step on which last update occurred.
    en_external : float
        Total external energy.
    en_pair : float
        Total pair energy.
    
    Methods
    -------
    update_all : 
        Update forces, energies, etc. for all particles, pairs, etc.
    update_some :
        
        
    '''
    
    def __init__(self,mc,at):
        self.mc = mc
        self.mc.force = self
        self.at = at
        self.at.force = self
        self.neigh = self.at.primary_neigh
        self.geom = self.at.geom
        self.last_update = -1
        self.en_external = 0.0
        self.en_pair     = 0.0
        self.en_total    = 0.0
        self.update_all()
        return
    
    def update_all(self,**kwargs):
        '''
        Update forces, energies, etc.
        
        Parameters
        ----------
        
        kwargs
        ------
        force : boolean
            Force update when forces have already been updated on the current
            step.

        Returns
        -------
        None.

        '''
        if self.last_update >= self.mc.stepnum and kwargs.get('force',False):
            return
        
        self.at.f[:self.at.n,:].fill(0.0)
        self.at.en_pair[:self.at.n].fill(0.0)
        for i in range(0,self.at.n):
            itype = self.at.atype[i]
            ext = self.at.external_type[itype]
            en,f = ext.phi(self.at.x[i])
            self.at.en_external[i] = en
            self.at.f[i,:] += f[:]
            for j in self.neigh.l[i,:self.neigh.nn[i]]:
                if j > i:
                    jtype = self.at.atype[j]
                    d,dsq,dr = self.geom.distance(self.at.x[i],self.at.x[j])
                    pair = self.at.pair_type[(itype,jtype)]
                    en,f = pair.phi(dsq)
                    self.at.en_pair[i] += en
                    self.at.en_pair[j] += en
                    self.at.f[i] -= dr*f
                    self.at.f[j] += dr*f
        
        self.at.en_total = self.at.en_external + self.at.en_pair
        self.en_external = np.sum(self.at.en_external[:self.at.n])
        self.en_pair     = 1.0/2.0*np.sum(self.at.en_pair[:self.at.n])
        self.en_total    = self.en_external + self.en_pair
        
        return

    def update_some(self,some):
        '''
        Update forces, energies, etc. for a subset of particles.
        
        Will also update contributions to forces, energies, etc. of other
        particles from interactions with members of some.

        Parameters
        ----------
        some : sequence or set
            Indexes of particles to be updated.

        Returns
        -------
        None.

        '''
        print('Request to update {}'.format(some))
        # Also need to update for any neighbor of any member of some
        s = []
        for i in some:
            s = s + [i] + list(self.neigh.l[i,:self.neigh.nn[i]])
            print('s: {}'.format(s))
        some = set(s)
        print('some: {}'.format(some))
        some = list(some)
        print('some: {}'.format(some))
        some.sort()
        # some = list(set(s)).sort()  # Remove duplicates
        print('Updating {}.'.format(some))
        
        self.at.en_pair[some] = 0.0
        self.at.f[some,:]       = 0.0
        
        for i in some:
            
            itype = self.at.atype[i]
            ext = self.at.external_type[itype]
            en,f = ext.phi(self.at.x[i])
            self.at.en_external[i] = en
            self.at.f[i] = f
            
            for j in self.neigh.l[i,:self.neigh.nn[i]]:
                jtype = self.at.atype[j]
                if j > i:
                    d,dsq,dr = self.geom.distance(self.at.x[i],self.at.x[j])
                    pair = self.at.pair_type[(itype,jtype)]
                    en,f = pair.phi(dsq)
                    self.at.en_pair[i] += en
                    self.at.en_pair[j] += en
                    self.at.f[i] -= dr*f
                    self.at.f[j] += dr*f
        
        self.at.en_total[some] = self.at.en_external[some] + self.at.en_pair[some]
        self.en_external = np.sum(self.at.en_external[:self.at.n])
        self.en_pair     = 1.0/2.0*np.sum(self.at.en_pair[:self.at.n])
        self.en_total    = self.en_external + self.en_pair
        
        return
    
        
        
        
        
        
        
        