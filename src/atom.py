#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 20:59:16 2021

@author: MattMansell
"""


import numpy as np

import pointers
import geometry
import force



class Atom(pointers.Pointers):
    '''
    
    
    Attributes
    ----------
    geom : 
        Associated geometry.
    nmax : 
        Maximum number of atoms (capacity of arrays).
    n :
        Actual current number of atoms.
    x :
        Position coordinates.
    im :
        Image coordinates.
    atype : numpy array (1D)
        Strings representing atom type.
    external_type : 
        Map from atom type to external potential
    pair_type : 
        Map from tuple of atom types to pair potential.
    listeners : list
        Anything that needs to know when an atom is added, removed, moved,
        or otherwise changed.
    f : numpy array (2D)
        Per-atom force vectors.
    en_external : numpy array (1D)
        Per-atom external energies.
    en_pair : numpy array (1D)
        Per-atom pair energies.
    primary_neigh : neighbor.Neighbor
        Primary neighbor.
    
        
    Methods
    -------
    add_atom(x,im,atype) : 
        
    delete_atom(ind) : 
        
    translate_atom(ind,x,ref) : 
        
    
    '''
    grow_fact = 1.5
    
    def __init__(self,mc,nmax):
        super().__init__(mc)
        self.mc.at = self
        if nmax < 1:
            raise ValueError()
        self.nmax = nmax
        self.n = 0
        self.x = np.empty((self.nmax,self.geom.ndim),dtype=float)
        self.f = self.x.copy()
        self.en_external = np.empty((self.nmax),dtype=float)
        self.en_pair = np.empty((self.nmax),dtype=float)
        self.en_total = np.empty((self.nmax),dtype=float)
        self.im = np.empty((self.nmax,self.geom.ndim),dtype=int)
        self.atype = np.empty((self.nmax),dtype=str)
        self.external_type = {}  # Map from atom type to external potential
        self.pair_type = {}  # Map from tuple of atom types to pair potential
        self.listeners = []  # Anything that needs to know when an atom is 
                             #  added, removed, moved, or otherwise changed
        self.primary_neigh = None
        return
    
    def add_atom(self,x,im,atype,f=None,en_external=None,en_pair=None,en_total=None):
        
        if self.n < self.nmax:
            self.x[self.n] = x
            self.im[self.n] = im
            self.atype[self.n] = atype
            self.n += 1
            
            for l in self.listeners:
                l.atom_added(self.n)
                
            if (f is None) or (en_external is None) or (en_pair is None) or (
                    en_total is None):
                if hasattr(self,'force') and self.force is not None:
                    self.force.update_some(np.array([self.n-1]))
            else:
                raise NotImplementedError('The current implementation on this ' +
                            'execution branch would fail to update particles ' +
                            'with which the added particle interacts.')
                self.f[self.n] = f
                self.en_external[self.n] = en_external
                self.en_pair[self.n] = en_pair
                self.en_total[self.n] = en_total
                
                if self.force is not None:
                    self.force.en_external += en_external
                    self.force.en_pair += en_pair
                    self.force.en_total += en_total
                    
        else:
            new_rows = np.ceil((self.grow_fact - 1.0)*self.nmax).astype(int)
            self.x = np.append(self.x,np.empty((new_rows,self.geom.ndim),
                                               dtype=self.x.dtype),axis=0)
            self.f = np.append(self.f,np.empty((new_rows,self.geom.ndim),
                                               dtype=self.f.dtype),axis=0)
            self.en_external = np.append(self.en_external,np.empty((new_rows),
                                                 dtype=self.en_external.dtype),axis=0)
            self.en_pair = np.append(self.en_pair,np.empty((new_rows),
                                                 dtype=self.en_pair.dtype),axis=0)
            self.en_total = np.append(self.en_total,np.empty((new_rows),
                                                 dtype=self.en_total.dtype),axis=0)
            self.im = np.append(self.im,np.empty((new_rows,self.geom.ndim),
                                               dtype=self.im.dtype),axis=0)
            self.atype = np.append(self.atype,np.empty((new_rows),self.atype.dtype),axis=0)
            self.nmax = self.x.shape[0]
            self.add_atom(x,im,atype,f=f,en_external=en_external,
                          en_pair=en_pair,en_total=en_total)
            
        return
    
    def delete_atom(self,ind):
        
        self.x[ind:self.n-1]  = self.x[ind+1:self.n]
        self.x[self.n:] = np.nan
        self.f[ind:self.n-1]  = self.f[ind+1:self.n]
        self.f[self.n:] = np.nan
        self.en_external[ind:self.n-1] = self.en_external[ind+1:self.n]
        self.en_external[self.n:] = np.nan
        self.en_pair[ind:self.n-1] = self.en_pair[ind+1:self.n]
        self.en_pair[self.n:] = np.nan
        self.en_total[ind:self.n-1] = self.en_total[ind+1:self.n]
        self.en_total[self.n:] = np.nan
        self.im[ind:self.n-1] = self.im[ind+1:self.n]
        self.atype[ind:self.n-1] = self.atype[ind+1:self.n]
        self.n -= 1
        for l in self.listeners:
            l.atom_removed(ind)
        
        return
        
    def translate_atom(self,ind,x,ref=None,f=None,en_external=None,
                       en_pair=None,en_total=None):
        if ref == 'origin':
            x0 = np.zeros(self.geom.ndim,dtype=float)
            im0 = np.array([0,0,0],dtype=int)
        elif ref == 'current':
            x0 = self.x[ind]
            im0 = self.im[ind]
        else:
            x0 = np.nan
        
        xnew,imnew = self.geom.wrap_position(x0+x,im0)
        xold = self.x[ind]
        dx = xnew - xold
        self.x[ind] = xnew
        self.im[ind] = imnew
        
        if (f is None) or (en_external is None) or (en_pair is None) or (
                    en_total is None):
                if hasattr(self,'force') and self.force is not None:
                    self.force.update_some(np.array([ind]))
        else:
            raise NotImplementedError('The current implementation of ' +
                    'translate_atom with f,en_external,en_pair, or en_total ' +
                    'not None fails to update neighbors of the translated ' +
                    'particle, and therefore, yields unexpected or incorrect ' +
                    'results.')
            f0 = self.f[ind]
            self.f[ind] = f
            
            en_external0 = self.en_external[ind]
            self.en_external[ind] = en_external
            
            en_pair0 = self.en_pair[ind]
            self.en_pair[ind] = en_pair
            
            en_total0 = self.en_total[ind]
            self.en_total[ind] = en_total
            
            if self.force is not None:
                self.force.en_external += en_external - en_external0
                self.force.en_pair += en_pair - en_pair0
                self.force.en_total += en_total - en_total0
                
        for l in self.listeners:
            l.atom_translated(ind,xold,xnew,dx)
        return
    
    
    


