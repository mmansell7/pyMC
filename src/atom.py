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
        
        # Per-atom arrays
        self.x = np.empty((nmax,self.geom.ndim),dtype=float)
        self.x.fill(np.nan)
        self.im = np.empty((nmax,self.geom.ndim),dtype=int)
        self.atype = np.empty((nmax),dtype=str)
        self.atype.fill('')
        
        # Per atom-type and pair-type arrays
        self.external_type = {}  # Map from atom type to external potential
        self.pair_type = {}  # Map from tuple of atom types to pair potential
        
        
        self.listeners = []  # Anything that needs to know when an atom is 
                             #  added, removed, moved, or otherwise changed
        self.primary_neigh = None
        return
    
    @property
    def nmax(self):
        nmaxs = np.array([self.x.shape[0],self.im.shape[0],self.atype.shape[0]])
        test = nmaxs - nmaxs[0]
        if np.any(test != 0):
            raise Exception('Inconsistent lengths of Atom class\'s internal ' +
                            'arrays.')
        else:
            return nmaxs[0]
    
    @property
    def n(self):
        print('x,atype: {},{}'.format(self.x,self.atype))
        a = np.where(np.any(np.isnan(self.x),axis=1))[0]
        a = a[0] if a.shape[0] > 0 else self.x.shape[0]
        b = np.where(self.atype=='')[0]
        b = b[0] if b.shape[0] > 0 else self.atype.shape[0]
        ns = np.array([a,b])
        test = ns - ns[0]
        if np.any(test != 0):
            raise Exception('Inconsistent n deduced from Atom class\'s ' +
                            'internal arrays.')
        else:
            return ns[0]
    
    def add_atom(self,x,im,atype):
           
        if self.n < self.nmax:
            n = self.n
            self.x[n] = x
            self.im[n] = im
            self.atype[n] = atype
            
            for l in self.listeners:
                l.atom_added(self.n)
                
        else:
            new_rows = np.ceil((self.grow_fact - 1.0)*self.nmax).astype(int)
            self.grow(new_rows)
            self.add_atom(x,im,atype)
        
        return
       
           
    def grow(self,new_rows):
            self.x = np.append(self.x,np.empty((new_rows,self.x.shape[1]),
                                               dtype=self.x.dtype),axis=0)
            self.x[-new_rows:].fill(np.nan)
            self.im = np.append(self.im,np.empty((new_rows,self.im.shape[1]),
                                               dtype=self.im.dtype),axis=0)
            self.atype = np.append(self.atype,np.empty((new_rows),self.atype.dtype),axis=0)
            self.atype[-new_rows:].fill('')
            
            for l in self.listeners:
                l.grow(self.nmax)
                if l.nmax != self.nmax:
                    raise Exception(('Atom listener {} has value of nmax that is ' +
                                    'inconsistent with atom object.').format(l))
           
    def delete_atom(self,ind):
        n_start = self.n
        print('Deleting from index {}'.format(n_start))
        self.x[ind:n_start-1]  = self.x[ind+1:n_start]
        self.x[n_start-1:] = np.nan
        self.im[ind:n_start-1] = self.im[ind+1:n_start]
        self.atype[ind:n_start-1] = self.atype[ind+1:n_start]
        self.atype[n_start-1:] = ''
        
        for l in self.listeners:
            l.atom_removed(ind)
            if l.n != self.n:
                raise Exception(('Atom listender {} has n value inconsistent ' +
                                'with Atom object.').format(l))
        
        return
        
    def translate_atom(self,ind,x,ref=None):
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
        
        for l in self.listeners:
            l.atom_translated(ind,xold,xnew,dx)
        return
    
    
    


