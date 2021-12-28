#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 20:59:16 2021

@author: MattMansell
"""


import numpy as np
import geometry
import neighbor




class Atom():
    
    grow_fact = 1.5
    
    def __init__(self,geom,nmax):
        if not isinstance(geom,geometry.Geometry):
            raise ValueError()
        self.geom = geom
        if nmax < 1:
            raise ValueError()
        self.nmax = nmax
        self.n = 0
        self.x = np.empty((self.nmax,self.geom.ndim),dtype=float)
        self.im = np.empty((self.nmax,self.geom.ndim),dtype=int)
        self.atype = np.empty((self.nmax),dtype=str)
        self.external_type = {}  # Map from atom type to external potential
        self.pair_type = {}  # Map from tuple of atom types to pair potential
        self.listeners = []  # Anything that needs to know when an atom is 
                             #  added, removed, moved, or otherwise changed
        
        return
    
    def add_atom(self,x,im,atype):
        
        if self.n < self.nmax:
            self.x[self.n] = x
            self.im[self.n] = im
            self.atype[self.n] = atype
            self.n += 1
            for l in self.listeners:
                l.atom_added(self.n)
        else:
            new_rows = np.ceil((self.grow_fact - 1.0)*self.nmax).astype(int)
            self.x = np.append(self.x,np.empty((new_rows,self.geom.ndim),
                                               dtype=self.x.dtype),axis=0)
            self.im = np.append(self.im,np.empty((new_rows,self.geom.ndim),
                                               dtype=self.im.dtype),axis=0)
            self.atype = np.append(self.atype,np.empty((new_rows),self.atype.dtype),axis=0)
            self.nmax = self.x.shape[0]
            self.add_atom(x,im,atype)
            
        return
    
    def delete_atom(self,ind):
        
        self.x[ind:self.n-1]  = self.x[ind+1:self.n]
        self.im[ind:self.n-1] = self.im[ind+1:self.n]
        self.atype[ind:self.n-1] = self.atype[ind+1:self.n]
        self.n -= 1
        for l in self.listeners:
            l.atom_removed(ind)
        
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
    
    
    


