#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 17:06:20 2021

@author: mattmansell
"""

import numpy as np

import pointers
import atom_listener

class Force(pointers.Pointers):
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
    
    def __init__(self,mc):
        super().__init__(mc)
        self.mc.force = self
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
        if self.last_update >= self.grat.stepnum and kwargs.get('force',False):
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

    def calculate_hypothetical(self,ind=None,itype=None,x=None):
        '''
        Calculate EVERYTHING that would be, if a hypothetical change was made.

        Parameters
        ----------
        ind : TYPE, optional
            DESCRIPTION. The default is None.
        itype : TYPE, optional
            DESCRIPTION. The default is None.
        x : TYPE, optional
            DESCRIPTION. The default is None.
        
        Returns
        -------
        None.

        '''
        
        raise NotImplementedError('Calculate hypothetical may be implemented ' +
                                  'in the future but is not at present.')
        
        qind = np.argwhere(ind==-1).reshape((-1))
        qitype = np.argwhere(itype=='').reshape((-1))
        qx = np.argwhere(np.all(np.isnan(x),axis=1)).reshape((-1))
        
        code = ( 1.0*qind + 2.0*qitype + 4.0*qx )
        # code key
        # 0: ind, itype, and x given        --> ERROR
        # 1: itype and x given, but not ind --> INSERT particle of type itype at position x
        # 2: ind and x given, but not itype --> TRANSLATE particle ind to x
        # 3: x given, but not ind or itype  --> ERROR
        # 4: ind and itype given, but not x --> CHANGE particle ind to type itype
        # 5: itype given, but not ind or x  --> ERROR
        # 6: ind given, but not itype or x  --> DELETE particle ind
        # 7: none of ind, itype, or x given --> ERROR
        
        if np.any(np.isin(code,[0,3,5,7])):
            raise ValueError('ind, itype, and x must, at each index, be a ' +
                             'meaningful combination.')
        
        ind[qind] = np.array(range(self.at.n,self.at.n+qind.shape[0]))
        itype[qitype] = self.at.itype[ind[qitype]]
        x[qx] = self.at.x[ind[qitype]]
        
        num_inds     = ind.shape[0]
        
        en_external1 = np.empty((num_inds),dtype=self.at.en_external.dtype)
        en_external0 = en_external1.copy()
        en_external0[~qind] = self.at.en_external[ind[~qind]]
        en_external0[qind]  = 0.0
        
        en_pair1     = np.empty((num_inds),dtype=self.at.en_pair.dtype)
        en_pair0     = en_pair1.copy()
        en_pair0[~qind] = self.at.en_pair[ind[~qind]]
        en_pair0[qind]  = np.nan
        
        en_total1    = np.empty((num_inds),dtype=self.at.en_total)
        en_total0    = en_total1.copy()
        en_total0[~qind] = self.at.en_total[ind[~qind]]
        en_tota0[qind]  = np.nan
        
        f1           = np.empty((num_inds,self.at.f.shape[1]),dtype=self.at.f.dtype)
        f0           = f1.copy()
        f0[~qind]    = self.at.f[ind[~qind]]
        f0[qind]     = np.nan
        
        neigh1       = np.empty((num_inds,self.neigh.l.shape[1]),dtype=self.neigh.l.dtype)
        neigh0       = neigh1.copy()
        neigh0[~qind] = self.neigh.l[ind[~qind]]
        neigh0[qind] = neighbor.Neighbor.minus_one
        
        
        # Insertions
        inds = ind[np.argwhere(code==1)]
        en_external1 = np.array((inds.shape[0]),dtype=self.at.en_external.dtype)
        en_pair     = np.array((inds.shape[0]),dtype=self.at.en_pair.dtype)
        en_total    = np.array((inds.shape[0]),dtype=self.at.en_total)
        f           = np.array((inds.shape[0],self.at.f.shape[1]),dtype=self.at.f.dtype)
        
        for i in ind[np.argwhere(code == 1)]:
            pass
        
        # Deletions
        for i in ind[np.argwhere(code == 1)]:
            pass
        
        # Translations
        for i in ind[np.argwhere(code == 5)]:
            pass
        
        # Errors
        (~np.isin(ind,list(range(0,self.at.n))))
        
        (np.isin(itype,))
        code = 0
        if ind is not None:
            if not np.all(np.isin(ind,list(range(0,self.at.n)))):
                raise ValueError('Invalid index value.')
            code += 1
        if itype is not None:
            if itype not in list(set(self.at.atype)):
                raise ValueError('Invalid atom type.')
            code += 2
        if x is not None:
            code += 4
        if ref not in [None,'origin','current']:
            raise ValueError('Invalid value of ref.')
            
        
        if code == 0:
            raise ValueError('Invalid combination of ind,itype,x,ref.')
        elif code == 1:
            # Only ind given
            raise ValueError('Atom index was given, but no type to change ' +
                             'it to, and no location to move it to.')
        elif code == 2:
            # Only itype given
            raise ValueError('Invalid combination of ind,itype,x,ref.')
        elif code == 3:
            # ind and itype given
            # Change atom at index ind to type itype
            raise NotImplementedError
        elif code == 4:
            # only x given
            raise ValueError('Invalid combination of ind,itype,x,ref.')
        elif code == 5:
            # Move atom at index ind to position x
            raise NotImplementedError
        elif code == 6:
            # x and itype given
            # Insert a new particle of type itype at position given by x
            pass
        
        
        
        
        
        
        
        