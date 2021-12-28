#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:23:19 2021

@author: MattMansell
"""

import warnings
import numpy as np
import atom_listener


class Neighbor(atom_listener.AtomListener):
    '''
    Base class.
    '''
    
    minus_one = np.array([-1],dtype='uint32')[0]
    
    def __init__(self,mc,at):
        super().__init__(at)
        self.mc = mc
        return
    
    def build(self):
        raise Exception('subclasses of Neighbor must implement a build method.')
        
    
    
class NeighborClass0(Neighbor):
    '''
    Calculates distance between every pair on every build.
    '''
        
    def __init__(self,mc,at,geom,nmax):
        '''
        nmax = maximum number of neighbors for any atom
        '''
        self.geom = geom
        super().__init__(mc,at)
        self.at.neighbors = [self]
        
        if nmax < 0:
            raise ValueError
        self.nmax = nmax
        self.l    = np.empty((self.at.nmax,self.nmax),dtype='uint32')
        print(self.at.nmax,self.nmax,self.l.shape)
        self.nn   = np.zeros(self.at.nmax,dtype='uint8')
        
        return
    
    def build(self):
        
        self.l.fill(self.minus_one)
        self.nn.fill(0)
        warnflag = 0
        for i in range(0,self.at.n-1):
            for j in range(i+1,self.at.n):
                atype   = self.at.pair_type[self.at.atype[i],self.at.atype[j]]
                d,dsq,_ = self.geom.distance(self.at.x[i],self.at.x[j])
                if d < (1.5*atype.rc):
                    if self.nn[i] >= self.nmax:
                        warnflag += 1
                    else:
                        self.l[i,self.nn[i]] = j
                        self.nn[i] += 1
                    if self.nn[j] >= self.nmax:
                        warnflag += 1
                    else:
                        self.l[j,self.nn[j]] = i
                        self.nn[j] += 1
        if warnflag:
            warnings.warn('There were {} instances of attempts to add ' +
                          'neighbors beyond the set maximum neighbor number.\n' +
                          ' Those neighbors will be neglected.')
        self.last_build = self.mc.stepnum
        
        return
    
    def calc_one_atom(self,i):
        '''
        Return the neighbor list of a single atom.
        
        Does not modify the neighbor list.  Only returns a new array
        representing the neighbor list that would be associated with atom
        i in its current position.

        Parameters
        ----------
        i : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        l = []
        for j in np.delete(np.array(range(0,self.at.n)),i):
            atype   = self.at.pair_type[self.at.atype[i],self.at.atype[j]]
            d,dsq,_ = self.geom.distance(self.at.x[i],self.at.x[j])
            if d < (1.5*atype.rc):
                l.append(j)
        l = np.array(l,dtype=self.l.dtype)
        if l.shape[0] > self.nmax:
            warnings.warn('There were {} instances of attempts to add ' +
                          'neighbors beyond the set maximum neighbor number.\n' +
                          ' Those neighbors will be neglected.')
            l = l[:self.nmax]
        
        return
    
    def grow(self,nmax):
        new_rows = nmax - self.l.shape[0]
        self.l = np.append(self.l,np.empty((new_rows,self.nmax),
                                               dtype=self.l.dtype),axis=0)
        self.nn = np.append(self.nn,np.empty((new_rows),dtype=self.nn.dtype),axis=0)
        self.nmax = nmax
        return
    
    def atom_added(self):
        i = self.at.n-1
        self.l[i].fill(self.minus_one)
        self.nn[i] = 0
        warnflag = 0
        for j in range(0,i):
            atype   = self.at.pair_type[self.at.atype[i],self.at.atype[j]]
            d,dsq,_ = self.geom.distance(self.at.x[i],self.at.x[j])
            if d < (1.5*atype.rc):
                if self.nn[i] >= self.nmax:
                    warnflag += 1
                else:
                    self.l[i,self.nn[i]] = j
                    self.nn[i] += 1
                if self.nn[j] >= self.nmax:
                    warnflag += 1
                else:
                    self.l[j,self.nn[j]] = i
                    self.nn[j] += 1
        if warnflag:
            warnings.warn('There were {} instances of attempts to add ' +
                          'neighbors beyond the set maximum neighbor number.\n' +
                          ' Those neighbors will be neglected.')
                    
        return
    
    
    def atom_deleted(self,ind):
        self.build()
        return
        
        
class NeighborClassMC(Neighbor):
    '''
    Efficient neighbor making use of knowlege about which particle(s) just moved.
    
    Attritubes
    ----------
    geom : geometry.Geometry
        Associated Geometry object
    at : atom.Atom
        Associated Atom object
    nmax : int
        Maximum number of neighbors for any particle in the list
    l : numpy array (2D)
        List of neighbor pairs. First index is the index of the "i" particle.
        l[i] is then an array of the indexes of i's neighbors. If l[i][q] = 
    nn : numpy array (1D)
        Array of number of neighbors associated with each particle
    cell_corners : numpy array, None
        The "lowest" corner of each cell.  Cells are used for the initial
        step of determining whether two particles are neighbors. If None,
        cells are ignored.
    cell_links : list of lists
        cell_links[i] is a list of cells with which the cell at index i is 
        "linked" (a particle from the cell at index i can have non-zero
        interaction energy with a particle in the cell at index j, for each j
        in cell_links[i], and otherwise cannot.).
    cell_list : numpy array (1D)
        at.nmax-length array.  Index of cell in which each particle is located.
    cutoff : float
        Cut-off distance
    
    
    Public Methods
    --------------
    build
        Fresh build of the neighbor list.
    atom_moved
        Function to be executed whenever a particle(s) in at has moved. Should
        update the neighbors of the moved particle(s).
    calc_one_atom
        ???
    grow
        Re-size array-like attributes.  Executed whenever the number of 
        particles in at exceeds the number that can be stored in these
        attributes at the their current sizes.
    atom_added
        Function to be executed whenever a particle(s) has been added to at.
        Should incorporate the added particle(s).
    atom_deleted
        Function to be executed whenver a particle(s) has been deleted from at.
        Should remove the deleted particle(s) and adjust the attributes of this
        instance as necessary.
    
    '''
    
    
    def __init__(self,mc,at,geom,nmax,cutoff=None,cell_corners=None):
        '''
        Instantiate an instance of NeighborClassMC.
        
        Parameters
        ----------
        mc : TYPE
            DESCRIPTION.
        at : TYPE
            DESCRIPTION.
        geom : TYPE
            DESCRIPTION.
        nmax : int (> 0)
            maximum number of neighbors for any atom.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.geom = geom
        super().__init__(mc,at)
        if self.at.neighbors:
            raise ValueError('Attempt to assign NeighborClassMC to atom.Atom' +
                ' that already has a Neighbor attribute. This is currently ' +
                'not supported.')
        self.at.neighbors = [self]
        
        if nmax < 0:
            raise ValueError
        self.nmax = nmax
        if not cutoff:
            self.cutoff = np.max([pt.cutoff for pt in self.at.pair_type.values()])
        self.l    = np.empty((self.at.nmax,self.nmax),dtype='uint32')
        print(self.at.nmax,self.nmax,self.l.shape)
        self.nn   = np.zeros(self.at.nmax,dtype='uint8')
        
        # Build cell list
        # Cell list is a 2D array in which element i,j is the 
        #   coordinate of the lower face of cell i in the j-th dimension
        if cell_corners is None:
            self.cell_corners = cell_corners
            self.cell_links = None
            self.cell_list = None
        elif cell_corners.dtype == int and cell_corners.ndim == self.geom.ndim:
            numcells = np.product(cell_corners)
            self.cell_corners = np.arange(0,numcells,
                                          dtype=float).reshape((-1,1)) * np.ones((1,self.geom.ndim))
            dim = 0
            b = 1
            while dim < self.geom.ndim:
                self.cell_corners[:,dim] = np.mod(np.floor(
                    self.cell_corners[:,dim]/b),cell_corners[dim])
                b = b*cell_corners[dim]
                dim += 1
            dx = (self.geom.L/cell_corners)
            self.cell_corners = self.x[0] + self.cell_corners * dx
            
            self.cell_links = [[]]*numcells
            cutsq = np.square(self.cutoff)
            for i in range(0,numcells):
                for j in range(i+1,numcells):
                    dist,dr2,dr = self.geom.dist(self.cell_corners[i],self.cell_corners[j])
                    dr = np.abs(dr)
                    dr = np.where(dr>0.,dr-dx,dr)
                    dr2 = np.dot(dr,dr)
                    if dr2 > cutsq:
                        self.cell_links[i].append(j)
                        self.cell_links[j].append(i)
            for i in range(0,numcells):
                self.cell_links(i).sort()
                    
        else:
            raise ValueError('NeighborClassMC: this type for cells is not ' + 
                             'supported (yet).')
        
        self.build()
        
        return
    
    def build(self):
        
        # Build cell list - must be done prior to pairs list
        
        # Build pairs list
        self.l.fill(self.minus_one)
        self.nn.fill(0)
        warnflag = 0
        for i in range(0,self.at.n-1):
            linki = self.cell_links[self.cell_list[i]]
            for j in range(i+1,self.at.n):
                if self.cell_list[j] not in linki: continue
                d,dsq,_ = self.geom.distance(self.at.x[i],self.at.x[j])
                if d <= (self.cutoff):
                    if self.nn[i] >= self.nmax:
                        warnflag += 1
                    else:
                        self.l[i,self.nn[i]] = j
                        self.nn[i] += 1
                    if self.nn[j] >= self.nmax:
                        warnflag += 1
                    else:
                        self.l[j,self.nn[j]] = i
                        self.nn[j] += 1
        if warnflag:
            warnings.warn('There were {} instances of attempts to add ' +
                          'neighbors beyond the set maximum neighbor number.\n' +
                          ' Those neighbors will be neglected.')
                    
        return
    
    def atom_moved(self,index,xold,xnew,dx):
        if move > skin: raise ValueError('NeighborClassMC error: attempt to move' +
                    ' a particle by more than the skin distance!')
        
        return None
        
    def calc_one_atom(self,i):
        '''
        Return the neighbor list of a single atom.
        
        Does not modify the neighbor list.  Only returns a new array
        representing the neighbor list that would be associated with atom
        i in its current position.
        
        Parameters
        ----------
        i : TYPE
            DESCRIPTION.
            s
        Returns
        -------
        None.
        
        '''
        l = []
        for j in np.delete(np.array(range(0,self.at.n)),i):
            atype   = self.at.pair_type[self.at.atype[i],self.at.atype[j]]
            d,dsq,_ = self.geom.distance(self.at.x[i],self.at.x[j])
            if d < (1.5*atype.rc):
                l.append(j)
        l = np.array(l,dtype=self.l.dtype)
        if l.shape[0] > self.nmax:
            warnings.warn('There were {} instances of attempts to add ' +
                          'neighbors beyond the set maximum neighbor number.\n' +
                          ' Those neighbors will be neglected.')
            l = l[:self.nmax]
        
        return
    
    def grow(self,nmax):
        new_rows = nmax - self.l.shape[0]
        self.l = np.append(self.l,np.empty((new_rows,self.nmax),
                                               dtype=self.l.dtype),axis=0)
        self.nn = np.append(self.nn,np.empty((new_rows),dtype=self.nn.dtype),axis=0)
        self.nmax = nmax
        return
    
    def atom_added(self):
        i = self.at.n-1
        self.l[i].fill(self.minus_one)
        self.nn[i] = 0
        warnflag = 0
        for j in range(0,i):
            atype   = self.at.pair_type[self.at.atype[i],self.at.atype[j]]
            d,dsq,_ = self.geom.distance(self.at.x[i],self.at.x[j])
            if d < (1.5*atype.rc):
                if self.nn[i] >= self.nmax:
                    warnflag += 1
                else:
                    self.l[i,self.nn[i]] = j
                    self.nn[i] += 1
                if self.nn[j] >= self.nmax:
                    warnflag += 1
                else:
                    self.l[j,self.nn[j]] = i
                    self.nn[j] += 1
        if warnflag:
            warnings.warn('There were {} instances of attempts to add ' +
                          'neighbors beyond the set maximum neighbor number.\n' +
                          ' Those neighbors will be neglected.')
                    
        return
    
    def atom_deleted(self,ind):
        self.build()
        return
        
        




