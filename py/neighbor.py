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
    Abstract base class.
    
    Builds, rebuilds, and stores neighbor lists and associated information.
    
    Attributes
    ----------
    mc : 
        Instance of class mc (i.e., a Monte Carlo engine) to which this 
        object is attached.
    last_build : int
        Step on which last (re-)build occurred.
    minus_one [class attribute]: 
        Unsigned long representation of -1.  Used to indicate invalid values of
        integer variables.
    nmax : int
        Maximum number of neighbors for any particle in the list
    l : numpy array (2D)
        List of neighbor pairs. First index is the index of the "i" particle.
        l[i] is then an array of the indexes of i's neighbors. If l[i][q] = 
    
    Methods
    -------
    build()
        Build or re-build the neighbor list.
    calc_one_atom(i)
        Calculate the neighbor list element for the single particle identified
        by index i, without actually modifying the neighbor list. This is 
        useful when a particle insertion is under consideration.  It is more
        appropriate to leave calculate the possible neighbors of the inserted
        particle within this class than to leave it to some other class.
    
    '''
    
    minus_one = np.array([-1],dtype='uint32')[0]
    
    def __init__(self,mc,at):
        super().__init__(at)
        self.mc = mc
        self.last_build = None
        self.nmax = None
        self.l = None
        return
    
    def build(self):
        '''
        (Re-)build the neighbor list.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        raise NotImplementedError('Subclasses of Neighbor must implement a ' +
                                  'build method.')
    
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
        
        raise NotImplementedError('Subclasses of Neighbor must implement a ' +
                                  'calc_one_atom method.')
    
    
class NeighborClass0(Neighbor):
    '''
    Calculates distance between every pair on every build.
    
    This is the "dumb and slow, but reliable" implementation.
    
    Attributes
    ----------
    geom : 
        Instance of class geometry to which this object is associated.
    
    Methods
    -------
    
    
    '''
        
    def __init__(self,mc,at,geom,nmax):
        '''
        
        Parameters
        ----------
        mc : 
            Instance of class mc (Monte Carlo engine) to which this object
            will be associated.
        at :
            Instance of class Atom to which this object will be associated.
        geom :
            Instance of class Geometry to which this object will be associated.
        nmax : int
            Maximum number of neighbors for any atom
        
        '''
        self.geom = geom
        super().__init__(mc,at)
        self.at.neighbors = [self]
        self.last_build = -1
        
        if nmax < 0:
            raise ValueError
        self.nmax = nmax
        self.l    = np.empty((self.at.nmax,self.nmax),dtype='uint32')
        print(self.at.nmax,self.nmax,self.l.shape)
        self.nn   = np.zeros(self.at.nmax,dtype='uint8')
        
        return
    
    def build(self,**kwargs):
        '''
        (Re-)build the list.
        
        Parameters
        ----------
        --None--
        
        ...kwargs...
        force : boolean
            If true, neighbor list will be (re-)built, regardless of whether it
            has already been (re-)built on this step.

        Returns
        -------
        None.

        '''
        self.l.fill(self.minus_one)
        self.nn.fill(0)
        self.nn[self.at.n:] = -1
        print(self.nn)
        warnflag = 0
        if (self.mc.stepnum > self.last_build) or kwargs.get('force',False):
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
                warnings.warn(('There were {} instances of attempts to add ' +
                              'neighbors beyond the set maximum neighbor number.\n' +
                              ' Those neighbors will be neglected.').format(warnflag))
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
    
    
    def atom_removed(self,ind):
        self.build()
        return
        
        
class NeighborClassMC(Neighbor):
    '''
    Neighbor that uses of knowlege about which particle(s) just moved.
    
    Should be more efficient than NeighborClass0.
    
    Attributes
    ----------
    geom : geometry.Geometry
        Associated Geometry object
    nn : numpy array (1D)
        Array of number of neighbors associated with each particle
    cell_corners : numpy array (2D), None
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
    edge_lengths : numpy array
        Edge lengths of each of the cells.
    distance : numpy array (2D)
        Distance between neighbors. Element i,j is the distance between 
        particle i and its j-th neighbor.
    dist_sq : numpay array (2D)
        Squared distance between neighbors.  Element i,j is the distance
        between particle i and its j-th neighbor.
    displ : numpy array (3D)
        Displacement coordinates between neighbors.  Element i,j,k is the
        displacement between particle i and its j-th neighbor, in dimension k.
    
    Public Methods
    --------------
    build
        Fresh build of the neighbor list.
    atom_moved
        Function to be executed whenever a particle(s) in at has moved. Should
        update the neighbors of the moved particle(s).
    calc_one_atom
        
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
    
    
    def __init__(self,mc,at,geom,nmax,cell_corners,cutoff):
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
        self.at.neighbors = [self]
        self.last_build = -1
        self.cutoff = cutoff
        # self.distance = np.empty((self.at.nmax,self.nmax),dtype=float)
        # self.distance.fill(np.nan)
        # self.dist_sq  = np.empty((self.at.nmax,self.nmax),dtype=float)
        # self.dist_sq.fill(np.nan)
        # self.displ    = np.empty((self.at.nmax,self.nmax,self.geom.ndim),dtype=float)
        # self.displ.fill(np.nan)
        
        if nmax < 0:
            raise ValueError
        self.nmax = nmax
        self.l    = np.empty((self.at.nmax,self.nmax),dtype='uint32')
        print(self.at.nmax,self.nmax,self.l.shape)
        self.nn   = np.zeros(self.at.nmax,dtype='uint8')
        self.nn.fill(-1)
        self.cell_list = np.empty((self.at.nmax,),dtype=int)
        self.cell_list.fill(-1)
        
        # Build cell_corners and cell_links
        if cell_corners is None:
            self.cell_corners = cell_corners
            self.cell_links = None
            self.cell_list = None
        elif cell_corners.dtype == int and cell_corners.shape[0] == self.geom.ndim:
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
            self.edge_lengths = (self.geom.L/cell_corners)
            self.cell_corners = self.geom.x[0] + self.cell_corners * self.edge_lengths
            
            self.cell_links = [[]]*numcells
            cutsq = np.square(cutoff)
            for i in range(0,numcells):
                self.cell_links[i].append(i)
                for j in range(i+1,numcells):
                    dist,dr2,dr = self.geom.dist(self.cell_corners[i],self.cell_corners[j])
                    dr = np.abs(dr)
                    dr = np.where(dr>0.,dr-self.edge_lengths,dr)
                    dr2 = np.dot(dr,dr)
                    if dr2 > cutsq:
                        self.cell_links[i].append(j)
                        self.cell_links[j].append(i)
            for i in range(0,numcells):
                self.cell_links[i].sort()
        else:
            raise ValueError('NeighborClassMC: this type for cells is not ' + 
                             'supported (yet).')
        # print('Cell links: {}'.format(self.cell_links))
        
        self.build(force=True)
        
        return
    
    def build(self,**kwargs):
        '''
        Complete (re)build of the lists.
        
        Should only have to be run once, to create the initial lists.  Any
        subsequent changes can be efficiently handled by atom_moved,
        atom_added, atom_removed, etc.  This function can be used to check
        the lists at any point, though.

        Parameters
        ----------
        
        KWARGS
        ------
        force : boolean
            Force a re-build, even if the last build was done on the same step.

        Returns
        -------
        None.

        '''
        
        # Build cell list - must be done prior to pairs list
        # print('Building cell list...')
        self.cell_list.fill(-1)
        for i in range(0,self.at.n):
            for cornind in range(0,self.cell_corners.shape[0]):
                q = self.at.x[i] - self.cell_corners[cornind]
                if np.all(q > 0.0) and np.all(q < self.edge_lengths):
                    self.cell_list[i] = cornind
        # print('Cell List: ')
        # print(self.cell_list)
        
        # Build pairs list
        # print('Building pair list...')
        # print(kwargs.get('force',False))
        if (self.mc.stepnum > self.last_build) or kwargs.get('force',False):
            self.l.fill(self.minus_one)
            self.nn[:self.at.n] = 0
            warnflag = 0
            # for i in range(0,self.at.nmax):
                # print('ind,cellind = {},{}'.format(i,self.cell_list[i]))
                
            for i in range(0,self.at.n-1):
                # print('Neighboring particle {}'.format(i))
                linki = self.cell_links[self.cell_list[i]]
                # print('Linked cells: {}'.format(linki))
                for j in range(i+1,self.at.n):
                    # print('Considering particle {} as neighbor...'.format(j))
                    if self.cell_list[j] not in linki:
                        # print(('Particle {} is in cell {} which is not linked ' + 
                        #       'to particle {}\'s cell').format(i,self.cell_list[j],i))
                        continue
                    d,dsq,_ = self.geom.distance(self.at.x[i],self.at.x[j])
                    if d <= (self.cutoff):
                        # print(('Particles {} (at {}) and {} (at {}) are ' +
                        #        'neighbors.').format(i,self.at.x[i],j,self.at.x[j]))
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
                    # else:
                        # print(('Particles {} (at {}) and {} (at {}) are NOT' +
                        #        'neighbors.').format(i,self.at.x[i],j,self.at.x[j]))
                        
            if warnflag:
                warnings.warn('There were {} instances of attempts to add ' +
                              'neighbors beyond the set maximum neighbor number.\n' +
                              ' Those neighbors will be neglected.')
                    
        return
    
    def atom_translated(self,index,xold,xnew,dx):
        
        self.cell_list[index] = -1
        for cornind in range(0,self.cell_corners.shape[0]):
            q = self.at.x[index] - self.cell_corners[cornind]
            if np.all(q > 0.0) and np.all(q < self.edge_lengths):
                self.cell_list[index] = cornind
        
        # Find neighbors after the move
        linki = self.cell_links[self.cell_list[index]]
        old_neighbors = list(self.l[index,np.nonzero(np.logical_and(
                                self.l[index]>-1,
                                self.l[index]!=Neighbor.minus_one))]
                             .reshape((-1,)))
        # print('old_neighbors: {}'.format(old_neighbors))
        new_neighbors = []
        for j in list(range(0,index)) + list(range(index+1,self.at.n)):
            if self.cell_list[j] not in linki:
                continue
            d,dsq,_ = self.geom.distance(self.at.x[index],self.at.x[j])
            if d <= (self.cutoff):
                new_neighbors.append(j)
        # print('new_neighbors: {}'.format(new_neighbors))
        
        # Remove "index" from list for each particle that was a neighbor prior
        # to the move but is not a neighbor after the move
        for j in (set(old_neighbors)-set(new_neighbors)):
            ind = np.argwhere(self.l[j,:]==index)[0,0]
            self.l[j,ind:-1] = self.l[j,ind+1:]
            self.l[j,-1] = -1
            self.nn[j] = self.nn[j] - 1
        
        # Insert "index" into list for each particle that was not a neighbor
        # before the move but is a neighbor after the move
        for j in (set(new_neighbors)-set(old_neighbors)):
            if self.l[j,-1] != Neighbor.minus_one:
                raise Exception(('Too many neighbors for particle {}, which ' +
                                 'already has neighbors {}.').format(j,
                                                            self.l[j]))
            ind = np.argwhere(self.l[j,:]>index)[0,0]
            self.l[j,ind+1:] = self.l[j,ind:-1]
            self.l[j,ind] = index
            self.nn[j] = self.nn[j] + 1
        
        # Update the list entries for "index" particle
        if len(new_neighbors) >= self.nmax:
            raise Exception('Too many neighbors for particle {}'.format(index))
        self.nn[index] = len(new_neighbors)
        # print('index,nn: {},{}'.format(index,self.nn[index]))
        self.l[index,:self.nn[index]] = new_neighbors
        self.l[index,self.nn[index]:] = Neighbor.minus_one
        
        return None
        
    # def calc_virtual(self,x):
    #     '''
    #     Return the neighbor list of a single virtual atom located at x.
        
    #     Does not modify the neighbor list.  Only returns a new array
    #     representing the neighbor list that would be associated with atom
    #     i in its current position.
        
    #     Parameters
    #     ----------
    #     x : TYPE
    #         DESCRIPTION.
    #         s
    #     Returns
    #     -------
    #     None.
        
    #     '''
    #     l = []
    #     for j in np.delete(np.array(range(0,self.at.n)),i):
    #         atype   = self.at.pair_type[self.at.atype[i],self.at.atype[j]]
    #         d,dsq,_ = self.geom.distance(self.at.x[i],self.at.x[j])
    #         if d < (1.5*atype.rc):
    #             l.append(j)
    #     l = np.array(l,dtype=self.l.dtype)
    #     if l.shape[0] > self.nmax:
    #         warnings.warn('There were {} instances of attempts to add ' +
    #                       'neighbors beyond the set maximum neighbor number.\n' +
    #                       ' Those neighbors will be neglected.')
    #         l = l[:self.nmax]
        
    #     return
    
    def atom_added(self,n_new):
        
        # Resize arrays, if they are not sufficiently sized
        while self.l.shape[0] < n_new:
            addsize = int(0.5*self.l.shape[0]+0.51)
            self.l = np.append(self.l,np.empty((addsize,self.l.shape[1]),
                                               dtype=self.l.dtype),
                               axis=0)
            self.nn = np.append(self.nn,np.empty((addsize),dtype=self.nn.dtype))
            self.cell_list = np.append(self.cell_list,np.empty((addsize),
                                              dtype=self.cell_list.dtype))
        
        i = n_new-1
        self.cell_list[i] = -1
        for cornind in range(0,self.cell_corners.shape[0]):
            q = self.at.x[i] - self.cell_corners[cornind]
            if np.all(q > 0.0) and np.all(q < self.edge_lengths):
                self.cell_list[i] = cornind
        # print('Cell List: ')
        # print(self.cell_list)
        
        # Build pairs list
        # print('Building pair list...')
        # print(kwargs.get('force',False))
        warnflag = 0
        # for i in range(0,self.at.nmax):
            # print('ind,cellind = {},{}'.format(i,self.cell_list[i]))
        
        self.nn[i] = 0
        self.l[i].fill(-1)
        linki = self.cell_links[self.cell_list[i]]
        # print('Linked cells: {}'.format(linki))
        for j in range(0,i):
            # print('Considering particle {} as neighbor...'.format(j))
            if self.cell_list[j] not in linki:
                # print(('Particle {} is in cell {} which is not linked ' + 
                #       'to particle {}\'s cell').format(i,self.cell_list[j],i))
                continue
            d,dsq,_ = self.geom.distance(self.at.x[i],self.at.x[j])
            if d <= (self.cutoff):
                # print(('Particles {} (at {}) and {} (at {}) are ' +
                #        'neighbors.').format(i,self.at.x[i],j,self.at.x[j]))
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
            # else:
                # print(('Particles {} (at {}) and {} (at {}) are NOT' +
                #        'neighbors.').format(i,self.at.x[i],j,self.at.x[j]))
                
        if warnflag:
            warnings.warn('There were {} instances of attempts to add ' +
                          'neighbors beyond the set maximum neighbor number.\n' +
                          ' Those neighbors will be neglected.')
                
        return
    
    def atom_removed(self,ind):
        
        # Remove "index" from lists for each particle that was a neighbor prior
        # to the deletion but is not a neighbor after the move
        for j in self.l[ind,self.l[ind]!=Neighbor.minus_one]:
            k = np.argwhere(self.l[j,:]==ind)[0,0]
            self.l[j,k:-1] = self.l[j,k+1:]
            self.l[j,-1] = -1
            self.nn[j] -= 1
        
        # Remove elements corresponding to ind
        self.cell_list[ind:-1] = self.cell_list[ind+1:]
        self.cell_list[-1] = -1
        self.nn[ind:-1] = self.nn[ind+1:]
        self.nn[-1] = -1
        self.l[ind:-1] = self.l[ind+1:]
        self.l[-1,:] = -1
        
        # Reindex
        self.l[np.logical_and(self.l>ind,self.l!=Neighbor.minus_one)] -= 1
        
        return
        
        



