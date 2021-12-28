#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:21:49 2021

@author: MattMansell
"""


import numpy as np
import pytest

import sys
sys.path.append('../py')
import neighbor
import atom
import geometry_cartesian_box
import pair_lj_cut

class TestNeighbor():
    
    def test_create_1(self):
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,3)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([11,12,0],dtype='int'),'1')
        neigh = neighbor.Neighbor(g,at,5)
        assert neigh.geom == g
        assert neigh.at == at
        assert neigh.at.neighbors[0] == neigh
        assert neigh.nmax == 5
        assert neigh.l.shape == (3,5)
        assert neigh.nn.shape == (3,)
        
        
        
    def test_build_1(self):
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,3)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([11,12,0],dtype='int'),'1')
        neigh = neighbor.Neighbor(g,at,5)
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        neigh.at.pair_type[('0','1')] = p
        neigh.at.pair_type[('1','0')] = p
        neigh.build()
        mo = neigh.minus_one
        assert np.all(neigh.nn == np.array([1,1,0]))
        assert np.all(neigh.l == np.array([[1,mo,mo,mo,mo],
                                          [0,mo,mo,mo,mo],
                                          [mo,mo,mo,mo,mo]]))
        
    def test_grow_1(self):
        assert True
    
    def test_add_atom_1(self):
        assert True
        
    def test_remove_atom_1(self):
        assert True





