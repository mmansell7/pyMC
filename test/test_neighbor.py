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
import integrator

class TestNeighbor():
    
    def test_fail_to_create_abstract_1(self):
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,3)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([11,12,0],dtype='int'),'1')
        neigh = neighbor.Neighbor(None,at)
        with pytest.raises(NotImplementedError) as excinfo:
            neigh.calc_one_atom(0)
        with pytest.raises(NotImplementedError) as excinfo:
            neigh.build()
        return
    
    def test_grow_1(self):
        assert True
    
    def test_add_atom_1(self):
        assert True
        
    def test_remove_atom_1(self):
        assert True


class TestNeighborClass0():
    
    def test_instantiate(self):
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,3)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([11,12,0],dtype='int'),'1')
        neigh = neighbor.NeighborClass0(None,at,g,5)
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
        mc = integrator.IntegratorMC_mu_geom_T_1(at,1.0,g,1.0,0.1)
        neigh = neighbor.NeighborClass0(mc,at,g,5)
        mo = neigh.minus_one
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        neigh.at.pair_type[('0','1')] = p
        neigh.at.pair_type[('1','0')] = p
        neigh.build()
        assert np.all(neigh.nn == np.array([1,1,-1],dtype='uint8'))
        assert np.all(neigh.l == np.array([[1,mo,mo,mo,mo],
                                          [0,mo,mo,mo,mo],
                                          [mo,mo,mo,mo,mo]]))
        neigh.build(force=True)
        
        return
        
    def test_build_2(self):
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,8)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([4.3,4.3,4.3]),np.array([11,12,0],dtype='int'),'0')
        at.add_atom(np.array([4.4,4.4,4.4]),np.array([11,12,0],dtype='int'),'0')
        at.add_atom(np.array([5.4,5.4,5.4]),np.array([11,12,0],dtype='int'),'0')
        at.add_atom(np.array([9.7,9.7,9.7]),np.array([11,12,0],dtype='int'),'0')
        at.add_atom(np.array([9.8,9.8,9.8]),np.array([11,12,0],dtype='int'),'0')
        mc = integrator.IntegratorMC_mu_geom_T_1(at,1.0,g,1.0,0.1)
        neigh = neighbor.NeighborClass0(mc,at,g,5)
        mo = neighbor.Neighbor.minus_one
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        neigh.at.pair_type[('0','0')] = p
        neigh.build()
        assert np.all(neigh.nn == np.array([1,3,2,3,2,1,-1,-1],dtype='uint8'))
        assert np.all(neigh.l == np.array([[ 1,mo,mo,mo,mo],
                                           [ 0, 2, 3,mo,mo],
                                           [ 1, 3,mo,mo,mo],
                                           [ 1, 2, 4,mo,mo],
                                           [ 3, 5,mo,mo,mo],
                                           [ 4,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo]]))
        at.delete_atom(3)
        neigh.build(force=True)
        assert np.all(neigh.l == np.array([[ 1,mo,mo,mo,mo],
                                           [ 0, 2,mo,mo,mo],
                                           [ 1,mo,mo,mo,mo],
                                           [ 4,mo,mo,mo,mo],
                                           [ 3,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo]]))
        assert np.all(neigh.nn == np.array([1,2,1,1,1,-1,-1,-1],dtype='uint8'))
        
        return
        
    
class TestNeighborClassMC():
    
    def test_instantiate(self):
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,3)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([11,12,0],dtype='int'),'1')
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        at.pair_type[('0','1')] = p
        mc = integrator.IntegratorMC_mu_geom_T_1(at,1.0,g,1.0,0.1)
        neigh = neighbor.NeighborClassMC(mc,at,g,5,np.array([1,1,1]),7.5)
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
        mc = integrator.IntegratorMC_mu_geom_T_1(at,1.0,g,1.0,0.1)
        neigh = neighbor.NeighborClassMC(mc,at,g,5,np.array([1,1,1]),7.5)
        mo = neigh.minus_one
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        neigh.at.pair_type[('0','1')] = p
        neigh.at.pair_type[('1','0')] = p
        neigh.build()
        assert np.all(neigh.l == np.array([[1,mo,mo,mo,mo],
                                          [0,mo,mo,mo,mo],
                                          [mo,mo,mo,mo,mo]]))
        assert np.all(neigh.nn == np.array([1,1,-1],dtype='uint8'))
        neigh.build(force=True)
        
        return
        
    def test_build_2(self):
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,8)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([4.3,4.3,4.3]),np.array([11,12,0],dtype='int'),'0')
        at.add_atom(np.array([4.4,4.4,4.4]),np.array([11,12,0],dtype='int'),'0')
        at.add_atom(np.array([5.4,5.4,5.4]),np.array([11,12,0],dtype='int'),'0')
        at.add_atom(np.array([9.7,9.7,9.7]),np.array([11,12,0],dtype='int'),'0')
        at.add_atom(np.array([9.8,9.8,9.8]),np.array([11,12,0],dtype='int'),'0')
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        at.pair_type[('0','0')] = p
        mc = integrator.IntegratorMC_mu_geom_T_1(at,1.0,g,1.0,0.1)
        mo = neighbor.Neighbor.minus_one
        
        neigh = neighbor.NeighborClassMC(mc,at,g,5,np.array([1,1,1]),7.5)
        
        # 0:  0.0,0.0,0.0
        # 1:  4.3,4.3,4.3
        # 2:  4.4,4.4,4.4
        # 3:  5.4,5.4,5.4
        # 4:  9.7,9.7,9.7
        # 5:  9.8,9.8,9.8
        assert np.all(neigh.l == np.array([[ 1,mo,mo,mo,mo],
                                           [ 0, 2, 3,mo,mo],
                                           [ 1, 3,mo,mo,mo],
                                           [ 1, 2, 4,mo,mo],
                                           [ 3, 5,mo,mo,mo],
                                           [ 4,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo]]))
        assert np.all(neigh.nn == np.array([1,3,2,3,2,1,-1,-1],dtype='uint8'))
        
        
        at.translate_atom(0,np.array([-0.1,-0.1,-0.1]),ref='current')
        # 0:  -0.1,-0.1,-0.1
        # 1:   4.3, 4.3, 4.3
        # 2:   4.4, 4.4, 4.4
        # 3:   5.4, 5.4, 5.4
        # 4:   9.7, 9.7, 9.7
        # 5:   9.8, 9.8, 9.8
        assert np.all(neigh.l == np.array([[mo,mo,mo,mo,mo],
                                           [ 2, 3,mo,mo,mo],
                                           [ 1, 3,mo,mo,mo],
                                           [ 1, 2, 4,mo,mo],
                                           [ 3, 5,mo,mo,mo],
                                           [ 4,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo]]))
        assert np.all(neigh.nn == np.array([0,2,2,3,2,1,-1,-1],dtype='uint8'))
        
        
        at.translate_atom(0,np.array([0.1,0.1,0.1]),ref='current')
        # 0:  0.0,0.0,0.0
        # 1:  4.3,4.3,4.3
        # 2:  4.4,4.4,4.4
        # 3:  5.4,5.4,5.4
        # 4:  9.7,9.7,9.7
        # 5:  9.8,9.8,9.8
        assert np.all(neigh.l == np.array([[ 1,mo,mo,mo,mo],
                                           [ 0, 2, 3,mo,mo],
                                           [ 1, 3,mo,mo,mo],
                                           [ 1, 2, 4,mo,mo],
                                           [ 3, 5,mo,mo,mo],
                                           [ 4,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo]]))
        assert np.all(neigh.nn == np.array([1,3,2,3,2,1,-1,-1],dtype='uint8'))
        
        
        at.delete_atom(3)
        # 0:  0.0,0.0,0.0
        # 1:  4.3,4.3,4.3
        # 2:  4.4,4.4,4.4
        # 3:  9.7,9.7,9.7
        # 4:  9.8,9.8,9.8
        assert np.all(neigh.l == np.array([[ 1,mo,mo,mo,mo],
                                           [ 0, 2,mo,mo,mo],
                                           [ 1,mo,mo,mo,mo],
                                           [ 4,mo,mo,mo,mo],
                                           [ 3,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo]]))
        assert np.all(neigh.nn == np.array([1,2,1,1,1,-1,-1,-1],dtype='uint8'))
        
        
        at.add_atom(np.array([5.4,5.4,5.4]),np.array([0,0,0]),'0')
        # 0:  0.0,0.0,0.0
        # 1:  4.3,4.3,4.3
        # 2:  4.4,4.4,4.4
        # 3:  9.7,9.7,9.7
        # 4:  9.8,9.8,9.8
        # 5:  5.4,5.4,5.4
        assert np.all(neigh.l == np.array([[ 1,mo,mo,mo,mo],
                                           [ 0, 2, 5,mo,mo],
                                           [ 1, 5,mo,mo,mo],
                                           [ 4, 5,mo,mo,mo],
                                           [ 3,mo,mo,mo,mo],
                                           [ 1, 2, 3,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo]]))
        assert np.all(neigh.nn == np.array([1,3,2,2,1,3,-1,-1],dtype='uint8'))
        
        neigh.build(force=True)
        assert np.all(neigh.l == np.array([[ 1,mo,mo,mo,mo],
                                           [ 0, 2, 5,mo,mo],
                                           [ 1, 5,mo,mo,mo],
                                           [ 4, 5,mo,mo,mo],
                                           [ 3,mo,mo,mo,mo],
                                           [ 1, 2, 3,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo]]))
        assert np.all(neigh.nn == np.array([1,3,2,2,1,3,-1,-1],dtype='uint8'))
        
        
        return
        
    
    def test_add_many_atoms(self):
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,8)
        at.add_atom(np.array([-10.0,-10.0,-10.0]),np.array([0,0,0],dtype='int'),'0')
        at.add_atom(np.array([ -9.99,-9.99, -9.99]),np.array([0,0,0],dtype='int'),'0')
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        at.pair_type[('0','0')] = p
        mc = integrator.IntegratorMC_mu_geom_T_1(at,1.0,g,1.0,0.1)
        mo = neighbor.Neighbor.minus_one
        
        neigh = neighbor.NeighborClassMC(mc,at,g,5,np.array([1,1,1]),1.01*0.02*np.sqrt(3))
        
        # 0:  0.0,0.0,0.0
        # 1:  4.3,4.3,4.3
        # 2:  4.4,4.4,4.4
        # 3:  5.4,5.4,5.4
        # 4:  9.7,9.7,9.7
        # 5:  9.8,9.8,9.8
        assert np.all(neigh.l == np.array([[ 1,mo,mo,mo,mo],
                                           [ 0,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           [mo,mo,mo,mo,mo],
                                           ]))
        assert np.all(neigh.nn == np.array([1,1,-1,-1,-1,-1,-1,-1],dtype='uint8'))
        
        
        for j in range(2,1001):
            q = -10.0+0.01*j
            at.add_atom(np.array([q,q,q]),np.array([0,0,0],dtype='int'),'0')
        
        assert np.all(neigh.l[5] == np.array([3,4,6,7,mo]))
        assert np.all(neigh.l[500] == np.array([498,499,501,502,mo]))
        assert np.all(neigh.l[900] == np.array([898,899,901,902,mo]))
        assert np.all(neigh.l[999] == np.array([997,998,1000,mo,mo]))
        assert np.all(neigh.l[1000] == np.array([998,999,mo,mo,mo]))
        
        neigh.build(force=True)
        
        assert np.all(neigh.l[5] == np.array([3,4,6,7,mo]))
        assert np.all(neigh.l[500] == np.array([498,499,501,502,mo]))
        assert np.all(neigh.l[900] == np.array([898,899,901,902,mo]))
        assert np.all(neigh.l[999] == np.array([997,998,1000,mo,mo]))
        assert np.all(neigh.l[1000] == np.array([998,999,mo,mo,mo]))
        
        
        return
        
        
        
        
        
        
        
        
    
