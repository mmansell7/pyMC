#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 18:58:31 2021

@author: mattmansell
"""

import numpy as np
import pytest

import sys
sys.path.append('../src')

import sim
import geometry_cartesian_box
import atom
import integrator
import neighbor
import external_wall
import pair_lj_cut
import force

class TestForce():
    
    def test_instantiate(self):
        mc = sim.Sim(0)
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        at = atom.Atom(mc,3)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([11,12,0],dtype='int'),'0')
        # mc = integrator.IntegratorMC_mu_geom_T_1(  at,1.0,g,1.0,0.1,0)
        grat = integrator.Integrator(mc)
        mo = neighbor.Neighbor.minus_one
        ext = external_wall.ExternalWall(mc,['xlo','xhi'])
        at.external_type['0'] = ext
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        at.pair_type[('0','0')] = p
        neigh = neighbor.NeighborClassMC(mc,5,np.array([1,1,1]),7.5)
        # neigh.build()
        assert np.all(neigh.l == np.array([[1,mo,mo,mo,mo],
                                          [0,mo,mo,mo,mo],
                                          [mo,mo,mo,mo,mo]]))
        assert np.all(neigh.nn == np.array([1,1,-1],dtype='uint8'))
        
        frc = force.Force(mc)
        
    def test_update_all(self):
        mc = sim.Sim(0)
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        at = atom.Atom(mc,3)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([11,12,0],dtype='int'),'0')
        # mc = integrator.IntegratorMC_mu_geom_T_1(at,1.0,g,1.0,0.1,0)
        grat = integrator.Integrator(mc)
        mo = neighbor.Neighbor.minus_one
        ext1 = external_wall.ExternalWall(mc,['xlo','xhi'])
        at.external_type['0'] = ext1
        p1 = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        at.pair_type[('0','0')] = p1
        neigh = neighbor.NeighborClassMC(mc,5,np.array([1,1,1]),7.5)
        # neigh.build()
        assert np.all(neigh.l == np.array([[1,mo,mo,mo,mo],
                                          [0,mo,mo,mo,mo],
                                          [mo,mo,mo,mo,mo]]))
        assert np.all(neigh.nn == np.array([1,1,-1],dtype='uint8'))
        
        frc = force.Force(mc)
        frc.update_all()
        # assert np.abs(frc.en_external - 0.0) < 1.0E-8
        # assert np.abs(frc.en_pair + 0.14266117969821684) < 1.0E-8
        # assert np.abs(frc.en_total + 0.14266117969821684) < 1.0E-8
        assert np.abs(frc.en +  0.14266117969821684) < 1.0E-8
        
        at.translate_atom(1,np.array([2.0,2.0,2.0]),ref='origin')
        
        frc.update_all()
        # assert np.abs(frc.en_external - 0.0) < 1.0E-8
        # assert np.abs(frc.en_pair + 0.0023134752229080954) < 1.0E-8
        # assert np.abs(frc.en_total + 0.0023134752229080954) < 1.0E-8
        assert np.abs(frc.en + 0.0023134752229080954) < 1.0E-8
        
        ext2 = external_wall.ExternalWall(mc,['xlo','xhi'])
        at.external_type['1'] = ext2
        p2 = pair_lj_cut.PairLJCut(1.5,2.0,10.0)
        at.pair_type[('0','1')] = p2
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([0,0,0]),'1')
        frc.update_all()
        # assert np.abs(frc.en_external - 0.0) < 1.0E-8
        # assert np.abs(frc.en_pair - (-0.0023134752229080954 + 2*19.48971193415642)) < 1.0E-8
        # assert np.abs(frc.en_total - frc.en_external - frc.en_pair) < 1.0E-8
        assert np.abs(frc.en - 0.0 - (-0.0023134752229080954 + 2*19.48971193415642)) < 1.0E-8
        


