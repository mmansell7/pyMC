#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 22:06:51 2022

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
import trial

class TestTrialTranslate():
    
    def test_instantiate(self):
        mc = sim.Sim(0)
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        at = atom.Atom(mc,3)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([11,12,0],dtype='int'),'0')
        ext = external_wall.ExternalWall(mc,['xlo','xhi'])
        at.external_type['0'] = ext
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        at.pair_type[('0','0')] = p
        # grat = integrator.Integrator(at,seed=0)
        grat = integrator.Integrator(mc)
        neigh = neighbor.NeighborClassMC(mc,5,np.array([1,1,1]),7.5)
        frc = force.Force(mc)
        
        grat.add_trial(trial.Trial_Translation(mc,1.0,'0'),1)
        
    def test_energy_decrease_1(self):
        mc = sim.Sim(0)
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        at = atom.Atom(mc,3)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([11,12,0],dtype='int'),'0')
        ext = external_wall.ExternalWall(mc,['xlo','xhi'])
        at.external_type['0'] = ext
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        at.pair_type[('0','0')] = p
        # grat = integrator.Integrator(at,seed=0)
        grat = integrator.Integrator(mc)
        neigh = neighbor.NeighborClassMC(mc,5,np.array([1,1,1]),7.5)
        frc = force.Force(mc)
        
        grat.add_trial(trial.Trial_Translation(mc,1.0,'0'),1)
        mc.kB  = 1.0
        grat.T = 100.0
        
        # Move particle 0 from:
        #  Start
        #    x0: (0.0,0.0,0.0)
        #    external energy: 0.0
        #    pair distance (0->1): d = sqrt(3)
        #    pair energy (0->1): 4.0*1.0*( (1.0/d)**12 - (1.0/d**6) ) = -0.14266117969821676
        #    
        #  End:
        #    x0: (0.5,0.0,0.0)
        #    external energy: 0.0
        #    pair distance (0->1): d = np.sqrt(2.25)
        #    pair energy (0->1): 4.0*1.0*( (1.0/d)**12 - (1.0/d**6) ) = -0.32033659427857464
        #
        #  Energy change: -0.177675
        #  Boltzmann factor: np.exp(-en_change/(1.0*100.0)) = 1.001778329355516
        #  
        bf,test_random = grat.trials[0].execute(ind=0,dx=np.array([0.5,0.0,0.0]))
        
        assert np.abs(bf - 1.001778329355516) < 1.0E-8
        # 
        
        return
    
    
    def test_energy_increase_1(self):
        mc = sim.Sim(0)
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        at = atom.Atom(mc,3)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([11,12,0],dtype='int'),'0')
        ext = external_wall.ExternalWall(mc,['xlo','xhi'])
        at.external_type['0'] = ext
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        at.pair_type[('0','0')] = p
        # grat = integrator.Integrator(at,seed=0)
        grat = integrator.Integrator(mc)
        neigh = neighbor.NeighborClassMC(mc,5,np.array([1,1,1]),7.5)
        neigh.build(force=True)
        frc = force.Force(mc)
        
        grat.add_trial(trial.Trial_Translation(mc,1.0,'0'),1)
        mc.kB  = 1.0
        grat.T = 100.0
        
        # Move particle 0 from:
        #  Start
        #    x0: (0.0,0.0,0.0)
        #    external energy: 0.0
        #    pair distance (0->1): d = sqrt(3)
        #    pair energy (0->1): 4.0*1.0*( (1.0/d)**12 - (1.0/d**6) ) = -0.14266117969821676
        #    
        #  End:
        #    x0: (0.773,1.0,1.0)
        #    external energy: 0.0
        #    pair distance (0->1): d = 0.356
        #    pair energy (0->1): 4.0*1.0*( (1.0/d)**12 - (1.0/d**6) ) = 69.55241833894998
        #
        #  Energy change: 69.6950795186482
        #  Boltzmann factor: np.exp(-en_change/(1.0*100.0)) = 0.4981018049764179
        #  
        bf,test_random = grat.trials[0].execute(ind=0,dx=np.array([0.227,1.0,1.0]))
        
        assert np.abs(bf - 0.5001901776441272) < 1.0E-8
        assert np.abs(test_random - 0.5) <= 0.5
        
        return
    
    
    
    
    
    
    
    
    
    
    
    
