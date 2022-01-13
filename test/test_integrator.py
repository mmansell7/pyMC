#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:14:15 2021

@author: MattMansell
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

class TestIntegrator():
    
    def test_instantiate(self):
        mc = sim.Sim()
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        at = atom.Atom(mc,3)
        # grat = integrator.Integrator(at,seed=0)
        grat = integrator.Integrator(mc,0)
    
    def test_step(self):
        mc = sim.Sim()
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
        grat = integrator.Integrator(mc,0)
        neigh = neighbor.NeighborClassMC(mc,5,np.array([1,1,1]),7.5)
        frc = force.Force(mc)
        
        grat.add_trial(trial.Trial_Translation(mc,1.0,'0'),1)
        
        assert grat.at == at
        assert grat.stepnum == 0
        grat.step()
        
        return
    

class TestIntegratorMC_mu_geom_T_1():
    
    def test_instantiate(self):
        mc = sim.Sim()
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        at = atom.Atom(mc,3)
        at.external_type['0'] = external_wall.ExternalWall(mc,['xlo','xhi'])
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([0,0,0],dtype='int'),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([0,0,0],dtype='int'),'0')
        at.add_atom(np.array([2.0,2.0,2.0]),np.array([0,0,0],dtype='int'),'0')
        grat = integrator.IntegratorMC_mu_geom_T_1(mc,1.0,1.0,0.1,0)
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        at.pair_type[('0','0')] = p
        neigh = neighbor.NeighborClassMC(mc,5,np.array([1,1,1]),7.5)
        
        assert grat.at == at
        assert grat.stepnum == 0
        
        
    def test_attempt_translation(self):
        mc = sim.Sim()
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        at = atom.Atom(mc,3)
        at.external_type['0'] = external_wall.ExternalWall(mc,['xlo','xhi'])
        at.add_atom(np.array([-0.01, 0.00, 0.00]),np.array([0,0,0],dtype='int'),'0')
        at.add_atom(np.array([ 0.00, 0.00, 0.00]),np.array([0,0,0],dtype='int'),'0')
        at.add_atom(np.array([ 0.01, 0.00, 0.00]),np.array([0,0,0],dtype='int'),'0')
        grat = integrator.IntegratorMC_mu_geom_T_1(mc,1.0,1.0,0.1,0)
        grat.kB = 1.0
        neigh = neighbor.NeighborClassMC(mc,5,np.array([1,1,1]),7.5)
        p = pair_lj_cut.PairLJCut(1.0,1.0,5.0)
        at.pair_type[('0','0')] = p
        frc = force.Force(mc)
        
        p.epsilon = 1.0
        p.sigma   = 1.0
        p.sigma2  = p.sigma**2
        p.rc      = 5.0
        p.rc2     = p.rc**2
        r01 = (at.x[1] - at.x[0])
        r01 = np.sqrt(np.dot(r01,r01))
        r02 = (at.x[2] - at.x[0])
        r02 = np.sqrt(np.dot(r02,r02))
        r12 = (at.x[2] - at.x[1])
        r12 = np.sqrt(np.dot(r12,r12))
        frc.update_all(force=True)
        en = np.array([4.0*p.epsilon*((p.sigma/r01)**12 - (p.sigma/r01)**6),
                        4.0*p.epsilon*((p.sigma/r02)**12 - (p.sigma/r02)**6),
                        4.0*p.epsilon*((p.sigma/r12)**12 - (p.sigma/r12)**6)])
        assert at.en_external[0] == 0.0
        assert at.en_external[1] == 0.0
        assert at.en_external[2] == 0.0
        print(at.en_pair)
        assert np.abs((at.en_pair[0] - (en[0] + en[1]))/at.en_pair[0]) < 1.0E-8
        assert np.abs((at.en_pair[1] - (en[0] + en[2]))/at.en_pair[1]) < 1.0E-8
        assert np.abs((at.en_pair[2] - (en[1] + en[2]))/at.en_pair[2]) < 1.0E-8
        
        f = np.array([4.0*p.epsilon/r01*(12.0*(p.sigma/r01)**12 - 6.0*(p.sigma/r01)**6),
                       4.0*p.epsilon/r02*(12.0*(p.sigma/r02)**12 - 6.0*(p.sigma/r02)**6),
                       4.0*p.epsilon/r01*(12.0*(p.sigma/r12)**12 - 6.0*(p.sigma/r12)**6)],dtype=float).reshape((3,1))
        print(at.f)
        print(f)
        print(np.array([[-f[0]-f[1]],[f[0]-f[2]],[f[1]+f[2]]]))
        assert np.abs(-f[0] - f[1] - at.f[0,0]) < 1.0E-8
        assert np.abs(f[0] - f[2] - at.f[1,0]) < 1.0E-8
        assert np.abs(f[1] + f[2] - at.f[2,0]) < 1.0E-8
        
        en0 = {'external':at.en_external.copy(),
               'pair':at.en_pair.copy(),
               'total':at.en_total.copy(),
               'force':at.f.copy()}
        grat.attempt_translation(0,np.array([0.0,0.0,0.0]))
        assert grat.last_accept == True
        assert np.all(at.en_external == en0['external'])
        assert np.all(at.en_pair     == en0['pair'])
        assert np.all(at.en_total    == en0['total'])
        assert np.all(at.f           == en0['force'])
        assert frc.en_external == np.sum(at.en_external)
        assert frc.en_pair == 1.0/2.0*np.sum(at.en_pair)
        assert frc.en_total == frc.en_external + frc.en_pair
        
        
        grat.attempt_translation(0,np.array([-1.0,0.0,0.0])-at.x[0])
        assert grat.last_accept == True
        grat.attempt_translation(2,np.array([ 1.0,0.0,0.0])-at.x[2])
        assert grat.last_accept == True
        r01 = (at.x[1] - at.x[0])
        r01 = np.sqrt(np.dot(r01,r01))
        r02 = (at.x[2] - at.x[0])
        r02 = np.sqrt(np.dot(r02,r02))
        r12 = (at.x[2] - at.x[1])
        r12 = np.sqrt(np.dot(r12,r12))
        en = np.array([4.0*p.epsilon*((p.sigma/r01)**12 - (p.sigma/r01)**6),
                       4.0*p.epsilon*((p.sigma/r02)**12 - (p.sigma/r02)**6),
                       4.0*p.epsilon*((p.sigma/r12)**12 - (p.sigma/r12)**6)])
        assert at.en_external[0] == 0.0
        assert at.en_external[1] == 0.0
        assert at.en_external[2] == 0.0
        print(at.x)
        print([r01,r02,r12])
        print(at.en_pair)
        print(en)
        print(np.array([en[0]+en[1],en[0]+en[2],en[1]+en[2]]))
        assert np.abs(at.en_pair[0] - (en[0] + en[1])) < 1.0E-8
        assert np.abs(at.en_pair[1] - (en[0] + en[2])) < 1.0E-8
        assert np.abs(at.en_pair[2] - (en[1] + en[2])) < 1.0E-8
        
        f = np.array([4.0*p.epsilon/r01*(12.0*(p.sigma/r01)**12 - 6.0*(p.sigma/r01)**6),
                       4.0*p.epsilon/r02*(12.0*(p.sigma/r02)**12 - 6.0*(p.sigma/r02)**6),
                       4.0*p.epsilon/r01*(12.0*(p.sigma/r12)**12 - 6.0*(p.sigma/r12)**6)],dtype=float).reshape((3,1))
        print(at.f)
        print(f)
        print(np.array([[-f[0]-f[1]],[f[0]-f[2]],[f[1]+f[2]]]))
        assert np.abs(-f[0] - f[1] - at.f[0,0]) < 1.0E-8
        assert np.abs(f[0] - f[2] - at.f[1,0]) < 1.0E-8
        assert np.abs(f[1] + f[2] - at.f[2,0]) < 1.0E-8


    def test_attempt_translation_2(self):
        mc = sim.Sim()
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        at = atom.Atom(mc,3)
        at.external_type['0'] = external_wall.ExternalWall(mc,['xlo','xhi'])
        at.add_atom(np.array([-0.01,-0.01,-0.01]),np.array([0,0,0],dtype='int'),'0')
        at.add_atom(np.array([ 0.00, 0.00, 0.00]),np.array([0,0,0],dtype='int'),'0')
        at.add_atom(np.array([ 0.01, 0.01, 0.01]),np.array([0,0,0],dtype='int'),'0')
        grat = integrator.IntegratorMC_mu_geom_T_1(mc,1.0,1.0,0.1,0)
        grat.kB = 1.0
        neigh = neighbor.NeighborClassMC(mc,5,np.array([1,1,1]),10.0)
        p = pair_lj_cut.PairLJCut(0.6,2.3,5.0*2.3)
        at.pair_type[('0','0')] = p
        frc = force.Force(mc)
        
        p.epsilon = 0.6
        p.sigma   = 2.3
        p.sigma2  = p.sigma**2
        p.rc      = 5.0*p.sigma
        p.rc2     = p.rc**2
        
        print(at.f)
        grat.attempt_translation(0,np.array([-3.0,-2.7,2.7])-at.x[0])
        assert grat.last_accept == True
        grat.attempt_translation(2,np.array([ 1.8,-1.0,2.2])-at.x[2])
        assert grat.last_accept == True
        
        r01 = (at.x[1] - at.x[0])
        r01 = np.sqrt(np.dot(r01,r01))
        r02 = (at.x[2] - at.x[0])
        r02 = np.sqrt(np.dot(r02,r02))
        r12 = (at.x[2] - at.x[1])
        r12 = np.sqrt(np.dot(r12,r12))
        frc.update_all(force=True)
        en = np.array([4.0*p.epsilon*((p.sigma/r01)**12 - (p.sigma/r01)**6),
                       4.0*p.epsilon*((p.sigma/r02)**12 - (p.sigma/r02)**6),
                       4.0*p.epsilon*((p.sigma/r12)**12 - (p.sigma/r12)**6)])
        assert at.en_external[0] == 0.0
        assert at.en_external[1] == 0.0
        assert at.en_external[2] == 0.0
        print(at.en_pair)
        assert np.abs((at.en_pair[0] - (en[0] + en[1]))/at.en_pair[0]) < 1.0E-8
        assert np.abs((at.en_pair[1] - (en[0] + en[2]))/at.en_pair[1]) < 1.0E-8
        assert np.abs((at.en_pair[2] - (en[1] + en[2]))/at.en_pair[2]) < 1.0E-8
        
        f = np.array([4.0*p.epsilon/r01*(12.0*(p.sigma/r01)**12 - 6.0*(p.sigma/r01)**6)*(at.x[1]-at.x[0])/r01,
                      4.0*p.epsilon/r02*(12.0*(p.sigma/r02)**12 - 6.0*(p.sigma/r02)**6)*(at.x[2]-at.x[0])/r02,
                      4.0*p.epsilon/r12*(12.0*(p.sigma/r12)**12 - 6.0*(p.sigma/r12)**6)*(at.x[2]-at.x[1])/r12],dtype=float)
        f1 = np.array([-f[0]-f[1],f[0]-f[2],f[1]+f[2]])
        print(at.f)
        print(f)
        print(f1)
        assert np.all(np.abs(f1-at.f) < 1.0E-8)
        
        frc.update_all(force=True)
        print(at.f)
        assert np.all(np.abs(f1-at.f) < 1.0E-8)
        
        return
    




