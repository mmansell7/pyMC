#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 13:22:36 2021

@author: mattmansell
"""


import pytest
import numpy as np

import sys
sys.path.append('../src')

import sim
import geometry
import geometry_cartesian_box
import external_wall


class TestExternalWall():
    
    def test_create_1(self):
        mc = sim.Sim()
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        which_walls = ['xlo','xhi']
        w = external_wall.ExternalWall(mc,which_walls)
        assert w.geom == g
        return
    
    def test_create_fails1(self):
        mc = sim.Sim()
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry.Geometry(mc)
        which_walls = ['xlo','xhi']
        with pytest.raises(ValueError) as excinfo:
            w = external_wall.ExternalWall(mc,which_walls)
        return
    
    def test_create_fails2(self):
        mc = sim.Sim()
        x = np.array([[-1,-2,-3,-4],[1,2,3,4]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        which_walls = ['xlo','xhi']
        with pytest.raises(ValueError) as excinfo:
            w = external_wall.ExternalWall(mc,which_walls)
        return
    
    def test_create_fails3(self):
        mc = sim.Sim()
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry.Geometry(mc)
        which_walls = ['xlo','xmid']
        with pytest.raises(ValueError) as excinfo:
            w = external_wall.ExternalWall(mc,which_walls)
        return
    
    def test_create_fails4(self):
        mc = sim.Sim()
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry.Geometry(mc)
        which_walls = ['xlo','xhi']
        with pytest.raises(ValueError) as excinfo:
            w = external_wall.ExternalWall(mc,which_walls)
        return
    
    def test_phi(self):
        mc = sim.Sim()
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        
        which_walls = ['xlo','xhi','ylo','yhi','zlo','zhi']
        w = external_wall.ExternalWall(mc,which_walls)
        test_data = np.array(
            [[-1.001, 0.000, 0.000, np.infty, np.nan, np.nan, np.nan],
             [ 1.001, 0.000, 0.000, np.infty, np.nan, np.nan, np.nan],
             [ 0.000,-2.001, 0.000, np.infty, np.nan, np.nan, np.nan],
             [ 0.000, 2.001, 0.000, np.infty, np.nan, np.nan, np.nan],
             [ 0.000, 0.000,-3.001, np.infty, np.nan, np.nan, np.nan],
             [ 0.000, 0.000, 3.001, np.infty, np.nan, np.nan, np.nan],
             [-0.999, 1.999, 2.999,      0.0,    0.0,    0.0,    0.0],
             [ 0.999, 1.999, 2.999,      0.0,    0.0,    0.0,    0.0]])
        for t in test_data:
            q = w.phi(t[:3])
            print(t,q)
            assert np.all(q[0] == t[3])
            if np.any(np.isnan(t[4:7])):
                assert np.all(np.isnan(q[1]))
            else:
                assert np.all(q[1]==t[4:7])
        
        which_walls = ['xlo','xhi']
        w = external_wall.ExternalWall(g,which_walls)
        test_data[2:,3:] = 0.0
        for t in test_data:
            q = w.phi(t[:3])
            print(t,q)
            assert np.all(q[0] == t[3])
            if np.any(np.isnan(t[4:7])):
                assert np.all(np.isnan(q[1]))
            else:
                assert np.all(q[1]==t[4:7])
        
        which_walls = ['yhi']
        w = external_wall.ExternalWall(g,which_walls)
        test_data[:,3:] = 0.0
        test_data[3,3:] = np.array([np.infty,np.nan,np.nan,np.nan])
        for t in test_data:
            q = w.phi(t[:3])
            print(t,q)
            assert np.all(q[0] == t[3])
            if np.any(np.isnan(t[4:7])):
                assert np.all(np.isnan(q[1]))
            else:
                assert np.all(q[1]==t[4:7])
        
        
        return
        
    
    


