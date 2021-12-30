#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 13:21:34 2021

@author: mattmansell
"""

import pytest
import numpy as np

import sys
sys.path.append('../src')
import geometry_cartesian_box



class TestCartesianBox():
    
    def test_create1(self):
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        assert np.all(g.L == np.array([2,4,6],dtype=float))
        assert np.all(~(g.L == np.array([1.999999,3.999999,5.999999],dtype=float)))
        assert np.all(~(g.L == np.array([2.000001,4.000001,6.000001],dtype=float)))
        assert np.all((g.L == g.x[1]-g.x[0]))
        assert g.ndim == 3
        assert np.all(np.isin(g.bcs,['fixed','periodic']))
        assert g.bcs.shape[0] == g.ndim
        return
    
    def test_create2(self):
        x = np.array([[-11.1,-22.2,-33.3],[12.3,45.6,78.9]],dtype=float)
        bcs = np.array(['fixed','periodic','periodic'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        assert np.all(~(g.L == np.array([23.399999,67.799999,112.199999],dtype=float)))
        assert np.all(~(g.L == np.array([23.400001,67.800001,112.200001],dtype=float)))
        assert np.all(g.L == np.array([23.4,67.8,112.2],dtype=float))
        assert np.all((g.L == g.x[1]-g.x[0]))
        assert g.ndim == 3
        assert np.all(np.isin(g.bcs,['fixed','periodic']))
        assert g.bcs.shape[0] == g.ndim
        return
        
    def test_create_inverted_x(self):
        x = np.array([[1,-2,-3],[-1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        with pytest.raises(ValueError) as excinfo:
            g = geometry_cartesian_box.CartesianBox(x,bcs)
        return
        
    def test_create_invalid_bcs(self):
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixedi','periodic','fixed'])
        with pytest.raises(ValueError) as excinfo:
            g = geometry_cartesian_box.CartesianBox(x,bcs)
        return
    
    def test_distance_1(self):
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([-0.1,-0.2,-0.3])
        r2 = np.array([0.1,0.2,0.3])
        dist,dr2,dr = g.distance(r1,r2)
        dr_check = np.array([0.2,0.4,0.6])
        dr2_check = 0.56
        dist_check = 0.74833147735
        assert np.all(dr == dr_check)
        assert np.all(dr2 == dr2_check)
        assert np.all(np.abs(dist-dist_check)<1.0E-10)
        return
    
    def test_distance_2(self):
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([0.1,0.2,0.3])
        r2 = np.array([-0.1,-0.2,-0.3])
        dist,dr2,dr = g.distance(r1,r2)
        dr_check = np.array([-0.2,-0.4,-0.6])
        dr2_check = 0.56
        dist_check = 0.74833147735
        assert np.all(dr == dr_check)
        assert np.all(dr2 == dr2_check)
        assert np.all(np.abs(dist-dist_check)<1.0E-10)
        return
    
    def test_distance_3(self):
        x = np.array([[-10,-10,-10],[10,10,10]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([-9.0,-9.0,-9.0])
        r2 = np.array([9.0,9.0,9.0])
        dist,dr2,dr = g.distance(r1,r2)
        dr_check = np.array([18.0,18.0,18.0])
        dr2_check = 972.0
        dist_check = 31.1769145362
        assert np.all(dr == dr_check)
        assert np.all(dr2 == dr2_check)
        assert np.all(np.abs(dist-dist_check)<1.0E-10)
        return
    
    def test_distance_4x(self):
        x = np.array([[-10,-10,-10],[10,10,10]],dtype=float)
        bcs = np.array(['periodic','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([-9.0,-9.0,-9.0])
        r2 = np.array([9.0,9.0,9.0])
        dist,dr2,dr = g.distance(r1,r2)
        dr_check = np.array([-2.0,18.0,18.0])
        dr2_check = 652.0
        dist_check = 25.5342906696
        assert np.all(dr == dr_check)
        assert np.all(dr2 == dr2_check)
        assert np.all(np.abs(dist-dist_check)<1.0E-10)
        return
    
    def test_distance_4y(self):
        x = np.array([[-10,-10,-10],[10,10,10]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([-9.0,-9.0,-9.0])
        r2 = np.array([9.0,9.0,9.0])
        dist,dr2,dr = g.distance(r1,r2)
        dr_check = np.array([18.0,-2.0,18.0])
        dr2_check = 652.0
        dist_check = 25.5342906696
        assert np.all(dr == dr_check)
        assert np.all(dr2 == dr2_check)
        assert np.all(np.abs(dist-dist_check)<1.0E-10)
        return
    
    def test_distance_4z(self):
        x = np.array([[-10,-10,-10],[10,10,10]],dtype=float)
        bcs = np.array(['fixed','fixed','periodic'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([-9.0,-9.0,-9.0])
        r2 = np.array([ 9.0, 9.0, 9.0])
        dist,dr2,dr = g.distance(r1,r2)
        dr_check = np.array([18.0,18.0,-2.0])
        dr2_check = 652.0
        dist_check = 25.5342906696
        assert np.all(dr == dr_check)
        assert np.all(dr2 == dr2_check)
        assert np.all(np.abs(dist-dist_check)<1.0E-10)
        return
    
    def test_distance_4all(self):
        x = np.array([[-10,-10,-10],[10,10,10]],dtype=float)
        bcs = np.array(['periodic','periodic','periodic'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([-9.0,-9.0,-9.0])
        r2 = np.array([ 9.0, 9.0, 9.0])
        dist,dr2,dr = g.distance(r1,r2)
        dr_check = np.array([-2.0,-2.0,-2.0])
        dr2_check = 12
        dist_check = np.sqrt(12)
        assert np.all(dr == dr_check)
        assert np.all(dr2 == dr2_check)
        assert np.all(np.abs(dist-dist_check)<1.0E-10)
        return
    
    def test_distance_4allinverse(self):
        x = np.array([[-10,-10,-10],[10,10,10]],dtype=float)
        bcs = np.array(['periodic','periodic','periodic'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([ 9.0, 9.0, 9.0])
        r2 = np.array([-9.0,-9.0,-9.0])
        dist,dr2,dr = g.distance(r1,r2)
        dr_check = np.array([2.0,2.0,2.0])
        dr2_check = 12
        dist_check = np.sqrt(12)
        assert np.all(dr == dr_check)
        assert np.all(dr2 == dr2_check)
        assert np.all(np.abs(dist-dist_check)<1.0E-10)
        return
    
    def test_distance_5(self):
        x = np.array([[-10,-10,-10],[10,10,10]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([-20.0,30.0,-40.0])
        r2 = np.array([ 20.0,20.0, 20.0])
        dist,dr2,dr = g.distance(r1,r2)
        dr_check = np.array([40.0,-10.0,60.0])
        dr2_check = 5300.0
        dist_check = 72.8010988928
        assert np.all(dr == dr_check)
        assert np.all(dr2 == dr2_check)
        assert np.all(np.abs(dist-dist_check)<1.0E-10)
        return
    
    ##############################################################
    
    def test_wrap_position_1(self):
        x = np.array([[-10,-10,-10],[10,10,10]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([0.0,0.0,0.0])
        im1 = np.array([0,0,0],dtype=int)
        r2,im2 = g.wrap_position(r1,im1)
        r2_check = np.array([0.0,0.0,0.0])
        im2_check = np.array([0,0,0],dtype=int)
        assert np.all(np.abs(r2==r2_check)<1.0E10)
        assert np.all(im2==im2_check)
        return
    
    def test_wrap_position_2(self):
        x = np.array([[-10,-10,-10],[10,10,10]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([-100.0,100.0,1000.0])
        im1 = np.array([1,2,3],dtype=int)
        r2,im2 = g.wrap_position(r1,im1)
        r2_check = np.array([-100.0,100.0,1000.0])
        im2_check = np.array([1,2,3],dtype=int)
        assert np.all(np.abs(r2==r2_check)<1.0E10)
        assert np.all(im2==im2_check)
        return
    
    def test_wrap_position_edge_low(self):
        x = np.array([[-10,-10,-10],[10,10,10]],dtype=float)
        bcs = np.array(['periodic','periodic','periodic'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([-10.0,-10.0,-10.0])
        im1 = np.array([10,10,10],dtype=int)
        r2,im2 = g.wrap_position(r1,im1)
        r2_check = np.array([-10.0,-10.0,-10.0])
        im2_check = np.array([10,10,10],dtype=int)
        assert np.all(np.abs(r2==r2_check)<1.0E10)
        assert np.all(im2==im2_check)
        return
    
    def test_wrap_position_edge_high(self):
        x = np.array([[-10,-10,-10],[10,10,10]],dtype=float)
        bcs = np.array(['periodic','periodic','periodic'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([10.0000000,10.000000,10.000000])
        im1 = np.array([10,10,10],dtype=int)
        r2,im2 = g.wrap_position(r1,im1)
        r2_check = np.array([-10.0,-10.0,-10.0])
        im2_check = np.array([11,11,11],dtype=int)
        assert np.all(np.abs(r2==r2_check)<1.0E10)
        assert np.all(im2==im2_check)
        return
    
    def test_wrap_position_3(self):
        x = np.array([[-10,-10,-10],[10,10,10]],dtype=float)
        bcs = np.array(['periodic','fixed','periodic'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        r1 = np.array([-90.1,90.1,89.9])
        im1 = np.array([0,0,0],dtype=int)
        r2,im2 = g.wrap_position(r1,im1)
        r2_check = np.array([9.9,90.1,9.9])
        im2_check = np.array([-5,0,4],dtype=int)
        assert np.all(np.abs(r2==r2_check)<1.0E10)
        assert np.all(im2==im2_check)
        return
    
