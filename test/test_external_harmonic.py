#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 13:23:43 2021

@author: mattmansell
"""

import pytest
import numpy as np

import sys
sys.path.append('../src')

import sim
import geometry_cartesian_box
import external_harmonic




class TestExternalHarmonic():
    
    def test_create1(self):
        mc = sim.Sim(0)
        x = np.array([[-11.1,-22.2,-33.3],[12.3,45.6,78.9]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        
        k  = 1.1
        r0 = np.array([0.0,1.0,2.0])
        p  = external_harmonic.ExternalHarmonic(mc,k,r0)
        assert p.k == 1.1
        assert np.all(p.r0 == np.array([0.0,1.0,2.0]))
        return
    
    def test_create_fail1(self):
        mc = sim.Sim(0)
        mc.geom = np.empty(3)
        
        k  = 1.1
        r0 = np.array([0.0,1.0,2.0])
        with pytest.raises(ValueError) as excinfo:
            p  = external_harmonic.ExternalHarmonic(mc,k,r0)
        return
    
    def test_create_fail2(self):
        mc = sim.Sim(0)
        x = np.array([[-11.1,-22.2,-33.3],[12.3,45.6,78.9]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        
        k  = 1.1
        r0 = np.array([0.0,1.0])
        with pytest.raises(ValueError) as excinfo:
            p  = external_harmonic.ExternalHarmonic(mc,k,r0)
        return
    
    def test_phi(self):
        mc = sim.Sim(0)
        x = np.array([[-11.1,-22.2,-33.3],[12.3,45.6,78.9]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(mc,x,bcs)
        
        k_ar = np.array([1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,4.70,4.70,4.70,4.70,4.70,4.70,4.70,4.70,4.70,4.70])
        
        r0_ar = np.array([0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,3.20,3.20,3.20,3.20,3.20,3.20,3.20,3.20,3.20,3.20,
                          0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.10,1.10,1.10,1.10,1.10,1.10,1.10,1.10,1.10,1.10,
                          0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80]).reshape((3,-1)).T
        
        r_ar = np.array([1.000000000,0.000000000,0.000000000,
                        0.500000000,0.000000000,0.000000000,
                        0.750000000,0.000000000,0.000000000,
                        1.250000000,0.000000000,0.000000000,
                        1.500000000,0.000000000,0.000000000,
                        2.000000000,0.000000000,0.000000000,
                        10.000000000,0.000000000,0.000000000,
                        0.577350269,0.577350269,0.577350269,
                        0.288675135,0.288675135,0.288675135,
                        0.433012702,0.433012702,0.433012702,
                        0.721687836,0.721687836,0.721687836,
                        0.866025404,0.866025404,0.866025404,
                        1.154700538,1.154700538,1.154700538,
                        5.773502692,5.773502692,5.773502692,
                        6.400000000,1.100000000,0.800000000,
                        5.900000000,1.100000000,0.800000000,
                        6.200000000,1.100000000,0.800000000,
                        7.500000000,1.100000000,0.800000000,
                        8.900000000,1.100000000,0.800000000,
                        5.047520861,2.947520861,2.647520861,
                        4.758845727,2.658845727,2.358845727,
                        4.932050808,2.832050808,2.532050808,
                        5.682606158,3.582606158,3.282606158,
                        6.490896534,4.390896534,4.090896534]).reshape((-1,3))

        en_check_ar = np.array([0.500000,0.125000,0.281250,0.781250,1.125000,2.000000,
                                50.000000,0.500000,0.125000,0.281250,0.781250,1.125000,
                                2.000000,50.000000,24.064000,17.131500,21.150000,43.451500,
                                76.351500,24.064000,17.131500,21.150000,43.451500,76.351500])
        
        f_check_ar = np.array([ -1.000000000,0.000000000,0.000000000,
                                -0.500000000,0.000000000,0.000000000,
                                -0.750000000,0.000000000,0.000000000,
                                -1.250000000,0.000000000,0.000000000,
                                -1.500000000,0.000000000,0.000000000,
                                -2.000000000,0.000000000,0.000000000,
                                -10.000000000,0.000000000,0.000000000,
                                -0.577350269,-0.577350269,-0.577350269,
                                -0.288675135,-0.288675135,-0.288675135,
                                -0.433012702,-0.433012702,-0.433012702,
                                -0.721687836,-0.721687836,-0.721687836,
                                -0.866025404,-0.866025404,-0.866025404,
                                -1.154700538,-1.154700538,-1.154700538,
                                -5.773502692,-5.773502692,-5.773502692,
                                -15.040000000,0.000000000,0.000000000,
                                -12.690000000,0.000000000,0.000000000,
                                -14.100000000,0.000000000,0.000000000,
                                -20.210000000,0.000000000,0.000000000,
                                -26.790000000,0.000000000,0.000000000,
                                -8.683348049,-8.683348049,-8.683348049,
                                -7.326574916,-7.326574916,-7.326574916,
                                -8.140638796,-8.140638796,-8.140638796,
                                -11.668248940,-11.668248940,-11.668248940,
                                -15.467213712,-15.467213712,-15.467213712]).reshape((-1,3))
        
        phi_check_ar = np.concatenate((en_check_ar.reshape((-1,1)),
                                       f_check_ar),axis=1)
            
        phi_ar = np.empty(phi_check_ar.shape)
        for ii in range(0,phi_check_ar.shape[0]):
            p  = external_harmonic.ExternalHarmonic(mc,k_ar[ii],r0_ar[ii])
            rr = r_ar[ii]
            phi_tmp = p.phi(r_ar[ii])
            phi_ar[ii,0]  = phi_tmp[0]
            phi_ar[ii,1:] = phi_tmp[1][:]
        assert np.all(np.abs(phi_ar[ii]-phi_check_ar[ii])<1.0E-7)
        
        #                       k  ,    r0x,r0y,r0z,        rx,ry,rz,    phi,    fx,fy,fz
        test_array = np.array([[1.0,  np.nan,0.0,0.0,     1.0,1.0,1.0,   1.0,  0.0,-1.0,-1.0],
                               [1.0,  0.0,np.nan,0.0,     1.0,1.0,1.0,   1.0,  -1.0,0.0,-1.0],
                               [1.0,  0.0,0.0,np.nan,     1.0,1.0,1.0,   1.0,  -1.0,-1.0,0.0],
                               [1.0,  np.nan,np.nan,0.0,  1.0,1.0,1.0,   0.5,  0.0,0.0,-1.0],
                               [1.0,  np.nan,0.0,np.nan,  1.0,1.0,1.0,   0.5,  0.0,-1.0,0.0],
                               [1.0,  0.0,np.nan,np.nan,  1.0,1.0,1.0,   0.5,  -1.0,0.0,0.0]])
        for t in test_array:
            p = external_harmonic.ExternalHarmonic(mc,t[0],t[1:4])
            phi_tmp = p.phi(t[4:7])
            assert np.abs(phi_tmp[0] - t[7]) < 1.0E-7
            assert np.all(np.abs(phi_tmp[1] - t[8:]) < 1.0E-7)
            
        return
        

    
    
    
    
    


