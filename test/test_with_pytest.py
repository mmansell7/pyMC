#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:38:08 2021

@author: MattMansell
"""

import pytest
import numpy as np

import sys
sys.path.append('../py')
import geometry
import geometry_cartesian_box
import pair_lj_cut
import external_harmonic
import external_wall
import atom

def test_always_passes():
    assert True
    
# def test_always_fails():
#     assert False

# with pytest.raises(RuntimeError) as excinfo:
# with pytest.raises(ValueError, match=r".* 123 .*"):
    

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
    
class TestPairLJCut():
    
    def test_create1(self):
        eps = 1.1
        sig = 2.0
        rc  = 3.0
        p = pair_lj_cut.PairLJCut(eps,sig,rc)
        assert p.epsilon == 1.1
        assert p.sigma == 2.0
        assert p.sigma2 == 4.0
        assert p.rc == 3.0
        assert p.rc2 == 9.0
        return
    
    def test_create_fail1(self):
        eps = -1.1
        sig = 2.0
        rc  = 3.0
        with pytest.raises(ValueError) as excinfo:
            p = pair_lj_cut.PairLJCut(eps,sig,rc)
        return
    
    def test_create_fail2(self):
        eps = 1.1
        sig = -2.0
        rc  = 3.0
        with pytest.raises(ValueError) as excinfo:
            p = pair_lj_cut.PairLJCut(eps,sig,rc)
        return
    
    def test_create_fail3(self):
        eps = 1.1
        sig = 2.0
        rc  = -3.0
        with pytest.raises(ValueError) as excinfo:
            p = pair_lj_cut.PairLJCut(eps,sig,rc)
        return
    
    def test_phi1(self):
        eps = 1.0
        sig = 1.0
        rc  = 5.0
        p   = pair_lj_cut.PairLJCut(eps,sig,rc)
        
        rsq1 = 1.0
        phi1 = p.phi(rsq1)
        en_check1 = 0.0
        f_check1 = 24.0
        assert phi1 == (en_check1,f_check1)
        
        rsq2 = 0.1
        phi2 = p.phi(rsq2)
        phi2_check = (3996000.0,479760000.0)
        assert np.abs(phi2[0] - phi2_check[0]) < 1.0
        assert np.abs(phi2[1] - phi2_check[1]) < 1.0
        
        rsq3 = 25.0001
        phi3 = p.phi(rsq3)
        phi3_check = (0.0,0.0)
        assert phi3[0] == phi3_check[0]
        assert phi3[1] == phi3_check[1]
        
        rsq4 = 2.3
        phi4 = p.phi(rsq4)
        phi4_check = (-0.30173764147,-0.71665347841)
        assert np.abs(phi4[0] - phi4_check[0]) < 1.0
        assert np.abs(phi4[1] - phi4_check[1]) < 1.0
        return
        
    def test_phi2(self):
        eps = 2.4
        sig = 7.7
        rc  = 22.2
        p   = pair_lj_cut.PairLJCut(eps,sig,rc)
        
        rsq_ar = np.array([1.0,0.1,25.0001,2.3,100.0,380.25])
        phi_ar = np.empty((rsq_ar.shape[0],2))
        phi_check_ar = np.array([[4.170209290E+11,  5.004263153E+12],
                         [4.170229278E+17,  5.004275146E+19],
                         [1.580031752E+03,  7.891448462E+02],
                         [2.816874936E+09,  1.469716780E+10],
                         [-1.583831919E+00,  -7.000853935E-02],
                         [-3.625420966E-02,  -5.698816944E-04]])
        phi_acceptable_err_mag_ar = np.array([[1.0E2,1.0E3],
                                              [1.0E8,1.0E10],
                                              [1.0E-6,1.0E-7],
                                              [1.0E0,1.0E1],
                                              [1.0E-9,1.0E-11],
                                              [1.0E-11,1.0E-13]])
        
        for ii in range(0,rsq_ar.shape[0]):
            phi_ar[ii] = p.phi(rsq_ar[ii])
        assert np.all(np.abs(phi_ar-phi_check_ar)<phi_acceptable_err_mag_ar)
        
        return
        
class TestExternalHarmonic():
    
    def test_create1(self):
        x = np.array([[-11.1,-22.2,-33.3],[12.3,45.6,78.9]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        
        k  = 1.1
        r0 = np.array([0.0,1.0,2.0])
        p  = external_harmonic.ExternalHarmonic(g,k,r0)
        assert p.k == 1.1
        assert np.all(p.r0 == np.array([0.0,1.0,2.0]))
        return
    
    def test_create_fail1(self):
        g = np.empty(3)
        
        k  = 1.1
        r0 = np.array([0.0,1.0,2.0])
        with pytest.raises(ValueError) as excinfo:
            p  = external_harmonic.ExternalHarmonic(g,k,r0)
        return
    
    def test_create_fail2(self):
        x = np.array([[-11.1,-22.2,-33.3],[12.3,45.6,78.9]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        
        k  = 1.1
        r0 = np.array([0.0,1.0])
        with pytest.raises(ValueError) as excinfo:
            p  = external_harmonic.ExternalHarmonic(g,k,r0)
        return
    
    def test_phi(self):
        x = np.array([[-11.1,-22.2,-33.3],[12.3,45.6,78.9]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        
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
            p  = external_harmonic.ExternalHarmonic(g,k_ar[ii],r0_ar[ii])
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
            p = external_harmonic.ExternalHarmonic(g,t[0],t[1:4])
            phi_tmp = p.phi(t[4:7])
            assert np.abs(phi_tmp[0] - t[7]) < 1.0E-7
            assert np.all(np.abs(phi_tmp[1] - t[8:]) < 1.0E-7)
            
        return
        
class TestExternalWall():
    
    def test_create_1(self):
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        which_walls = ['xlo','xhi']
        w = external_wall.ExternalWall(g,which_walls)
        assert w.geom == g
        return
    
    def test_create_fails1(self):
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry.Geometry()
        which_walls = ['xlo','xhi']
        with pytest.raises(ValueError) as excinfo:
            w = external_wall.ExternalWall(g,which_walls)
        return
    
    def test_create_fails2(self):
        x = np.array([[-1,-2,-3,-4],[1,2,3,4]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        which_walls = ['xlo','xhi']
        with pytest.raises(ValueError) as excinfo:
            w = external_wall.ExternalWall(g,which_walls)
        return
    
    def test_create_fails3(self):
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry.Geometry()
        which_walls = ['xlo','xmid']
        with pytest.raises(ValueError) as excinfo:
            w = external_wall.ExternalWall(g,which_walls)
        return
    
    def test_create_fails4(self):
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry.Geometry()
        which_walls = ['xlo','xhi']
        with pytest.raises(ValueError) as excinfo:
            w = external_wall.ExternalWall(g,which_walls)
        return
    
    def test_phi(self):
        x = np.array([[-1,-2,-3],[1,2,3]],dtype=float)
        bcs = np.array(['fixed','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        
        which_walls = ['xlo','xhi','ylo','yhi','zlo','zhi']
        w = external_wall.ExternalWall(g,which_walls)
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
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    