#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 13:20:06 2021

@author: mattmansell
"""

import pytest
import numpy as np

import sys
sys.path.append('../src')
import pair_lj_cut



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
        


