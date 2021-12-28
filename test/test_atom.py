#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 08:10:46 2021

@author: MattMansell
"""

import numpy as np
import pytest

import sys
sys.path.append('../py')
import geometry_cartesian_box
import atom
import neighbor

class TestAtom():
    
    def test_create_1(self):
        x = np.array([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,100)
        return
        
    def test_create_fails_1(self):
        g = np.array([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]],dtype=float)
        with pytest.raises(ValueError) as excinfo:
            at = atom.Atom(g,100)
        return
    
    def test_create_fails_2(self):
        x = np.array([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        with pytest.raises(ValueError) as excinfo:
            at = atom.Atom(g,-1)
        return
    
    def test_init_1(self):
        x = np.array([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,113)
        assert at.n == 0
        assert at.nmax == 113
        assert at.x.shape[0] == 113
        assert at.x.shape[1] == 3
        assert at.atype.shape == (113,)
        return
    
    def test_add_atom_1(self):
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['fixed','fixed','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,3)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([0,0,0],dtype=int),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([1,1,1],dtype=int),'1')
        at.add_atom(np.array([2.0,2.0,2.0]),np.array([2,2,2],dtype=int),'2')
        at.add_atom(np.array([3.0,3.0,3.0]),np.array([3,3,3],dtype=int),'3')
        at.add_atom(np.array([4.0,4.0,4.0]),np.array([4,4,4],dtype=int),'4')
        at.add_atom(np.array([5.0,5.0,5.0]),np.array([5,5,5],dtype=int),'5')
        at.add_atom(np.array([6.0,6.0,6.0]),np.array([6,6,6],dtype=int),'6')
        at.add_atom(np.array([7.0,7.0,7.0]),np.array([7,7,7],dtype=int),'7')
        at.add_atom(np.array([8.0,8.0,8.0]),np.array([8,8,8],dtype=int),'8')
        at.add_atom(np.array([9.0,9.0,9.0]),np.array([9,9,9],dtype=int),'9')
        at.add_atom(np.array([10.0,10.0,10.0]),np.array([10,10,10],dtype=int),'10')
        at.add_atom(np.array([11.0,11.0,11.0]),np.array([11,11,11],dtype=int),'11')
        assert at.n == 12
        assert at.x.shape[0] >= 12
        assert at.x.shape[1] == 3
        assert at.im.shape[0] >= 12
        assert at.im.shape[1] == 3
        assert at.atype.shape[0] >= 12
        
        at.delete_atom(2)
        at.delete_atom(5)
        assert at.n == 10
        assert np.all(at.x[2] == np.array([3.0,3.0,3.0]))
        assert np.all(at.im[2] == np.array([3,3,3],dtype=int))
        assert at.atype[2] == '3'
        assert np.all(at.x[5] == np.array([7.0,7.0,7.0]))
        assert np.all(at.im[5] == np.array([7,7,7],dtype=int))
        assert at.atype[5] == '7'
        
        return

    def test_translate_atom_1(self):
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,3)
        at.add_atom(np.array([0.0,0.0,0.0]),np.array([1,2,0],dtype='int'),'0')
        at.add_atom(np.array([1.0,1.0,1.0]),np.array([11,12,0],dtype='int'),'1')
        
        at.translate_atom(0,np.array([6.0,6.0,6.0]),ref='origin')
        assert np.all(at.x[0] == np.array([6.0,6.0,6.0]))
        assert np.all(at.im[0] == np.array([0,0,0],dtype=int))
        at.translate_atom(0,np.array([5.0,5.0,5.0]),ref='origin')
        assert np.all(at.x[0] == np.array([5.0,5.0,5.0]))
        assert np.all(at.im[0] == np.array([0,0,0],dtype=int))
        at.translate_atom(0,np.array([6.0,6.0,6.0]),ref='current')
        assert np.all(at.x[0] == np.array([-9.0,-9.0,11.0]))
        assert np.all(at.im[0] == np.array([1,1,0],dtype=int))
        
        at.translate_atom(1,np.array([-6.0,-6.0,-6.0]),ref='current')
        assert np.all(at.x[1] == np.array([-5.0,-5.0,-5.0]))
        assert np.all(at.im[1] == np.array([11,12,0],dtype=int))
        at.translate_atom(1,np.array([-6.0,-6.0,-6.0]),ref='current')
        assert np.all(at.x[1] == np.array([9.0,9.0,-11.0]))
        assert np.all(at.im[1] == np.array([10,11,0],dtype=int))
        
        assert np.all(at.x[0] == np.array([-9.0,-9.0,11.0]))
        assert np.all(at.im[0] == np.array([1,1,0],dtype=int))
        
        return
    
    
        
        
    
        
        
    
        
        