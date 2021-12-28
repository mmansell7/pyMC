#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:14:15 2021

@author: MattMansell
"""



import numpy as np
import pytest

import sys
sys.path.append('../py')
import integrator
import geometry_cartesian_box
import atom

class TestIntegrator():
    
    def test_integrator_1(self):
        x = np.array([[-10.0,-10.0,-10.0],[10.0,10.0,10.0]],dtype=float)
        bcs = np.array(['periodic','periodic','fixed'])
        g = geometry_cartesian_box.CartesianBox(x,bcs)
        at = atom.Atom(g,3)
        grat = integrator.Integrator(at)
        assert grat.at == at
        assert grat.stepnum == 0
        for i in range(1,13):
            grat.step()
        assert grat.stepnum == 12
        return
    




