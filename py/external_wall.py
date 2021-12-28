#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 10:59:53 2021

@author: MattMansell
"""

import numpy as np
import external
import geometry
import geometry_cartesian_box

class ExternalWall(external.External):
    
    def __init__(self,geom,which_walls):
        if not isinstance(geom,geometry_cartesian_box.CartesianBox):
            raise ValueError('geometry for external_wall must be of class ' +
                             'CartesianBox')
        if geom.ndim == 2:
            wall_strs = np.array([['xlo','ylo'],['xhi','yhi']])
        elif geom.ndim == 3:
            wall_strs = np.array([['xlo','ylo','zlo'],['xhi','yhi','zhi']])
        else:
            raise ValueError('external_wall is implemented only for 2 and 3 dimensions')
        self.geom = geom
        
        if not np.all(np.isin(which_walls,wall_strs)):
            raise ValueError('which_walls must be one of ' + ','.join(wall_strs.flatten()))
        self.walls = np.isin(wall_strs,which_walls)
    
    def phi(self,r):
        c = np.stack([r < self.geom.x[0],r >= self.geom.x[1]])
        
        if np.any(np.logical_and(c,self.walls)):
            en = np.infty
            f  = np.empty(self.geom.ndim)
            f[:] = np.nan
            return en,f
        en = 0.0
        f = np.zeros(self.geom.ndim)
        return en,f
    
    
    
        
        
        