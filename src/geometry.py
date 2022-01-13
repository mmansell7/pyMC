#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:39:45 2021

@author: MattMansell
"""

import numpy as np

import pointers

class Geometry(pointers.Pointers):
    
    def __init__(self,mc):
        super().__init__(mc)
        self.mc.geom = self
        return
    
    def distance(self,r1,r2):
        pass
    
    def wrap_position(self,r1,im1):
        return r1,im1
    
    
    
