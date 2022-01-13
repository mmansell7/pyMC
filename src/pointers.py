#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:16:57 2022

@author: mattmansell
"""



class Pointers():
    
    def __init__(self,mc):
        self.mc = mc
    
    @property
    def at(self):
        return self.mc.at
    
    @property
    def neigh(self):
        return self.mc.neigh
    
    @property
    def geom(self):
        return self.mc.geom
    
    @property
    def force(self):
        return self.mc.force
    
    @property
    def grat(self):
        return self.mc.grat
    
    
    