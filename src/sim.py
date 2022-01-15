#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:15:00 2022

@author: mattmansell
"""



class Sim():
    
    def __init__(self):
        self.__kB = None
        self.__h  = None
    
    @property
    def at(self):
        return self.__at
    
    @at.setter
    def at(self,at):
        if hasattr(self,'at'):
            raise Exception("Cannot reset at once it is set.")
        else:
            self.__at = at
        return
    
    @property
    def force(self):
        return self.__force
    
    @force.setter
    def force(self,force):
        if hasattr(self,'force'):
            raise Exception("Cannot reset force once it is set.")
        else:
            self.__force = force
        return
    
    @property
    def neigh(self):
        print('neigh attribute')
        return self.__neigh
    
    @neigh.setter
    def neigh(self,neigh):
        print('neigh setter')
        if hasattr(self,'neigh'):
            raise Exception("Cannot reset neigh once it is set.")
        else:
            self.__neigh = neigh
        return
    
    @property
    def geom(self):
        return self.__geom
    
    @geom.setter
    def geom(self,geom):
        if hasattr(self,'geom'):
            raise Exception("Cannot reset geom once it is set.")
        else:
            self.__geom = geom
        return
    
    @property
    def grat(self):
        return self.__grat
    
    @grat.setter
    def grat(self,grat):
        if hasattr(self,'grat'):
            raise Exception("Cannot reset grat once it is set.")
        else:
            self.__grat = grat
        return
    
    @property
    def kB(self):
        return self.__kB
    
    @kB.setter
    def kB(self,val):
        if self.__kB is None:
            self.__kB = val
        else:
            raise Exception('Cannot re-set kB.')
        return
    
    @property
    def h(self):  # Planck constant
        return self.__h
    
    @h.setter
    def h(self,val):
        if self.__h is None:
            self.__h = val
        else:
            raise Exception('Cannot re-set h.')
        return
    
    
    
    
    
    
    
    
    
    
    
    