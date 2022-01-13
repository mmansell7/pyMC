#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:55:49 2021

@author: MattMansell
"""

import pointers
import atom

class AtomListener(pointers.Pointers):
    
    def __init__(self,mc):
        '''
        '''
        super().__init__(mc)
        self.at.listeners.append(self)
        
        return
    
    def atom_added(self):
        raise Exception('atom_added method must be implemented by subclasses of AtomListener.')
    
    def atom_deleted(self):
        raise Exception('atom_deleted method must be implemented by subclasses of AtomListener.')
    
    def atom_moved(self,ind,xold,xnew,dx):
        raise Exception('atom_moved method must be implemented by subclasses of AtomListener.')
    
    def atom_type_changed(self,ind,old_type,new_type):
        raise Exception('atom_type_changed method must be implemented by subclasses of AtomListener.')
    
    