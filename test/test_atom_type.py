#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:17:23 2021

@author: MattMansell
"""




import pytest

import sys
sys.path.append('./')
import atom_type


class TestAtomType():
    
    def test_create_1(self):
        atype = atom_type.AtomType('1')
        assert True






