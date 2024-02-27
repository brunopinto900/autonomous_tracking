#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 08:30:50 2024

@author: bruno
"""
import numpy as np

p1 = np.array([1,1])
p2 = np.array([3,3])
p3 = np.array([2,2])

print(np.abs(np.linalg.norm(np.cross(p2-p1, p1-p3)))/np.linalg.norm(p2-p1))