#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:28:20 2019

@author: miguel
"""
import numpy as np

# Soft thresholding function
def softThreshold(x, threshold):
    j = np.abs(x) <= threshold
    x[j] = 0
    j = np.abs(x) > threshold
    x[j] = x[j] - np.sign(x[j])*threshold
    return x

# Hard thresholding function
def hardThreshold(x, threshold):
    j = np.abs(x) < threshold
    x[j] = 0
    return x