#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:25:07 2019

@author: miguel
"""

import numpy as np
from transforms import *
import scipy.signal as sci

def RM_CLEAN(P, R, W, K, phi, lambda2, lambda2_ref, m, n, iterations, gain):
    dirty_F = form_F_dirty(K, P, phi, lambda2, lambda2_ref, n)
    faraday_model = np.zeros(n) + 1j*np.zeros(n)
    dirty_padded = np.zeros(2*n-1) + 1j*np.zeros(2*n-1)
    center_idx = np.arange(n-1-(n/2), n-1+(n/2)).astype(int)
    
    dirty_padded[center_idx] = dirty_F
    for i in range(0, iterations):
        dirty_F = dirty_padded[center_idx]
        correlation = sci.convolve(dirty_F, R, 'full', 'auto')
        
        centercorr = correlation[center_idx]
        peak_idx = np.where(centercorr  == np.max(centercorr))
        peak_idx = peak_idx[0][0]
        peak = dirty_F[peak_idx]
        faraday_model[peak_idx] = faraday_model[peak_idx] + gain * peak
        
        indx = np.arange(peak_idx, peak_idx + n).astype(int)
        dirty_padded[indx] = dirty_padded[indx] - gain*(peak)*R

    
    residuals = dirty_padded[center_idx]
    return faraday_model + residuals