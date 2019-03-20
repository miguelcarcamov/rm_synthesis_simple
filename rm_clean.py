#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:25:07 2019

@author: miguel
"""

import numpy as np
from transforms import *
import matplotlib.pyplot as plt
import scipy.signal as sci
from scipy.interpolate import splrep, sproot, splev
from astropy.modeling import models

class MultiplePeaks(Exception): pass
class NoPeaksFound(Exception): pass

def gaussian(x, amplitude, mean, stddev):
    f = amplitude*np.exp(-(x-mean)**2/(2*stddev**2))
    return f
def fwhm(x, y, k=3):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """

    half_max = np.max(y)/2.0
    s = splrep(x, y - half_max, k=k)
    roots = sproot(s)

    if len(roots) > 2:
        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
                "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                "the dataset is flat (e.g. all zeros).")
    else:
        return abs(roots[1] - roots[0])

def RM_CLEAN(P, R, W, K, phi, lambda2, lambda2_ref, m, n, iterations, gain):
    dirty_F = form_F_dirty(K, P, phi, lambda2, lambda2_ref, n)
    fwhm_R = fwhm(phi, abs(R))
    g = gaussian(phi, 1.0, 0.0, fwhm_R/2)
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
    model_conv = sci.convolve(faraday_model, g, 'same', 'auto')
    return model_conv + residuals