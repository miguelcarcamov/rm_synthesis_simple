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

def RM_CLEAN(P, R, W, K, phi, lambda2, lambda2_ref, m, n, iterations, gain, threshold):
    dirty_F = form_F_dirty(K, P, phi, lambda2, lambda2_ref, n)
    rmsf = abs(R)
    fwhm_R = fwhm(phi, abs(R))
    g = gaussian(phi, 1.0, 0.0, fwhm_R/2)
    faraday_model = np.zeros(n) + 1j*np.zeros(n)
    i = 0
    while i < iterations and np.sum(np.abs(dirty_F)) > threshold:
        correlation = sci.convolve(dirty_F, rmsf, 'same', 'auto')
        corr_real = correlation.real
        corr_imag = correlation.imag
        
        peak_idx_real = np.where(corr_real  == np.max(corr_real))
        peak_idx_imag = np.where(corr_imag  == np.max(corr_imag))
        peak_idx_real = peak_idx_real[0][0]
        peak_idx_imag = peak_idx_imag[0][0]
        
        peak_real = dirty_F[peak_idx_real].real
        peak_imag = dirty_F[peak_idx_imag].imag
        #Storing a delta component at that location
        spike = faraday_model[peak_idx_real].real + gain * peak_real + faraday_model[peak_idx_imag].imag + gain * peak_imag
        faraday_model[peak_idx_real] = spike
        
        dif_real = int(np.floor(peak_idx_real - (n/2)))
        dif_imag = int(np.floor(peak_idx_imag - (n/2)))
        shifted_R_real = np.roll(rmsf,dif_real)
        shifted_R_imag = np.roll(rmsf,dif_imag)

        dirty_F.real = dirty_F.real - gain*peak_real*shifted_R_real
        dirty_F.imag = dirty_F.imag - gain*peak_imag*shifted_R_imag
        i = i+1
    residuals = dirty_F
    model_conv = sci.convolve(faraday_model, g, 'same', 'auto')
    return model_conv + residuals