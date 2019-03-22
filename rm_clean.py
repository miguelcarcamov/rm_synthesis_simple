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

def RM_CLEAN(P, R, W, K, phi, lambda2, lambda2_ref, m, n, iterations, gain, threshold, cross_corr=False):
    dirty_F = form_F_dirty(K, P, phi, lambda2, lambda2_ref, n)
    
    #idx_center = np.arange(n-1-(n/2), n-1+(n/2)).astype(int)
    rmsf = R
    
    fwhm_R = fwhm(phi, np.abs(rmsf))/2.0/np.sqrt(2.0 * np.log(2.0))
    g = gaussian(phi, 1.0, 0.0, fwhm_R)
    
    faraday_model = np.zeros(n) + 1j*np.zeros(n)
    #dirty_padded = np.zeros(2*n-1) + 1j*np.zeros(2*n-1)

    #dirty_padded[idx_center] = dirty_F
    
    i = 0
    
    if cross_corr:
        correlation = np.correlate(dirty_F-np.mean(dirty_F), rmsf-np.mean(rmsf), 'same')
        peak_idx = np.argmax(np.abs(correlation))
    else:
        peak_idx = np.argmax(np.abs(dirty_F))
        
    peak = dirty_F[peak_idx]
    
    while i < iterations and np.abs(peak) >= threshold:
        #print("Iteration: ", i, "- Peak ", np.abs(peak), "at position: ", peak_idx)
        #Scaled peak value
        #Storing a delta component at that location
        faraday_model[peak_idx] = faraday_model[peak_idx] + gain * peak
         
        #Calculate how many pixels to shift the rmsf       

        shft = peak_idx - np.argmax(np.abs(rmsf))
        shifted_rmsf = np.roll(rmsf, shft)
        #plt.cla()
        #plt.plot(np.abs(correlation), 'r--')
        #plt.plot(np.abs(dirty_F), 'b--', linewidth =0.5)
        #plt.plot(dirty_value , 'g-.', linewidth=5.0)
        #plt.savefig('Frame%03d.png' %i)
        #Substract a shifted and scaled rmsf from the dirty F
        dirty_F = dirty_F - gain*peak*shifted_rmsf
        i = i+1
        
        #Search the maximum value of the cross-correlation between dirty_F and the rmsf. Or find the maximum in the dirty_F
        if cross_corr:
            correlation = np.correlate(dirty_F-np.mean(dirty_F), rmsf-np.mean(rmsf), 'same')
            peak_idx = np.argmax(np.abs(correlation))
        else:
            peak_idx = np.argmax(np.abs(dirty_F))
        
        #The peak is the value of the dirty_F in that index
        peak = dirty_F[peak_idx]
        
        
       
    residuals = dirty_F
    model_conv = sci.convolve(faraday_model, g, 'same', 'auto')
    return faraday_model
    #return faraday_model