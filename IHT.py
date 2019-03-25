#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:47:35 2019

@author: miguel
"""
import numpy as np
from transforms import form_P_meas, form_F_li, form_F_dirty
from thresholds import *

def largestElement(x, n):
    # returns the nth largest element of the vector x
    N = x.shape[0]
    if n > N:
        n = N
    elif n < 1:
        n = 1
    t = np.sort(x)[::-1]
    return t[n-1] # python index starts at 0

def reconstructIHT(P, W, K, phi, lambda2, lambda2_ref, m, n, s, Its=500, tol=1e-12, x=0, verbose=False):
    # recovers a sparse vector x from y using Iterative Hard thresholding Algorithm
    # xhat = reconstructIHT(A, t, T, tol, x, verbose)
    #  Arguments:
    #       A - measurement matrix
    #       y - measurements
    #       s - sparsity level require in reconstruction
    #       Its - max number of iterations (optional)
    #       tol - stopping criteria (optional)
    #       x - original vector used to print progress of MSE (optional)
    #       verbose - print progress (optional)

    # Initial estimate
    F_d = form_F_dirty(K, P, phi, lambda2, lambda2_ref, n)
    xhat = np.zeros(n) + 1j*np.zeros(n)
    # Initial residue
    y = form_P_meas(W, F_d, phi, lambda2, m)
    r = y

    for t in range(0, Its):
        # Pre-threshold value
        gamma = xhat + form_F_li(r, phi, lambda2, lambda2_ref, n)

        # Find the s-th largest coefficient of gamma
        threshold_real = largestElement(gamma.real, s)
        threshold_imag = largestElement(gamma.imag, s)

        # Estimate the signal (by hard thresholding)
        xhat_real = hardThreshold(gamma.real, threshold_real)
        xhat_imag = hardThreshold(gamma.imag, threshold_imag)
        xhat = xhat_real + 1j*xhat_imag
        
        # Compute error, print and plot
        #if verbose:
            #err = np.mean((x-xhat)**2)
            #print("iter# = ",str(t))

        # update the residual
        r = y - form_P_meas(W, xhat, phi, lambda2, m)

        # Stopping criteria
        if np.linalg.norm(r)/np.linalg.norm(y) < tol:
            break

    return xhat