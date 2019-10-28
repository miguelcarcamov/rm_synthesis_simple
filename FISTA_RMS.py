#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:52:15 2019

@author: miguel
"""

import numpy as np
import sys
import pywt
from transforms import form_P_meas, form_F_li, form_F_dirty
from thresholds import softThreshold, hardThreshold

def FISTA_Thin(P, W, K, phi, lambda2, lambda2_ref, m, n, lambda_threshold, delta_noise, tol):
    dirty_F = form_F_dirty(K, W, P, phi, lambda2, lambda2_ref, n)
    X_temp = dirty_F
    X = X_temp
    t_new = 1
    niter = int(np.floor(lambda_threshold/delta_noise))
    for i in range(0, niter):
            X_old = X_temp
            t_old = t_new
            #Gradient
            comb = X
            F_comb = form_P_meas(W, comb, phi, lambda2, lambda2_ref, m)
            D = F_comb - P
            if i%1000==0:
                objf = 0.5*np.sqrt(np.sum(np.abs(D)**2))
                print("Iteration - ", i,": ", objf)
            comb = comb - form_F_li(K, D, phi, lambda2, lambda2_ref, n)

            aux_Xreal = comb.real
            aux_Ximag = comb.imag

            aux_Xreal = softThreshold(aux_Xreal, lambda_threshold)
            aux_Ximag = softThreshold(aux_Ximag, lambda_threshold)

            X_temp = aux_Xreal + 1j*aux_Ximag

            norm = np.sum(np.abs(X_temp - X_old))
            #if norm <= tol:
            #    print("Iterations: ", i)
            #    print("Exit due to tolerance: ", norm, "<= ", tol)
            #    break;

            #Step using the Lipschitz constant
            t_new = (1+np.sqrt(1 + 4*t_old**2))/2
            aux_Xreal = X_temp.real + (t_old-1)/t_new*(X_temp.real-X_old.real)
            aux_Ximag = X_temp.imag + (t_old-1)/t_new*(X_temp.imag-X_old.imag)
            X = aux_Xreal + 1j*aux_Ximag
            lambda_threshold = lambda_threshold - delta_noise
    return X_temp

def FISTA_Thick(P, W, K, phi, lambda2, lambda2_ref, m, n, lambda_threshold, delta_noise, tol):
    db4 = pywt.Wavelet('db4')
    dirty_F = form_F_dirty(K, W, P, phi, lambda2, lambda2_ref, n)
    X_temp = dirty_F
    X = X_temp
    t_new = 1
    niter = int(np.floor(lambda_threshold/delta_noise))
    for i in range(0, niter):
            X_old = X_temp
            t_old = t_new
            #Gradient
            comb = X
            F_comb = form_P_meas(W, comb, phi, lambda2, lambda2_ref, m)
            D = F_comb - P
            if i%1000==0:
                objf = 0.5*np.sqrt(np.sum(np.abs(D)**2))
                print("Iteration - ", i,": ", objf)
            comb = comb - form_F_li(K, D, phi, lambda2, lambda2_ref, n)

            aux_Xreal = comb.real
            aux_Ximag = comb.imag

            re_coeffs = pywt.wavedec(aux_Xreal, db4, level=3, mode='zpd')
            im_coeffs = pywt.wavedec(aux_Ximag, db4, level=3, mode='zpd')

            thres_re_coeffs = []
            for j in re_coeffs:
                thres_j = pywt.threshold(j, lambda_threshold, 'soft')
                thres_re_coeffs.append(thres_j)

            thres_im_coeffs = []
            for k in im_coeffs:
                thres_k = pywt.threshold(k, lambda_threshold, 'soft')
                thres_im_coeffs.append(thres_k)

            aux_Xreal = pywt.waverec(thres_re_coeffs, 'db4', mode='zpd')
            aux_Ximag = pywt.waverec(thres_im_coeffs, 'db4', mode='zpd')

            X_temp  = aux_Xreal + 1j*aux_Ximag

            norm = np.sum(np.abs(X_temp - X_old))
            if norm <= tol:
                print("Iterations: ", i)
                print("Exit due to tolerance: ", norm, "<= ", tol)
                break;

            #Step using the Lipschitz constant
            t_new = (1+np.sqrt(1 + 4*t_old**2))/2
            aux_Xreal = X_temp.real + (t_old-1)/t_new*(X_temp.real-X_old.real)
            aux_Ximag = X_temp.imag + (t_old-1)/t_new*(X_temp.imag-X_old.imag)
            X = aux_Xreal + 1j*aux_Ximag
            lambda_threshold = lambda_threshold - delta_noise
    print("Max iterations reached")
    return X_temp


def FISTA_Mix(P, W, K, phi, lambda2, lambda2_ref, m, n, lambda_threshold, delta_noise, tol):
    db4 = pywt.Wavelet('db4')
    F_recon = np.zeros(n) + 1j*np.zeros(n)

    F_thin = form_F_dirty(K, W, P, phi, lambda2, lambda2_ref, n)
    F_thick = np.zeros(n) + 1j*np.zeros(n)
    F_temp = F_thin + F_thick
    F = F_temp
    t_new= 1

    niter = int(np.floor(lambda_threshold/delta_noise))
    for i in range(0, niter):
        F_old = F_temp
        t_old = t_new
        #Search thin structures
        F_comb = form_P_meas(W, F, phi, lambda2, lambda2_ref, m)
        D = P - F_comb
        if i%1000==0:
            objf = 0.5*np.sqrt(np.sum(np.abs(D)**2))
            print("Iteration - ", i,": ", objf)
        F_thin += form_F_li(K, D, phi, lambda2, lambda2_ref, n)

        aux_Xreal = softThreshold(F_thin.real, lambda_threshold)
        aux_Ximag = softThreshold(F_thin.imag, lambda_threshold)

        F_thin = aux_Xreal + 1j*aux_Ximag

        F_comb = form_P_meas(W, F_thin+F_thick, phi, lambda2, lambda2_ref, m)
        D = P - F_comb
        F_thick += form_F_li(K, D, phi, lambda2, lambda2_ref, n)

        aux_Xreal = F_thick.real
        aux_Ximag = F_thick.imag

        re_coeffs = pywt.wavedec(aux_Xreal, db4, level=3, mode='zpd')
        im_coeffs = pywt.wavedec(aux_Ximag, db4, level=3, mode='zpd')

        thres_re_coeffs = []
        for j in re_coeffs:
            thres_j = pywt.threshold(j, lambda_threshold, 'soft')
            thres_re_coeffs.append(thres_j)

        thres_im_coeffs = []
        for k in im_coeffs:
            thres_k = pywt.threshold(k, lambda_threshold, 'soft')
            thres_im_coeffs.append(thres_k)

        aux_Xreal = pywt.waverec(thres_re_coeffs, 'db4', mode='zpd')
        aux_Ximag = pywt.waverec(thres_im_coeffs, 'db4', mode='zpd')

        F_thick = aux_Xreal + 1j*aux_Ximag

        F_temp = F_thin + F_thick

        norm = np.sum(np.abs(F_temp - F_old))
        if norm <= tol:
            print("Iterations: ", i)
            print("Exit due to tolerance: ", norm, "<= ", tol)
            break;

        #Step using the Lipschitz constant
        t_new = (1+np.sqrt(1 + 4*t_old**2))/2
        aux_Xreal = F_temp.real + (t_old-1)/t_new*(F_temp.real-F_old.real)
        aux_Ximag = F_temp.imag + (t_old-1)/t_new*(F_temp.imag-F_old.imag)
        F = aux_Xreal + 1j*aux_Ximag
        lambda_threshold = lambda_threshold - delta_noise
    print("Max iterations reached")
    return F_temp

def Ultimate_FISTAMix(P, W, K, phi, lambda2, lambda2_ref, m, n, lambda_threshold, delta_noise, structure, tol):

    if structure=="Thin":
        F_recon = FISTA_Thin(P, W, K, phi, lambda2, lambda2_ref, m, n, lambda_threshold, delta_noise, tol)
    elif structure=="Thick":
        F_recon = FISTA_Thick(P, W, K, phi, lambda2, lambda2_ref, m, n, lambda_threshold, delta_noise, tol)
    else:
        F_recon = FISTA_Mix(P, W, K, phi, lambda2, lambda2_ref, m, n, lambda_threshold, delta_noise, tol)
    return F_recon
