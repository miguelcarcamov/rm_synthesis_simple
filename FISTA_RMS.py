#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:52:15 2019

@author: miguel
"""

import numpy as np
import pywt
from transforms import form_P_meas, form_F_li, form_F_dirty
from thresholds import softThreshold

def FISTA_Thin(P, W, K, phi, lambda2, lambda2_ref, m, n, soft_threshold, iterations):
    dirty_F = form_F_dirty(K, P, phi, lambda2, lambda2_ref, n)
    #F_meas = F_meas/len(F_meas)
    X_temp = dirty_F
    X = X_temp
    t_new = 1
    for i in range(0, iterations):
            X_old = X_temp
            t_old = t_new
            #Gradient
            comb = X
            F_comb = form_P_meas(W, comb, phi, lambda2, m)
            D = F_comb - P
            comb = comb - form_F_li(D, phi, lambda2, lambda2_ref, n)
            
            Xreal = comb.real
            Ximag = comb.imag
            
            X_temp.real = softThreshold(Xreal, soft_threshold)
      
            X_temp.imag = softThreshold(Ximag, soft_threshold)
            
            t_new = (1+np.sqrt(1 + 4*t_old**2))/2
            X.real = X_temp.real + (t_old-1)/t_new*(X_temp.real-X_old.real)
            X.imag = X_temp.imag + (t_old-1)/t_new*(X_temp.imag-X_old.imag)

    return X_temp

def FISTA_Thick(P, W, K, phi, lambda2, lambda2_ref, m, n, soft_threshold, iterations):
    dirty_F = form_F_dirty(K, P, phi, lambda2, lambda2_ref, n)
    #F_meas = F_meas/len(F_meas)
    X_temp = dirty_F
    X = X_temp
    t_new = 1
    db8 = pywt.Wavelet('db8')
    for i in range(0, iterations):
            X_old = X_temp
            t_old = t_new
            #Gradient
            comb = X
            F_comb = form_P_meas(W, comb, phi, lambda2, m)
            D = F_comb - P
            comb = comb - form_F_li(D, phi, lambda2, lambda2_ref, n)
            
            A_re, B_re, C_re, D_re = pywt.wavedec(comb.real, db8, level=3, mode='zpd')
            A_im, B_im, C_im, D_im = pywt.wavedec(comb.imag, db8, level=3, mode='zpd')
            
            A_re = softThreshold(A_re, soft_threshold)
            B_re = softThreshold(B_re, soft_threshold)
            C_re = softThreshold(C_re, soft_threshold)
            D_re = softThreshold(D_re, soft_threshold)
            A_im = softThreshold(A_im, soft_threshold)
            B_im = softThreshold(B_im, soft_threshold)
            C_im = softThreshold(C_im, soft_threshold)
            D_im = softThreshold(D_im, soft_threshold)
            
            X_temp.real = pywt.waverec([A_re, B_re, C_re, D_re], 'db8', mode='zpd')       
            X_temp.imag = pywt.waverec([A_im, B_im, C_im, D_im], 'db8', mode='zpd')
            
            t_new = (1+np.sqrt(1 + 4*t_old**2))/2
            X.real = X_temp.real + (t_old-1)/t_new*(X_temp.real-X_old.real)
            X.imag = X_temp.imag + (t_old-1)/t_new*(X_temp.imag-X_old.imag)

    return X_temp

def FISTA_Mix(P, W, K, phi, lambda2, lambda2_ref, m, n, soft_threshold, iterations):
    F_thin = form_F_dirty(K, P, phi, lambda2, lambda2_ref, n)
    F_thick = np.zeros(n) + 1j*np.zeros(n)
    db8 = pywt.Wavelet('db8')
    for i in range(0, iterations):
            #Thin structures
            F_comb = form_P_meas(W, F_thin+F_thick, phi, lambda2, m)
            residual = P - F_comb
            F_thin = F_thin + form_F_li(residual, phi, lambda2, lambda2_ref, n)
            Xreal = F_thin.real
            Ximag = F_thin.imag
            X_tempreal = softThreshold(Xreal, soft_threshold)
            X_tempimag = softThreshold(Ximag, soft_threshold)
            F_thin = X_tempreal + 1j* X_tempimag
            
            #Thick structures
            F_comb = form_P_meas(W, F_thin+F_thick, phi, lambda2, m)
            residual = P - F_comb
            F_thick = F_thick + form_F_li(residual, phi, lambda2, lambda2_ref, n)
            
            A_re, B_re, C_re, D_re = pywt.wavedec(F_thick.real, db8, level=3, mode='zpd')
            A_im, B_im, C_im, D_im = pywt.wavedec(F_thick.imag, db8, level=3, mode='zpd')
            
            A_re = softThreshold(A_re, soft_threshold)
            B_re = softThreshold(B_re, soft_threshold)
            C_re = softThreshold(C_re, soft_threshold)
            D_re = softThreshold(D_re, soft_threshold)
            A_im = softThreshold(A_im, soft_threshold)
            B_im = softThreshold(B_im, soft_threshold)
            C_im = softThreshold(C_im, soft_threshold)
            D_im = softThreshold(D_im, soft_threshold)
            
            real_Xthick = pywt.waverec([A_re, B_re, C_re, D_re], 'db8', mode='zpd') 
            imag_Xthick = pywt.waverec([A_im, B_im, C_im, D_im], 'db8', mode='zpd')
            
            F_thick = real_Xthick + 1j* imag_Xthick
    return F_thin+F_thick


def FISTA_Mix_General(P, W, K, phi, lambda2, lambda2_ref, m, n, soft_threshold, iterations):
    F_thin = form_F_dirty(K, P, phi, lambda2, lambda2_ref, n)
    F_thick = np.zeros(n) + 1j*np.zeros(n)

    for i in range(0, iterations):
            #Thin structures
            F_comb = form_P_meas(W, F_thin+F_thick, phi, lambda2, m)
            residual = P - F_comb
            F_thin = F_thin + form_F_li(residual, phi, lambda2, lambda2_ref, n)
            Xreal = F_thin.real
            Ximag = F_thin.imag
            X_tempreal = softThreshold(Xreal, soft_threshold)
            X_tempimag = softThreshold(Ximag, soft_threshold)
            F_thin = X_tempreal + 1j* X_tempimag
            
            #Thick structures
            F_comb = form_P_meas(W, F_thin+F_thick, phi, lambda2, m)
            residual = P - F_comb
            F_thick = F_thick + form_F_li(residual, phi, lambda2, lambda2_ref, n)
            
            A_re, D_re  = pywt.dwt(F_thick.real, 'db8', pywt.Modes.zero)
            A_im, D_im = pywt.dwt(F_thick.imag, 'db8', pywt.Modes.zero)
            
            A_re = softThreshold(A_re, soft_threshold)
            D_re = softThreshold(D_re, soft_threshold)
            A_im = softThreshold(A_im, soft_threshold)
            D_im = softThreshold(D_im, soft_threshold)
            
            real_Xthick = pywt.idwt(A_re, D_re, 'db8', pywt.Modes.zero) 
            imag_Xthick = pywt.idwt(A_im, D_im, 'db8', pywt.Modes.zero)
            
            F_thick = real_Xthick + 1j* imag_Xthick
    return F_thin+F_thick


def FISTA_Mix2(P, W, K, phi, lambda2, lambda2_ref, m, n, soft_threshold, iterations):
    F_thin = form_F_dirty(K, P, phi, lambda2, lambda2_ref, n)
    F_thick = np.zeros(n) + 1j*np.zeros(n)
    db8 = pywt.Wavelet('db8')
    for i in range(0, iterations):
            #Thin structures
            F_comb = form_P_meas(W, F_thin+F_thick, phi, lambda2, m)
            residual = P - F_comb
            F_thin = F_thin + form_F_li(residual, phi, lambda2, lambda2_ref, n)
            Xreal = F_thin.real
            Ximag = F_thin.imag
            X_tempreal = softThreshold(Xreal, soft_threshold)
            X_tempimag = softThreshold(Ximag, soft_threshold)
            F_thin = X_tempreal + 1j* X_tempimag
            
            #Thick structures
            F_comb = form_P_meas(W, F_thin+F_thick, phi, lambda2, m)
            residual = P - F_comb
            F_thick = F_thick + form_F_li(residual, phi, lambda2, lambda2_ref, n)
            
            A_re, B_re, C_re, D_re = pywt.wavedec(F_thick.real, db8, level=3, mode='zpd')
            A_im, B_im, C_im, D_im = pywt.wavedec(F_thick.imag, db8, level=3, mode='zpd')
            
            A_re = softThreshold(A_re, soft_threshold)
            B_re = softThreshold(B_re, soft_threshold)
            C_re = softThreshold(C_re, soft_threshold)
            D_re = softThreshold(D_re, soft_threshold)
            A_im = softThreshold(A_im, soft_threshold)
            B_im = softThreshold(B_im, soft_threshold)
            C_im = softThreshold(C_im, soft_threshold)
            D_im = softThreshold(D_im, soft_threshold)
            
            real_Xthick = pywt.waverec([A_re, B_re, C_re, D_re], 'db8', mode='zpd') 
            imag_Xthick = pywt.waverec([A_im, B_im, C_im, D_im], 'db8', mode='zpd')
            
            F_thick = real_Xthick + 1j* imag_Xthick
    return F_thin+F_thick, F_thin, F_thick