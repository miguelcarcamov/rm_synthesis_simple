#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:34:19 2019

@author: miguel
"""

import numpy as np

def form_P_meas(W, F, phi, lambda2, m):
    P = np.zeros(m) + 1j*np.zeros(m)
    for i in range(0,m):
        P[i] = W[i]*np.sum(F*np.exp(2j*phi*lambda2[i]))
    return P

def form_F_meas(K, P_meas, phi, lambda2, lambda2_ref, n):
    F = np.zeros(n) + 1j*np.zeros(n)
    for i in range(0,n):
        F[i] = K*np.sum(P_meas*np.exp(-2j*phi[i]*(lambda2-lambda2_ref)))
    return F

def form_F_meas_li(K, P_meas, phi, lambda2, lambda2_ref, n):
    F = np.zeros(n) + 1j*np.zeros(n)
    for i in range(0,n):
        F[i] = K*np.sum(P_meas*np.exp(-2j*phi[i]*lambda2))
    return F

def form_P(F, phi, lambda2, m):
    P = np.zeros(m) + 1j*np.zeros(m)
    for i in range(0,m):
        P[i] = 1.0*np.sum(F*np.exp(2j*phi*lambda2[i]))
    return P

def form_R(K, W, phi, lambda2, lambda2_ref, n):
    R = np.zeros(n) + 1j*np.zeros(n)
    for i in range(0,n):
        R[i] = K*np.sum(W*np.exp(-2j*phi[i]*(lambda2-lambda2_ref)))
    return R
