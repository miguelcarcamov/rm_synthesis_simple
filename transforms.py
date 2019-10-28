#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:34:19 2019

@author: miguel
"""

import numpy as np

def form_F_dirty(K, W, P_meas, phi, lambda2, lambda2_ref, n):
    F = np.zeros(n) + 1j*np.zeros(n)
    for i in range(0,n):
        F[i] = np.sum(W*P_meas*np.exp(-2j*phi[i]*(lambda2-lambda2_ref)))
    return K*F

def form_P_meas(W, F, phi, lambda2, lambda2_ref, m):
    P = np.zeros(m) + 1j*np.zeros(m)
    for i in range(0,m):
        P[i] = np.sum(F*np.exp(2j*phi*(lambda2[i]-lambda2_ref)))
    return W*P

def form_F_li(K, P_meas, phi, lambda2, lambda2_ref, n):
    F = np.zeros(n) + 1j*np.zeros(n)
    for i in range(0,n):
        F[i] = np.sum(P_meas*np.exp(-2j*phi[i]*(lambda2-lambda2_ref)))
    return F/n

def form_P(F, phi, lambda2, lambda2_ref, m):
    P = np.zeros(m) + 1j*np.zeros(m)
    for i in range(0,m):
        P[i] = 1.0 * np.sum(F*np.exp(2j*phi*(lambda2[i]-lambda2_ref)))
    return P

def form_R(K, W, phi, lambda2, lambda2_ref, n):
    R = np.zeros(n) + 1j*np.zeros(n)
    for i in range(0,n):
        R[i] = np.sum(W*np.exp(-2j*phi[i]*(lambda2-lambda2_ref)))
    return K*R
