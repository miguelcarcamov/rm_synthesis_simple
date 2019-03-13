#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:34:19 2019

@author: miguel
"""

import numpy as np

def form_P(W, F, phi, lambda2, m):
    P = np.zeros(m) + 1j*np.zeros(m)
    for i in range(0,m):
        P[i] = W[i]*np.sum(F*np.exp(2j*phi*lambda2[i]));
    return P


def form_F(K, P, W, phi, lambda2, lambda2_ref, n):
    F = np.zeros(n) + 1j*np.zeros(n)
    for i in range(0,n):
        F[i] = K*np.sum(W*P*np.exp(-2j*phi[i]*(lambda2-lambda2_ref)));
    return F

def form_R(K, W, phi, lambda2, lambda2_ref, n):
    R = np.zeros(n) + 1j*np.zeros(n)
    for i in range(0,n):
        R[i] = K*np.sum(W*np.exp(-2j*phi[i]*(lambda2-lambda2_ref)));
    return R