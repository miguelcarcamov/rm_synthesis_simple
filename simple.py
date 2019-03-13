# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from transforms import *

c = 2.99792458e8
n = 200 # -> phi_space
m = 300 # -> lambda2_space

start_range = 3.6 #cm
end_range = 95.0 #cm

stop_a_range = 3.6 #cm
stop_b_range = 50.0 #cm

w_range = np.array([end_range, start_range])/100.0 #wavelength in metres

nu_range = c/(w_range) # frequency in Hz

lambda1 = np.linspace(w_range[1], w_range[0], m)

lambda2 = lambda1*lambda1

#print(lambda2)

lambda2_ref = np.median(lambda2)

start = -150
end = 150

ps_F = [10,2,3] #Jy
ps_1_pos = -10

ps_2_pos = [30,50]

ps_3_pos = [90,100]

phi = np.linspace(start, end, n)
F = np.zeros(n)
W = np.zeros(m)

#simulated spikes and sources
ps_1_idx = (np.abs(phi-ps_1_pos)).argmin()
ps_2_idx = [(np.abs(phi-ps_2_pos[0])).argmin(), (np.abs(phi-ps_2_pos[1])).argmin()]
ps_3_idx = [(np.abs(phi-ps_3_pos[0])).argmin(), (np.abs(phi-ps_3_pos[1])).argmin()]

#weighting function
pos_start_w = (np.abs(lambda2-(stop_a_range/100.0)**2)).argmin()
pos_end_w = (np.abs(lambda2-(stop_b_range/100.0)**2)).argmin()

W[pos_start_w:pos_end_w] = 1

K = 1/np.sum(W);

F[ps_1_idx] = ps_F[0];
F[ps_2_idx[0]:ps_2_idx[1]] = ps_F[1];
F[ps_3_idx[0]:ps_3_idx[1]] = ps_F[2];

P = form_P(W, F, phi, lambda2, m)
R = form_R(K, W, phi, lambda2, lambda2_ref, n)
F_meas = form_F(K, P, W, phi, lambda2, lambda2_ref, n)

f, axarr = plt.subplots(3, 2)
axarr[0,0].plot(phi, abs(F))
axarr[0,1].plot(lambda2, abs(P))
axarr[1,0].plot(phi, abs(R))
axarr[1,1].plot(lambda2, abs(W))
axarr[2,0].plot(phi, abs(F_meas))
axarr[2,1].plot(lambda2, abs(P))
#plt.plot(phi, abs(F_meas))
#plt.plot(phi, true_emission);
#plt.show()
#print(idx)

#print(phi[idx])