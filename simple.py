# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from transforms import *
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

c = 2.99792458e8
n = 500 # -> phi_space
m = 126 # -> lambda2_space

start_range = 3.6 #cm
end_range = 100.0 #cm

stop_a_range = 3.6 #cm
stop_b_range = 50.0 #cm

w_range = np.array([end_range, start_range])/100.0 #wavelength in metres

#nu_range = c/(w_range) # frequency in Hz

lambda1 = np.linspace(w_range[1], w_range[0], m)

lambda2 = lambda1*lambda1

delta_lambda2 = lambda2[1] - lambda2[0]

#lambda2_ref = np.median(lambda2)

start = -150
end = 150

ps_F = [10,2,3] #Jy
ps_1_pos = -10

ps_2_pos = [30,50]

ps_3_pos = [90,100]

phi = np.linspace(start, end, n)
F = np.zeros(n)
W = np.zeros(m)

#simulated spikes and sources indexes
ps_1_idx = (np.abs(phi-ps_1_pos)).argmin()
ps_2_idx = [(np.abs(phi-ps_2_pos[0])).argmin(), (np.abs(phi-ps_2_pos[1])).argmin()]
ps_3_idx = [(np.abs(phi-ps_3_pos[0])).argmin(), (np.abs(phi-ps_3_pos[1])).argmin()]

#weighting function
pos_start_w = (np.abs(lambda2-(stop_a_range/100.0)**2)).argmin()
pos_end_w = (np.abs(lambda2-(stop_b_range/100.0)**2)).argmin()

W[pos_start_w:pos_end_w] = 1

K = 1/np.sum(W);

lambda2_ref = np.sum(W*lambda2)/np.sum(W);
#print(lambda2, lambda2_ref)

F[ps_1_idx] = ps_F[0];
F[ps_2_idx[0]:ps_2_idx[1]] = ps_F[1];
F[ps_3_idx[0]:ps_3_idx[1]] = ps_F[2];

P = form_P(F, phi, lambda2, m)
R = form_R(K, W, phi, lambda2, lambda2_ref, n)
P_meas = form_P_meas(W, F, phi, lambda2, m)
F_meas = form_F_meas(K, P_meas, phi, lambda2, lambda2_ref, n)

f, axarr = plt.subplots(3, 2)

#Simulated F
axarr[0,0].plot(phi, np.abs(F))
axarr[0,0].set_ylim([0, None])
axarr[0,0].set_xlim([start, end])

# P
axarr[0,1].plot(lambda2, np.abs(P))
axarr[0,1].set_ylim([0, None])
axarr[0,1].set_xlim([0, 1])

# R
axarr[1,0].plot(phi, np.abs(R))
axarr[1,0].set_ylim([0, 1])
axarr[1,0].set_xlim([start, end])

# Weight function
axarr[1,1].plot(lambda2, np.abs(W) , '.')
axarr[1,1].set_ylim([0, 2])
axarr[1,1].set_xlim([0, 1])

#Measured F
axarr[2,0].plot(phi, np.abs(F_meas))
axarr[2,0].set(xlabel=r'$\phi$[rad m$^{-2}$]')
axarr[2,0].set_ylim([0, None])
axarr[2,0].set_xlim([start, end])

#Measured P
axarr[2,1].plot(lambda2, np.abs(P_meas), '+')
axarr[2,1].set(xlabel=r'$\lambda^2$ [m$^{2}$]')
axarr[2,1].set_ylim([0, None])
axarr[2,1].set_xlim([0, 1])

plt.subplots_adjust(right=1.0)
plt.subplots_adjust(top=3.0)
plt.subplots_adjust(bottom=1.0)