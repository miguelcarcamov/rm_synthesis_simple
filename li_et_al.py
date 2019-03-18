import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from transforms import *
from FISTA_RMS import *
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#Scales and constants

c = 2.99792458e8

m = 126 # -> lambda2_space

w_min = 3.6 #lambda start cm
w_max = 50.0 #lambda end cm

w2_min = (w_min/100.0)*(w_min/100.0)
w2_max = (w_max/100.0)*(w_max/100.0)

#delta_lambda = lambda2[1]-lambda2[0]
lambda2_ref = (w2_max+w2_min)/2
delta_lambda2 = (w2_max-w2_min)/(m-1)

lambda2 = np.arange(w2_min, w2_max, delta_lambda2)

delta_phi = 2*np.sqrt(3)/(w2_max-w2_min)

phi_max = np.sqrt(3)/(delta_lambda2)

times = 4

phi_r = delta_phi/times

temp = np.int(np.floor(2*phi_max/phi_r))
n = temp-np.mod(temp,32)

phi_r = 2*phi_max/n;
phi = phi_r*np.arange(-(n/2),(n/2), 1)

#Simulated Sources
sources_F = [10-1j*4, -7+1j*5, 9-1j*7, -4+1j*3] #Jy
pos_F = [-10, -17, 40, 88]

#simulated spikes and sources indexes
F = np.zeros(n) + 1j*np.zeros(n)
W = np.ones(m)
K = 1/np.sum(W);

for i in range(0,len(sources_F)):
    ps_idx = (np.abs(phi-pos_F[i])).argmin()
    F[ps_idx] = sources_F[i]

P = form_P(F, phi, lambda2, m)
R = form_R(K, W, phi, lambda2, lambda2_ref, n)
P_meas = form_P_meas(W, F, phi, lambda2, m)
F_dirty = form_F_meas(K, P_meas, phi, lambda2, lambda2_ref, n)

soft_threshold = 1.0
iterations = 600

F_recon_thin = FISTA_Thin(P_meas, W, phi, lambda2, lambda2_ref, m, n, soft_threshold, 600)
F_recon_thick = FISTA_Thick(P_meas, W, phi, lambda2, lambda2_ref, m, n, soft_threshold, 600)
F_recon_mix = FISTA_Mix(P_meas, W, phi, lambda2, lambda2_ref, m, n, soft_threshold, 600)

f, axarr = plt.subplots(2, 3)

axarr[0,0].plot(phi, np.abs(F), 'k-')
axarr[0,0].plot(phi, F.real, 'k-.')
axarr[0,0].plot(phi, F.imag, 'k--')
axarr[0,0].set_ylim([None, None])
axarr[0,0].set_xlim([-200, 200])
axarr[0,0].set(title='Original')

axarr[0,1].plot(phi, np.abs(F_dirty), 'k-')
axarr[0,1].plot(phi, F_dirty.real, 'k-.')
axarr[0,1].plot(phi, F_dirty.imag, 'k--')
axarr[0,1].set_ylim([None, None])
axarr[0,1].set_xlim([-200, 200])
axarr[0,1].set(title='Dirty curve')

axarr[1,0].plot(phi, np.abs(F_recon_thin), 'k-')
axarr[1,0].plot(phi, F_recon_thin.real, 'k-.')
axarr[1,0].plot(phi, F_recon_thin.imag, 'k--')
axarr[1,0].set_ylim([None, None])
axarr[1,0].set_xlim([-200, 200])
axarr[1,0].set(title='CS-RM-Thin')

axarr[1,1].plot(phi, np.abs(F_recon_thick), 'k-')
axarr[1,1].plot(phi, F_recon_thick.real, 'k-.')
axarr[1,1].plot(phi, F_recon_thick.imag, 'k--')
axarr[1,1].set_ylim([None, None])
axarr[1,1].set_xlim([-200, 200])
axarr[1,1].set(title='CS-RM-Thick')


axarr[1,2].plot(phi, np.abs(F_recon_mix), 'k-')
axarr[1,2].plot(phi, F_recon_mix.real, 'k-.')
axarr[1,2].plot(phi, F_recon_mix.imag, 'k--')
axarr[1,2].set_ylim([None, None])
axarr[1,2].set_xlim([-200, 200])
axarr[1,2].set(title='CS-RM-Mix')

plt.subplots_adjust(right=3.0)
plt.subplots_adjust(top=2.0)
plt.subplots_adjust(bottom=0.5)