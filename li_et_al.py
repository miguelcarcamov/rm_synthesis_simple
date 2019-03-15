import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from transforms import *
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#Scales and constants

c = 2.99792458e8

m = 126 # -> lambda2_space

start_range = 3.6 #lambda start cm
end_range = 50.0 #lambda end cm

w_a_range = 3.6 #weights start cm
w_b_range = 50.0 #weights end cm

w_range = np.array([end_range, start_range])/100.0 #wavelength ranges in metres

lambda1 = np.linspace(w_range[1], w_range[0], m)

lambda2 = lambda1*lambda1

#lambda2_ref = np.median(lambda2)
lambda2_ref = 0

delta_lambda2 = lambda2[1] - lambda2[0]

delta_total_lambda2 = lambda2[len(lambda2)-1] - lambda2[0]

phi_max = np.sqrt(3)/delta_lambda2

#phi_max = np.pi/lambda2[0]

delta_phi = 2*np.sqrt(3)/delta_total_lambda2

times = 4

phi_r = delta_phi/times

n = np.int(np.floor(2*phi_max/phi_r))

phi = np.linspace((-n/2)+1, n/2, n)

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
F_meas = form_F_meas(K, P_meas, phi, lambda2, lambda2_ref, n)

plt.plot(phi, abs(F_meas), 'k-')
plt.plot(phi, F_meas.real, 'k-.')
plt.plot(phi, F_meas.imag , 'k--')
plt.xlim([-200, 200])





