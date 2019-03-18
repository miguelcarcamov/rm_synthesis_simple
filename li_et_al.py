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
#F_meas = form_F_meas(K, P_meas, phi, lambda2, lambda2_ref, n)

plt.plot(lambda2, abs(P_meas), 'k-')
plt.plot(lambda2, P_meas.real, 'k-.')
plt.plot(lambda2, P_meas.imag , 'k--')
#plt.xlim([-200, 200])

#soft_threshold = 1.0
#stop_threshold = 0.001
#iterations = 0

#Fthin = F_meas
#Fthick = np.zeros(n) + 1j*np.zeros(n)

#for i in range(0, iterations):
    #Faraday thin sources
    #residual = P_meas - form_P_meas(W, Fthin, phi, lambda2, m) - form_P_meas(W, Fthick, phi, lambda2, m)
    #d = form_F_meas_li(K, residual, phi, lambda2, lambda2_ref, n)
    #Fthin = Fthin + d
    #re_pos = np.where(Fthin.real < soft_threshold)
    #im_pos = np.where(Fthin.imag < soft_threshold)
    #Fthin[re_pos].real = 0.0
    #Fthin[im_pos].imag = 0.0
    #Faraday thick sources
    #Fthick = Fthick + d
    #coeffA_re, coeffD_re  = pywt.dwt(Fthick.real, 'db8', pywt.Modes.zero)
    #coeffA_im, coeffD_im = pywt.dwt(Fthick.imag, 'db8', pywt.Modes.zero)
    #Soft thresholding in wavelet space real part
    #reA_pos = np.where(coeffA_re < soft_threshold)
    #reD_pos = np.where(coeffD_re < soft_threshold)
    #Applying the thresholding real part
    #coeffA_re[reA_pos] = 0.0
    #coeffD_re[reD_pos] = 0.0
    #Soft thresholding in wavelet space imaginary part
    #imA_pos = np.where(coeffA_im < soft_threshold)
    #imD_pos = np.where(coeffD_im < soft_threshold)
    #Applying the thresholding imaginary part
    #coeffA_im[imA_pos] = 0.0
    #coeffD_im[imD_pos] = 0.0
    #Update Fthick
    #inv_real = pywt.idwt(coeffA_re, coeffD_re, 'db8', pywt.Modes.zero)
    #inv_imag = pywt.idwt(coeffA_im, coeffD_im, 'db8', pywt.Modes.zero)
    #Fthick.real = inv_real[0:len(inv_real)-1]
    #Fthick.imag = inv_imag[0:len(inv_real)-1]
    

#F_recon = Fthin + Fthick
#plt.plot(phi, abs(F_recon), 'k-')
#plt.plot(phi, F_recon.real, 'k-.')
#plt.plot(phi, F_recon.imag , 'k--')
#plt.xlim([-200, 200])
