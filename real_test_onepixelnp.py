#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:24:18 2019

@author: miguel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:00:57 2019

@author: miguel
"""
import multiprocessing
from astropy.io import fits
import numpy as np
import sys
from transforms import *
from FISTA_RMS import *
from rm_clean import *

RPDEG = (np.pi/180.0) #Radians per degree
c = 2.99792458e8


def getFileNFrequencies(filename):
    f_filename = filename
    try:
        with open(f_filename, "r") as f:
            freqs = f.readlines()
            m = len(freqs)
            freqs[:] = [freq.rstrip("\n") for freq in freqs]
            freqs[:] = [float(freq) for freq in freqs]
    except IOError:
        print("Cannot open file")
        sys.exit(1)
    freqs = np.array(freqs)
    return m, freqs

n_id = int(sys.argv[1])
freq_text_file = sys.argv[2]
file_name = sys.argv[3]
noise = float(sys.argv[4])
soft_t = float(sys.argv[5])
plotOn = sys.argv[6]
structure = sys.argv[7]
# Get number of frequencies and values
m, freqs = getFileNFrequencies(freq_text_file)
# Calculate the scales for lambda2 and phi
lambda2 = (c/freqs)**2
lambda2 = lambda2[::-1]

w2_min = lambda2[0]
w2_max = lambda2[m-1]

print("l2 min: ", w2_min)
print("l2 max: ", w2_max)

lambda2_ref = (w2_max+w2_min)/2.0

print("Lambda2 ref: ", lambda2_ref)

delta_lambda2 = np.abs(lambda2[1]-lambda2[0]) #delta lambda2

delta_phi_fwhm = 2.0*np.sqrt(3.0)/(w2_max-w2_min) #FWHM of the FPSF
delta_phi_theo = np.pi/w2_min

delta_phi = min(delta_phi_fwhm, delta_phi_theo)
print("delta phi: ", delta_phi)

phi_max = np.sqrt(3)/(delta_lambda2) #Maximum observable phi / Maximal Faraday depth
print(phi_max)

times = 4 # In general the resolution is one fourth of the FWHM of FPSF
phi_r = delta_phi/times

temp = int(np.floor(2.0*phi_max/phi_r)) #Since phi_r = 2*phi_max/n, we can get n
n = temp-np.mod(temp,32)
print("Nphi: ", n)

phi_r = 2*phi_max/n; # Then we get our real resolution
print("Phi_r: ", phi_r)

phi = phi_r*np.arange(-(n/2),(n/2), 1) # We make our phi axis

np_array = np.load(file_name)

Q = np_array[n_id,:,0]
U = np_array[n_id,:,1]

# Build P, F, W and K
P = Q + 1j*U

P = P[::-1]

W = np.ones(m)
K = 1.0/np.sum(W)

F_dirty = form_F_dirty(K, W, P, phi, lambda2, lambda2_ref, n)
#P_back = form_P_meas(W, F_dirty, phi, lambda2, lambda2_ref, m)
F_recon = Ultimate_FISTAMix(P, W, K, phi, lambda2, lambda2_ref, m, n, soft_t, noise, structure, 1e-12)
if plotOn:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.figure(1)
    #min_y = min(np.min(np.abs(P)), np.min(P.real), np.min(P.imag))
    #max_y = max(np.max(np.abs(P)), np.max(P.real), np.max(P.imag))
    #axarr[0].plot(lambda2, np.abs(P), 'k-')
    plt.plot(lambda2, P.real, 'k-')
    plt.plot(lambda2, P.imag, 'k-.')
    plt.xlabel(r'$\lambda^2$ [m$^{2}$]')
    plt.ylabel(r"Flux (Jy)")
    #axarr[0].set_ylim([min_y, max_y])
    #axarr[0].set_xlim([-200, 200])
    #title='P')

    #min_y = min(np.min(np.abs(F_dirty)), np.min(F_dirty.real), np.min(F_dirty.imag))
    #max_y = max(np.max(np.abs(F_dirty)), np.max(F_dirty.real), np.max(F_dirty.imag))

    #axarr[1].plot(phi, np.abs(F_dirty), 'k-')
    plt.figure(2)
    amp_F_dirty = np.abs(F_dirty)
    plt.plot(phi, amp_F_dirty, 'r-')
    plt.plot(phi, F_dirty.real, 'k-')
    plt.plot(phi, F_dirty.imag, 'k-.')
    plt.xlabel(r'$\phi$[rad m$^{-2}$]')
    plt.ylabel(r'Jy m$^2$ rad$^{-1}$')
    plt.legend((r"Amplitude", r"Real part", r"Imaginary part"), loc='upper right')
    plt.xlim([-500, 500])
    print("Dirty amplitude peak:", np.max(amp_F_dirty))
    pospeak = np.argmax(amp_F_dirty)
    print("Position peak at: ", pospeak, "or: ", phi[pospeak])
    #axarr[1].set_ylim([min_y, max_y])
    #title='Dirty F')

    plt.figure(3)
    #axarr[2].plot(phi, np.abs(F_recon), 'k-')
    amp_F_recon = np.abs(F_recon)
    plt.plot(phi, amp_F_recon, 'r-')
    plt.plot(phi, F_recon.real, 'k-')
    plt.plot(phi, F_recon.imag, 'k-.')
    plt.xlabel(r'$\phi$[rad m$^{-2}$]')
    plt.ylabel(r'Jy m$^2$ rad$^{-1}$')
    plt.legend((r"Amplitude", r"Real part", r"Imaginary part"), loc='upper right')
    plt.xlim([-500, 500])
    print("FISTA recon magnitude peak:", np.max(amp_F_recon))
    pospeak = np.argmax(amp_F_recon)
    print("Position peak at: ", pospeak, "or: ", phi[pospeak])
    #axarr[2].set_ylim([min_y, max_y])
    #plt.title('Reconstructed F with FISTA')

    #min_y = min(np.min(np.abs(P_back)), np.min(P_back.real), np.min(P_back.imag))
    #max_y = max(np.max(np.abs(P_back)), np.max(P_back.real), np.max(P_back.imag))

    #axarr[2].plot(lambda2, np.abs(P_back), 'k-')
    #axarr[2].plot(lambda2, P_back.real, 'k-.')
    #axarr[2].plot(lambda2, P_back.imag, 'k--')
    #axarr[2].set(xlabel=r'$\lambda^2$ [m$^{2}$]')
    #axarr[2].set_ylim([min_y, max_y])
    #axarr[2].set(title='P back')

    plt.show(block=True)
np.save(output_file, F_recon)
