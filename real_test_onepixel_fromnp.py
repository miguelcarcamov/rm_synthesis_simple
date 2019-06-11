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
import matplotlib.pyplot as plt
import numpy as np
import sys
from transforms import *
from FISTA_RMS import *
from rm_clean import *

RPDEG = (np.pi/180.0) #Radians per degree
c = 2.99792458e8

def readHeader(fitsfile):
    f_filename = fitsfile
    i_image = fits.open(f_filename)
    i_header = i_image[0].header
    M = i_header['NAXIS1']
    N = i_header['NAXIS2']
    #bmaj = i_header['BMAJ']
    #bmin = i_header['BMIN']
    #bpa = i_header['BPA']
    #cdelt1 = i_header['CDELT1']
    #cdelt2 = i_header['CDELT2']
    #ra = i_header['CRVAL1']
    #dec = i_header['CRVAL2']
    #crpix1 = i_header['CRPIX1']
    #crpix2 = i_header['CRPIX2']
    
    return [M,N]
    
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

def getFileData(filename):
    f_filename = filename
    array_par = np.loadtxt(f_filename, comments='%', usecols=1)
    dec_min = int(array_par[0])
    dec_max = int(array_par[1])
    ra_min = int(array_par[2])
    ra_max = int(array_par[3])

    gain = float(array_par[4])
    niter = int(array_par[5])
    cutoff = float(array_par[6])
    threshold = float(array_par[7])
    
    cutoff_params = [dec_min, dec_max, ra_min, ra_max]
    clean_params = [gain, niter, cutoff, threshold]
    
    return clean_params, cutoff_params
    
def readCube_path(path, M, N, m, stokes):
    cube = np.zeros([m, M, N])
    for i in range(0,m):
        f_filename = path+'BAND03_CHAN0'+str(i)+'_'+stokes+'image.restored.corr_conv.fits'
        print("Reading FITS File: ", f_filename)
        i_image = fits.open(f_filename)
        data = np.squeeze(i_image[0].data)
        cube[i] = data
        i_image.close()
    return cube

def readCube(file1, file2, M, N, m):
    Q = np.zeros([m, M, N])
    U = np.zeros([m, M, N])
    
    hdu1 = fits.open(file1)
    hdu2 = fits.open(file2)
    
    for i in range(m):
        Q = hdu1[0].data
        U = hdu2[0].data
    
    hdu1.close()
    hdu2.close()
    return Q,U

def writeCube(cube, output):
    hdu_new = fits.PrimaryHDU(cube)
    hdu_new.writeto(output)
    
def find_pixel(M, N, contiguous_id):
    for i in range(M):
        for j in range(N):
            if contiguous_id == N*i+j:
                return i,j
        
freq_text_file = sys.argv[1]
params_file = sys.argv[2]
path_Q = sys.argv[3]
path_U = sys.argv[4]
fits_file = sys.argv[5]
output_file = sys.argv[6]
pixel_id = int(sys.argv[7])
isCube = sys.argv[8]
noise = float(sys.argv[9])
soft_t = float(sys.argv[10])
plotOn = sys.argv[11]
structure = sys.argv[12]
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
# Get information from header
header = readHeader(fits_file)
M = header[0]
N = header[1]
#dx = -1.0*header[5]*RPDEG #to radians
#dy = header[6]*RPDEG #to radians
#ra = header[7]*RPDEG #to radians
#dec= header[8]*RPDEG #to radians
#crpix1 = header[9] #center in pixels
#crpix2 = header[10] #center in pixels
# Get cutoff and RM-CLEAN params
print("Reading params file: ", params_file)
clean_params, cutoff_params = getFileData(params_file)
# Read cubes Q and U
print("Reading FITS files")
if isCube:
    Q,U = readCube(path_Q, path_U, M, N, m)
    Q = np.flipud(Q)
    U = np.flipud(U)
else:
    Q = readCube_path(path_Q, M, N, m, "Q")
    Q = np.flipud(Q)
    U = readCube_path(path_U, M, N, m, "U")
    U = np.flipud(U)

# Build P, F, W and K
i, j = find_pixel(M, N, pixel_id)
print("Iterating in pixel (", i, ",", j, ")")
P = Q[:, i, j] + 1j*U[:, i, j]

W = np.ones(m)
K = 1.0/np.sum(W)

F_dirty = form_F_dirty(K, P, phi, lambda2, lambda2_ref, n)
#P_back = form_P_meas(W, F_dirty, phi, lambda2, lambda2_ref, m)
F_recon = Ultimate_FISTAMix(P, W, K, phi, lambda2, lambda2_ref, m, n, soft_t, noise, structure)
if plotOn:
    f, axarr = plt.subplots(1, 3)
    
    min_y = min(np.min(np.abs(P)), np.min(P.real), np.min(P.imag))
    max_y = max(np.max(np.abs(P)), np.max(P.real), np.max(P.imag))
    axarr[0].plot(lambda2, np.abs(P), 'k-')
    axarr[0].plot(lambda2, P.real, 'k-.')
    axarr[0].plot(lambda2, P.imag, 'k--')
    axarr[0].set(xlabel=r'$\lambda^2$ [m$^{2}$]')
    axarr[0].set_ylim([min_y, max_y])
    #axarr[0].set_xlim([-200, 200])
    axarr[0].set(title='P')
    
    min_y = min(np.min(np.abs(F_dirty)), np.min(F_dirty.real), np.min(F_dirty.imag))
    max_y = max(np.max(np.abs(F_dirty)), np.max(F_dirty.real), np.max(F_dirty.imag))
    
    axarr[1].plot(phi, np.abs(F_dirty), 'k-')
    axarr[1].plot(phi, F_dirty.real, 'k-.')
    axarr[1].plot(phi, F_dirty.imag, 'k--')
    axarr[1].set(xlabel=r'$\phi$[rad m$^{-2}$]')
    axarr[1].set_ylim([min_y, max_y])
    axarr[1].set_xlim([-1000, 1000])
    axarr[1].set(title='Dirty F')
    
    axarr[2].plot(phi, np.abs(F_recon), 'k-')
    axarr[2].plot(phi, F_recon.real, 'k-.')
    axarr[2].plot(phi, F_recon.imag, 'k--')
    axarr[2].set(xlabel=r'$\phi$[rad m$^{-2}$]')
    axarr[2].set_ylim([min_y, max_y])
    #axarr[2].set_xlim([-200, 200])
    axarr[2].set(title='Reconstructed F with FISTA')
    
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
