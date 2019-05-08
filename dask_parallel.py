#!/usr/bin/env python3
#!/usr/bin/env JOBLIB_TEMP_FOLDER=/tmp
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:00:57 2019

@author: miguel
"""
import multiprocessing
from astropy.io import fits
import numpy as np
import sys
import ctypes
from transforms import *
from FISTA_RMS import *
from rm_clean import *
from time import time
from dask import delayed

RPDEG = (np.pi/180.0) #Radians per degree
c = 2.99792458e8

def readHeader(fitsfile):
    f_filename = fitsfile
    i_image = fits.open(f_filename)
    i_header = i_image[0].header
    
    i_image.close()
    return i_header
    
def getFileNFrequencies(filename):
    f_filename = filename
    try:
        with open(f_filename, "r") as f:     
            freqs = f.readlines()
            m = len(freqs)
            freqs[:] = [freq.rstrip("\n") for freq in freqs]
            freqs[:] = [float(freq) for freq in freqs]
            f.close()
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
    cube = np.zeros([M, N, m])
    for i in range(0,m):
        f_filename = path+'BAND03_CHAN0'+str(i)+'_'+stokes+'image.restored.corr_conv.fits'
        print("Reading FITS File: ", f_filename)
        i_image = fits.open(f_filename)
        data = np.squeeze(i_image[0].data)
        cube[:,:,i] = data
        i_image.close()
    return cube

def readCube(file1, file2, M, N, m):
    Q = np.zeros([M, N, m])
    U = np.zeros([M, N, m])
    
    hdu1 = fits.open(file1)
    hdu2 = fits.open(file2)
    
    for i in range(m):
        Q[:, :, i] = hdu1[0].data[i,:,:]
        U[:, :, i] = hdu2[0].data[i,:,:]
    
    hdu1.close()
    hdu2.close()
    return Q,U

def writeCube(cube, output, nphi, phi, dphi, header):
    header['NAXIS3'] = (nphi, 'Length of Faraday depth axis')
    header['CTYPE3'] = 'Phi'
    header['CDELT3'] = dphi
    header['CUNIT3'] = 'rad/m/m'
    header['CRVAL3'] = phi[0]
    #header['CRVAL3'] = 'Phi
    #cube = np.reshape(cube, (nphi, M, N))
    hdu_new = fits.PrimaryHDU(cube, header)
    hdu_new.writeto(output, overwrite=True)
        
freq_text_file = sys.argv[1]
params_file = sys.argv[2]
path_Q = sys.argv[3]
path_U = sys.argv[4]
fits_file = sys.argv[5]
output_file = sys.argv[6]
nprocs = int(sys.argv[7])
niter = int(sys.argv[8])
isCube = sys.argv[9]
if nprocs < 1 or nprocs > multiprocessing.cpu_count():
    print("You cannot use more than", multiprocessing.cpu_count(), "processors and less than 1")
    sys.exit(-1)
# Get number of frequencies and values
m, freqs = getFileNFrequencies(freq_text_file)
# Calculate the scales for lambda2 and phi
lambda2 = (c/freqs)**2
lambda2 = np.flipud(lambda2)

w2_min = lambda2[0]
w2_max = lambda2[m-1]
print("l2 min: ", w2_min)
print("l2 max: ", w2_max)
lambda2_ref = (w2_max+w2_min)/2.0
print("Lambda2 ref: ", lambda2_ref)
delta_lambda2 = (w2_max-w2_min)/(m-1)

delta_phi = 2*np.sqrt(3)/(w2_max-w2_min)
print("delta phi: ", delta_phi)

phi_max = np.sqrt(3)/(delta_lambda2)

times = 4

phi_r = delta_phi/times

temp = np.int(np.floor(2*phi_max/phi_r))
n = int(temp-np.mod(temp,32))

phi_r = 2*phi_max/n;
print("Phi_r: ", phi_r)
phi = phi_r*np.arange(-(n/2),(n/2), 1)
# Get information from header
header = readHeader(fits_file)
M = header['NAXIS1']
N = header['NAXIS2']
print("Image size: ", M, "x", N)
print("Frecuencies: ", m)
print("Float Memory for Q and U: ", 2*M*N*m*4/(2**30), "GB")
print("Double Memory for Q and U: ", 2*M*N*m*8/(2**30), "GB")
# Get cutoff and RM-CLEAN params
print("Reading params file: ", params_file)
clean_params, cutoff_params = getFileData(params_file)
# Read cubes Q and U
print("Reading FITS files")
if isCube:
    Q,U = readCube(path_Q, path_U, M, N, m)
else:
    Q = readCube(path_Q, M, N, m, "Q")
    Q = np.flipud(Q)
    U = readCube(path_U, M, N, m, "U")
    U = np.flipud(U)
# Build P, F, W and K
P = Q + 1j*U
W = np.ones(m)
K = 1.0/np.sum(W)

#FISTA arguments
soft_t = 0.00001

i_min = cutoff_params[0]
i_max = cutoff_params[1]
j_min = cutoff_params[2]
j_max = cutoff_params[3]

start = time()
results = []
for i in range(imin, imax):
    for j in range(jmin, jmax):
        x = delayed(FISTA_Thin)(P[i,j], W, K, phi, lambda2, lambda2_ref, m, n, soft_t, niter)
        results.append(x)

F = results.compute(scheduler='processes', num_workers=nprocs)

time_taken = time()-start
print ('Process took', time_taken, 'seconds')
print("Writing solution to FITS")
F = np.reshape(F, (n, M, N))

writeCube(np.abs(F), output_file+"_abs.fits", n, phi, phi_r, M, N,header)
writeCube(F.real, output_file+"_real.fits", n, phi, phi_r, M, N, header)
writeCube(F.imag, output_file+"_imag.fits", n, phi, phi_r, M, N, header)