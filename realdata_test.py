#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:00:57 2019

@author: miguel
"""
from astropy.io import fits
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
    bmaj = i_header['BMAJ']
    bmin = i_header['BMIN']
    bpa = i_header['BPA']
    cdelt1 = i_header['CDELT1']
    cdelt2 = i_header['CDELT2']
    ra = i_header['CRVAL1']
    dec = i_header['CRVAL2']
    crpix1 = i_header['CRPIX1']
    crpix2 = i_header['CRPIX2']
    
    return [M,N, bmaj, bmin, bpa, cdelt1, cdelt2, ra, dec, crpix1, crpix2]
    
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
    dec_min = array_par[0]
    dec_max = array_par[1]
    ra_min = array_par[2]
    ra_max = array_par[3]
    gain = array_par[4]
    niter = array_par[5]
    cutoff = array_par[6]
    threshold = array_par[7]
    
    cutoff_params = [dec_min, dec_max, ra_min, ra_max]
    clean_params = [gain, niter, cutoff, threshold]
    
    return clean_params, cutoff_params
    
def readCube(path, M, N, m, stokes):
    cube = np.zeros([m, M, N])
    for i in range(0,m):
        f_filename = path+'BAND03_CHAN'+str(m)+'_'+stokes+'image.restored.corr_conv.fits'
        i_image = fits.open(f_filename)
        data = np.squeeze(i_image[0].data)
        cube[i] = data
        
    return cube

def writeCube(cube, output):
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    for img in cube:
        hdul.append(fits.ImageHDU(data=img))
    hdul.writeto(output)
    
    
freq_text_file = sys.argv[1]
params_file = sys.argv[2]
path_Q = sys.argv[3]
path_U = sys.argv[4]
fits_file = sys.argv[5]
output_file = sys.argv[6]
# Get number of frequencies and values
m, freqs = getFileNFrequencies(freq_text_file)
# Calculate the scales for lambda2 and phi
lambda2 = (c/freqs)**2

w2_min = lambda2[m-1]
w1_max = lambda2[0]

lambda2_ref = (w2_max+w2_min)/2.0
delta_lambda2 = (w2_max-w2_min)/(m-1)

delta_phi = 2*np.sqrt(3)/(w2_max-w2_min)

phi_max = np.sqrt(3)/(delta_lambda2)

times = 4

phi_r = delta_phi/times

temp = np.int(np.floor(2*phi_max/phi_r))
n = temp-np.mod(temp,32)

phi_r = 2*phi_max/n;
phi = phi_r*np.arange(-(n/2),(n/2), 1)

# Get information from header
header = readHeader(fits_file)
M = header[0]
N = header[1]
dx = -1.0*header[5]*RPDEG #to radians
dy = header[6]*RPDEG #to radians
ra = header[7]*RPDEG #to radians
dec= header[8]*RPDEG #to radians
crpix1 = header[9] #center in pixels
crpix2 = header[10] #center in pixels
# Get cutoff and RM-CLEAN params
cutoff_params, clean_params = getFileData(params_file)
# Read cubes Q and U
Q = readCube(path_Q, M, N, m, "Q")
U = readCube(path_U, M, N, m, "U")
# Build P, F, W and K
P = Q + 1j*U
F = np.zeros([m, M, N])+1j*np.zeros([m, M, N])
W = np.ones([m, M, N])
K = np.zeros([M,N])
for i in range(0,M):
    for j in range(0,N):
        K[i,j] = 1.0/np.sum(W[:,i,j])

#FISTA arguments
soft_t = 0.05
niter = 2000
for i in range(0,M):
    for j in range(0,N):
        if i<=params_file[1] and i>params_file[0] and j<=params_file[3] and j>params_file[2]:
            F[:,i,j] = FISTA_Mix(P[:,i,j], W[:,i,j], K[i,j], phi, lambda2, lambda2_ref, m, n, soft_t, niter)#Optimize P[:,i,j]
        else:
            F[:,i,j] = 0
writeCube(F, output_file)