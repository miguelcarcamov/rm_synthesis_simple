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
import ctypes
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
    
def readCube(path, M, N, m, stokes):
    cube = np.zeros([m, M, N])
    for i in range(0,m):
        f_filename = path+'BAND03_CHAN0'+str(i)+'_'+stokes+'image.restored.corr_conv.fits'
        print("Reading FITS File: ", f_filename)
        i_image = fits.open(f_filename)
        data = np.squeeze(i_image[0].data)
        cube[i] = data
        
    return cube

def writeCube(cube, output):
    hdu_new = fits.PrimaryHDU(cube)
    hdu_new.writeto(output)
    
def ParallelFISTA(lock, z, chunks_start, chunks_end, j_min, j_max, F, P, W, K, phi, lambda2, lambda2_ref, m, n, soft_t, niter, N):
    for i in range(chunks_start[z], chunks_end[z]):
        for j in range(j_min, j_max):
            F[:,i,j] = FISTA_Mix_General(P[:,i,j], W, K, phi, lambda2, lambda2_ref, m, n, soft_t, niter)#Optimize P[:,i,j]
        #print("Processor: ", z, " - Chunk percentage: ", 100.0*(i/chunks_end[z]))
    
freq_text_file = sys.argv[1]
params_file = sys.argv[2]
path_Q = sys.argv[3]
path_U = sys.argv[4]
fits_file = sys.argv[5]
output_file = sys.argv[6]
nprocs = int(sys.argv[7])
niter = int(sys.argv[8])
# Get number of frequencies and values
m, freqs = getFileNFrequencies(freq_text_file)
# Calculate the scales for lambda2 and phi
lambda2 = (c/freqs)**2

w2_min = lambda2[m-1]
w2_max = lambda2[0]

lambda2_ref = (w2_max+w2_min)/2.0
delta_lambda2 = (w2_max-w2_min)/(m-1)

delta_phi = 2*np.sqrt(3)/(w2_max-w2_min)
print("delta phi: ", delta_phi)

phi_max = np.sqrt(3)/(delta_lambda2)

times = 4

phi_r = delta_phi/times

temp = np.int(np.floor(2*phi_max/phi_r))
n = int(temp-np.mod(temp,32))

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
print("Reading params file: ", params_file)
clean_params, cutoff_params = getFileData(params_file)
# Read cubes Q and U
print("Reading FITS files")
Q = readCube(path_Q, M, N, m, "Q")
U = readCube(path_U, M, N, m, "U")
# Build P, F, W and K
P = Q + 1j*U

F_base = multiprocessing.Array(ctypes.c_double, M*N*2*n)
F = np.ctypeslib.as_array(F_base.get_obj())
F = F.view(np.complex128).reshape(n, M, N)

W = np.ones(m)
K = 1.0/np.sum(W)

#FISTA arguments
soft_t = 0.00001

i_min = cutoff_params[0]
i_max = cutoff_params[1]
j_min = cutoff_params[2]
j_max = cutoff_params[3]
pixels = 0
for i in range(i_min,i_max):
    for j in range(j_min,j_max):
            pixels = pixels+1

ids = np.arange(i_min,i_max)
items = len(ids)
print("Total pixels: ", items)
print("Min ra: ", j_min)
print("Max ra: ", j_max)
print("Min dec: ", i_min)
print("Max dec: ", i_max)
iterated_pixels = 0
#Call parallel function

id_procs = np.arange(0, nprocs)
chunk_size = int(items/nprocs)
print("Chunk size: ", chunk_size)
rest_chunk = items % nprocs
print("Rest: ", rest_chunk)
chunks_start = ids[id_procs*chunk_size]
print(chunks_start)
chunks_end = ids[id_procs*chunk_size] + chunk_size
chunks_end[nprocs-1] = chunks_end[nprocs-1] + rest_chunk
print(chunks_end)

jobs = []
lock = multiprocessing.Lock()
print("Going to parallel")
for z in range(0,nprocs):
    process = multiprocessing.Process(target=ParallelFISTA, args=(lock, z, chunks_start, chunks_end, j_min, j_max, F, P, W, K, phi, lambda2, lambda2_ref, m, n, soft_t, niter, N))
    jobs.append(process)
    process.start()

# Ensure all of the processes have finished
for j in range(0, nprocs):
    jobs[j].join()
    print("Process ", jobs[j].pid, " ended - Start: ",chunks_start[j], " - End: ", chunks_end[j])
    
print("Writing solution to FITS")
writeCube(np.abs(F), output_file+"_abs.fits")
writeCube(F.real, output_file+"_real.fits")
writeCube(F.imag, output_file+"_imag.fits")
