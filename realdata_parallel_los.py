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
import matplotlib.pyplot as plt
from transforms import *
from FISTA_RMS import *
from rm_clean import *
from time import time

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
    Q = np.zeros([M, N, m])
    U = np.zeros([M, N, m])
    
    hdu1 = fits.open(file1)
    hdu2 = fits.open(file2)
    
    for i in range(m):
        Q = hdu1[0].data
        U = hdu2[0].data
    
    hdu1.close()
    hdu2.close()
    return Q,U

def writeCube(cube, output, nphi, phi, dphi, M, N, header):
    header['NAXIS3'] = (nphi, 'Length of Faraday depth axis')
    header['CTYPE3'] = 'Phi'
    header['CDELT3'] = dphi
    header['CUNIT3'] = 'rad/m/m'
    header['CRVAL3'] = phi[0]
    #header['CRVAL3'] = 'Phi
    hdu_new = fits.PrimaryHDU(cube, header)
    hdu_new.writeto(output, overwrite=True)

def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError

def find_pixel(M, N, contiguous_id):
    for i in range(M):
        for j in range(N):
            if contiguous_id == N*i+j:
                return i,j
  
def ParallelFISTA(z, chunks_start, chunks_end, F, P, W, K, phi, lambda2, lambda2_ref, m, n, soft_t, noise, structure):
    for i in range(chunks_start[z], chunks_end[z]):
            F[:,i] = Ultimate_FISTAMix(P[:,i], W, K, phi, lambda2, lambda2_ref, m, n, soft_t, noise, structure)#Optimize P[:,i,j]
        #print("Processor: ", z, " - Chunk percentage: ", 100.0*(i/chunks_end[z]))

def ParallelDirty(z, chunks_start, chunks_end, j_min, j_max, F, P, K, phi, lambda2, lambda2_ref, n):
    for i in range(chunks_start[z], chunks_end[z]):
        F[:,i] = form_F_dirty(K, P[:,i], phi, lambda2, lambda2_ref, n)#Optimize P[:,i,j]
        #print("Processor: ", z, " - Chunk percentage: ", 100.0*(i/chunks_end[z]))
def test(z, chunks_start, chunks_end):
    print("I am process ",z, "I work from LOS: ", chunks_start[z], "to ", chunks_end[z])
    
        
freq_text_file = sys.argv[1]
st_los = int(sys.argv[2])
end_los = int(sys.argv[3])
path_Q = sys.argv[4]
path_U = sys.argv[5]
fits_file = sys.argv[6]
path_output = sys.argv[7]
nprocs = int(sys.argv[8])
isCube = str_to_bool(sys.argv[9])
noise = float(sys.argv[10])
soft_t = float(sys.argv[11])
structure = sys.argv[12]
plotOn = str_to_bool(sys.argv[13])
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

delta_lambda2 = np.abs(lambda2[1]-lambda2[0])

delta_phi_fwhm = 2.0*np.sqrt(3.0)/(w2_max-w2_min) #FWHM of the FPSF
delta_phi_theo = np.pi/w2_min

delta_phi = min(delta_phi_fwhm, delta_phi_theo)
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
n_los = (end_los - st_los) + 1
print("Total LOS: ", n_los)
print("Image size: ", M, "x", N)
print("Frecuencies: ", m)
print("Float Memory for Q and U: ", 2*M*N*m*4/(2**30), "GB")
print("Double Memory for Q and U: ", 2*M*N*m*8/(2**30), "GB")
#dx = -1.0*header['CDELT1']*RPDEG #to radians
#dy = header['CDELT2']*RPDEG #to radians
#ra = header['CRVAL1']*RPDEG #to radians
#dec= header['CRVAL2']*RPDEG #to radians
#crpix1 = header['CRPIX1'] #center in pixels
#crpix2 = header['CRPIX2'] #center in pixels
# Get cutoff and RM-CLEAN params
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

#LOS IDs
ids = np.arange(st_los,end_los)
    
#Find pixels
xy_pos = [find_pixel(M, N, x) for x in ids]

# Build P, F, W and K
P = np.zeros((m, n_los)) + 1j * np.zeros((m, n_los)) 
los_count = 0
for xy in xy_pos:
    P[:,los_count] = Q[:,xy[0], xy[1]] + 1j * U[:, xy[0], xy[1]]
    los_count += 1

F_base = multiprocessing.Array(ctypes.c_double, n*2*n_los)
F = np.ctypeslib.as_array(F_base.get_obj())
F = F.view(np.complex128).reshape(n, n_los)

W = np.ones(m)
K = 1.0/np.sum(W)

#Call parallel function

id_procs = np.arange(0, nprocs)
chunk_size = int(n_los/nprocs)
print("Chunk size: ", chunk_size)
rest_chunk = n_los % nprocs
print("Rest: ", rest_chunk)
chunks_start = ids[id_procs*chunk_size]
print(chunks_start)
chunks_end = ids[id_procs*chunk_size] + chunk_size
chunks_end[nprocs-1] = chunks_end[nprocs-1] + rest_chunk
print(chunks_end)

jobs = []
#lock = multiprocessing.Lock()
print("Going to parallel")
start = time()
for z in range(0,nprocs):
    process = multiprocessing.Process(target=ParallelFISTA, args=(z, chunks_start, chunks_end, F, P, W, K, phi, lambda2, lambda2_ref, m, n, soft_t, noise, structure))
    jobs.append(process)
    process.start()

# Ensure all of the processes have finished
for j in range(0, nprocs):
    jobs[j].join()
    print("Process ", jobs[j].pid, " ended - Start: ",chunks_start[j], " - End: ", chunks_end[j])

time_taken = time()-start
print ('Process took', time_taken, 'seconds')


if plotOn:
    P_selected = P[:,0]
    F_selected = F[:,0]
    f, axarr = plt.subplots(1, 2)
    
    min_y = min(np.min(np.abs(P_selected)), np.min(P_selected.real), np.min(P_selected.imag))
    max_y = max(np.max(np.abs(P_selected)), np.max(P_selected.real), np.max(P_selected.imag))
    axarr[0].plot(lambda2, np.abs(P_selected), 'k-')
    axarr[0].plot(lambda2, P_selected.real, 'k-.')
    axarr[0].plot(lambda2, P_selected.imag, 'k--')
    axarr[0].set(xlabel=r'$\lambda^2$ [m$^{2}$]')
    axarr[0].set_ylim([min_y, max_y])
    #axarr[0].set_xlim([-200, 200])
    axarr[0].set(title='P')
    
    #min_y = min(np.min(np.abs(F_selected)), np.min(F_selected.real), np.min(F_selected.imag))
    #max_y = max(np.max(np.abs(F_selected)), np.max(F_selected.real), np.max(F_selected.imag))
    
    #axarr[1].plot(phi, np.abs(F_dirty), 'k-')
    #axarr[1].plot(phi, F_dirty.real, 'k-.')
    #axarr[1].plot(phi, F_dirty.imag, 'k--')
    #axarr[1].set(xlabel=r'$\phi$[rad m$^{-2}$]')
    #axarr[1].set_ylim([min_y, max_y])
    #axarr[1].set_xlim([-1000, 1000])
    #axarr[1].set(title='Dirty F')
    
    axarr[1].plot(phi, np.abs(F_selected), 'k-')
    axarr[1].plot(phi, F_selected.real, 'k-.')
    axarr[1].plot(phi, F_selected.imag, 'k--')
    axarr[1].set(xlabel=r'$\phi$[rad m$^{-2}$]')
    #axarr[2].set_ylim([min_y, max_y])
    #axarr[2].set_xlim([-200, 200])
    axarr[1].set(title='Reconstructed F with FISTA')
    
    #min_y = min(np.min(np.abs(P_back)), np.min(P_back.real), np.min(P_back.imag))
    #max_y = max(np.max(np.abs(P_back)), np.max(P_back.real), np.max(P_back.imag))
    
    #axarr[2].plot(lambda2, np.abs(P_back), 'k-')
    #axarr[2].plot(lambda2, P_back.real, 'k-.')
    #axarr[2].plot(lambda2, P_back.imag, 'k--')
    #axarr[2].set(xlabel=r'$\lambda^2$ [m$^{2}$]')
    #axarr[2].set_ylim([min_y, max_y])
    #axarr[2].set(title='P back')
    
    plt.show(block=True)
print("Writing solution to a numpy array")
#st_los,end_los
np.save(path_output+"LOS_"+str(st_los)+"_to_"+str(end_los), F)
#writeCube(np.abs(F), output_file+"_abs.fits", n, phi, phi_r, M, N,header)
#writeCube(F.real, output_file+"_real.fits", n, phi, phi_r, M, N, header)
#writeCube(F.imag, output_file+"_imag.fits", n, phi, phi_r, M, N, header)
