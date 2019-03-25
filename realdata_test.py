#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:00:57 2019

@author: miguel
"""
from astropy.io import fits
import numpy as np
import sys

RPDEG = (np.pi/180.0) #Radians per degree

def readHeader(path, fitsfile):
    f_filename = path + fitsfile
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
    
    header = [M,N, bmaj, bmin, bpa, cdelt1, cdelt2, ra, dec, crpix1, crpix2]
    
    return header
    
def getFileNFrequencies(path, filename):
    f_filename = path+filename
    try:
        with open(f_filename, "r") as f:     
            freqs = f.readlines()
            m = len(freqs)
            freqs[:] = [freq.rstrip("\n") for freq in freqs]
            freqs[:] = [float(freq) for freq in freqs]
    except IOError:
        print("Cannot open file")
        sys.exit(1)
    return m, freqs

def getFileData(path, filename):
    f_filename = path+filename
    array_par = np.loadtxt(f_filename, comments='%', usecols=1)
    dec_min = array_par[0]
    dec_max = array_par[1]
    ra_min = array_par[2]
    ra_max = array_par[3]
    gain = array_par[4]
    niter = array_par[5]
    cutoff = array_par[6]
    threshold = array_par[7]
    
    return dec_min, dec_max, ra_min, ra_max, gain, niter, cutoff, threshold
    
def readCubes(path, M, N, m, stokes):
    cube = np.zeros([m,M,N])
    for i in range(0,m):
        f_filename = path+'BAND03_CHAN'+str(m)+'_'+stokes+'image.restored.corr_conv.fits'
        i_image = fits.open(f_filename)
        data = np.squeeze(i_image[0].data)
        cube[i] = data
        
    return cube
def writeCubes():