import pickle
import numpy as np
from itertools import islice

import pyratbay as pb
import pyratbay.constants as pc
import pyratbay.spectrum as ps


def parse_list(array, n, fmt='.8f'):
    """Pretty parsing list of numbers into text"""
    iter_values = iter(array)
    s = []
    while True:
        values = list(islice(iter_values, n))
        if len(values) == 0:
            break
        s += ['  '.join(f'{val:{fmt}}' for val in values)]
    return s


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Generate HST filter files:

with np.load('inputs/data/WASP43b_Stevenson2017_Mendonca2018.npz') as d:
    phase = d['phase']
    wavelength = d['wavelength']
    flux = d['flux']
    uncert = d['uncert']

# Generate HST filter files for Stevenson et al. (2017) observations:
width = wavelength[1] - wavelength[0]
filters = ['filters =']
for wl in wavelength:
    if wl > 2.0:
        break
    margin = 0.1*width
    ffile = f"inputs/filters/WASP43b_hst_wfc3_g141_{wl:.3f}um.dat"
    ps.tophat(wl, width, margin, resolution=20000.0, ffile=ffile)
    filters.append(ffile)
#print('\n    '.join(filters))


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Generate JWST filter files:

pandexo_file = 'inputs/data/WASP43b_3D_synthetic_pandexo_flux_ratios.npz'
with np.load(pandexo_file) as d:
    pandexo_flux = d['noiseless_flux_ratio']
    pandexo_uncert = d['pandexo_uncert']
    pandexo_wl = d['pandexo_wl']
    obs_phase = d['phase']

nphase = len(obs_phase)
mask = pandexo_uncert[0,0] < 1.0

# Make the filters:
n_soss = np.sum(pandexo_wl<2.81)
n_nirspec = np.sum(pandexo_wl>2.81)
n_pandexo = n_soss + n_nirspec

bounds = np.zeros((n_pandexo,2))
bounds[1:, 0] = 0.5*(pandexo_wl[1:] + pandexo_wl[:-1])
bounds[:-1,1] = 0.5*(pandexo_wl[1:] + pandexo_wl[:-1])
bounds[0,0] = 2*pandexo_wl[0] -  bounds[0,1]
bounds[n_soss-1,1] = 2*pandexo_wl[n_soss-1] - bounds[n_soss-1,0]
bounds[n_soss,0] = 2*pandexo_wl[n_soss] - bounds[n_soss,1]
bounds[n_pandexo-1,1] = 2*pandexo_wl[n_pandexo-1] - bounds[n_pandexo-1,0]

filters = ''
for i in range(n_pandexo):
    wl = 0.5 * (bounds[i,1] + bounds[i,0])
    width = bounds[i,1] - bounds[i,0]
    margin = 0.1*width
    if wl < 2.81:
        inst = 'NIRISS_SOSS'
    else:
        inst = 'NIRSpec_G395H'
    filter_file = f"inputs/filters/JWST_{inst}_{pandexo_wl[i]:5.3f}um.dat"
    ps.tophat(wl, width, margin, resolution=20000.0, ffile=filter_file)
    if mask[i]:
        filters += f"    {filter_file}\n"
# print(filters)
