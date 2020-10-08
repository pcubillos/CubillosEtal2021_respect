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
# HST observations:
hst_obs = [
    'inputs/data/WASP43b_day_disk-int.csv',
    'inputs/data/WASP43b_day_long-res.csv',
    'inputs/data/WASP43b_east_disk-int.csv',
    'inputs/data/WASP43b_east_long-res.csv',
    'inputs/data/WASP43b_west_disk-int.csv',
    'inputs/data/WASP43b_west_long-res.csv',
    ]


for obs in hst_obs:
    wls, widths, flux, error = [], [], [], []
    for line in open(obs, 'r'):
        if line.startswith('#') or line.strip() == '':
            continue
        if line.startswith('@'):
            instrument = line[1:].strip()
            continue
        info = line.strip().split(',')
        wls.append(info[0])
        widths.append(info[1])
        flux.append(info[2])
        error.append(info[3])

    flux = np.array(flux, np.double)
    error = np.array(error, np.double)
    data = 'data =\n' + '\n'.join(
        parse_list(flux/pc.ppt, 7, '8.3f'))
    uncert = 'uncert =\n' + '\n'.join(
        parse_list(error/pc.ppt, 7, '8.3f'))
    print(obs)
    print(f'dunits = ppt\n{data}\n{uncert}')

    wls = np.array(wls, np.double)
    widths = np.array(widths, np.double)

filters = ['filters =']
for wl, width in zip(wls, widths):
    if wl > 2.0:
        break
    margin = 0.1*width
    ffile = f"inputs/filters/WASP43b_hst_wfc3_g141_{wl:.3f}um.dat"
    ps.tophat(wl, width, margin, resolution=20000.0, ffile=ffile)
    filters.append(ffile)
print('\n    '.join(filters))


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Generate JWST filter files and data/uncert texts for config files:


# Read disk-integrated and spatially-resolved data:
with np.load(f'run_simulation/WASP43b_3D_synthetic_pandexo_flux_ratios.npz') as d:
    pflux = d['noiseless_flux_ratio']
    puncert = d['pandexo_uncert']
    pwl = d['pandexo_wl']
    obs_phase = d['phase']
nphase = len(obs_phase)
mask = puncert[0] < 1.0


# Make the filters:
nsoss = np.sum(pwl<2.81)
nnirspec = np.sum(pwl>2.81)
npandexo = nsoss + nnirspec

bounds = np.zeros((npandexo,2))
bounds[1:, 0] = 0.5*(pwl[1:] + pwl[:-1])
bounds[:-1,1] = 0.5*(pwl[1:] + pwl[:-1])
bounds[0,0] = 2*pwl[0] -  bounds[0,1]
bounds[nsoss-1,1] = 2*pwl[nsoss-1] - bounds[nsoss-1,0]
bounds[nsoss,0] = 2*pwl[nsoss] - bounds[nsoss,1]
bounds[npandexo-1,1] = 2*pwl[npandexo-1] - bounds[npandexo-1,0]

filters = ''
for i in range(npandexo):
    wl = 0.5 * (bounds[i,1] + bounds[i,0])
    width = bounds[i,1] - bounds[i,0]
    margin = 0.1*width
    if wl < 2.81:
        ffile = f"../inputs/filters/JWST_NIRISS_SOSS_{pwl[i]:5.3f}um.dat"
    else:
        ffile = f"../inputs/filters/JWST_NIRSpec_G395H_{pwl[i]:5.3f}um.dat"
    ps.tophat(wl, width, margin, resolution=20000.0, ffile=ffile)
    if mask[i]:
        filters += f"    {ffile}\n"


# Pretty-print the data:
# Integrated:
sim_file = open(f'retrieval_WASP43b_integrated_jwst_phase.txt', 'w')
for i in range(nphase):
    data   = '    ' + '\n    '.join(
        parse_list(pflux[i][mask]/pc.ppm, 5, '11.6f'))
    uncert = '    ' + '\n    '.join(
        parse_list(puncert[i][mask]/pc.ppm, 5, '11.6f'))

    sim_file.write(f'\nphase = {obs_phase[i]}:\n')
    sim_file.write(f'dunits = ppm\ndata =\n{data}')
    sim_file.write(f'\nuncert =\n{uncert}\n')
sim_file.write(f'\nfilters =\n{filters}')
sim_file.close()


# Resolved:
with open('../inputs/data/longspectra_noiseless.pkl', 'rb') as f:
    d = pickle.load(f)
resolved_flux = np.roll(d['long_flux_ratio'], 1, axis=0)
resolved_unc = np.roll(d['long_flux_uncert'], 1, axis=0)
obs_phase = d['phase']
nphase = len(obs_phase)

sim_file = open(f'retrieval_WASP43b_resolved_jwst_phase.txt', 'w')
for i in range(nphase):
    data   = '    ' + '\n    '.join(
        parse_list(resolved_flux[i][mask]/pc.ppm, 5, '11.6f'))
    uncert = '    ' + '\n    '.join(
        parse_list(resolved_unc[i][mask]/pc.ppm, 5, '11.6f'))

    sim_file.write(f'\nphase = {obs_phase[i]}:\n')
    sim_file.write(f'dunits = ppm\ndata =\n{data}')
    sim_file.write(f'\nuncert =\n{uncert}\n')
sim_file.write(f'\nfilters =\n{filters}')
sim_file.close()


