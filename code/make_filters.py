import pickle
import numpy as np
from itertools import islice

import pyratbay as pb
import pyratbay.constants as pc


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

# Generate JWST filter files and data/uncert texts for config files:
run_labels = [f'run{k+1:02}' for k in range(3)]
for k in run_labels:
    # Read the disk-integrated and spatially-resolved data:
    with np.load(f'WASP43b_3D_synthetic_pandexo_flux_ratios_{k}.npz') as d:
        pflux = d['pandexo_flux_ratio']
        puncert = d['pandexo_uncert']
        pwl = d['pandexo_wl']
        obs_phase = d['phase']

    with open(f'../inputs/data/model_WASP43b_phase_{k}.pkl', 'rb') as f:
        d = pickle.load(f)
    resolved_flux = d['long_flux_ratio']
    resolved_unc = d['long_flux_uncert']

    # Pretty-print the data:
    # Integrated:
    mask = puncert[0] < 1.0
    sim_file = open(f'retrieval_WASP43b_integrated_jwst_phase_{k}.txt', 'w')
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
    sim_file = open(f'retrieval_WASP43b_resolved_jwst_phase_{k}.txt', 'w')
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


nphase = len(obs_phase)
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


# Make the filters:
filters = ''
for i in range(npandexo):
    wl = 0.5 * (bounds[i,1] + bounds[i,0])
    width = bounds[i,1] - bounds[i,0]
    margin = 0.1*width
    if wl < 2.81:
        ffile = f"../inputs/filters/JWST_NIRISS_SOSS_{pwl[i]:5.3f}um.dat"
    else:
        ffile = f"../inputs/filters/JWST_NIRSpec_G395H_{pwl[i]:5.3f}um.dat"
    pb.tools.tophat(wl, width, margin, resolution=20000.0, ffile=ffile)
    if mask[i]:
        filters += f"    {ffile}\n"

