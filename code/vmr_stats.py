from itertools import product
import numpy as np
import mc3


model_names = [
    'run_jwst_02TP_01Q',
    'run_jwst_09TP_01Q',
    'run_jwst_16TP_16Q',
    ]
methods = [
    'integrated',
    'resolved',
    ]
molecs = 'H2O CO CO2 CH4'.split()

nmodels = len(model_names)
nmodes = len(methods)
nmol = len(molecs)

k = 2

with np.load('inputs/data/WASP43b_3D_synthetic_pandexo_flux_ratios.npz') as d:
    obs_phase = d['obs_phase']
    abundances = d['abundances']
    temperatures = d['temperatures']

nphase = len(obs_phase)
true_q = np.log10(abundances[k,::4])

mname = model_names[k][9:]
posteriors = np.zeros((nmodels, nmodes, nphase), dtype=object)
for j,i in product(range(nmodes), range(nphase)):
    if k != 2 and i > 8:
        continue
    mcmc_file = \
        f'/MCMC_model_WASP43b_{mname}_{methods[j]}_phase{obs_phase[i]:.2f}.npz'
    with np.load(model_names[k]+mcmc_file) as mcmc:
        posterior, zchain, zmask = mc3.utils.burn(mcmc)
        posteriors[k,j,i] = posterior[:,-4:]


medians = np.zeros((nmodes, nphase, nmol))
means = np.zeros((nmodes, nphase, nmol))
modes = np.zeros((nmodes, nphase, nmol))
stds = np.zeros((nmodes, nphase, nmol))
ups = np.zeros((nmodes, nphase, nmol))
los = np.zeros((nmodes, nphase, nmol))

nbins = 40
for j,i in product(range(nmodes), range(nphase)):
    post = posteriors[k,j,i]
    stds[j,i] = np.std(post, axis=0)
    means[j,i] = np.mean(post, axis=0)
    medians[j,i] = np.median(post, axis=0)
    for m in range(nmol):
        hist, xhist = np.histogram(post[:,m], bins=nbins)
        modes[j,i,m] = np.mean(xhist[np.argmax(hist):np.argmax(hist)+2])
        pdf, x_pdf, hpd_min = mc3.stats.cred_region(post[:,m])
        ups[j,i,m] = np.amin(x_pdf[pdf>hpd_min])
        los[j,i,m] = np.amax(x_pdf[pdf>hpd_min])


zscore_mean   = (true_q - means)   / stds
zscore_median = (true_q - medians) / stds
zscore_mode   = (true_q - modes)   / stds

mean_zscore_mode = np.mean(np.abs(zscore_mode), axis=1)
mean_zscore_mean = np.mean(np.abs(zscore_mean), axis=1)
mean_zscore_median = np.mean(np.abs(zscore_median), axis=1)

print('           H2O     CO      CO2     CH4')
print('Mode:')
for j,zscore in enumerate(mean_zscore_mode):
    print(f"Z-score &  {' &  '.join(f'{z:.2f}' for z in zscore)} {methods[j]}")

print('\nMean:')
for j,zscore in enumerate(mean_zscore_mean):
    print(f"Z-score &  {' &  '.join(f'{z:.2f}' for z in zscore)} {methods[j]}")

print('\nMedian')
for j,zscore in enumerate(mean_zscore_median):
    print(f"Z-score &  {' &  '.join(f'{z:.2f}' for z in zscore)} {methods[j]}")

