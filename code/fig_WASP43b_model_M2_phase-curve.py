import numpy as np
import matplotlib.pyplot as plt

import pyratbay.constants as pc


with np.load('inputs/data/WASP43b_3D_synthetic_pandexo_flux_ratios.npz') as d:
    obs_phase = d['obs_phase']
    pandexo_wl = d['pandexo_wl']
    loc_flux_ratio = d['local_flux_ratio']
    int_flux_ratio = d['noiseless_flux_ratio']

with np.load('inputs/data/RESPECT_3D_synthetic_pandexo_flux_ratios.npz') as d:
    res_flux_ratio = d['longspectra']


labels = [
    r'1.4 $\mu$m (H2O)',
    r'3.3 $\mu$m (CH4)',
    r'4.15 $\mu$m',
    r'4.5 $\mu$m (CO/CO2)',
    ]
nwl = len(labels)

windows = [
    (1.35,1.41),
    (3.10,3.50),
    (4.10,4.20),
    (4.25,4.75),
    ]

window_masks = [
    (pandexo_wl > wleft) & (pandexo_wl < wright)
    for wleft, wright in windows]

loc_phase = [
    np.mean(loc_flux_ratio[2,:,mask], axis=0)/pc.ppt
    for mask in window_masks]
int_phase = [
    np.mean(int_flux_ratio[2,:,mask], axis=0)/pc.ppt
    for mask in window_masks]
res_phase = [
    np.mean(res_flux_ratio[2,:,mask], axis=0)/pc.ppt
    for mask in window_masks]

wide_phase = np.hstack([obs_phase-1, obs_phase, obs_phase+1])
wide_loc = np.hstack([loc_phase, loc_phase, loc_phase])
wide_res = np.hstack([res_phase, res_phase, res_phase])
wide_int = np.hstack([int_phase, int_phase, int_phase])


mew = 1.5
fs = 12

plt.figure(12, (6,7))
plt.clf()
plt.subplots_adjust(0.1, 0.07, 0.99, 0.99, hspace=0.18)
for i in range(nwl):
    ax = plt.subplot(nwl, 1, 1+i)
    plt.plot(wide_phase, wide_loc[i], lw=1.5, alpha=0.4, c='firebrick')
    plt.plot(wide_phase, wide_int[i], lw=1.5, alpha=0.4, c='royalblue')
    plt.plot(wide_phase, wide_res[i], lw=1.5, alpha=0.4, c='orange')
    plt.plot(
        obs_phase, loc_phase[i], '*-', lw=1.5, c='firebrick',
        ms=10, label='Local flux ratio')
    plt.plot(
        obs_phase, int_phase[i], 'D-', lw=1.5, mec='royalblue',
        mfc='none', mew=mew, label='Disk integrated')
    plt.plot(
        obs_phase, res_phase[i], 'o-', lw=1.5, mec='orange',
        mfc='none', mew=mew, label='Spatially resolved')
    ax.set_xlim(-obs_phase[1]*0.5, 1.0-obs_phase[1]*0.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    plt.text(
        0.02, 0.85, labels[i], ha='left', transform=ax.transAxes, fontsize=fs)
    if i == 0:
        plt.legend(loc='upper right')
    plt.ylabel('Fp/Fs (ppt)', fontsize=fs)
    if ax.get_ylim()[0] > 0:
        ax.set_ylim(bottom=0.0)
plt.xlabel('Orbital phase', fontsize=fs)
plt.savefig('plots/model_WASP43b_16TP_phase-curve.pdf')

