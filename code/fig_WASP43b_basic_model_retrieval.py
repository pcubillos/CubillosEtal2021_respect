import os
import sys
from itertools import product

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from scipy.ndimage.filters import gaussian_filter1d as gaussf
import scipy.interpolate as si

import pyratbay as pb
import pyratbay.atmosphere as pa
import pyratbay.tools as pt
import pyratbay.plots as pp
import pyratbay.spectrum as ps
import pyratbay.constants as pc

import mc3
import mc3.plots as mp
import mc3.utils as mu

sys.path.append('code')
from legend_handler import *


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Temperature profile posteriors:
models = [
    'run_jwst_02TP_01Q',
    'run_jwst_09TP_01Q',
    ]

modes = [
    'integrated',
    'resolved',
]

with np.load('run_simulation/WASP43b_3D_synthetic_emission_spectra.npz') as emission_model:
    #spectra = emission_model['spectra']
    obs_phase = emission_model['obs_phase']
    abundances = emission_model['abundances']
    temperatures = emission_model['temperatures']

atm_models = [
    {'abund': q, 'temp': t}
    for q,t in zip(abundances, temperatures)
]


with np.load('run_simulation/WASP43b_3D_synthetic_pandexo_flux_ratios.npz') as emission_model:
    local_flux_ratio = emission_model['local_flux_ratio']
    pandexo_wl = emission_model['pandexo_wl']
    #spectra = emission_model['noiseless_flux_ratio']
    #abundances = emission_model['abundances']
    #temperatures = emission_model['temperatures']
local_flux_ratio[0,4] = 0.5*(local_flux_ratio[0,3]+local_flux_ratio[0,5])

nmodels = len(models)
nmodes = len(modes)
nphase = len(obs_phase)
obs_phase = obs_phase[0:9]
#nmol = 4
nspec = 346
nlayers = 61
nprofiles = 5


data       = np.zeros((nmodels, nmodes, nphase, nspec))
uncerts    = np.zeros((nmodels, nmodes, nphase, nspec))
tpost      = np.zeros((nmodels, nmodes, nphase, nprofiles, nlayers))
posteriors = np.zeros((nmodels, nmodes, nphase), dtype=object)

for k,j,i in product(range(nmodels), range(nmodes), range(nphase)):
    if k != 2 and i > 8:
        continue
    model = f'mcmc_model_WASP43b_{modes[j]}_phase{obs_phase[i]:.2f}.cfg'
    with pt.cd(models[k]):
        pyrat = pb.run(model, init=True, no_logfile=True)
    with np.load(pyrat.ret.mcmcfile) as mcmc:
        posterior, zchain, zmask = mc3.utils.burn(mcmc)
        posteriors[k,j,i] = posterior[:,-4:]
        bestp = mcmc['bestp']

    data[k,j,i] = pyrat.obs.data
    uncerts[k,j,i] = pyrat.obs.uncert
    ifree = pyrat.ret.pstep[pyrat.ret.itemp] > 0
    itemp = np.arange(np.sum(ifree))
    temp_best = pyrat.atm.tmodel(bestp[pyrat.ret.itemp])
    tpost[k,j,i] = pa.temperature_posterior(
        posterior[:,itemp], pyrat.atm.tmodel,
        pyrat.ret.params[pyrat.ret.itemp], ifree, pyrat.atm.press)


band_wl = 1.0 / (pyrat.obs.bandwn * pc.um)
wl = 1.0 / (pyrat.spec.wn * pc.um)

contrib = np.zeros((nphase, nlayers))
k = 1
abunds = atm_models[k]['abund']
temps  = atm_models[k]['temp']
for i in range(nphase):
    if k != 2 and i > 8:
        continue
    q2 = pa.qscale(
        pyrat.atm.qbase, pyrat.mol.name, pyrat.atm.molmodel,
        pyrat.atm.molfree, np.log10(abunds[4*i]),
        pyrat.atm.bulk, iscale=pyrat.atm.ifree, ibulk=pyrat.atm.ibulk,
        bratio=pyrat.atm.bulkratio, invsrat=pyrat.atm.invsrat)
    pyrat.run(temps[4*i], q2)
    cf = ps.contribution_function(pyrat.od.depth, pyrat.atm.press, pyrat.od.B)
    bcf = ps.band_cf(cf, pyrat.obs.bandtrans, pyrat.spec.wn, pyrat.obs.bandidx)
    bcf = np.sum(bcf, axis=1)
    contrib[i] = bcf / np.amax(bcf)

labels = 'H2O CO CO2 CH4'.split()
themes = ['blue', 'green', 'red', 'orange']
rc =   'navy darkgreen darkred darkorange'.split()
hpdc = 'cornflowerblue limegreen red gold'.split()

true_q = atm_models[0]['abund'][0]
nmol = len(labels)

ranges = [(-5.5, -1.0), (-5.5, -1.0), (-12.0,-1.0), (-12.0, -1.0)]
nbins = 100
nxpdf = 1000
x    = np.zeros((nmodels, nmodes, nphase, nmol, nbins))
post = np.zeros((nmodels, nmodes, nphase, nmol, nbins))
xpdf = [np.linspace(ran[0], ran[1], nxpdf) for ran in ranges]
fpdf = np.zeros((nmodels, nmodes, nphase, nmol, nxpdf))

for k,j,i,m in product(range(nmodels), range(nmodes), range(nphase), range(nmol)):
    if k != 2 and i > 8:
        continue
    vals, bins = np.histogram(
        posteriors[k,j,i][:,m], bins=nbins, range=ranges[m], density=True)
    vals = gaussf(vals, 1.5)
    vals = vals/np.amax(vals) * 0.8
    bins = 0.5 * (bins[1:] + bins[:-1])
    PDF, Xpdf, HPDmin = mc3.stats.cred_region(
        posteriors[k,j,i][:,m], quantile=0.683)
    f = si.interp1d(
        bins, vals, kind='nearest', bounds_error=False, fill_value=0.0)
    x[k,j,i,m] = bins
    post[k,j,i,m] = vals
    fpdf[k,j,i,m] = f(xpdf[m])


acorn = pb.plots.alphatize('cornflowerblue', 0.7)
arancia = pb.plots.alphatize('orange', 0.6)
c_model = ['navy', 'maroon']
c_data  = ['royalblue', 'darkorange']
c_error = [acorn, arancia]
col_true = 'red'
cf_color = 'lightgreen'

ttheme = 'royalblue', '0.4'

fs = 10
lw1 = 1.25
lw2 = 0.75
f0 = 1.0 / pc.ppt
ms = 2.0


k = 1
model = models[k][9:]
nphase = 9

xmin1, xmax1 = 0.045, 0.385
xmin2, xmax2 = 0.445, 0.62
xmin3, xmax3 = 0.70, 0.99
ymin, ymax = 0.055, 0.99
margin = 0.01
margin2 = 0.03
plot_models = [0, 4, 6, 8]
ny = len(plot_models)


plt.figure(41, (8.5, 8.0))
plt.clf()
for j,i in enumerate(plot_models):
    rect = [xmin1, ymin, xmax1, ymax]
    ax = mc3.plots.subplotter(rect, margin, j+1, nx=1, ny=ny)
    di_eb = ax.errorbar(
        band_wl, data[k,0,i]/pc.ppt, uncerts[k,0,i]/pc.ppt,
        lw=lw1, fmt='o', ms=ms, mew=0.0, c=c_data[0], ecolor=c_error[0],
        label='Disk integrated')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(pyrat.inputs.logxticks)
    lr_eb = plt.errorbar(
        band_wl, data[k,1,i]/pc.ppt, uncerts[k,1,i]/pc.ppt,
        lw=lw1, fmt='o', ms=ms, mew=0.0, c=c_data[1], ecolor=c_error[1],
        label='Long. resolved')
    true, = plt.plot(
        pandexo_wl, local_flux_ratio[k,i]/pc.ppt, c=col_true,
        zorder=10, lw=lw1, label='True (local) model')
    plt.xlim(0.83, 5.2)
    if j == 0:
        plt.legend(markerscale=2.0,
            handles=[di_eb, lr_eb, true], loc=(0.03, 0.6), fontsize=fs-2)
    ax.text(
        0.04, 0.98, f'phase = {obs_phase[i]:.2f}', weight='bold',
        va='top', ha='left', transform=ax.transAxes)
    ax.set_ylabel(r'$F_{\rm p}/F_{\rm s}$ (ppt)', fontsize=fs, labelpad=0.5)
    ax.tick_params(labelsize=fs-1, direction='in')
    plt.xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(pyrat.inputs.logxticks)
    if j == ny-1:
        plt.xlabel('Wavelength (um)', fontsize=fs)
    else:
        ax.set_xticklabels([])
    ylim = ax.get_ylim()
    yticks = [ytick for ytick in ax.get_yticks() if ytick%1.0==0]
    ax.set_yticks(yticks)
    ax.set_ylim(ylim)

for j,i in enumerate(plot_models):
    rect = [xmin2, ymin, xmax2, ymax]
    ax = mc3.plots.subplotter(rect, margin, j+1, nx=1, ny=ny)
    pp.temperature(
        pyrat.atm.press, bounds=tpost[k,0,i,1:3],
        ax=ax, theme=ttheme, alpha=[1.0,0.6], fs=fs)
    pp.temperature(
        pyrat.atm.press, bounds=tpost[k,1,i,1:3],
        ax=ax, theme='orange', alpha=[0.7, 0.3], fs=fs)
    for l in range(nphase):
        ax.plot(
            atm_models[k]['temp'][4*l], pyrat.atm.press/pc.bar,
            alpha=0.4, lw=1.0, c='0.4')
    ax.plot(
        atm_models[k]['temp'][4*i], pyrat.atm.press/pc.bar,
        c=col_true, lw=1.5, zorder=10)
    ax.tick_params(labelsize=fs-1, axis='x', direction='in')
    ax.tick_params(labelsize=fs-2, axis='y', direction='in')
    ax.set_ylabel('Pressure (bar)', fontsize=fs, labelpad=0.5)
    ax.set_xlim(200, 2300)
    ax.set_ylim(ymax=1e-7)
    ax.fill_betweenx(pyrat.atm.press/pc.bar, 200+250*contrib[i], lw=1.5,
        color=cf_color, ec='forestgreen')
    if j != ny-1:
        ax.set_xticklabels([])

for m in range(nmol):
    rect = [xmin3, ymin, xmax3, ymax]
    ax0 = mc3.plots.subplotter(rect, margin2, m+1, nx=1, ny=nmol)
    rect = ax0.get_position().extents
    axes = [
        mp.subplotter(rect, 0.0, i+1, nx=nphase, ny=1)
        for i in range(nphase)]
    ax0.axhline(np.log10(true_q[m]), c=rc[m], lw=1.5)
    ax0.set_xticks(np.linspace(0,1,5))
    dphase = obs_phase[1] - obs_phase[0]
    ax0.set_xlim(obs_phase[0]-dphase/2, obs_phase[nphase-1]+dphase/2)
    ax0.tick_params(axis='both', direction='in', right=True)
    ax0.set_ylim(ranges[m])
    ax0.set_ylabel(f'$\\log_{{10}}(X_{{\\rm {labels[m]} }})$', fontsize=fs)
    if m == nmol-1:
        ax0.set_xlabel('Orbital phase', fontsize=fs)
    for i in range(nphase):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_frame_on(False)
        axes[i].plot(0, np.log10(true_q[m]), 'o', ms=3, c=rc[m])
        axes[i].plot(-post[k,0,i,m], x[k,0,i,m], 'k', lw=lw2)
        axes[i].plot( post[k,1,i,m], x[k,1,i,m], themes[m], lw=lw2)
        axes[i].fill_betweenx(
            xpdf[m], 0, -fpdf[k,0,i,m], facecolor='0.3',
            edgecolor='none', interpolate=False, zorder=-2, alpha=0.7)
        axes[i].fill_betweenx(
            xpdf[m], 0, fpdf[k,1,i,m], facecolor=hpdc[m],
            edgecolor='none', interpolate=False, zorder=-2, alpha=0.6)
        axes[i].set_xlim(-1, 1)
        axes[i].set_ylim(ranges[m])
    if m == 0:
        handles = [Disk(), Resolved()]
        handle_labels = ['Disk integrated', 'Longitudinally resolved']
        handler_map = {Disk: Handler('black'), Resolved: Handler(themes[m])}
    else:
        handles = [Resolved()]
        handle_labels = ['Longitudinally resolved']
        handler_map = {Resolved: Handler(themes[m])}
    axes[-1].legend(
        handles, handle_labels, handler_map=handler_map,
        loc=(1.25-nphase, 0.95-0.1*len(handles)), fontsize=fs-1,
        framealpha=0.8, borderpad=0.25, labelspacing=0.25)

plt.savefig(f'plots/model_WASP43b_09TP_retrieved_spectra_temp_abundances.pdf')
