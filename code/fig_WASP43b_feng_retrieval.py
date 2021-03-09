import os
import sys
from itertools import product

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gaussf
import scipy.interpolate as si

import pyratbay as pb
import pyratbay.atmosphere as pa
import pyratbay.constants as pc
import pyratbay.io as io
import pyratbay.plots as pp
import pyratbay.spectrum as ps
import pyratbay.tools as pt

import mc3
import mc3.plots as mp
import mc3.utils as mu

sys.path.append('code')
from legend_handler import *



modes = [
    'integrated',
    'resolved',
    ]

obs_phase = np.array([
    0.06, 0.12, 0.19, 0.25, 0.31, 0.38, 0.44, 0.50,
    0.56, 0.62, 0.69, 0.75, 0.81, 0.88, 0.94,
    ])

nmodes = len(modes)
nphase = len(obs_phase)
nspec = 17
nprofiles = 3
nlayers = 61
nwave = 17048

data    = np.zeros((nmodes, nphase, nspec))
uncerts = np.zeros((nmodes, nphase, nspec))
tpost = np.zeros((nmodes, nphase, nprofiles, nlayers))
posteriors = np.zeros((nmodes, nphase), dtype=object)
median_fr = np.zeros((nmodes, nphase, nwave))
hi_fr     = np.zeros((nmodes, nphase, nwave))
lo_fr     = np.zeros((nmodes, nphase, nwave))


for j,i in product(range(nmodes), range(nphase)):
    #model = f'mcmc_WASP43b_{modes[j]}_phase{obs_phase[i]:.2f}.cfg'
    #with pt.cd('run_feng2020/'):
    #    pyrat = pb.run(model, init=True, no_logfile=True)
    #with np.load(pyrat.ret.mcmcfile) as mcmc:
    #    posterior, zchain, zmask = mc3.utils.burn(mcmc)
    #    bestp = mcmc['bestp']
    pickle_file = f'MCMC_WASP43b_{modes[j]}_phase{obs_phase[i]:.2f}.pickle'
    print(pickle_file)
    with pt.cd('run_feng2020/'):
        pyrat = io.load_pyrat(pickle_file)
    posteriors[j,i] = pyrat.ret.posterior[:,-4:]
    rprs = pyrat.phy.rplanet/pyrat.phy.rstar
    starflux = pyrat.spec.starflux
    median_fr[j,i] = pyrat.ret.spec_median / starflux * rprs**2
    lo_fr[j,i] = pyrat.ret.spec_low1 / starflux * rprs**2
    hi_fr[j,i] = pyrat.ret.spec_high1 / starflux * rprs**2

    data[j,i] = pyrat.obs.data
    uncerts[j,i] = pyrat.obs.uncert
    ifree = pyrat.ret.pstep[pyrat.ret.itemp] > 0
    itemp = np.arange(np.sum(ifree))
    tpost[j,i] = pa.temperature_posterior(
        pyrat.ret.posterior[:,itemp], pyrat.atm.tmodel,
        pyrat.ret.params[pyrat.ret.itemp], ifree, pyrat.atm.press)[0:3]


mol_themes = ['blue', 'green', 'red', 'orange']
rc =   'navy darkgreen darkred darkorange'.split()
hpdc = 'cornflowerblue limegreen red gold'.split()
molecs = 'H2O CO CO2 CH4'.split()
ranges = [(-12.0, -1.0) for _ in molecs]
nmol = len(molecs)


nbins = 100
nxpdf = 100
x    = np.zeros((nmodes, nphase, nmol, nbins))
post = np.zeros((nmodes, nphase, nmol, nbins))
xpdf = [np.linspace(ran[0], ran[1], nxpdf) for ran in ranges]
fpdf = np.zeros((nmodes, nphase, nmol, nxpdf))

for j,i,m in product(range(nmodes), range(nphase), range(nmol)):
    vals, bins = np.histogram(
        posteriors[j,i][:,m], bins=nbins, range=ranges[m], density=True)
    vals = gaussf(vals, 1.5)
    vals = vals/np.amax(vals) * 0.8
    bins = 0.5 * (bins[1:] + bins[:-1])
    PDF, Xpdf, HPDmin = mc3.stats.cred_region(
        posteriors[j,i][:,m], quantile=0.683)
    f = si.interp1d(
        bins, vals, kind='nearest', bounds_error=False, fill_value=0.0)
    x[j,i,m] = bins
    post[j,i,m] = vals
    fpdf[j,i,m] = f(xpdf[m])

band_wl = 1.0 / (pyrat.obs.bandwn * pc.um)
wl = 1.0 / (pyrat.spec.wn * pc.um)

labels = [
    "Disk integrated",
    "Long. resolved",
    ]


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

amedium = pb.plots.alphatize('mediumblue', 0.85)
c_data  = ['mediumblue', 'darkorange']
c_error  = ['royalblue', 'orange']
c_model = [amedium, 'red']
cf_color = 'lightgreen'

alpha   = [1.0, 0.7]
themes = (
    [c_error[0],''],
    [c_error[1], ''],
    )
ttheme = 'royalblue', '0.4'


fs = 10
lw1 = 1.25
lw2 = 0.75
ms = 4.0
ms = 3.5
sigma = 30.0
offset = [1.0, 1.0025]

xmin1, xmax1 = 0.045, 0.342
xmin2, xmax2 = 0.405, 0.574
xmin3, xmax3 = 0.65, 0.99
ymin, ymax = 0.055, 0.99
margin = 0.01
margin2 = 0.03
plot_models = [2, 4, 7, 12]
ny = len(plot_models)


plt.figure(41, (8.5, 8.0))
plt.clf()
legs = []
for j,i in enumerate(plot_models):
    rect = [xmin1, ymin, xmax1, ymax]
    ax = mc3.plots.subplotter(rect, margin, j+1, nx=1, ny=ny)
    for k in range(2):
        cr = ax.fill_between(
            wl, gaussf(lo_fr[k,i], sigma)/pc.ppt,
            gaussf(hi_fr[k,i], sigma)/pc.ppt,
            facecolor=c_error[k], edgecolor='none', alpha=alpha[k])
        line, = ax.plot(
            wl, gaussf(median_fr[k,i],sigma)/pc.ppt, c=c_model[k], lw=lw1)
        eb = ax.errorbar(
            band_wl*offset[k], data[k,i]/pc.ppt, uncerts[k,i]/pc.ppt,
            zorder=90, mew=0.25, mec='k', lw=lw1, fmt='o', ms=ms, c=c_data[k],
            ecolor=c_model[k], label='Disk integrated')
        legs.append((cr, line, eb))
    if j == 0:
        ax.legend(legs, labels, fontsize=fs-1, loc=(0.03, 0.63))
    ax.text(
        0.04, 0.98, f'phase = {obs_phase[i]:.2f}', weight='bold',
        va='top', ha='left', transform=ax.transAxes)
    ax.set_ylabel(r'$F_{\rm p}/F_{\rm s}$ (ppt)', fontsize=fs, labelpad=0.5)
    ax.tick_params(labelsize=fs-1, direction='in')
    plt.xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(pyrat.inputs.logxticks)
    ax.set_xlim(1.05, 5.1)
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
        pyrat.atm.press, bounds=tpost[0,i,1:3],
        ax=ax, theme=themes[0], alpha=[1.0,0.6], fs=fs)
    pp.temperature(
        pyrat.atm.press, bounds=tpost[1,i,1:3],
        ax=ax, theme='orange', alpha=[0.7, 0.3], fs=fs)
    for k in range(2):
        ax.plot(tpost[k,i,0], pyrat.atm.press/pc.bar, c=c_model[k], lw=1.5)
    ax.tick_params(labelsize=fs-1, axis='x', direction='in')
    ax.tick_params(labelsize=fs-2, axis='y', direction='in')
    ax.set_ylabel('Pressure (bar)', fontsize=fs, labelpad=0.5)
    ax.set_xlim(200, 2300)
    ax.set_ylim(ymax=1e-7)
    #ax.fill_betweenx(pyrat.atm.press/pc.bar, 200+250*contrib[i], lw=1.5,
    #    color=cf_color, ec='forestgreen')
    if j != ny-1:
        ax.set_xticklabels([])
        ax.set_xlabel('')

for m in range(nmol):
    rect = [xmin3, ymin, xmax3, ymax]
    ax0 = mc3.plots.subplotter(rect, margin2, m+1, nx=1, ny=nmol)
    rect = ax0.get_position().extents
    axes = [
        mp.subplotter(rect, 0.0, i+1, nx=nphase, ny=1)
        for i in range(nphase)]
    ax0.set_xticks(np.array([0.06, 0.25, 0.5, 0.75, 0.94]))
    dphase = 0.0625
    ax0.set_xlim(obs_phase[0]-dphase/2, obs_phase[-1]+dphase/2)
    ax0.tick_params(axis='both', direction='in', right=True)
    ax0.set_ylim(ranges[m])
    ax0.set_ylabel(f'$\\log_{{10}}(X_{{\\rm {molecs[m]} }})$')
    if m >= 2:
        ax0.set_xlabel('Orbital phase')
    for i in range(nphase):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_frame_on(False)
        axes[i].plot(-post[0,i,m], x[0,i,m], 'k', lw=lw2)
        axes[i].plot( post[1,i,m], x[1,i,m], mol_themes[m], lw=lw2)
        axes[i].fill_betweenx(
            xpdf[m], 0, -fpdf[0,i,m], facecolor='0.3',
            edgecolor='none', interpolate=False, zorder=-2, alpha=0.7)
        axes[i].fill_betweenx(
            xpdf[m], 0, fpdf[1,i,m], facecolor=hpdc[m],
            edgecolor='none', interpolate=False, zorder=-2, alpha=0.6)
        axes[i].set_xlim(-1, 1)
        axes[i].set_ylim(ranges[m])
    if m == 0:
        handles = [Disk(), Resolved()]
        handle_labels = ['Disk integrated', 'Longitudinally resolved']
        handler_map = {Disk: Handler('black'), Resolved: Handler(mol_themes[m])}
    else:
        handles = [Resolved()]
        handle_labels = ['Longitudinally resolved']
        handler_map = {Resolved: Handler(mol_themes[m])}
    axes[-1].legend(
        handles, handle_labels, handler_map=handler_map,
        loc=(1.25-nphase, 0.05), fontsize=fs-1, framealpha=0.8,
        borderpad=0.25, labelspacing=0.25)
plt.savefig('plots/WASP43b_feng_spectra_temps_abundances.pdf')

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Abundances:

themes = ['blue', 'green', 'red', 'orange']
rc =   'navy darkgreen darkred darkorange'.split()
hpdc = 'cornflowerblue limegreen red gold'.split()
labels = 'H2O CO CO2 CH4'.split()
ranges = [(-12.0, -1.0) for _ in molecs]
nmol = len(labels)


nbins = 100
nxpdf = 100
x    = np.zeros((nmodes, nphase, nmol, nbins))
post = np.zeros((nmodes, nphase, nmol, nbins))
xpdf = [np.linspace(ran[0], ran[1], nxpdf) for ran in ranges]
fpdf = np.zeros((nmodes, nphase, nmol, nxpdf))

for j,i,m in product(range(nmodes), range(nphase), range(nmol)):
    vals, bins = np.histogram(
        posteriors[j,i][:,m], bins=nbins, range=ranges[m], density=True)
    vals = gaussf(vals, 1.5)
    vals = vals/np.amax(vals) * 0.8
    bins = 0.5 * (bins[1:] + bins[:-1])
    PDF, Xpdf, HPDmin = mc3.stats.cred_region(
        posteriors[j,i][:,m], quantile=0.683)
    f = si.interp1d(
        bins, vals, kind='nearest', bounds_error=False, fill_value=0.0)
    x[j,i,m] = bins
    post[j,i,m] = vals
    fpdf[j,i,m] = f(xpdf[m])


horizontal = True
lw = 0.75
fs = 10

plt.figure(40, (4.5, 8.0))
plt.clf()
plt.subplots_adjust(0.14, 0.06, 0.99, 0.99, hspace=0.135)
for m in range(nmol):
    ax0 = plt.subplot(nmol, 1, m+1, zorder=-100)
    rect = ax0.get_position().extents
    axes = [
        mp.subplotter(rect, 0.0, i+1, nx=nphase, ny=1)
        for i in range(nphase)]
    #ax0.set_xticks(np.linspace(0,1,5))
    ax0.set_xticks(np.array([0.06, 0.25, 0.5, 0.75, 0.94]))
    dphase = 0.0625
    ax0.set_xlim(obs_phase[0]-dphase/2, obs_phase[-1]+dphase/2)
    ax0.tick_params(axis='both', direction='in')
    ax0.set_ylim(ranges[m])
    ax0.set_ylabel(f'$\\log_{{10}}(X_{{\\rm {labels[m]} }})$')
    if m == nmol-1:
        ax0.set_xlabel('Orbital phase')
    for i in range(nphase):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_frame_on(False)
        axes[i].plot(-post[0,i,m], x[0,i,m], 'k', lw=lw)
        axes[i].plot( post[1,i,m], x[1,i,m], themes[m], lw=lw)
        axes[i].fill_betweenx(
            xpdf[m], 0, -fpdf[0,i,m], facecolor='0.3',
            edgecolor='none', interpolate=False, zorder=-2, alpha=0.7)
        axes[i].fill_betweenx(
            xpdf[m], 0, fpdf[1,i,m], facecolor=hpdc[m],
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
        loc=(1.25-nphase, 0.05), fontsize=fs-1, framealpha=0.8,
        borderpad=0.25, labelspacing=0.25)
plt.savefig(f'plots/WASP43b_feng_retrieved_abundances.pdf')

plt.figure(40, (8.5, 4.2))
plt.clf()
plt.subplots_adjust(0.07, 0.1, 0.99, 0.99, hspace=0.135, wspace=0.18)
for m in range(nmol):
    ax0 = plt.subplot(2, 2, m+1, zorder=-100)
    rect = ax0.get_position().extents
    axes = [
        mp.subplotter(rect, 0.0, i+1, nx=nphase, ny=1)
        for i in range(nphase)]
    ax0.set_xticks(np.array([0.06, 0.25, 0.5, 0.75, 0.94]))
    dphase = 0.0625
    ax0.set_xlim(obs_phase[0]-dphase/2, obs_phase[-1]+dphase/2)
    ax0.tick_params(axis='both', direction='in', right=True)
    ax0.set_ylim(ranges[m])
    ax0.set_ylabel(f'$\\log_{{10}}(X_{{\\rm {labels[m]} }})$')
    if m >= 2:
        ax0.set_xlabel('Orbital phase')
    for i in range(nphase):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_frame_on(False)
        axes[i].plot(-post[0,i,m], x[0,i,m], 'k', lw=lw)
        axes[i].plot( post[1,i,m], x[1,i,m], themes[m], lw=lw)
        axes[i].fill_betweenx(
            xpdf[m], 0, -fpdf[0,i,m], facecolor='0.3',
            edgecolor='none', interpolate=False, zorder=-2, alpha=0.7)
        axes[i].fill_betweenx(
            xpdf[m], 0, fpdf[1,i,m], facecolor=hpdc[m],
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
        loc=(1.25-nphase, 0.05), fontsize=fs-1, framealpha=0.8,
        borderpad=0.25, labelspacing=0.25)
plt.savefig(f'plots/WASP43b_feng_retrieved_abundances_horizontal.pdf')

