import sys
from itertools import product

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

sys.path.append('code')
from legend_handler import Disk, Resolved, Handler


model_names = [
    'run_jwst_02TP_01Q',
    'run_jwst_09TP_01Q',
    'run_jwst_16TP_16Q',
    ]

modes = [
    'integrated',
    'resolved',
    ]

with np.load('inputs/data/WASP43b_3D_temperature_madhu_model.npz') as d:
    press = d['press']/pc.bar

with np.load('inputs/data/WASP43b_3D_synthetic_pandexo_flux_ratios.npz') as d:
    obs_phase = d['obs_phase']
    abundances = d['abundances']
    temperatures = d['temperatures']
    local_flux_ratio = d['local_flux_ratio']
    pandexo_wl = d['pandexo_wl']

atm_models = [
    {'abund': q, 'temp': t}
    for q,t in zip(abundances, temperatures)
    ]

nmodels = len(model_names)
nmodes = len(modes)
nphase = len(obs_phase)
nlayers = len(press)
nspec = 346
nprofiles = 5

data       = np.zeros((nmodels, nmodes, nphase, nspec))
uncerts    = np.zeros((nmodels, nmodes, nphase, nspec))
tpost      = np.zeros((nmodels, nmodes, nphase, nprofiles, nlayers))
posteriors = np.zeros((nmodels, nmodes, nphase), dtype=object)

#for k,j,i in product(range(nmodels), range(nmodes), range(nphase)):
k = 2
for j,i in product(range(nmodes), range(nphase)):
    if k != 2 and i > 8:
        continue
    model = f'mcmc_model_WASP43b_{modes[j]}_phase{obs_phase[i]:.2f}.cfg'
    folder = model_names[k] if k != 2 else 'run_jwst/'  # TMP HACK
    with pt.cd(folder):
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

molecs = 'H2O CO CO2 CH4'.split()
themes = 'blue green red orange'.split()
rc = 'navy darkgreen darkred darkorange'.split()
hpdc = 'cornflowerblue limegreen red gold'.split()
nmol = len(molecs)

ranges = [(-5.5, -1.0), (-12.0, -1.0), (-12.0,-1.0), (-12.0, -1.0)]
nbins = 100
nxpdf = 300
x    = np.zeros((nmodes, nphase, nmol, nbins))
post = np.zeros((nmodes, nphase, nmol, nbins))
xpdf = [np.linspace(ran[0], ran[1], nxpdf) for ran in ranges]
fpdf = np.zeros((nmodes, nphase, nmol, nxpdf))

for j,i,m in product(range(nmodes), range(nphase), range(nmol)):
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
    x[j,i,m] = bins
    post[j,i,m] = vals
    fpdf[j,i,m] = f(xpdf[m])

plot_phase = np.concatenate([obs_phase-1, obs_phase, obs_phase+1])
true_q = atm_models[k]['abund'][::4]
plot_q = np.vstack([true_q, true_q, true_q])


# Colors:
acorn = pb.plots.alphatize('cornflowerblue', 0.7)
arancia = pb.plots.alphatize('orange', 0.6)
c_data  = ['royalblue', 'darkorange']
c_error = [acorn, arancia]
col_true = 'red'
cf_color = 'lightgreen'

fs = 10
lw1 = 1.25
lw2 = 0.75
ms = 1.5

xmin1, xmax1 = 0.04, 0.78
xmin2, xmax2 = 0.34, 0.995
ymin1, ymax1 = 0.05, 0.98
margin1 = 0.26
margin2 = 0.35
ymargin = 0.01

plot_models = [
    [0, 4, 6, 8],
    [0, 4, 6, 8],
    [0, 1, 3, 5, 7, 8, 10, 12, 13, 15],
    ]

npanels = len(plot_models[k])
nx = 2
ny = npanels//nx


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Spectra and temperatures:

fig = plt.figure(20, (8.5, 9.0))
plt.clf()
for j,i in enumerate(plot_models[k]):
    rect = [xmin1, ymin1, xmax1, ymax1]
    ax = mc3.plots.subplotter(rect, margin1, j+1, nx, ny, ymargin)
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
        plt.legend(markerscale=2.5,
            handles=[di_eb, lr_eb, true], loc=(0.03, 0.57), fontsize=fs-2)
    ax.text(
        0.04, 0.98, f'phase = {obs_phase[i]:.2f}', weight='bold',
        va='top', ha='left', transform=ax.transAxes)
    ax.set_ylabel(r'$F_{\rm p}/F_{\rm s}$ (ppt)', fontsize=fs, labelpad=0.5)
    ax.tick_params(labelsize=fs-1, direction='in')
    plt.xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(pyrat.inputs.logxticks)
    if j >= npanels-2:
        ax.set_xlabel('Wavelength (um)', fontsize=fs)
    else:
        ax.set_xticklabels([])
    ylim = ax.get_ylim()
    yticks = [ytick for ytick in ax.get_yticks() if ytick%1.0==0]
    ax.set_yticks(yticks)
    ax.set_ylim(ylim)

for j,i in enumerate(plot_models[k]):
    rect = [xmin2, ymin1, xmax2, ymax1]
    ax = mc3.plots.subplotter(rect, margin2, j+1, nx, ny, ymargin)
    pp.temperature(
        pyrat.atm.press, bounds=tpost[k,0,i,1:3],
        ax=ax, theme=[c_data[0],'0.5'], alpha=[1.0, 0.6], fs=fs)
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
    ax.fill_betweenx(pyrat.atm.press/pc.bar, 200+250*contrib[i], lw=1.5,
        color=cf_color, ec='forestgreen')
    ax.tick_params(direction='in')
    ax.set_xlim(200, 2300)
    ax.set_ylim(ymax=1e-7)
    if j < npanels-2:
        ax.set_xticklabels([])
    else:
        ax.tick_params(labelsize=fs-1, axis='x', direction='in')
    ax.set_ylabel('Pressure (bar)', fontsize=fs, labelpad=0.5)
plt.savefig(f'plots/model_WASP43b_{model_names[k][9:13]}_retrieved_spectra_temperatures.pdf')


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Abundances:

plt.figure(21, (4.5, 8.0))
plt.clf()
plt.subplots_adjust(0.13, 0.05, 0.99, 0.99, hspace=0.135)
for m in range(nmol):
    ax0 = plt.subplot(nmol, 1, m+1, zorder=-10)
    rect = ax0.get_position().extents
    axes = [
        mp.subplotter(rect, 0.0, i+1, nx=nphase, ny=1)
        for i in range(nphase)]
    ax0.plot(plot_phase, np.log10(plot_q[:,m]), c=rc[m], lw=1.5)
    ax0.set_xticks(np.linspace(0,1,5))
    dphase = obs_phase[1] - obs_phase[0]
    ax0.set_xlim(obs_phase[0]-dphase/2, obs_phase[nphase-1]+dphase/2)
    ax0.tick_params(axis='both', direction='in', right=True)
    ax0.set_ylim(ranges[m])
    ax0.set_ylabel(f'$\\log_{{10}}(X_{{\\rm {molecs[m]} }})$', fontsize=fs)
    if m == nmol-1:
        ax0.set_xlabel('Orbital phase', fontsize=fs)
    for i in range(nphase):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_frame_on(False)
        axes[i].plot(0, np.log10(true_q[i,m]), 'o', ms=4, c=rc[m])
        axes[i].plot(-post[0,i,m], x[0,i,m], 'k', lw=lw2)
        axes[i].plot( post[1,i,m], x[1,i,m], themes[m], lw=lw2)
        axes[i].fill_betweenx(
            xpdf[m], 0, -fpdf[0,i,m], facecolor='0.3', edgecolor='none',
            interpolate=False, zorder=-2, alpha=0.7)
        axes[i].fill_betweenx(
            xpdf[m], 0, fpdf[1,i,m], facecolor=hpdc[m], edgecolor='none',
            interpolate=False, zorder=-2, alpha=0.6)
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
        loc=(1.25-nphase, 0.025), fontsize=fs-1,
        framealpha=0.8, borderpad=0.25, labelspacing=0.25)
plt.savefig(f'plots/model_WASP43b_{model_names[k][9:13]}_retrieved_abundances.pdf')

