import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gaussf
import scipy.interpolate as si

sys.path.append('rate')
import rate

import pyratbay as pb
import pyratbay.atmosphere as pa
import pyratbay.tools as pt
import pyratbay.plots as pp
import pyratbay.spectrum as ps
import pyratbay.constants as pc

import mc3
import mc3.plots as mp
import mc3.utils as mu


with np.load('run_simulation/WASP43b_3D_temperature_madhu_model.npz') as gcm:
    temps = gcm['temp']
    tpars = gcm['tpars']
    press = gcm['press']
    lats  = gcm['lats']
    lons  = gcm['lons']
    obs_phase = gcm['obs_phase']

nlat = len(lats)
nlon = len(lons)
nobs = len(obs_phase)
nlayers = len(press)

# Initialize object with solar composition:
r = rate.Rate(C=2.5e-4, N=1.0e-4, O=5.0e-4, fHe=0.0851)
ipress = 42  # ~0.1 bar
ilat   =  0  # -60 deg  (Nice switch from CO--CH4 dominated atm)
p = np.tile(press[ipress], nobs)  # bars
rtemp = temps[:,ilat,ipress]     # kelvin
Q = r.solve(rtemp, p/pc.bar)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Temperature profile posteriors:
integrated = sorted([
    cfile for cfile in os.listdir('run_jwst') 
    if cfile.endswith('.cfg') and 'integrated' in cfile
    ])

resolved = sorted([
    cfile for cfile in os.listdir('run_jwst')
    if cfile.endswith('.cfg') and 'resolved' in cfile
    ])
datasets = [integrated, resolved]
phases = [cfg[-8:-4] for cfg in resolved]


nsets, nphase = np.shape(datasets)
nmol = 4
nspec = 346

data    = np.zeros((nsets,nphase,nspec))
uncerts = np.zeros((nsets,nphase,nspec))
models  = np.zeros((nsets,nphase,nspec))
tpost   = np.zeros((nsets,nphase,5,nlayers))

posteriors = [
    [None] * nphase,
    [None] * nphase,
    ]


for j in range(nsets):
    for i in range(nphase):
        with pt.cd('run_jwst'):
            pyrat = pb.run(datasets[j][i], init=True, no_logfile=True)
        with np.load(pyrat.ret.mcmcfile) as mcmc:
            posterior, zchain, zmask = mc3.utils.burn(mcmc)
            posteriors[j][i] = posterior[:,-4:]
            bestp = mcmc['bestp']
            models[j,i] = mcmc['best_model']

        ifree = pyrat.ret.pstep[pyrat.ret.itemp] > 0
        itemp = np.arange(np.sum(ifree))
        temp_best = pyrat.atm.tmodel(bestp[pyrat.ret.itemp])
        tpost[j,i] = pa.temperature_posterior(
            posterior[:,itemp], pyrat.atm.tmodel,
            pyrat.ret.params[pyrat.ret.itemp], ifree, pyrat.atm.press)
        data[j,i] = pyrat.obs.data
        uncerts[j,i] = pyrat.obs.uncert


band_wl = 1.0 / (pyrat.obs.bandwn * pc.um)
wl = 1.0 / (pyrat.spec.wn * pc.um)

# 1D models at local conditions:
labels = r.species
starflux = pyrat.spec.starflux
rprs = pyrat.phy.rplanet/pyrat.phy.rstar
imol = [labels.index(molfree) for molfree in pyrat.atm.molfree]
obs_lat = 1

uniform_flux = np.zeros((nphase, nspec))
for iphase in range(nphase):
    q2 = pa.qscale(
        pyrat.atm.qbase, pyrat.mol.name, pyrat.atm.molmodel,
        pyrat.atm.molfree, np.log10(Q[imol,iphase]),
        pyrat.atm.bulk, iscale=pyrat.atm.ifree, ibulk=pyrat.atm.ibulk,
        bratio=pyrat.atm.bulkratio, invsrat=pyrat.atm.invsrat)
    temp = temps[iphase, obs_lat]
    pyrat.run(temp, q2)
    uniform_flux[iphase] = pyrat.band_integrate()


acorn = pb.plots.alphatize('cornflowerblue', 0.7)
arancia = pb.plots.alphatize('orange', 0.6)
c_model = ['navy', 'maroon']
c_data  = ['royalblue', 'chocolate']
c_error = [acorn, arancia]
col_true = 'red'


plot_spec = [1, 3, 7, 8, 12, 15]
nplot = len(plot_spec)
plot_temps = [0, 1, 3, 5, 7, 8, 10, 12, 13, 15]
ntemps = len(plot_temps)
weight = ['bold' if index in plot_spec else 'normal' for index in plot_temps]
subtext = [
    '(hot spot)' if i == 7 else
    '(secondary eclipse)' if i == 8
    else ''
    for i in range(nphase)]

fs = 10
lw = 1.25
f0 = 1.0 / pc.ppt
ms = 3.0

xs = 0.05
dxs = 0.62 - xs
rect = 0.065, 0.045, 0.99, 0.355
margin = 0.01

fig = plt.figure(201, (8.5,9))
plt.clf()
plt.subplots_adjust(xs, 0.4, 0.99, 0.99, hspace=0.14, wspace=0.12)
for i,k in enumerate(plot_spec):
    ax = plt.subplot(3, 2, 1+i)
    di_eb = ax.errorbar(
        band_wl, data[0,k]/pc.ppt, uncerts[0,k]/pc.ppt, lw=lw, fmt='o', ms=ms,
        mew=0.0, c=c_data[0], ecolor=c_error[0],
        label='disk-integrated dataset')
    di_fit, = ax.plot(band_wl, models[0,k]/pc.ppt,
        c=c_model[0], zorder=10, lw=lw, label='disk-integrated fit')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(pyrat.inputs.logxticks)
    lr_eb = plt.errorbar(
        band_wl, data[1,k]/pc.ppt, uncerts[1,k]/pc.ppt, lw=lw, fmt='o', ms=ms,
        mew=0.0, c=c_data[1], ecolor=c_error[1], label='long. resolved dataset')
    lr_fit, = plt.plot(band_wl, models[1,k]/pc.ppt,
        c=c_model[1], zorder=10, lw=lw, label='long. resolved fit')
    true, = plt.plot(band_wl, uniform_flux[k]/pc.ppt,
        c=col_true, zorder=10, lw=lw, label='Model at local properties')
    plt.xlim(0.83, 5.25)
    if i == 0:
        plt.legend(handles=[di_eb, di_fit, lr_eb, lr_fit, true],
        loc='upper left', fontsize=fs-2)
    xtext = 0.5 if i==0 else 0.03
    ax.text(xtext, 0.95, f'phase = {phases[k]}\n{subtext[k]}',
        va='top', ha='left', transform=ax.transAxes)
    plt.xlabel('Wavelength (um)', fontsize=fs)
    plt.ylabel(r'$F_{\rm p}/F_{\rm s}$ (ppt)', fontsize=fs)
    ax.tick_params(labelsize=fs-1, direction='in')
    plt.xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(pyrat.inputs.logxticks)
    if ax.get_ylim()[1] < 3.1:
        ax.set_ylim(ymax=3.1)

for j,i in enumerate(plot_temps):
    ax = mp.subplotter(rect, margin, j+1, ntemps//2, 2, ymargin=0.01)
    pp.temperature(pyrat.atm.press,
        bounds=tpost[0,i,1:3], ax=ax, alpha=[1.0,0.6], fs=fs)
    pp.temperature(pyrat.atm.press,
        bounds=tpost[1,i,1:3], ax=ax, theme='orange', alpha=[0.7, 0.3], fs=fs)
    for k in range(nobs):
        if obs_phase[k] == obs_phase[i]:
            ax.plot(temps[k,1], press/pc.bar, c=col_true,lw=1.5, zorder=10)
        else:
            ax.plot(temps[k,1], press/pc.bar, alpha=0.2, lw=1.0, c='0.4')
    ax.tick_params(direction='in')
    ax.set_xlim(200, 2400)
    ax.set_ylim(ymax=1e-7)
    ax.text(0.98, 0.9, f'phase = {phases[i]}', fontweight=weight[j],
        transform=ax.transAxes, ha='right', fontsize=fs-2)
    if j%(ntemps//2) != 0:
        ax.set_ylabel('')
        ax.set_yticklabels([])
    if j<(ntemps//2) != 0:
        ax.set_xlabel('')
        ax.set_xticklabels([])
    plt.subplots_adjust(xs, 0.4, 0.99, 0.99, hspace=0.14, wspace=0.12)
plt.savefig('plots/model_WASP43b_retrieved_spectra_temperatures.pdf')


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Abundances:

themes = ['blue', 'green', 'red', 'orange']
rc =   'navy darkgreen darkred darkorange'.split()
hpdc = 'cornflowerblue limegreen red gold'.split()

iq = [0, 2, 3, 1]
labels = 'H2O CO CO2 CH4'.split()
ranges = [(-4.5, -1.5), (-9, -1), (-11,-5), (-9, -1.5)]

plot_phase = np.concatenate([obs_phase-1, obs_phase, obs_phase+1])
plot_q = np.hstack([Q,Q,Q])


quantile = np.tile(0.683, (nsets, nmol, nphase))
# CO
quantile[:,1,np.array([0,1,14,15])] = 0.9545
# CO2
quantile[:,2,np.array([0,1,14,15])] = 0.9545
quantile[1,2,np.array([2,13])] = 0.9545
# CH4
quantile[1,3,np.array([5,6,7,8])] = 0.9545

nbins = 100
nxpdf = 3000
x    = np.zeros((nsets, nphase, nmol, nbins))
post = np.zeros((nsets, nphase, nmol, nbins))
xpdf = np.zeros((nsets, nphase, nmol, nxpdf))
fpdf = np.zeros((nsets, nphase, nmol, nxpdf))
hpd = np.zeros((nsets, nphase, nmol, nxpdf), bool)

ranges = [(-4.1, -1.5), (-8.5, -1.0), (-10.3,-5.5), (-8.5, -2.0)]
for k in range(nsets):
    for j in range(nmol):
        for i in range(nphase):
            vals, bins = np.histogram(
                posteriors[k][i][:,j], bins=nbins, range=ranges[j],density=True)
            vals = gaussf(vals, 1.5)
            vals = vals/np.amax(vals) * 0.8
            bins = 0.5*(bins[1:]+bins[:-1])
            PDF, Xpdf, HPDmin = mc3.stats.cred_region(
                posteriors[k][i][:,j], quantile[k,j,i])
            f = si.interp1d(bins, vals, kind='nearest', bounds_error=False)
            x[k,i,j] = bins
            post[k,i,j] = vals
            xpdf[k,i,j] = Xpdf
            fpdf[k,i,j] = f(Xpdf)
            hpd[k,i,j] = PDF >= HPDmin


lw = 1.0

plt.figure(400, (4.5, 9.0))
plt.clf()
plt.subplots_adjust(0.13, 0.05, 0.99, 0.99, hspace=0.135)
for j in range(nmol):
    ax0 = plt.subplot(nmol, 1, j+1, zorder=-10)
    rect = ax0.get_position().extents
    axes = [
        mp.subplotter(rect, 0.0, i+1, nx=nphase, ny=1)
        for i in range(nphase)]
    ax0.plot(plot_phase, np.log10(plot_q[iq[j]]), c=rc[j], lw=1.5)
    ax0.set_xticks(np.linspace(0,1,5))
    ax0.set_xlim(-obs_phase[1]/2, 1-obs_phase[1]/2)
    ax0.tick_params(axis='y', direction='in')
    ax0.set_ylim(ranges[j])
    ax0.set_ylabel(f'$\\log_{{10}}(X_{{\\rm {labels[j]} }})$')
    if j == nmol-1:
        ax0.set_xlabel('Orbital phase')
    if j == 0:
        ax0.set_yticks([-4, -3, -2])
    for i in range(nphase):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_frame_on(False)
        axes[i].plot([0,0], ranges[j], lw=0.75, c='0.8', zorder=-3)
        axes[i].plot([0],  np.log10(Q[iq[j],i]), 'o', ms=4, c=rc[j])
        axes[i].plot(-post[0,i,j], x[0,i,j], 'k', lw=lw)
        axes[i].plot( post[1,i,j], x[1,i,j], themes[j], lw=lw)
        axes[i].fill_betweenx(xpdf[0,i,j], 0, -fpdf[0,i,j], where=hpd[0,i,j],
              facecolor='0.3', edgecolor='none', interpolate=False,
              zorder=-2, alpha=0.7-0.35*(quantile[0,j,i]>0.7))
        axes[i].fill_betweenx(xpdf[1,i,j], 0, fpdf[1,i,j], where=hpd[1,i,j],
              facecolor=hpdc[j], edgecolor='none', interpolate=False,
              zorder=-2, alpha=0.6-0.35*(quantile[1,j,i]>0.7))
        axes[i].set_xlim(-1, 1)
        axes[i].set_ylim(ranges[j])
plt.savefig('plots/model_WASP43b_retrieved_abundances.pdf')
