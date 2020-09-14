import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gaussf
import scipy.interpolate as si

sys.path.append('rate')
import rate

import pyratbay as pb
import pyratbay.tools as pt
import pyratbay.plots as pp
import pyratbay.constants as pc

import mc3
import mc3.plots as mp
import mc3.utils as mu


def posterior_pt(posterior, tmodel, tpars, ifree, pressure):
    """
    Plot the posterior PT profile.

    Parameters
    ----------
    posterior: 2D float ndarray
        MCMC posterior distribution for tmodel (of shape [nparams, nfree]).
    tmodel: Callable
        Temperature-profile model.
    tpars: 1D float ndarray
        Temperature-profile parameters (including fixed parameters).
    ifree: 1D bool ndarray
        Mask of free (True) and fixed (False) parameters in tpars.
        The number of free parameters must match nfree in posterior.
    pressure: 1D float ndarray
        The atmospheric pressure profile in barye.

    Returns
    -------
    ax: AxesSubplot instance
        The matplotlib Axes of the figure.
    """
    nlayers = len(pressure)

    u, uind, uinv = np.unique(posterior[:,0], return_index=True,
        return_inverse=True)
    nsamples = len(u)

    # Evaluate posterior PT profiles:
    profiles = np.zeros((nsamples, nlayers), np.double)
    for i in range(nsamples):
        tpars[ifree] = posterior[uind[i]]
        profiles[i] = tmodel(tpars)

    # Get percentiles (for 1,2-sigma boundaries and median):
    low1   = np.zeros(nlayers, np.double)
    low2   = np.zeros(nlayers, np.double)
    median = np.zeros(nlayers, np.double)
    high1  = np.zeros(nlayers, np.double)
    high2  = np.zeros(nlayers, np.double)
    for i in range(nlayers):
        tpost = profiles[uinv,i]
        low2[i]   = np.percentile(tpost,  2.275)
        low1[i]   = np.percentile(tpost, 15.865)
        median[i] = np.percentile(tpost, 50.000)
        high1[i]  = np.percentile(tpost, 84.135)
        high2[i]  = np.percentile(tpost, 97.725)
    return median, low1, high1, low2, high2


# Bicolor map:
viridis = plt.cm.get_cmap('viridis_r', 256)
gray = plt.cm.get_cmap('gray', 256)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[0:128] = gray(np.linspace(0.0, 0.825, 128))
newcolors[128:256] = viridis(np.linspace(0.0, 1.0, 128))
bicolor = matplotlib.colors.ListedColormap(newcolors)



with np.load('run_simulation/WASP43b_3D_temperature_madhu_model.npz') as gcm_data:
    temps = gcm_data['temp']
    tpars = gcm_data['tpars']
    press = gcm_data['press']
    lats  = gcm_data['lats']
    lons  = gcm_data['lons']
    obs_phase = gcm_data['obs_phase']

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
rate_spec = np.array([0, 1, 2, 3])
rc = 'navy orange limegreen r deepskyblue'.split()
labels = r.species


# Three latitudes temperature:
fs = 12
bot = 0.05
dy = 0.267
dx = 0.83
lat_index = 5, 10, 15
lat_labels = -60, -30, 0

fig = plt.figure(32, (6, 10))
plt.clf()
for j,ilat in enumerate([0,1]):
    ax = fig.add_axes([0.13, bot+1.05*dy*j, dx, dy])
    for i in range(nobs):
        plt.semilogy(temps[i,ilat], press, lw=2.0, c=bicolor(obs_phase[i]))
    plt.ylim(np.amax(press), np.amin(press))
    plt.xlim(300, 2600)
    plt.ylabel('Pressure (bar)', fontsize=fs)
    plt.text(0.98, 0.98, rf'Lat = ${lat_labels[j]}\degree$',
        ha='right', va='top', transform=ax.transAxes, fontsize=fs)
    if j == 0:
        plt.xlabel('Temperature (K)', fontsize=fs)
    else:
        ax.set_xticklabels([])
    ax.tick_params(labelsize=fs-1)

ax1 = fig.add_axes([0.13, 0.93, dx, 0.025])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
cb1 = matplotlib.colorbar.ColorbarBase(
    ax1, cmap=bicolor, norm=norm, orientation='horizontal')
cb1.set_label("Orbital Phase", fontsize=fs)
par = ax1.twiny()
par.set_xlim(180.0, -180.0)
par.set_xticks(np.linspace(180.0, -180.0, 7))
par.set_xlabel('Longitude (deg)', fontsize=fs)
ax1.tick_params(labelsize=fs-1)
par.tick_params(labelsize=fs-1)
#plt.savefig(f'../plots/model_WASP43b_temperature_profiles.pdf')


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Temperature profile posteriors:
integrated = [
    'mcmc_model_WASP43b_integrated_phase0.00.cfg',
    'mcmc_model_WASP43b_integrated_phase0.12.cfg',
    'mcmc_model_WASP43b_integrated_phase0.25.cfg',
    'mcmc_model_WASP43b_integrated_phase0.38.cfg',
    'mcmc_model_WASP43b_integrated_phase0.50.cfg',
    'mcmc_model_WASP43b_integrated_phase0.62.cfg',
    'mcmc_model_WASP43b_integrated_phase0.75.cfg',
    'mcmc_model_WASP43b_integrated_phase0.88.cfg',
    ]

resolved = [
    'mcmc_model_WASP43b_resolved_phase0.00.cfg',
    'mcmc_model_WASP43b_resolved_phase0.12.cfg',
    'mcmc_model_WASP43b_resolved_phase0.25.cfg',
    'mcmc_model_WASP43b_resolved_phase0.38.cfg',
    'mcmc_model_WASP43b_resolved_phase0.50.cfg',
    'mcmc_model_WASP43b_resolved_phase0.62.cfg',
    'mcmc_model_WASP43b_resolved_phase0.75.cfg',
    'mcmc_model_WASP43b_resolved_phase0.88.cfg',
    ]

nret = len(resolved)
nmol = 4
ret_phase = obs_phase[::2]
tpost = [None] * nret


tpost_int = np.zeros((nret,5,nlayers))
for i in range(nret):
    with pt.cd('run_jwst'):
        pyrat = pb.run(integrated[i], True, True)
    with np.load(pyrat.ret.mcmcfile) as mcmc:
        posterior, zchain, zmask = mc3.utils.burn(mcmc)

    ifree = pyrat.ret.pstep[pyrat.ret.itemp] > 0
    itemp = np.arange(np.sum(ifree))
    tpost_int[i] = np.array(posterior_pt(
        posterior[:,itemp], pyrat.atm.tmodel,
        pyrat.ret.params[pyrat.ret.itemp], ifree, pyrat.atm.press))

tpost_res = np.zeros((nret,5,nlayers))
for i in range(nret):
    with pt.cd('run_jwst'):
        pyrat = pb.run(resolved[i], True, True)
    with np.load(pyrat.ret.mcmcfile) as mcmc:
        posterior, zchain, zmask = mc3.utils.burn(mcmc)

    ifree = pyrat.ret.pstep[pyrat.ret.itemp] > 0
    itemp = np.arange(np.sum(ifree))
    tpost_res[i] = np.array(posterior_pt(
        posterior[:,itemp], pyrat.atm.tmodel,
        pyrat.ret.params[pyrat.ret.itemp], ifree, pyrat.atm.press))


plt.figure(1500, (8,8))
plt.clf()
for i in range(nret):
    ax = plt.subplot(4, 2, 2*i+1 - 7*(i>=4))
    ax = pp.temperature(pyrat.atm.press, #median=tpost_int[0],
        bounds=tpost_int[i,1:], ax=ax, alpha=[1.0,0.6])
    pp.temperature(pyrat.atm.press, #median=tpost_int[0],
        bounds=tpost_res[i,1:], ax=ax, theme='orange', alpha=[0.5, 0.3])
    for k in range(nobs):
        if obs_phase[k] == ret_phase[i]:
            ax.plot(temps[k,1,:], press/pc.bar, c='r', lw=2.0, zorder=10)
        else:
            dphase = obs_phase[k] - ret_phase[i]
            dphase = np.abs(dphase - 1*(dphase>0.5) + 1*(dphase<-0.5))
            alpha = 0.1 + 0.6 * (dphase<0.25) *(0.25-dphase)/0.25
            ax.plot(temps[k,1,:], press/pc.bar, alpha=alpha, lw=1.5, c='0.2')
    ax.set_xlim(200, 2600)
    ax.text(0.98, 0.9, f'phase = {phase[i*2]:.2f}',
        transform=ax.transAxes, ha='right')
    ax.legend_.remove()
    if i >= 4:
        ax.set_ylabel('')
    if (i+1)%4 != 0:
        ax.set_xlabel('')
        ax.set_xticklabels([])
    plt.subplots_adjust(0.1, 0.08, 0.98, 0.98, wspace=0.18, hspace=0.1)
    plt.savefig('plots/model_WASP43b_retrieved_temperature.pdf')



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Collect posteriors:
integrated = [
    'run_jwst/MCMC_model_WASP43b_run01_integrated_phase0.00.npz',
    'run_jwst/MCMC_model_WASP43b_run01_integrated_phase0.12.npz',
    'run_jwst/MCMC_model_WASP43b_run01_integrated_phase0.25.npz',
    'run_jwst/MCMC_model_WASP43b_run01_integrated_phase0.38.npz',
    'run_jwst/MCMC_model_WASP43b_run01_integrated_phase0.50.npz',
    'run_jwst/MCMC_model_WASP43b_run01_integrated_phase0.62.npz',
    'run_jwst/MCMC_model_WASP43b_run01_integrated_phase0.75.npz',
    'run_jwst/MCMC_model_WASP43b_run01_integrated_phase0.88.npz',
    ]

resolved = [
    'run_jwst/MCMC_model_WASP43b_run01_resolved_phase0.00.npz',
    'run_jwst/MCMC_model_WASP43b_run01_resolved_phase0.12.npz',
    'run_jwst/MCMC_model_WASP43b_run01_resolved_phase0.25.npz',
    'run_jwst/MCMC_model_WASP43b_run01_resolved_phase0.38.npz',
    'run_jwst/MCMC_model_WASP43b_run01_resolved_phase0.50.npz',
    'run_jwst/MCMC_model_WASP43b_run01_resolved_phase0.62.npz',
    'run_jwst/MCMC_model_WASP43b_run01_resolved_phase0.75.npz',
    'run_jwst/MCMC_model_WASP43b_run01_resolved_phase0.88.npz',
    ]


nret = len(resolved)
nmol = 4
ret_phase = obs_phase[::2]
post = [None] * nret
ipost = [None] * nret

bestp = np.zeros((2, nret, nmol))
for i in range(nret):
    with np.load(resolved[i]) as d:
        posterior, zchain, zmask = mu.burn(d)
        post[i] = posterior[:,-4:]
        bestp[0,i] = d['bestp'][-4:]

for i in range(nret):
    with np.load(integrated[i]) as d:
        posterior, zchain, zmask = mu.burn(d)
        ipost[i] = posterior[:,-4:]
        bestp[1,i] = d['bestp'][-4:]

themes = ['blue', 'green', 'red', 'orange']
rc = 'navy darkgreen red darkorange'.split()
hpdc = 'cornflowerblue limegreen red gold'.split()
iq = [0, 2, 3, 1]
labels = 'H2O CO CO2 CH4'.split()
ranges = [(-5, -1), (-12, -1), (-12,-5), (-12, -1)]
ranges = [(-4.5, -1), (-9, -1), (-11,-5), (-9, -1)]

plot_phase = np.concatenate([phase-1, phase, phase+1])
plot_q = np.hstack([Q,Q,Q])


quantile = np.tile(0.683, (2,4,nret))
quantile[:,1,0] = quantile[:,1,7] = quantile[1,1,6] = 0.9545
quantile[:,2,0] = quantile[:,2,7] = quantile[1,2,6] = 0.9545
quantile[1,3,2] = quantile[:,3,3] = 0.9545

for j in range(4):
    plt.figure(40+j)
    plt.clf()
    rect = [0.1, 0.1, 0.95, 0.95]
    axes = [mp.subplotter(rect, 0.0, i+1, nx=nret, ny=1) for i in range(nret)]

    ax0 = plt.axes([0.1, 0.1, 0.85, 0.85], zorder=-10)
    ax0.plot(plot_phase, np.log10(plot_q[iq[j]]), c=rc[j], lw=2.0)
    ax0.set_xticks(np.linspace(0,1,5))
    ax0.set_xlim(-phase[1], 1-phase[1])
    ax0.set_ylim(ranges[j])
    ax0.set_ylabel(f'{labels[j]} log(volume mixing ratio)')
    ax0.set_xlabel('Orbital phase')
    for i in range(nret):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_frame_on(False)
        axes[i].plot([0,0], ranges[j], lw=0.75, c='0.8')
        axes[i].plot([0],  np.log10(Q[iq[j],2*i]), 'o', ms=5, c=rc[j])
        vals, bins = np.histogram(
            ipost[i][:,j], bins=100, range=ranges[j], density=True)
        vals = gaussf(vals, 1.5)
        vals = vals/np.amax(vals) * 0.8
        bins = 0.5*(bins[1:]+bins[:-1])
        axes[i].plot(-vals, bins, 'k')
        PDF, Xpdf, HPDmin = mc3.stats.cred_region(ipost[i][:,j],quantile[0,j,i])
        f = si.interp1d(bins, -vals, kind='nearest', bounds_error=False)
        axes[i].fill_betweenx(Xpdf, 0, f(Xpdf), where=PDF>=HPDmin,
              facecolor='0.3', edgecolor='none',
              interpolate=False, zorder=-2, alpha=0.4)

        vals, bins = np.histogram(
            post[i][:,j], bins=100, range=ranges[j], density=True)
        vals = gaussf(vals, 1.5)
        vals = vals/np.amax(vals) * 0.8
        bins = 0.5*(bins[1:]+bins[:-1])
        axes[i].plot(vals, bins, themes[j])

        PDF, Xpdf, HPDmin = mc3.stats.cred_region(post[i][:,j], quantile[1,j,i])
        f = si.interp1d(bins, vals, kind='nearest', bounds_error=False)
        axes[i].fill_betweenx(Xpdf, 0, f(Xpdf), where=PDF>=HPDmin,
              facecolor=hpdc[j], edgecolor='none',
              interpolate=False, zorder=-2, alpha=0.4)
        axes[i].set_xlim(-1, 1)
        axes[i].set_ylim(ranges[j])
    plt.savefig(f'plots/model_WASP43b_retrieved_abundance_{labels[j]}.pdf')

