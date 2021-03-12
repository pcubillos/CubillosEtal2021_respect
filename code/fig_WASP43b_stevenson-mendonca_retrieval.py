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
import mc3

sys.path.append('code')
from legend_handler import Disk, Resolved, Handler


pickles = [
    ['run_stevenson_mendonca/MCMC_WASP43b_integrated_phase0.25.pickle',
     'run_stevenson_mendonca/MCMC_WASP43b_integrated_phase0.50.pickle',
     'run_stevenson_mendonca/MCMC_WASP43b_integrated_phase0.75.pickle',],
    ['run_stevenson_mendonca/MCMC_WASP43b_resolved_phase0.25.pickle',
     'run_stevenson_mendonca/MCMC_WASP43b_resolved_phase0.50.pickle',
     'run_stevenson_mendonca/MCMC_WASP43b_resolved_phase0.75.pickle',],
    ]

modes = [
    'integrated',
    'resolved',
]

obs_phase = np.array([0.25, 0.50, 0.75])

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
# For stats:
npars = 8
medians = np.zeros((nmodes,nphase, npars))
p_lows  = np.zeros((nmodes,nphase, npars))
p_highs = np.zeros((nmodes,nphase, npars))

for j,i in product(range(nmodes), range(nphase)):
    pyrat = io.load_pyrat(pickles[j][i])
    with np.load(pyrat.ret.mcmcfile) as mcmc:
        posterior, zchain, zmask = mc3.utils.burn(mcmc)

    medians[j,i] = np.median(posterior, axis=0)
    plows[j,i]  = np.percentile(posterior, 15.865, axis=0)
    phighs[j,i] = np.percentile(posterior, 84.135, axis=0)

    posteriors[j,i] = posterior[:,-5:-1]
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
        posterior[:,itemp], pyrat.atm.tmodel,
        pyrat.ret.params[pyrat.ret.itemp], ifree, pyrat.atm.press)[0:3]


# Stats:
decimals = [1, 1, 0, 1, 1, 1, 1, 1]
max_name_len = max([len(name) for name in pyrat.ret.texnames])
text = ""
for k in range(npars):
    dec = decimals[k]
    text += f'{pyrat.ret.texnames[k]:{max_name_len}}'
    for t in range(nmodes*nphase):
        j, i = t%nmodes, t//nmodes
        med = medians[j,i,k]
        low = med - plows[j,i,k]
        high = phighs[j,i,k] - med
        text += f' & ${med:.{dec}f}_{{-{low:.{dec}f}}}^{{+{high:.{dec}f}}}$'
    text += ' \\\\\n'
print(text)


band_wl = 1.0 / (pyrat.obs.bandwn * pc.um)
wl = 1.0 / (pyrat.spec.wn * pc.um)
press = pyrat.atm.press

labels = [
    "Disk integrated",
    "Long. resolved",
    ]

fs = 10
ms = 4.5
lw = 1.25
sigma = 30.0

xmin1, xmax1 = 0.05, 0.68
xmin2, xmax2 = xmax1+0.075, 0.99
ymin, ymax = 0.05, 0.99
margin = 0.04
plot_phases = [3, 7, 11]
offset = [1.0, 1.0025]


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


fig = plt.figure(21, (8.5, 8.0))
plt.clf()
legs = []
for i in range(nphase):
    rect = [xmin1, ymin, xmax1, ymax]
    ax = mc3.plots.subplotter(rect, margin, i+1, nx=1, ny=3)
    rect = [xmin1+0.04, ymin+0.13, 0.48, ymax-0.005]
    margin2 = margin - (rect[3] - rect[1]) + (ymax-ymin)
    ax2 = mc3.plots.subplotter(rect, margin2, i+1, nx=1, ny=3)
    for k in range(2):
        cr = ax.fill_between(
            wl, gaussf(lo_fr[k,i], sigma)/pc.ppt,
            gaussf(hi_fr[k,i], sigma)/pc.ppt,
            facecolor=c_error[k], edgecolor='none', alpha=alpha[k])
        line, = ax.plot(
            wl, gaussf(median_fr[k,i],sigma)/pc.ppt, c=c_model[k], lw=lw)
        eb = ax.errorbar(
            band_wl*offset[k], data[k,i]/pc.ppt, uncerts[k,i]/pc.ppt,
            zorder=90, mew=0.25, mec="k",
            lw=lw, fmt='o', ms=ms, c=c_data[k], ecolor=c_model[k])
        legs.append((cr, line, eb))
        ax2.fill_between(
            wl, gaussf(lo_fr[k,i], sigma)/pc.ppt,
            gaussf(hi_fr[k,i], sigma)/pc.ppt,
            facecolor=c_error[k], edgecolor='none', alpha=alpha[k])
        ax2.plot(
            wl, gaussf(median_fr[k,i],sigma)/pc.ppt, c=c_model[k], lw=lw)
        ax2.errorbar(
            band_wl*offset[k], data[k,i]/pc.ppt, uncerts[k,i]/pc.ppt,
            zorder=90, mew=0.25, mec="k",
            lw=lw, fmt='o', ms=ms, c=c_data[k], ecolor=c_model[k])
        ax2.set_xlim(1.12, 1.65)
        ax2.set_ylim(
           np.amin(data[k,i,:-2]/pc.ppt)-0.1, np.amax(data[k,i,:-2]/pc.ppt)+0.1)
    ax.plot(wl[pyrat.obs.bandidx[-1]], pyrat.obs.bandtrans[-1]*150, '0.5')
    ax.plot(wl[pyrat.obs.bandidx[-2]], pyrat.obs.bandtrans[-2]*150, '0.5')
    ax.set_xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(labelsize=fs, direction='in')
    ax2.tick_params(labelsize=fs-1, direction='in')
    ax.set_xticks([1.1, 1.5, 2.0, 3.0, 4.0, 5.0])
    ax.set_xlim(1.05, 5.1)
    ax.set_ylim(bottom=0.0)
    ax2.text(
        0.08, 0.97, f'phase = {obs_phase[i]:.2f}', weight='bold',
        va='top', ha='left', transform=ax.transAxes)
    if i == 0:
        ax.legend(legs, labels, fontsize=fs, loc=(0.69, 0.78))
        ax.set_ylim(top=4.5)
    if i == 2:
        ax.set_xlabel("Wavelength (um)", fontsize=fs)
    ax.set_ylabel(r"$F_{\rm p}/F_{\rm s}$ (ppt)", fontsize=fs)
    if ax.get_ylim()[1] < 4:
        ylim = ax.get_ylim()
        yticks = [ytick for ytick in ax.get_yticks() if ytick%1.0==0]
        ax.set_yticks(yticks)
        ax.set_ylim(ylim)

for i in range(nphase):
    rect = [xmin2, ymin, xmax2, ymax]
    ax = mc3.plots.subplotter(rect, margin, i+1, nx=1, ny=3)
    ax.clear()
    for k in range(2):
        pp.temperature(
            pyrat.atm.press, bounds=tpost[k,i,1:3], theme=themes[k],
            alpha=[alpha[k], 0.5], ax=ax, lw=1.5, fs=fs)
        ax.plot(tpost[k,i,0], pyrat.atm.press/pc.bar, c=c_model[k], lw=1.5)
    ax.tick_params(direction='in', labelsize=fs, top=True, right=True)
    ax.set_xlim(400, 2400)
    ax.set_ylim(ymax=1e-6)
    if i != 2:
        ax.set_xlabel('')
plt.savefig('plots/WASP43b_stevenson-mendonca_spectra_temperatures.pdf')


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
themes = ['blue', 'green', 'red', 'orange']
rc = 'navy darkgreen darkred darkorange'.split()
hpdc = 'cornflowerblue limegreen red gold'.split()
molecs = 'H2O CO CO2 CH4'.split()
ranges = [(-12.0, -1.0) for _ in molecs]
ranges[0] = (-8.0, -1.0)
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
    vals = gaussf(vals, 3.0)
    vals = vals/np.amax(vals) * 0.8
    bins = 0.5 * (bins[1:] + bins[:-1])
    PDF, Xpdf, HPDmin = mc3.stats.cred_region(posteriors[j,i][:,m])
    f = si.interp1d(
        bins, vals, kind='nearest', bounds_error=False, fill_value=0.0)
    x[j,i,m] = bins
    post[j,i,m] = vals
    fpdf[j,i,m] = f(xpdf[m])


cm = plt.cm.viridis_r

lw = 0.75
pnames = [f'$\\log_{{10}}(X_{{\\rm {pname} }})$' for pname in molecs]
xmin = 0.49
xmax = 0.98
rect2 = xmin, 0.58, xmax, 0.98
rect3 = xmin, 0.08, xmax, 0.48

plt.figure(30, (8.5, 8.0))
plt.clf()
for m in range(nmol):
    ax0 = mc3.plots.subplotter(
        [0.07, 0.06, 0.4, 0.99], 0.03, m+1, nx=1, ny=nmol)
    rect = ax0.get_position().extents
    axes = [
        mc3.plots.subplotter(rect, 0.0, i+1, nx=nphase, ny=1)
        for i in range(nphase)]
    ax0.set_xticks(np.array([0.06, 0.25, 0.5, 0.75, 0.94]))
    dphase = obs_phase[1] - obs_phase[0]
    ax0.set_xlim(obs_phase[0]-dphase/2, obs_phase[-1]+dphase/2)
    ax0.tick_params(axis='both', direction='in')
    ax0.set_ylim(ranges[m])
    ax0.set_ylabel(f'$\\log_{{10}}(X_{{\\rm {molecs[m]} }})$')
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
        loc=(1.1-nphase, 0.05), fontsize=fs-1, framealpha=0.8,
        borderpad=0.25, labelspacing=0.25)

k = 0
i = 0
axes, cb = mc3.plots.pairwise(
    posteriors[k,i], ranges=ranges, pnames=pnames, rect=rect2,
    nbins=16, margin=0.008, fs=fs, palette=cm)
for ax in axes[:,2]:
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
cb.ax.set_position([xmax-0.02, 0.78, 0.02, 0.2])
axes[0,0].text(
    -1.03, 2.8, f'Disk integrated\nphase = {obs_phase[i]}',
    transform=ax.transAxes, weight='bold')
tcb = cb.ax.twinx()
tcb.set_yticks([])

i = 1
axes, cb = mc3.plots.pairwise(
    posteriors[k,i], ranges=ranges, pnames=pnames, rect=rect3,
    nbins=16, margin=0.007, fs=fs-0.5, palette=cm)
for ax in axes[:,2]:
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
cb.ax.set_position([xmax-0.02, 0.28, 0.02, 0.2])
tcb = cb.ax.twinx()
tcb.set_yticks([])
axes[0,0].text(
    -1.03, 2.8, f'Disk integrated\nphase = {obs_phase[i]}',
    transform=ax.transAxes, weight='bold')

plt.savefig('plots/WASP43b_stevenson-mendonca_abundances.pdf')

