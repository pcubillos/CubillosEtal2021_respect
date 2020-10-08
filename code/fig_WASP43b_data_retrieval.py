import os

import matplotlib
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gaussf
import matplotlib.pyplot as plt
plt.ion()

import pyratbay as pb
import pyratbay.atmosphere as pa
import pyratbay.constants as pc
import pyratbay.tools as pt

import mc3
import mc3.plots as mp


retrievals = [
    'run_resolved/mcmc_WASP43b_east_resolved.cfg',
    'run_resolved/mcmc_WASP43b_day_resolved.cfg',
    'run_resolved/mcmc_WASP43b_west_resolved.cfg',
    'run_integrated/mcmc_WASP43b_east_integrated.cfg',
    'run_integrated/mcmc_WASP43b_day_integrated.cfg',
    'run_integrated/mcmc_WASP43b_west_integrated.cfg',
    ]


bestp, best_spec, data, uncert = [], [], [], []
bestPT, pt_low, pt_high, pt_median = [], [], [], []
posterior = []
for cfg in retrievals:
    with pt.cd(os.path.dirname(cfg)):
        pyrat = pb.run(os.path.basename(cfg), init=True, no_logfile=True)

    with np.load(pyrat.ret.mcmcfile) as mcmc:
        post, zchain, zmask = mc3.utils.burn(mcmc)
        bestp.append(mcmc["bestp"])
        best_spec.append(pyrat.eval(mcmc["bestp"])[0])

    data.append(pyrat.obs.data)
    uncert.append(pyrat.obs.uncert)
    posterior.append(post)
    # Posterior PT profiles:
    ifree = pyrat.ret.pstep[pyrat.ret.itemp] > 0
    itemp = np.arange(np.sum(ifree))
    tpost = pa.temperature_posterior(
        post[:,itemp], pyrat.atm.tmodel, pyrat.ret.params[pyrat.ret.itemp],
        ifree, pyrat.atm.press)
    pt_median.append(tpost[0])
    pt_low.append(tpost[1])
    pt_high.append(tpost[2])


medians = [np.median(post, axis=0) for post in posterior]
plows  = [np.percentile(post, 15.865, axis=0) for post in posterior]
phighs = [np.percentile(post, 84.135, axis=0) for post in posterior]

for median, plow, phigh in zip(medians, plows, phighs):
    print("")
    for med,lo,hi in zip(median, plow, phigh):
        low = med-lo
        high = hi-med
        print(f'${med:.1f}_{{-{low:.1f}}}^{{+{high:.1f}}}$')



swest_posteriors = posterior[0][:,3:7] #[:,4:8]
sday_posteriors  = posterior[1][:,3:7] #[:,4:8]
seast_posteriors = posterior[2][:,3:7] #[:,4:8]

dwest_posteriors = posterior[3][:,3:7] #[:,4:8]
dday_posteriors  = posterior[4][:,3:7] #[:,4:8]
deast_posteriors = posterior[5][:,3:7] #[:,4:8]


bandwl = 1e4/pyrat.obs.bandwn
pressure = pyrat.atm.press/pc.bar
sflux = pyrat.spec.starflux
rprs = pyrat.phy.rplanet/pyrat.phy.rstar
wl = 1.0/(pyrat.spec.wn*pc.um)

labels = [
    "Phase = 0.25 (east)",
    "Phase = 0.50 (dayside)",
    "Phase = 0.75 (west)",
    ]

dcolors = ["mediumblue", "forestgreen", "red"]
scolors = ["cornflowerblue", "limegreen", "darkorange"]

themes = [
    {'edgecolor': 'mediumblue',
     'facecolor': 'mediumblue',
     'color':     'mediumblue', },
    {'edgecolor': 'cornflowerblue',
     'facecolor': 'cornflowerblue',
     'color':     'cornflowerblue', },
    {'edgecolor': 'forestgreen',
     'facecolor': 'forestgreen',
     'color':     'forestgreen', },
    {'edgecolor': 'limegreen',
     'facecolor': 'limegreen',
     'color':     'limegreen', },
]

pnames = [
    '$\\log_{10}(X_{\\rm H2O})$',
    '$\\log_{10}(X_{\\rm CO})$',
    '$\\log_{10}(X_{\\rm CO2})$',
    '$\\log_{10}(X_{\\rm CH4})$',
    ]


fs = 10
sigma = 15.0
lw = 1.0
f0 = 1.0 / pc.ppt

temp_lims = 400, 2400
press_lims = 100, 1e-7
xt, yt = 0.7, 0.055
dxt = 1.0 - xt - 0.01
dyt = 0.5 - yt - 0.04

xs, ys = 0.062, 0.03
dxs = 0.62 - xs 
dys = 1.0/3 - ys - 0.03

rect = 0.03, yt, 0.62, 0.34
margin = 0.01
ranges = [(-6, -1), (-12, -1), (-12, -1), (-12, -1)]


plt.figure(10, (8.5, 7.5))
plt.clf()
ax = plt.axes([xt, yt+0.505, dxt, dyt])
for tm, tl, th, col, lab in \
        zip(pt_median[3:], pt_low[3:], pt_high[3:], dcolors, labels):
    plt.plot(tm, pressure, c=col, label=lab)
    ax.fill_betweenx(pressure, tl, th, facecolor=col, alpha=0.25)
ax.tick_params(labelsize=fs, direction='in')
ax.tick_params(axis='both', which='minor', length=0)
ax.set_yscale("log")
ax.set_ylim(np.amax(pressure), np.amin(pressure))
ax.set_ylim(press_lims)
ax.set_xlim(temp_lims)
ax.set_xlabel("Temperature (K)", fontsize=fs)
ax.set_ylabel("Pressure (bar)", fontsize=fs)
ax.legend(loc='upper right', fontsize=fs-1)
ax.set_title('Disk integrated', fontsize=fs)

ax = plt.axes([xt, yt, dxt, dyt])
for tm, tl, th, col, lab in \
        zip(pt_median, pt_low, pt_high, scolors, labels):
    plt.plot(tm, pressure, c=col, label=lab)
    ax.fill_betweenx(pressure, tl, th, facecolor=col, alpha=0.25)
ax.tick_params(labelsize=fs, direction='in')
ax.tick_params(axis='both', which='minor', length=0)
ax.set_yscale("log")
ax.set_ylim(np.amax(pressure), np.amin(pressure))
ax.set_ylim(press_lims)
ax.set_xlim(temp_lims)
ax.set_xlabel("Temperature (K)", fontsize=fs)
ax.set_ylabel("Pressure (bar)", fontsize=fs)
ax.legend(loc='upper right', fontsize=fs-1)
ax.set_title('Longitudinally resolved', fontsize=fs)

# Spectra:
for bflux, datum, err, col in \
      zip(best_spec, data, uncert, scolors+dcolors):
    offset = 1.002 if col not in dcolors else 1.0
    ax = plt.axes([xs, ys+0.69, dxs, dys])
    ax.tick_params(labelsize=fs, direction='in')
    ax.plot(wl, f0*gaussf(bflux/sflux*rprs**2, sigma), c=col, lw=lw)
    ax.errorbar(bandwl*offset, f0*datum, f0*err, fmt="o", c=col, zorder=100,
                 ms=4.0, mew=0.5, mec="k")
    ax.set_xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([1.1, 1.4, 1.7, 2, 2.4, 3, 4, 5])
    ax.set_xlim(1.05, 5.2)
    ax.set_ylim(0.0, 5)
    ax.set_xlabel("Wavelength (um)", fontsize=fs)
    ax.set_ylabel(r"$F_{\rm p}/F_{\rm s}$ (ppt)", fontsize=fs)
    # WFC3
    ax = plt.axes([xs, ys+0.365, dxs, dys])
    ax.tick_params(labelsize=fs, direction='in')
    ax.plot(wl, f0*gaussf(bflux/sflux*rprs**2, sigma), c=col, lw=lw)
    ax.errorbar(bandwl*offset, f0*datum, f0*err, fmt="o", c=col, zorder=100,
                 ms=4.0, mew=0.5, mec="k")
    ax.set_xlim(1.1, 1.7)
    ax.set_ylim(0.0, 0.9)
    ax.set_xlabel("Wavelength (um)", fontsize=fs)
    ax.set_ylabel(r"$F_{\rm p}/F_{\rm s}$ (ppt)", fontsize=fs)

# Volume mixing ratios
axes = [mc3.plots.subplotter(rect, margin, i, nx=4, ny=3, ymargin=0.0)
    for i in range(1,13)]
margin = 0.01
mp.histogram(dwest_posteriors, ranges=ranges, axes=axes[0: 4], theme=themes[0])
mp.histogram(swest_posteriors, ranges=ranges, axes=axes[0: 4], theme=themes[1])
mp.histogram(dday_posteriors,  ranges=ranges, axes=axes[4: 8], theme=themes[2])
mp.histogram(sday_posteriors,  ranges=ranges, axes=axes[4: 8], theme=themes[3])
mp.histogram(deast_posteriors, ranges=ranges, axes=axes[8:12], theme='red')
mp.histogram(seast_posteriors, ranges=ranges, axes=axes[8:12], theme='orange',
    pnames=pnames, fs=fs)
for i in range(8):
    axes[i].set_xticklabels([])
for i in range(8,12):
    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=0)
for i in [0,4]:
    axes[i].set_xticks([-6, -4, -2])
plt.savefig("plots/WASP43b_retrieval.pdf")


