import matplotlib
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gaussf
import matplotlib.pyplot as plt
plt.ion()

import pyratbay.atmosphere as pa
import pyratbay.constants as pc
import pyratbay.io as io

import mc3
import mc3.plots as mp


retrievals = [
    'run_integrated/MCMC_WASP43b_east_integrated.pickle',
    'run_integrated/MCMC_WASP43b_day_integrated.pickle',
    'run_integrated/MCMC_WASP43b_west_integrated.pickle',
    'run_resolved/MCMC_WASP43b_east_resolved.pickle',
    'run_resolved/MCMC_WASP43b_day_resolved.pickle',
    'run_resolved/MCMC_WASP43b_west_resolved.pickle',
    ]


bestp, best_spec, data, uncert = [], [], [], []
median, low, high = [], [], []
bestPT, pt_low, pt_high, pt_median = [], [], [], []
posterior = []
for cfg in retrievals:
    pyrat = io.load_pyrat(cfg)
    median.append(pyrat.ret.spec_median)
    low.append(pyrat.ret.spec_low1)
    high.append(pyrat.ret.spec_high1)
    with np.load(pyrat.ret.mcmcfile) as mcmc:
        bestp.append(mcmc["bestp"])
        best_spec.append(pyrat.eval(mcmc["bestp"])[0])

    data.append(pyrat.obs.data)
    uncert.append(pyrat.obs.uncert)
    posterior.append(pyrat.ret.posterior)
    # Posterior PT profiles:
    ifree = pyrat.ret.pstep[pyrat.ret.itemp] > 0
    itemp = np.arange(np.sum(ifree))
    tpost = pa.temperature_posterior(
        pyrat.ret.posterior[:,itemp], pyrat.atm.tmodel,
        pyrat.ret.params[pyrat.ret.itemp], ifree, pyrat.atm.press)
    pt_median.append(tpost[0])
    pt_low.append(tpost[1])
    pt_high.append(tpost[2])


pmedians = [np.median(post, axis=0) for post in posterior]
plows  = [np.percentile(post, 15.865, axis=0) for post in posterior]
phighs = [np.percentile(post, 84.135, axis=0) for post in posterior]

for pmedian, plow, phigh in zip(pmedians, plows, phighs):
    print("")
    for med,lo,hi in zip(pmedian, plow, phigh):
        _low = med-lo
        _high = hi-med
        print(f'${med:.1f}_{{-{_low:.1f}}}^{{+{_high:.1f}}}$')


dwest_posteriors = posterior[0][:,3:7]
dday_posteriors  = posterior[1][:,3:7]
deast_posteriors = posterior[2][:,3:7]

swest_posteriors = posterior[3][:,3:7]
sday_posteriors  = posterior[4][:,3:7]
seast_posteriors = posterior[5][:,3:7]


bandwl = 1e4/pyrat.obs.bandwn
pressure = pyrat.atm.press/pc.bar
sflux = pyrat.spec.starflux
rprs = pyrat.phy.rplanet/pyrat.phy.rstar
wl = 1.0/(pyrat.spec.wn*pc.um)

labels = [
    "phase = 0.25",
    "phase = 0.50",
    "phase = 0.75",
    ]

all_labels = [
    "phase=0.25, disk",
    "phase=0.50, disk",
    "phase=0.75, disk",
    "phase=0.25, resolved",
    "phase=0.50, resolved",
    "phase=0.75, resolved",
    ]


#scolors = ["cornflowerblue", "limegreen", "darkorange"]
scolors = ["royalblue", "limegreen", "darkorange"]
dcolors = ["mediumblue", "forestgreen", "red"]
ecolors = "blue green sienna navy darkgreen maroon".split()

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

xt, yt = 0.7, 0.4
dxt = 1.0 - xt - 0.01
dyt = 0.295

xs, ys = 0.062, 0.03
dxs = 0.62 - xs
dys = 1.0/4 - ys - 0.03

rect1 = 0.03, 0.32, 0.62, 0.52
rect2 = 0.15, 0.04, 0.45, 0.265
rect3 = 0.60, 0.04, 0.90, 0.265
margin = 0.01
ranges = [(-6, -1), (-12, -1), (-12, -1), (-12, -1)]


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#plt.close(3)
plt.figure(3, (8.5, 9.5))
plt.clf()
ax = plt.axes([xt, 0.682, dxt, dyt])
for tm, tl, th, col, lab in \
        zip(pt_median, pt_low, pt_high, dcolors, labels):
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
ax.set_title('Disk integrated', fontsize=fs, weight='bold')

ax = plt.axes([xt, 0.312, dxt, dyt])
for tm, tl, th, col, lab in \
        zip(pt_median[3:], pt_low[3:], pt_high[3:], scolors, labels):
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
ax.set_title('Longitudinally resolved', fontsize=fs, weight='bold')

# Spectra:
legs = []
for bflux, lo, hi, datum, err, col, ecol in \
      zip(median, low, high, data, uncert, dcolors+scolors, ecolors):
    offset = 1.0025 if col not in dcolors else 1.0
    alpha = 0.25 if col not in dcolors else 0.2
    # WFC3
    ax = plt.axes([xs, 0.57, dxs, dys])
    ax.tick_params(labelsize=fs, direction='in')
    ax.plot(wl, f0*gaussf(bflux/sflux*rprs**2, sigma), c=col, lw=lw)
    ax.fill_between(
        wl, f0*gaussf(lo/sflux*rprs**2, sigma),
            f0*gaussf(hi/sflux*rprs**2, sigma),
        facecolor=col, edgecolor='none', alpha=alpha)
    ax.errorbar(bandwl*offset, f0*datum, f0*err, fmt="o", c=col, zorder=100,
                 ms=4.0, mew=0.5, mec="k", ecolor=ecol)
    ax.set_xlim(1.12, 1.66)
    ax.set_ylim(0.0, 0.9)
    ax.set_xlabel("Wavelength (um)", fontsize=fs)
    ax.set_ylabel(r"$F_{\rm p}/F_{\rm s}$ (ppt)", fontsize=fs)
    # WFC3 + Spitzer
    ax = plt.axes([xs, 0.805, dxs, dys])
    ax.tick_params(labelsize=fs, direction='in')
    line, = ax.plot(wl, f0*gaussf(bflux/sflux*rprs**2, sigma), c=col, lw=lw)
    fbet = ax.fill_between(
        wl, f0*gaussf(lo/sflux*rprs**2, sigma),
            f0*gaussf(hi/sflux*rprs**2, sigma),
        facecolor=col, edgecolor='none', alpha=alpha)
    eb = ax.errorbar(
        bandwl*offset, f0*datum, f0*err, fmt="o", c=col, zorder=100,
        ms=4.0, mew=0.5, mec="k", ecolor=ecol)
    ax.set_xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([1.1, 1.4, 1.7, 2, 2.4, 3, 4, 5])
    ax.set_xlim(1.05, 5.2)
    ax.set_ylim(0.0, 5.3)
    ax.set_xlabel("Wavelength (um)", fontsize=fs)
    ax.set_ylabel(r"$F_{\rm p}/F_{\rm s}$ (ppt)", fontsize=fs)
    if col == scolors[0]:
        ax.plot(wl[pyrat.obs.bandidx[-1]], pyrat.obs.bandtrans[-1]*150, '0.5')
        ax.plot(wl[pyrat.obs.bandidx[-2]], pyrat.obs.bandtrans[-2]*150, '0.5')
    legs.append((line, fbet, eb))
legend1 = plt.legend(
    legs[0:3], labels, fontsize=fs-2, title='disk integ.', loc=(0.015,0.58))
legend2 = plt.legend(
    legs[3:6], labels, fontsize=fs-2, title='long. resolved', loc=(0.27,0.58))
plt.gca().add_artist(legend1)


# Volume mixing ratios
axes = [mc3.plots.subplotter(rect1, margin, i, nx=4, ny=3, ymargin=0.0)
    for i in range(1,13)]
margin = 0.01
mp.histogram(deast_posteriors, ranges=ranges, axes=axes[0: 4], theme=themes[0])
mp.histogram(seast_posteriors, ranges=ranges, axes=axes[0: 4], theme=themes[1])
mp.histogram(dday_posteriors,  ranges=ranges, axes=axes[4: 8], theme=themes[2])
mp.histogram(sday_posteriors,  ranges=ranges, axes=axes[4: 8], theme=themes[3])
mp.histogram(dwest_posteriors, ranges=ranges, axes=axes[8:12], theme='red')
mp.histogram(swest_posteriors, ranges=ranges, axes=axes[8:12], theme='orange',
    pnames=pnames, fs=fs)
for i in range(8):
    axes[i].set_xticklabels([])
for i in range(8,12):
    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=0)
for i in [0,4]:
    axes[i].set_xticks([-6, -4, -2])

axes, cb = mp.pairwise(
    dday_posteriors, ranges=ranges, pnames=pnames, rect=rect2,
    nbins=20, margin=0.005, fs=fs-0.5, palette=plt.cm.YlGn)
for ax in axes[:,2]:
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
cb.ax.set_position([0.44, 0.13, 0.01, 0.1])
axes[0,0].text(
    -1.05, 2.95, labels[1]+', disk integ.',
    transform=ax.transAxes, weight='bold')
tcb = cb.ax.twinx()
tcb.set_yticks([])

axes, cb = mp.pairwise(
    dwest_posteriors, ranges=ranges, pnames=pnames, rect=rect3,
    nbins=20, margin=0.005, fs=fs-0.5, palette=plt.cm.YlOrRd)
for ax in axes[:,2]:
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
cb.ax.set_position([0.88, 0.13, 0.01, 0.1])
tcb = cb.ax.twinx()
tcb.set_yticks([])
axes[0,0].text(
    -1.05, 2.95, labels[2]+', disk integ.',
    transform=ax.transAxes, weight='bold')
plt.savefig("plots/WASP43b_retrieval.pdf")

