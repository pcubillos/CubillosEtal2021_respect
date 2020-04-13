import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gaussf

import pyratbay.constants as pc


# Bicolor map:
viridis = plt.cm.get_cmap('viridis_r', 256)
gray = plt.cm.get_cmap('gray', 256)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[0:128] = gray(np.linspace(0.0, 0.825, 128))
newcolors[128:256] = viridis(np.linspace(0.0, 1.0, 128))
bicolor = matplotlib.colors.ListedColormap(newcolors)


with np.load('WASP43b_3D_synthetic_emission_spectra.npz') as emission_model:
    flux = emission_model['spectra']
    wl = emission_model['wavelength']
    obs_phase = emission_model['phase']
    starflux = emission_model['starflux']
    rprs = emission_model['rprs']

with np.load('WASP43b_3D_synthetic_pandexo_flux_ratios.npz') as sim:
    pwl = sim['pandexo_wl']
    pflux = sim['pandexo_flux_ratio']
    puncert = sim['pandexo_uncert']


nphase = len(obs_phase)

fs = 13
bot, dy = 0.11, 0.84
left1, left2, dx1, dx2 = 0.09, 0.41, 0.25, 0.56
logxticks = 0.6, 1.0, 1.4, 2.0, 3.0, 5.0, 7.0, 10.0
sigma = 15.0


fig = plt.figure(47, (8,5))
plt.clf()
ax = fig.add_axes([0.11, bot, 0.77, dy])
plt.xscale('log')
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks(logxticks)
for iphase in range(nphase):
    plt.plot(wl, gaussf(flux[iphase],sigma), c=bicolor(obs_phase[iphase]))
plt.xlim(1.0, 6.0)
plt.xlim(0.6, 12.0)
plt.ylim(0, 90000)
plt.xlabel('Wavelength (um)', fontsize=fs)
plt.ylabel(r'Disk-integrated Flux (erg s$^{-1}$ cm$^{-2}$ cm)', fontsize=fs)

ax1 = fig.add_axes([0.9, bot, 0.025, dy])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
cb1 = matplotlib.colorbar.ColorbarBase(
    ax1, cmap=bicolor, norm=norm, ticks=np.linspace(0,1,11),
    orientation='vertical')
cb1.set_label('Orbital Phase', fontsize=fs)
cb1.update_ticks()
plt.savefig('../plots/model_WASP43b_disk_integrated_emission.pdf')


fig = plt.figure(48, (8,5))
plt.clf()
ax = fig.add_axes([0.11, bot, 0.77, dy])
plt.xscale('log')
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks(logxticks)
for iphase in range(nphase):
    flux_ratio = 1e6*flux[iphase]/starflux * rprs**2
    plt.plot(wl, gaussf(flux_ratio,sigma), c=bicolor(obs_phase[iphase]))
plt.xlim(0.6, 12.0)
plt.ylim(0, 8000)
plt.xlabel('Wavelength (um)', fontsize=fs)
plt.ylabel(r'Disk-integrated $F_{\rm p}/F_{\rm s}$ (ppm)', fontsize=fs)

ax1 = fig.add_axes([0.9, bot, 0.025, dy])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
cb1 = matplotlib.colorbar.ColorbarBase(
    ax1, cmap=bicolor, norm=norm, ticks=np.linspace(0,1,11),
    orientation='vertical')
cb1.set_label('Orbital Phase', fontsize=fs)
cb1.update_ticks()
plt.savefig('../plots/model_WASP43b_disk_integrated_flux_ratio.pdf')


sigma = 10.0
logxticks = 1.0, 1.4, 2.0, 3.0, 4.0, 5.0

col = iter('orange red limegreen blue'.split())
mcol = iter('saddlebrown maroon darkgreen navy'.split())
fig = plt.figure(101, (8,5))
plt.clf()
ax = plt.subplot(111)
for i in [0, 4, 8, 12]:
    flux_ratio = flux[i]/starflux * rprs**2
    plt.plot(wl, gaussf(flux_ratio,sigma)/pc.ppm, c='0.2', zorder=-10,lw=1)
    snr = puncert[i] < 1
    plt.errorbar(pwl[snr], pflux[i][snr]/pc.ppm, puncert[i][snr]/pc.ppm,
        fmt='o', ms=3.5, mew=0.0, label=f'phase={obs_phase[i]-0.006:.2f}',
        c=next(col), mfc=next(mcol))
plt.xlim(0.8, 5.5)
plt.ylim(-100, 5250)
plt.xlabel('Wavelength (um)', fontsize=fs)
plt.ylabel(r'Disk-integrated $F_{\rm p}/F_{\rm s}$ (ppm)',
     fontsize=fs)
ax.tick_params(labelsize=fs-1)
plt.xscale('log')
plt.legend(loc='upper left', fontsize=fs-1)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks(logxticks)
plt.tight_layout()
plt.savefig(f'../plots/model_WASP43b_pandexo_flux_ratio.pdf')

