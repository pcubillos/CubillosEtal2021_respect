import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gaussf
import pyratbay.constants as pc

sys.path.append('../rate')
import rate

# Make 3D sphere
nlon = 48
nlat = 24
n = nlon//4

u = np.linspace(0, 2 * np.pi, nlon)
v = np.linspace(0, np.pi, nlat)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

u_cell = u[34:36]
v_cell = v[8:10]
r_cell = 10.1
x_cell = r_cell * np.outer(np.cos(u_cell), np.sin(v_cell))
y_cell = r_cell * np.outer(np.sin(u_cell), np.sin(v_cell))
z_cell = r_cell * np.outer(np.ones(np.size(u_cell)), np.cos(v_cell))

r = 10.1
u_slice = u[32:39:4]
v_slice = v[0:19]
x_slice = r * np.outer(np.cos(u_slice), np.sin(v_slice))
y_slice = r * np.outer(np.sin(u_slice), np.sin(v_slice))
z_slice = r * np.outer(np.ones(np.size(u_slice)), np.cos(v_slice))

yy = np.linspace(-0.25*np.pi, 0.5*np.pi, 100)
xx = -np.cos(yy)/10.0

colors = np.zeros(z.shape + (4,))
for i in range(n):
    colors[4*i:4*(i+1)] = plt.cm.viridis(np.sin(np.pi*i/(n-1)))
wcolors = np.copy(colors)
wcolors[:,:,0:3] = 0.0
azim, elev = 22.5, 270.0


# Bicolor map:
viridis = plt.cm.get_cmap('viridis_r', 256)
gray = plt.cm.get_cmap('gray', 256)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[0:128] = gray(np.linspace(0.0, 0.825, 128))
newcolors[128:256] = viridis(np.linspace(0.0, 1.0, 128))
bicolor = matplotlib.colors.ListedColormap(newcolors)


with np.load('WASP43b_3D_temperature_madhu_model.npz') as gcm_data:
    temps = gcm_data['temp']
    press = gcm_data['press']/pc.bar
    phase = gcm_data['phase']
    obs_ilat = gcm_data['obs_ilat']

with np.load('WASP43b_3D_synthetic_emission_spectra.npz') as emission_model:
    flux = emission_model['spectra']
    wl = emission_model['wavelength']
    starflux = emission_model['starflux']
    rprs = emission_model['rprs']
    obs_phase = emission_model['obs_phase']

with np.load('WASP43b_3D_synthetic_pandexo_flux_ratios.npz') as sim:
    pwl = sim['pandexo_wl']
    pflux = sim['pandexo_flux_ratio']
    puncert = sim['pandexo_uncert']
    obs_phase = sim['phase']

nphase = len(obs_phase)

# Composition:
r = rate.Rate(C=2.5e-4, N=1.0e-4, O=5.0e-4, fHe=0.0851)
ipress = 42  # ~0.1 bar
ilat = 0  # -60 deg  (Nice switch from CO--CH4 dominated atm)
p = np.tile(press[ipress], nphase)  # bars
rtemp = temps[:,ilat,ipress]     # kelvin
Q = r.solve(rtemp, p)
rate_spec = np.array([0, 1, 2, 3])
rc = 'navy orange limegreen r deepskyblue'.split()
labels = r.species

ilon = np.concatenate([[nphase-1], np.arange(nphase), [0]])
phase = obs_phase[ilon]
phase[0] -= 1
phase[-1] += 1
logQ = np.log(Q[:, ilon])



fs = 11
bot, dy = 0.10, 0.88
left1, left2, dx1, dx2 = 0.09, 0.41, 0.25, 0.56
logxticks = 0.8, 1.0, 1.4, 2.0, 3.0, 4.0, 5.0
sigma = 15.0


fig = plt.figure(47, (8.5, 4.5))
plt.clf()

ax = plt.axes([-0.06, 0.55, 0.35, 0.55], projection='3d')
ax.axis('off')
ax.view_init(azim, elev)
ax.plot_surface(
    x, y, z, facecolors=colors, linewidth=0.0, antialiased=True, zorder=3)
ax.plot_wireframe(x_cell, y_cell, z_cell, color='k', linewidth=1.0, zorder=5)
ax.plot_wireframe(
    x_slice, y_slice, z_slice, color='k', linewidth=1.0, zorder=5, ccount=0)

ax2 = plt.axes([-0.03, 0.55, 0.35, 0.88])
ax2.axis('off')
ax2.plot(xx+0.01, yy, color='0.5')
ax2.plot(-xx+0.087, 1.1*yy+0.6, color='0.5')
ax2.set_xlim(-0.22, 0.5)
ax2.set_ylim(-2.5, 13.0)
ax2.text(-0.02, -0.75, r"32x64 lat$-$lon grid", fontsize=fs-2)
ax2.text(-0.09, -1.4, "16 longitudinal slices", fontsize=fs-2)

ax = fig.add_axes([0.068, 0.11, 0.15, 0.44])
for i in range(nphase):
    plt.semilogy(temps[i,1], press, lw=1.5, c=bicolor(obs_phase[i]))
ax.set_yticks(np.logspace(2, -6, 5))
ax.set_ylim(np.amax(press), 1e-7)
ax.set_xlim(250, 2200)
ax.set_ylabel('Pressure (bar)', fontsize=fs-1)
plt.xlabel('Temperature (K)', fontsize=fs-1)
ax.tick_params(labelsize=fs-2, direction='in', which='both')

ax = fig.add_axes([0.29, bot, 0.63, dy])
plt.xscale('log')
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks(logxticks)
for iphase in range(nphase):
    ax.plot(wl, gaussf(flux[iphase],sigma)*pc.ppt, c=bicolor(obs_phase[iphase]))
ax.set_xlim(0.8, 5.5)
ax.set_ylim(0, 110)
ax.set_xlabel('Wavelength (um)', fontsize=fs)
ax.set_ylabel(
    r'Disk-integrated flux ($10^{3}$ erg s$^{-1}$ cm$^{-2}$ cm)', fontsize=fs)
ax.tick_params(labelsize=fs-1, direction='in')

ax1 = fig.add_axes([0.925, bot, 0.015, dy])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
cb1 = matplotlib.colorbar.ColorbarBase(
    ax1, cmap=bicolor, norm=norm, ticks=np.linspace(0,1,11),
    orientation='vertical')
cb1.set_label('Orbital phase', fontsize=fs)
ax1.tick_params(labelsize=fs-1, direction='in')
cb1.update_ticks()

ax = fig.add_axes([0.364, 0.69, 0.22, 0.28])
for j,ir in enumerate(rate_spec):
    ax.plot(phase, Q[ir, ilon], 'o-', c=rc[j], ms=2, label=r.species[ir])
ax.tick_params(labelsize=fs-2, direction='in')
ax.set_yscale('log')
ax.set_xlabel("Orbital Phase", fontsize=fs-1)
ax.set_ylabel(r'Volume mixing ratio   ', fontsize=fs-1)
ax.set_xticks([0, 0.5, 1.0])
ax.set_yticks([1e-3, 1e-6, 1e-9])
ax.set_xlim(0, 1)
ax.set_ylim(3e-11, 3e-3)
ax.legend(fontsize=fs-2.5, loc=(1.03,0.46),
    labelspacing=0.2,
    handlelength=1.25,
    handletextpad=0.5,
    )
plt.savefig('../plots/WASP43b_disk_integrated_model_emission.pdf')


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
fig = plt.figure(48, (8.5, 4.5))
plt.clf()

ax = plt.axes([-0.06, 0.55, 0.35, 0.55], projection='3d')
ax.axis('off')
ax.view_init(azim, elev)
ax.plot_surface(
    x, y, z, facecolors=colors, linewidth=0.0, antialiased=True, zorder=3)
ax.plot_wireframe(x_cell, y_cell, z_cell, color='k', linewidth=1.0, zorder=5)
ax.plot_wireframe(
    x_slice, y_slice, z_slice, color='k', linewidth=1.0, zorder=5, ccount=0)

ax2 = plt.axes([-0.03, 0.55, 0.35, 0.88])
ax2.axis('off')
ax2.plot(xx+0.01, yy, color='0.5')
ax2.plot(-xx+0.087, 1.1*yy+0.6, color='0.5')
ax2.set_xlim(-0.22, 0.5)
ax2.set_ylim(-2.5, 13.0)
ax2.text(-0.02, -0.75, r"32x64 lat$-$lon grid", fontsize=fs-2)
ax2.text(-0.09, -1.4, "16 longitudinal slices", fontsize=fs-2)

ax = fig.add_axes([0.068, 0.11, 0.16, 0.44])
for i in range(nphase):
    plt.semilogy(temps[i,1], press, lw=1.5, c=bicolor(obs_phase[i]))
ax.set_yticks(np.logspace(2, -6, 5))
ax.set_ylim(np.amax(press), 1e-7)
ax.set_xlim(250, 2200)
ax.set_ylabel('Pressure (bar)', fontsize=fs-1)
plt.xlabel('Temperature (K)', fontsize=fs-1)
ax.tick_params(labelsize=fs-2, direction='in', which='both')

ax = fig.add_axes([0.29, bot, 0.63, dy])
plt.xscale('log')
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks(logxticks)
for iphase in range(nphase):
    flux_ratio = flux[iphase]/starflux * rprs**2 / pc.ppt
    plt.plot(wl, gaussf(flux_ratio,sigma), c=bicolor(obs_phase[iphase]))
ax.set_xlim(0.8, 5.5)
ax.set_ylim(0, 5.3)
ax.set_xlabel('Wavelength (um)', fontsize=fs)
ax.set_ylabel(r'Disk-integrated $F_{\rm p}/F_{\rm s}$ (ppt)', fontsize=fs)
ax.tick_params(labelsize=fs-1, direction='in')

ax1 = fig.add_axes([0.925, bot, 0.015, dy])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
cb1 = matplotlib.colorbar.ColorbarBase(
    ax1, cmap=bicolor, norm=norm, ticks=np.linspace(0,1,11),
    orientation='vertical')
cb1.set_label('Orbital phase', fontsize=fs)
ax1.tick_params(labelsize=fs-1, direction='in')
cb1.update_ticks()

ax = fig.add_axes([0.37, 0.65, 0.25, 0.3])
for j,ir in enumerate(rate_spec):
    ax.plot(phase, Q[ir, ilon], 'o-', c=rc[j], ms=2, label=r.species[ir])
ax.tick_params(labelsize=fs-2, direction='in')
ax.set_yscale('log')
ax.set_xlabel("Orbital Phase", fontsize=fs-1)
ax.set_ylabel(r'Volume mixing ratio', fontsize=fs-1)
ax.set_xticks([0, 0.5, 1.0])
ax.set_yticks([1e-3, 1e-6, 1e-9])
ax.set_xlim(0, 1)
ax.set_ylim(3e-11, 3e-3)
ax.legend(fontsize=fs-2.5, loc=(1.03,0.5),
    labelspacing=0.2,
    handlelength=1.25,
    handletextpad=0.5,
    )
plt.savefig('../plots/WASP43b_disk_integrated_model_flux_ratio.pdf')
