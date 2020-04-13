import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../rate')
import rate


# Bicolor map:
viridis = plt.cm.get_cmap('viridis_r', 256)
gray = plt.cm.get_cmap('gray', 256)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[0:128] = gray(np.linspace(0.0, 0.825, 128))
newcolors[128:256] = viridis(np.linspace(0.0, 1.0, 128))
bicolor = matplotlib.colors.ListedColormap(newcolors)



with np.load('WASP43b_3D_temperature_madhu_model.npz') as gcm_data:
    temps = gcm_data['temp']
    tpars = gcm_data['tpars']
    press = gcm_data['press']
    lats  = gcm_data['lat']
    lons  = gcm_data['lon']
nlat = len(lats)
nlon = len(lons)
phase = 0.5 - np.radians(lons)/(2*np.pi)

# Initialize object with solar composition:
r = rate.Rate(C=2.5e-4, N=1.0e-4, O=5.0e-4, fHe=0.0851)
ipress = 30  # ~0.1 bar
ilat   =  5  # -60 deg  (Nice switch from CO--CH4 dominated atm)
p = np.tile(press[ipress], nlon)  # bars
rtemp = temps[:,ilat,ipress]     # kelvin

Q = r.solve(rtemp, p)
rate_spec = np.array([0, 1, 2, 3, 9])
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
for j,ilat in enumerate(lat_index):
    ax = fig.add_axes([0.13, bot+1.05*dy*j, dx, dy])
    for i in range(nlon)[::2]:
        plt.semilogy(temps[i,ilat], press, lw=2.0, c=bicolor(phase[i]))
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
plt.savefig(f'../plots/model_WASP43b_temperature_profiles.pdf')


# Abundances at lat = -60 deg
plt.figure(4)
plt.clf()
ax = plt.subplot(111)
for j,ir in enumerate(rate_spec):
    ax.plot(phase, Q[ir], 'o-', c=rc[j], ms=4, label=r.species[ir])
ax.tick_params(labelsize=fs-1)
plt.yscale('log')
ax.set_xlabel("Orbital Phase", fontsize=fs)
ax.set_ylabel('Volume mixing fraction', fontsize=fs)
ax.set_xlim(0, 1)
ax.set_ylim(3e-11, 3)
ax.legend(loc='upper right', fontsize=fs-2)
par = ax.twiny()
par.set_xticks(np.arange(-180,181, 60))
par.set_xlim(180, -180)
par.tick_params(labelsize=fs-1)
par.set_xlabel('Longitude (deg)', fontsize=fs)
plt.tight_layout()
plt.savefig(f'../plots/model_WASP43b_abundances_60north.pdf')


