
import sys
import warnings
import time

import numpy as np
import pyratbay as pb
import pyratbay.atmosphere as pa
import pyratbay.constants as pc
import pyratbay.blackbody as bb
import pyratbay.io as io
import trapz as t

sys.path.append('../rate')
import rate


def get_area(lon, lat):
    """
    Calculate the (unprojected) surface area of the cells on a regular
    longitude-latitude grid.

    Parameters
    ----------
    lon: 1D float ndarray
       Longitude (radians) array on planet (0 at substellar point).
    lat: 1D float ndarray
       Latitude (radians) array on planet (-pi/2 at North pole,
       pi/2 at South pole).

    Returns
    -------
    area: 2D float array
    """
    nlon, nlat = len(lon), len(lat)
    # Assume regular grid:
    dlat = np.abs(lat[1] - lat[0])
    dlon = np.abs(lon[1] - lon[0])
    # delta cosine of latitude:
    dcos_lat = np.cos(lat+np.pi/2-dlat/2.0) - np.cos(lat+np.pi/2+dlat/2.0)
    # The are of the cells:
    return np.ones((nlon, nlat)) * dlon * dcos_lat


def project(lon, lat, phase):
    """
    Project normal angle relative to an observed orbital phase:

    Parameters
    ----------
    lon: 1D float ndarray
        Longitude (radians) array on planet (0 at substellar point).
    lat: 1D float ndarray
        Latitude (radians) array on planet (-pi/2 at North pole,
        pi/2 at South pole).
    phase: Scalar or 1D float ndarray
        Observed orbital phase (0.0 at transit, 0.5 at eclipse).

    Returns
    -------
    mu: 3D float ndarray
        Cosine of normal vector to observer of shape [nlon,nlat]
        (1.0 towards observer, -1.0 away from observer).
    """
    # Latitude and Longitude as 3D arrays:
    latitude  = lat[np.newaxis,np.newaxis,:]
    longitude = lon[np.newaxis,:,np.newaxis]

    # Orbital phase to radians:
    obs_lon = np.pi - 2*np.pi*np.atleast_1d(phase)[:,np.newaxis,np.newaxis]
    # Normal vector of the cells in cartesian coordinates
    mu = np.cos(latitude) * np.cos(longitude-obs_lon)
    return mu


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Now a 3D planet:

with np.load('WASP43b_3D_temperature_madhu_model.npz') as gcm_data:
    temps = gcm_data['temp']
    tpars = gcm_data['tpars']
    press = gcm_data['press']
    lats  = gcm_data['lat']
    lons  = gcm_data['lon']
nlat = len(lats)
nlon = len(lons)

# Initialize object with solar composition:
r = rate.Rate(C=2.5e-4, N=1.0e-4, O=5.0e-4, fHe=0.0851)
ipress = 30  # ~0.1 bar
ilat   =  5  # -60 deg  (Nice switch from CO--CH4 dominated atm)
p = np.tile(press[ipress], nlon)  # bars
rtemp = temps[:,ilat,ipress]     # kelvin

Q = r.solve(rtemp, p)
labels = r.species


pyrat = pb.run('model_WASP43b.cfg')
press = pyrat.atm.press/pc.bar
wl = 1/(pyrat.spec.wn*pc.um)
rprs = pyrat.phy.rplanet/pyrat.phy.rstar
q0 = np.copy(pyrat.atm.qbase)
imol = [labels.index(molfree) for molfree in pyrat.atm.molfree]
nwave = pyrat.spec.nwave

ilon_obs = np.arange(0, nlon, 4)
obs_phase = 0.5 - np.radians(lons[ilon_obs])/(2*np.pi)
nphase = len(obs_phase)
mu = project(np.radians(lons), np.radians(lats), phase=obs_phase)
area = get_area(np.radians(lons), np.radians(lats))
pyrat.log.verb = 0

t0 = time.time()
flux = np.zeros((nphase,nwave))
for ilon in range(nlon):
    vis_phase = mu[:,ilon,0] > 0
    if not np.any(vis_phase):
        continue
    iphase = np.where(vis_phase)[0]
    q2 = pa.qscale(
        q0, pyrat.mol.name, pyrat.atm.molmodel,
        pyrat.atm.molfree, np.log10(Q[imol,ilon]),
        pyrat.atm.bulk, iscale=pyrat.atm.ifree, ibulk=pyrat.atm.ibulk,
        bratio=pyrat.atm.bulkratio, invsrat=pyrat.atm.invsrat)
    for ilat in range(nlat//2):
        temp = temps[ilon,ilat]

        status = pb._ra.update_atm(pyrat, temp, q2, None)
        pb._cs.interpolate(pyrat)
        pb._ray.absorption(pyrat)
        pb._al.absorption(pyrat)
        pb._od.opticaldepth(pyrat)
        bb.blackbody_wn_2D(
            pyrat.spec.wn, pyrat.atm.temp, pyrat.od.B, pyrat.od.ideep)
        intensity = t.intensity(
            pyrat.od.depth, pyrat.od.ideep, pyrat.od.B,
            mu[iphase,ilon,ilat], pyrat.atm.rtop)
        flux[iphase] += 2*intensity * mu[iphase,ilon,ilat,np.newaxis] \
                        * area[ilon,ilat]
    print(f'Longitude [{ilon}]: {lons[ilon]:.1f} deg')
t1 = time.time()
print(t1-t0)

# Save model emission spectra to file:
np.savez('WASP43b_3D_synthetic_emission_spectra.npz',
    spectra=flux,
    wavelength=1e4/pyrat.spec.wn,
    starflux=pyrat.spec.starflux,
    rprs=rprs,
    phase=obs_phase)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Pandexo these monsters:

with np.load('WASP43b_3D_synthetic_emission_spectra.npz') as emission_model:
    flux = emission_model['spectra']
    wl = emission_model['wavelength']
    obs_phase = emission_model['phase']

pyrat = pb.run('model_WASP43b.cfg')
rprs = pyrat.phy.rplanet/pyrat.phy.rstar
nphase = len(obs_phase)

# Show JWST instruments:
# jdi.print_instruments()

# Hellier (2011)
period = 0.813475 * pc.day
tdur = 1.1592 * pc.hour

transit_duration = period/nphase
baseline = 2 * tdur
Kmag = 9.267
metal = 0.0

instruments = [
    'NIRISS SOSS',
#    'NIRSpec Prism',
#    'NIRSpec G140H',
#    'NIRSpec G235H',
    'NIRSpec G395H',
#    'NIRCam F322W2',
#    'NIRCam F444W',
    ]
# SOSS gives lower noise than PRISM
# PRISM reaches 0.6 um, but there's too little flux already.
# G395H is enough to complement SOSS
# So, I'll stich to those two

resolution = 100.0
noise = [20.0, 30.0]

pyrat.ncpu = 5  # 24 was breaking pandexo/pandeia
for k in range(3):
    pandexo_wl = []
    pandexo_flux = []
    pandexo_uncert = []
    for iphase in range(8, nphase):
        print(f'\nThis is phase {iphase+1}/{nphase}:')
        pyrat.spec.spectrum = flux[iphase]
        pflux, puncert, pwl = [], [], []
        for instrument, noise_floor in zip(instruments, noise):
            save_file = f'pandexo_WASP43b_phase{obs_phase[iphase]:.2f}_{"-".join(instrument.split())}.p'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pandexo_sim = io.export_pandexo(
                    pyrat, baseline, transit_duration,
                    Kmag=Kmag, metal=metal, instrument=instrument, n_transits=1,
                    resolution=resolution, noise_floor=noise_floor,
                    save_file=save_file)
            pwl.append(pandexo_sim[1][0])
            pflux.append(pandexo_sim[2][0])
            puncert.append(pandexo_sim[3][0])
        pandexo_wl    .append(pwl)
        pandexo_flux  .append(pflux)
        pandexo_uncert.append(puncert)

    # All phases have same wavelength, so, keep only the first one:
    pandexo_wl = pandexo_wl[0]

    pwl = np.concatenate(pandexo_wl)
    pflux   = np.array([np.concatenate(pflux) for pflux in pandexo_flux])
    puncert = np.array([np.concatenate(punc) for punc in pandexo_uncert])
    np.savez(f'WASP43b_3D_synthetic_pandexo_flux_ratios_run0{k+1}.npz',
        flux_ratio=flux/pyrat.spec.starflux * rprs**2,
        wavelength=wl,
        stellar_flux=pyrat.spec.starflux,
        pandexo_flux_ratio=pflux,
        pandexo_uncert=puncert,
        pandexo_wl=pwl,
        phase=obs_phase,
        flux_units='erg s-1 cm-2 cm',
        wl_units='micron')


