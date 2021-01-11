import sys
import warnings
import time
import os
import pickle as pk

import numpy as np
import pyratbay as pb
import pyratbay.atmosphere as pa
import pyratbay.constants as pc
import pyratbay.spectrum as ps
import pyratbay.tools as pt
import pyratbay.io as io
import _trapz as t

sys.path.append('../rate')
import rate
sys.path.append('../pandexo')
import pandexo.engine.justplotit as jpi


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
    pressure = gcm_data['press']
    lats  = gcm_data['lats']
    lons  = gcm_data['lons']
    phase = gcm_data['phase']
    obs_phase = gcm_data['obs_phase']
    obs_ilat = gcm_data['obs_ilat']

nlat = len(lats)
nlon = len(lons)


# Initialize object with solar composition:
r = rate.Rate(C=2.5e-4, N=1.0e-4, O=5.0e-4, fHe=0.0851)
ipress = 42  # ~0.1 bar
ilat = 0  # -60 deg  (Nice switch from CO--CH4 dominated atm)
p = np.tile(pressure[ipress]/pc.bar, len(obs_phase))  # bars
rtemp = temps[:, ilat, ipress]     # kelvin
Q = r.solve(rtemp, p)
labels = r.species


pyrat = pb.run('model_WASP43b.cfg')
press = pyrat.atm.press / pc.bar
wavelength = 1.0 / (pyrat.spec.wn*pc.um)
rprs = pyrat.phy.rplanet/pyrat.phy.rstar

q0 = np.copy(pyrat.atm.qbase)
imol = [labels.index(molfree) for molfree in pyrat.atm.molfree]
nwave = pyrat.spec.nwave

nphase = len(obs_phase)
mu = project(np.radians(lons), np.radians(lats), phase=obs_phase)
area = get_area(np.radians(lons), np.radians(lats))
pyrat.log.verb = 0


ilatt = 10  # Use temperature profiles at -30 deg
obs_lat = 1
t0 = time.time()
spectra = np.zeros((nphase,nwave))
for ilon in range(nlon):
    vis_phase = mu[:,ilon,0] > 0
    if not np.any(vis_phase):
        continue
    iphase = np.where(vis_phase)[0]

    ophase = (
        obs_phase
        - (phase[ilon]<0.5)*(obs_phase-phase[ilon]> 0.5)
        + (phase[ilon]>0.5)*(obs_phase-phase[ilon]<-0.5))
    ilon_discrete = np.argmin(np.abs(phase[ilon] - ophase))

    q2 = pa.qscale(
        q0, pyrat.mol.name, pyrat.atm.molmodel,
        pyrat.atm.molfree, np.log10(Q[imol,ilon_discrete]),
        pyrat.atm.bulk, iscale=pyrat.atm.ifree, ibulk=pyrat.atm.ibulk,
        bratio=pyrat.atm.bulkratio, invsrat=pyrat.atm.invsrat)
    # Discrete TP per slice:
    temp = temps[ilon_discrete, obs_lat]
    status = pb._ra.update_atm(pyrat, temp, q2, None)
    pb._cs.interpolate(pyrat)
    pb._ray.absorption(pyrat)
    pb._al.absorption(pyrat)
    pb._od.optical_depth(pyrat)
    ps.blackbody_wn_2D(
        pyrat.spec.wn, pyrat.atm.temp, pyrat.od.B, pyrat.od.ideep)
    for ilat in range(nlat//2):
        intensity = t.intensity(
            pyrat.od.depth, pyrat.od.ideep, pyrat.od.B,
            mu[iphase,ilon,ilat], pyrat.atm.rtop)
        spectra[iphase] += 2*intensity * mu[iphase,ilon,ilat,np.newaxis] \
                        * area[ilon,ilat]
    print(f'Longitude [{ilon:2d}]: {lons[ilon]:.1f} deg')
t1 = time.time()
print(t1-t0)

# Save disk-integrated model emission spectra to file:
np.savez('WASP43b_3D_synthetic_emission_spectra.npz',
    wavelength=wavelength,
    spectra=spectra,
    starflux=pyrat.spec.starflux,
    rprs=rprs,
    obs_phase=obs_phase)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Pandexo these monsters:

with np.load('WASP43b_3D_synthetic_emission_spectra.npz') as emission_model:
    wl = emission_model['wavelength']
    spectra = emission_model['spectra']
    starflux = emission_model['starflux']
    obs_phase = emission_model['obs_phase']
    rprs = float(emission_model['rprs'])

pyrat = pb.run('model_WASP43b.cfg')
nphase = len(obs_phase)

# Show JWST instruments:
# print(jdi.print_instruments())

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
# So, I'll stitch to those two

# Note, pandexo's resolution is R = 0.5 * lambda/dlambda
resolution = 100.0
noise = [20.0, 30.0]

pyrat.ncpu = 5  # 24 was breaking pandexo/pandeia
pandexo_wl = []
pandexo_flux = []
pandexo_uncert = []
for iphase in range(nphase):
    print(f'\nThis is phase {iphase+1}/{nphase}:')
    pyrat.spec.spectrum = spectra[iphase]
    pflux, puncert, pwl = [], [], []
    for instrument, noise_floor in zip(instruments, noise):
        save_file = f'new_pandexo_WASP43b_phase{obs_phase[iphase]:.2f}_{"-".join(instrument.split())}.p'
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


filters = sorted([
    f'../inputs/filters/{filt}' for filt in os.listdir('../inputs/filters')
    if filt.startswith('JWST')])
fwn, ftrans = [], []
for filt in filters:
    wn, trans = io.read_spectrum(filt)
    fwn.append(wn)
    ftrans.append(trans)

flux_ratio = flux/starflux * rprs**2

noiseless_flux_ratio = np.array([
    pt.band_integrate(flux_rat, 1e4/wl, ftrans, fwn)
    for flux_rat in flux_ratio])


#np.savez(f'WASP43b_3D_synthetic_pandexo_flux_ratios.npz',
#    flux_ratio=flux_ratio,
#    wavelength=wl,
#    stellar_flux=pyrat.spec.starflux,
#    pandexo_flux_ratio=pflux,
#    noiseless_flux_ratio=noiseless_flux_ratio,
#    pandexo_uncert=puncert,
#    pandexo_wl=pwl,
#    phase=obs_phase,
#    flux_units='erg s-1 cm-2 cm',
#    wl_units='micron')


pfiles = sorted([
    pfile for pfile in os.listdir()
    if pfile.endswith('.p')])

pandexo_flux = []
pandexo_uncert = []
for i in range(nphase):
    print(i)
    pwl = []
    pflux = []
    puncert = []
    for j in range(2):
        with open(pfiles[2*i+j], 'br') as pick:
            out = pk.load(pick)
        x, y, err = jpi.jwst_1d_spec(
            out, R=resolution, num_tran=1, model=False, plot=False)
        pwl += list(x[0])
        pflux += list(y[0])
        puncert += list(err[0])
    pandexo_flux.append(pflux)
    pandexo_uncert.append(puncert)
pandexo_wl = np.array(pwl)
pandexo_flux = np.array(pandexo_flux)
pandexo_uncert = np.array(pandexo_uncert)


np.savez('WASP43b_3D_synthetic_pandexo_flux_ratios.npz',
    flux_ratio=flux_ratio,
    wavelength=wl,
    stellar_flux=starflux,
    noiseless_flux_ratio=noiseless_flux_ratio,
    pandexo_flux_ratio=pandexo_flux,
    pandexo_uncert=pandexo_uncert,
    pandexo_wl=pandexo_wl,
    phase=obs_phase,
    flux_units='erg s-1 cm-2 cm',
    wl_units='micron')

