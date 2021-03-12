import sys
import warnings
import time
import os

import numpy as np

import pyratbay as pb
import pyratbay.atmosphere as pa
import pyratbay.constants as pc
import pyratbay.spectrum as ps
import pyratbay.io as io
import _trapz as t

sys.path.append('../rate')
import rate
sys.path.append('../pandexo')


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
# Load 3D TP model of WASP-43b:
# Originated from WASP-34b GCM from Venot et al. (2020), but with
# simplifiled TP profiles (Madhu TP fits to GCM)
with np.load('WASP43b_3D_temperature_madhu_model.npz') as gcm_data:
    temps = gcm_data['temp']
    pressure = gcm_data['press']
    obs_ilat = gcm_data['obs_ilat']
    obs_phase = gcm_data['obs_phase']

    lats = gcm_data['lats']
    lons = gcm_data['lons']
    phase = gcm_data['phase']


nlat = len(lats)
nlon = len(lons)
nlayers = len(pressure)
nphase = len(obs_phase)

# Chemistry in equilibrium for solar composition at 0.1 bar and lat 60 deg:
r = rate.Rate(C=2.5e-4, N=1.0e-4, O=5.0e-4, fHe=0.0851)
ipress = 42  # ~0.1 bar
ilat_q = 0  # -60 deg  (Nice switch from CO--CH4 dominated atm)
p = np.tile(pressure[ipress]/pc.bar, len(obs_phase))  # bars
rtemp = temps[:, ilat_q, ipress]
Q_equil = r.solve(rtemp, p)
labels = r.species

pyrat = pb.run('model_WASP43b.cfg')
nwave = pyrat.spec.nwave
press = pyrat.atm.press / pc.bar
wavelength = 1.0 / (pyrat.spec.wn*pc.um)
rprs = pyrat.phy.rplanet/pyrat.phy.rstar
q0 = np.copy(pyrat.atm.qbase)
imol = [labels.index(molfree) for molfree in pyrat.atm.molfree]
nmol = len(pyrat.atm.molfree)

mu = project(np.radians(lons), np.radians(lats), phase=obs_phase)
area = get_area(np.radians(lons), np.radians(lats))
pyrat.log.verb = 0


# Feng et al. (2020):      H2O    CO    CO2   CH4
Q_constant = 10**np.array([-3.37, -3.7, -9.0, -9.0])

# Discrete temperature slices:
temps_02S = np.zeros((nlon,nlayers))  # 2 hemispheres (day/night)
temps_09S = np.zeros((nlon,nlayers))  # 9 slices (symmetric day--night)
temps_16S = np.zeros((nlon,nlayers))  # 16 slices (GCM asymmetric+offset)

# Discrete abundance slices:
Q_01S = np.zeros((nlon,nmol))  # Uniform
Q_16S = np.zeros((nlon,nmol))  # 16 slices (equilibrium chemistry)

# Temperature profiles at lat 30 deg:
ilat_temp = 1
for ilon in range(nlon):
    if np.abs(phase[ilon]-0.5) < 0.25:
        temps_02S[ilon] = temps[ 7,ilat_temp]
    else:
        temps_02S[ilon] = temps[15,ilat_temp]

    ophase = (
        obs_phase
        - (phase[ilon]<0.5)*(obs_phase-phase[ilon]> 0.5)
        + (phase[ilon]>0.5)*(obs_phase-phase[ilon]<-0.5))
    ilon_16 = np.argmin(np.abs(phase[ilon] - ophase))
    ilon_09 = 7 + np.abs(ilon_16-8)

    temps_09S[ilon] = temps[ilon_09,ilat_temp]
    temps_16S[ilon] = temps[ilon_16,ilat_temp]

    Q_01S[ilon] = Q_constant
    Q_16S[ilon] = Q_equil[imol, ilon_16]


models = [
    {'abund': Q_01S, 'temp': temps_02S},
    {'abund': Q_01S, 'temp': temps_09S},
    {'abund': Q_16S, 'temp': temps_16S},
]
nmodels = len(models)

spectra = np.zeros((nmodels,nphase,nwave))
for i, model in enumerate(models):
    t0 = time.time()
    abunds = model['abund']
    temps  = model['temp']
    for ilon in range(nlon):
        vis_phase = mu[:,ilon,0] > 0
        if not np.any(vis_phase):
            continue
        iphase = np.where(vis_phase)[0]

        q2 = pa.qscale(
            q0, pyrat.mol.name, pyrat.atm.molmodel,
            pyrat.atm.molfree, np.log10(abunds[ilon]),
            pyrat.atm.bulk, iscale=pyrat.atm.ifree, ibulk=pyrat.atm.ibulk,
            bratio=pyrat.atm.bulkratio, invsrat=pyrat.atm.invsrat)
        status = pb._ra.update_atm(pyrat, temps[ilon], q2, None)
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
            spectra[i, iphase] += \
                2*intensity * mu[iphase,ilon,ilat,np.newaxis] * area[ilon,ilat]
        print(f'Longitude [{ilon:2d}]: {lons[ilon]:6.1f} deg')
    t1 = time.time()
    print(t1-t0)

# Save disk-integrated model emission spectra to file:
np.savez(
    'WASP43b_3D_synthetic_emission_spectra.npz',
    wavelength=wavelength,
    spectra=spectra,
    starflux=pyrat.spec.starflux,
    rprs=rprs,
    obs_phase=obs_phase,
    abundances=np.array([Q_01S, Q_01S, Q_16S]),
    temperatures=np.array([temps_02S, temps_09S, temps_16S]),
    molecules=pyrat.atm.molfree,
    phase=phase,
)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Pandexo these monsters:

with np.load('WASP43b_3D_synthetic_emission_spectra.npz') as emission_model:
    wl = emission_model['wavelength']
    spectra = emission_model['spectra']
    starflux = emission_model['starflux']
    obs_phase = emission_model['obs_phase']
    rprs = float(emission_model['rprs'])
    abundances = emission_model['abundances']
    temperatures = emission_model['temperatures']

pyrat = pb.run('model_WASP43b.cfg')
nmodels, nphase, nwave = np.shape(spectra)

models = [
    {'abund': q, 'temp': t}
    for q,t in zip(abundances, temperatures)
]


# Show JWST instruments:
# print(jdi.print_instruments())

# Hellier (2011):
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
# So, I'll stitch those two


filters = sorted([
    f'../inputs/filters/{filt}'
    for filt in os.listdir('../inputs/filters')
    if filt.startswith('JWST')])
filter_wn, filter_trans = [], []
for filt in filters:
    wn, trans = io.read_spectrum(filt)
    filter_wn.append(wn)
    filter_trans.append(trans)

nwave_obs = len(filters)
flux_ratio = spectra / starflux * rprs**2

noiseless_flux_ratio = np.zeros((nmodels, nphase, nwave_obs))
for i in range(nmodels):
    for j in range(nphase):
        noiseless_flux_ratio[i,j] = ps.band_integrate(
            flux_ratio[i,j], 1e4/wl, filter_trans, filter_wn)


# Note, pandexo's resolution is R = 0.5 * lambda/dlambda
resolution = 100.0
noise = [20.0, 30.0]
n_transits = 1
pyrat.ncpu = 5  # 24 was breaking pandexo/pandeia

pflux = np.zeros((nmodels, nphase, nwave_obs))
puncert = np.zeros((nmodels, nphase, nwave_obs))
for i in range(nmodels):
    for iphase in range(nphase):
        if iphase > 8 and i != 2:
            pflux[i,iphase] = pflux[i,nphase-iphase]
            puncert[i,iphase] = puncert[i,nphase-iphase]
            continue
        print(f'\nThis is phase {iphase+1}/{nphase} of model {i}:')
        pyrat.spec.spectrum = spectra[i,iphase]
        pandexo_wl = []
        pandexo_flux = []
        pandexo_uncert = []
        for instrument, noise_floor in zip(instruments, noise):
            save_file = (
                f'pandexo_WASP43b_model{i:02d}_'
                f'phase{obs_phase[iphase]:.2f}_'
                f'{"-".join(instrument.split())}.p')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pandexo_sim = io.export_pandexo(
                    pyrat, baseline, transit_duration,
                    Kmag=Kmag, metal=metal, instrument=instrument,
                    n_transits=n_transits, resolution=resolution,
                    noise_floor=noise_floor, save_file=save_file)
            pandexo_wl.append(pandexo_sim[1][0])
            pandexo_flux.append(pandexo_sim[2][0])
            pandexo_uncert.append(pandexo_sim[3][0])
        pflux[i,iphase] = np.concatenate(pandexo_flux)
        puncert[i,iphase] = np.concatenate(pandexo_uncert)


# Model evaluated at properties of local (sub-observer) longitude
# for each orbital phase:
local_flux_ratio = np.zeros((nmodels, nphase, nwave_obs))
for i in range(nmodels):
    abunds = models[i]['abund']
    temps  = models[i]['temp']
    for iphase in range(nphase):
        q2 = pa.qscale(
            pyrat.atm.qbase, pyrat.mol.name, pyrat.atm.molmodel,
            pyrat.atm.molfree, np.log10(abunds[4*iphase]),
            pyrat.atm.bulk, iscale=pyrat.atm.ifree, ibulk=pyrat.atm.ibulk,
            bratio=pyrat.atm.bulkratio, invsrat=pyrat.atm.invsrat)
        pyrat.run(temps[4*iphase], q2)
        local_fratio = pyrat.spec.spectrum / pyrat.spec.starflux * rprs**2
        local_flux_ratio[i,iphase] = ps.band_integrate(
            local_fratio, 1e4/wl, filter_trans, filter_wn)


np.savez(
    'WASP43b_3D_synthetic_pandexo_flux_ratios.npz',
    flux_ratio=flux_ratio,
    wavelength=wl,
    stellar_flux=pyrat.spec.starflux,
    pandexo_flux_ratio=pflux,
    pandexo_uncert=puncert,
    pandexo_wl=np.concatenate(pandexo_wl),
    noiseless_flux_ratio=noiseless_flux_ratio,
    local_flux_ratio=local_flux_ratio,
    phase=obs_phase,
    flux_units='erg s-1 cm-2 cm',
    wl_units='micron')

