#! /usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def read_gcm(path, molecs=['H2O','CH4','CO','CO2']):
    """
    Read GCM atmospheric models from V. Parmentier.

    Parameters
    ----------
    path: String
        Location where the input model files are.
    molecs: List of strings
        Requested molecules to extract

    Returns
    -------
    lon: 1D float ndarray
        Longitude array in radians (lon=0 at substellar point).
    lat: 1D float ndarray
        Latitude array in radians (lat=-pi/2 at north pole, pi/2 at south pole).
    press: 1D float ndarray
        Pressure profile in bars.
    temp: 3D float ndarray
        Temperature profiles with shape [nlon, nlat, nlayers]
    spec: 4D float ndarray
        log10(abundances) of requested species, with shape
        [nmol, nlon, nlat, nlayers].

    Example
    -------
    >>> import sys
    >>> sys.path.append('../code')
    >>> import read_gcm as rg
    >>> path = '/home/pcubillos/ast/compendia/CubillosEtal2020_SpatialRetrieval/inputs/'
    >>> molecs = 'H2 H He H2O CH4 CO CO2'.split()
    >>> lon, lat, press, temp, spec = rg.read_gcm(path, molecs)
    """
    # File names:
    files = [
        "WASP43b_PT_profiles_solar_clear_alpha000.dat",
        "WASP43b_PT_profiles_solar_clear_alpha180.dat",
        ]
    # Molecules in files:
    molecules = [
        'e-',   'H2',   'H',    'H+',  'H-',  'H2-', 'H2+', 'H3+', 'He',
        'H2O',  'CH4',  'CO',   'NH3', 'N2',  'PH3', 'H2S', 'TiO', 'VO',
        'Fe',   'FeH',  'CrH',  'Na',  'K',   'Rb',  'Cs',  'CO2', 'HCN',
        'C2H2', 'C2H4', 'C2H6', 'SiO', 'MgH', 'OCS', 'Li',
        ]

    # Hardcoded sizes:
    nlayers = 52
    nlon = 64
    nlat = 32
    # Degrees:
    degdlat = 180.0/nlat
    deglat = np.arange(0.5*degdlat- 90.0,  90.0, degdlat)
    deglon = np.arange(0.5*degdlat-180.0, 180.0, degdlat)
    # Radians:
    lat = deglat * np.pi/180.0
    lon = deglon * np.pi/180.0

    # Requested molecules:
    nmol = len(molecs)
    imol = np.zeros(nmol, int)
    for k in np.arange(nmol):
        imol[k] = molecules.index(molecs[k]) + 5

    # Allocate arrays:
    press = np.zeros(nlayers)
    temp  = np.zeros((nlon, nlat, nlayers))
    spec  = np.zeros((nmol, nlon, nlat, nlayers))

    for j in np.arange(len(files)):
        with open(path + "/" + files[j], "r") as f:
            data = f.readlines()
        nlines = len(data)

        for line in np.arange(nlines):
            if line[0].isdigit():
                info = np.array(line.strip().split(","), np.double)
                ilay = int(info[0]) - 1
                ilon = np.argmin(np.abs(deglon-info[1]))
                ilat = np.argmin(np.abs(deglat-info[2]))
                temp[ilon, ilat, ilay] = info[4]
                for k in np.arange(nmol):
                    spec[k, ilon, ilat, ilay] = info[imol[k]]
                if j == 1 and ilat == 0 and ilon == 0:
                    press[ilay] = info[3]

    # ln to log10:
    spec /= np.log(10)

    # Patch temperatures and abundances with values from neighbor longitudes:
    for k in np.arange(nmol+1):
        var = temp if k == nmol else spec[k]
        var[45,31] = np.copy(var[44,31])
        var[61,31] = np.copy(var[60,31])
        var[46] = var[45]*2/3 + var[48]*1/3
        var[47] = var[45]*1/3 + var[48]*2/3
        var[62] = var[61]*2/3 + var[ 0]*1/3
        var[63] = var[61]*1/3 + var[ 0]*2/3

    return lon, lat, press, temp, 10**spec


def area(lon, lat):
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


def profiles(lon, lat, press, var, frac=0.1, xran=None,
             xlabel=None, alpha=0.5, savefile=None):
    """
    Plot profiles from Vivien Parmentier GCM models.
    Modification History:    created   Jasmina Blecic on May 16th 2018

    Parameters
    ----------
    lon: 1D float ndarray
       Longitude (radians) array on planet (0 at substellar point).
    lat: 1D float ndarray
       Latitude (radians) array on planet (-pi/2 at North pole,
       pi/2 at South pole).
    press: 1D float ndarray
       Pressure profile in bars.
    var: 3D float ndarray
       Variable to plot (e.g., temp, spec[0]).
    frac: Float
       Fraction of profiles to plot.
    xran: 2-element array
       Ranges for X-axis.

    Example
    -------
    >>> import jb_read_Vivien_cleaned as jb
    >>> lon, lat, press, temp, spec = jb.readGCM("inputs/GCM")

    >>> var = temp
    >>> xlabel = "Temperature (K)"
    >>> rg.profiles(lon, lat, press, var, xlabel=xlabel,
                    savefile='plots/GCM_TP.png')
    >>> var = spec[1]
    >>> xlabel = r"$\log_{10}({\rm CH4})$"
    >>> jb.profiles(lon, lat, press, var, xlabel=xlabel,
                    savefile='plots/GCM_CH4.png')
    """
    nlon, nlat, nlay = np.shape(var)

    # Cosine of substellar angle:
    mu = project(lon, lat, phase=0.5)
    col, vmin, vmax = mu, -1.0, 1.0

    # lon/lat in degrees:
    #latitude  = np.ones((nlon, nlat)) * lat               * 180/np.pi
    #longitude = np.ones((nlon, nlat)) * lon[:,np.newaxis] * 180/np.pi
    # lat
    # col, vmin, vmax = latitude, -90, 90
    # Color scale as function of mu:

    # Create a ScalarMappable for colormap:
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    s_m = mpl.cm.ScalarMappable(cmap=plt.cm.seismic, norm=norm)
    s_m.set_array([])

    plt.figure(10)
    plt.clf()
    # Shuffle indices to avoide biasing the plot:
    arr = np.arange(nlat*nlon)
    np.random.shuffle(arr)
    # Print only a fraction to avoide over-crowding the plot:
    for k in arr[0:int(frac*nlat*nlon)]:
        i = k // nlat
        j = k % nlat
        plt.semilogy(var[i,j], press, alpha=0.5, color=s_m.to_rgba(col[i,j]))
    plt.ylim(np.amax(press), np.amin(press))
    if xran is not None:
        plt.xlim(xran)
    cb = plt.colorbar(s_m)
    cb.set_label(r"Cosine of sub-stellar angle, $\mu$")

    # plot the colorbar, usingScalarMappable
    plt.ylabel("Pressure (bar)")
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)


def phasevar(lon, lat, press, var, phase, xlabel, fignum=20, savefile=None):
    """
    >>> var = [spec[0], spec[1], spec[2], spec[3], temp]
    >>> xlabel = [r"$\log_{10}({\rm H2O})$", r"$\log_{10}({\rm CH4})$",
    >>>           r"$\log_{10}({\rm CO})$",  r"$\log_{10}({\rm CO2})$",
    >>>           'Temperature (K)']
    >>> sfile = ['plots/phase_H2O.pdf', 'plots/phase_CH4.pdf',
    >>>          'plots/phase_CO.pdf', 'plots/phase_CO2.pdf',
    >>>          'plots/phase_TP.pdf']
    >>> nphase = 19
    >>> phase = np.linspace(0, 1, nphase)

    >>> for j in np.arange(5):
    >>>   jb.phasevar(lon, lat, press, var[j], phase, xlabel[j], fignum=j,
    >>>               savefile=sfile[j])

    """
    # Custom colormap:
    # https://matplotlib.org/examples/pylab_examples/custom_cmap.html
    cdict = {
        'red':   ((0.0, 0.0, 0.0),
                  (0.25, 0.0, 0.0),
                  (0.5, 0.8, 1.0),
                  (0.75, 1.0, 1.0),
                  (1.0, 0.4, 1.0)),

        'green': ((0.0, 0.0, 0.0),
                  (0.25, 0.0, 0.0),
                  (0.5, 0.9, 0.8),
                  (0.75, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),

        'blue':  ((0.0, 0.0, 0.4),
                  (0.25, 1.0, 1.0),
                  (0.5, 1.0, 0.8),
                  (0.75, 0.0, 0.0),
                  (1.0, 0.0, 0.0))
        }
    blue_red2 = LinearSegmentedColormap('BlueRed2', cdict)
    plt.register_cmap(cmap=blue_red2)
    cm = plt.get_cmap('BlueRed2')

    nphase = len(phase)

    # Create a ScalarMappable for colormap:
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    s_m = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
    s_m.set_array([])

    plt.figure(fignum)
    plt.clf()
    plt.subplots_adjust(0.12, 0.1, 1.0, 0.95)
    for i in np.arange(nphase):
        mu = np.clip(project(lon, lat, phase[i]), 0, 1)
        factor = (mu * area(lon,lat))[:,:,np.newaxis]
        avg = np.sum(var*factor, axis=(0,1))/np.pi
        plt.semilogy(avg, press, alpha=0.9, color=s_m.to_rgba(phase[i]))
    plt.ylim(np.amax(press), np.amin(press))
    plt.ylabel('Pressure (bar)')

    if xlabel is not None:
        plt.xlabel(xlabel)

    cb = plt.colorbar(s_m)
    cb.set_label(r"Orbital phase")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    if savefile is not None:
        plt.savefig(savefile)

