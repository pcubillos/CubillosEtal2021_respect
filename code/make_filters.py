#! /usr/bin/env python

import sys
import os
import numpy as np


def tophat(wl0, width, margin, dlambda, file=None):
  """
  Generate a top-hat filter transmission function.

  Parameters
  ----------
  wl0:  Float
     Filter central wavelength in microns.
  width:
     Filter width in microns.
  margin:
     Margin with zero-valued transmission.
  dlambda:
     Spectral sampling rate in microns.
  file: String
     Name of the output file.

  Example
  -------
  >>> wl0     = 0.743
  >>> width   = 0.010
  >>> margin  = 0.003
  >>> dlambda = 0.0002
  >>> tophat(wl0, width, margin, dlambda, "tophat_filter.dat")
  """
  wl = np.arange(wl0-width/2.0-margin,
                 wl0+width/2.0+margin, dlambda)
  trans = np.ones(len(wl))
  trans[np.where(np.abs(wl-wl0)>width/2.0)] = 0.0

  if file is not None:
    with open(file, "w") as f:
      f.write("# Wavelength (um)  Transmission\n")
      for i in np.arange(len(wl)):
        f.write("{:10.6f}         {:6.4f}\n".format(wl[i], trans[i]))
  return wl, trans


if __name__ == "__main__":
  """
  """
  # Gather files:
  dpath = "../inputs/data/"
  fdata = []
  for f in os.listdir(dpath):
    if f.endswith(".csv"):
      fdata.append(f)
  fdata = sorted(fdata)

  # Prep up SPITZER filters:
  irac = ["../pyratbay/inputs/filters/spitzer_irac1_sa.dat",
          "../pyratbay/inputs/filters/spitzer_irac2_sa.dat",
          "../pyratbay/inputs/filters/spitzer_irac3_sa.dat",
          "../pyratbay/inputs/filters/spitzer_irac4_sa.dat"]
  swave = np.array([3.6, 4.5, 5.8, 8.0, 24])

  # Read data:
  for j in np.arange(len(fdata)):
    with open(dpath + fdata[j], "r") as f:
      lines = f.readlines()
    planet = fdata[j].split("_")[0]
    nlines = len(lines)

    data    = "data   = "
    error   = "uncert = "
    filters = ""
    k = 0

    for i in np.arange(nlines):
      if lines[i].strip() == "" or lines[i].strip().startswith("#"):
        continue
      elif lines[i].strip().startswith("@"):
        inst = lines[i].strip()[1:]
      else:
        wl, width, fpfs, uncert = lines[i].split(",")
        wl    = float(wl)
        width = float(width) * 2.0
        fpfs  = float(fpfs)
        uncert  = float(uncert)
        if inst.startswith("irac"):
          k = np.argmin(np.abs(swave-wl))
          ffile = irac[k]
        elif inst.startswith("mips"):
          ffile = "../pyratbay/inputs/filters/spitzer_mips24.dat"
        else:
          ffile = "../inputs/filters/{:s}_{:s}_{:5.3f}um.dat".format(planet,
                                                                   inst, wl)
        if (inst.startswith("stis") or inst.startswith("wfc3")   or
            inst.startswith("acs")  or inst.startswith("nicmos") ):
          tophat(wl, width, 0.1*width, width/500.0, ffile)
        data  += "{:.4e}  ".format(fpfs)
        error += "{:.4e}  ".format(uncert)
        if (k+1) % 5 == 0 and k != 0:
          data  += "\n         "
          error += "\n         "
        filters += "         {:s}\n".format(ffile)
        k += 1

    filters = filters.replace("        ", "filter =", 1)
    print(fdata[j])
    print(planet)
    print(data)
    print(error)
    print(filters)
