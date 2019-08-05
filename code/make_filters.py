#! /usr/bin/env python

import sys
import os
import numpy as np

sys.path.append('../pyratbay')
import pyratbay as pb


def main():
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
                continue

            wl, width, fpfs, uncert = lines[i].split(",")
            wl    = float(wl)
            width = float(width) * 2.0
            fpfs  = float(fpfs)
            uncert = float(uncert)
            if inst.startswith("irac"):
                k = np.argmin(np.abs(swave-wl))
                ffile = irac[k]
            elif inst.startswith("mips"):
                ffile = "../pyratbay/inputs/filters/spitzer_mips24.dat"
            else:
                ffile = "../inputs/filters/{:s}_{:s}_{:5.3f}um.dat".format(
                            planet, inst, wl)
            if inst.startswith("stis") or inst.startswith("wfc3") \
               or inst.startswith("acs") or inst.startswith("nicmos"):
                pb.tools.tophat(wl, width, 0.1*width, width/500.0, ffile=ffile)
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


if __name__ == "__main__":
    main()
