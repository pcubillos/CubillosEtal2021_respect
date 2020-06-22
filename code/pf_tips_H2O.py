#! /usr/bin/env python

import sys
import numpy as np

sys.path.append("../pyratbay/modules/pytips")
import pytips as p

temp = np.linspace(70, 3000, 294)

molname = "H2O"
p.to_file("./PF_tips_H2O.dat", molname, temp)
