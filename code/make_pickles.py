import os

import numpy as np

import pyratbay as pb
import pyratbay.io as io
import pyratbay.tools as pt
import mc3


retrievals = [
    'run_resolved/mcmc_WASP43b_east_resolved.cfg',
    'run_resolved/mcmc_WASP43b_day_resolved.cfg',
    'run_resolved/mcmc_WASP43b_west_resolved.cfg',
    'run_integrated/mcmc_WASP43b_east_integrated.cfg',
    'run_integrated/mcmc_WASP43b_day_integrated.cfg',
    'run_integrated/mcmc_WASP43b_west_integrated.cfg',
    ]


for cfg in retrievals:
    with pt.cd(os.path.dirname(cfg)):
        pyrat = pb.run(os.path.basename(cfg), init=True, no_logfile=True)

        mcmc = np.load(pyrat.ret.mcmcfile)
        pyrat.ret.posterior, _, _ = mc3.utils.burn(mcmc)
        pyrat.percentile_spectrum()
        pfile = pyrat.ret.mcmcfile.replace('.npz', '.pickle')
        io.save_pyrat(pyrat, pfile)
        mcmc.close()

