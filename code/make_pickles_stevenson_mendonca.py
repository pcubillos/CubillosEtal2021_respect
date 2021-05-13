import os

import numpy as np

import pyratbay as pb
import pyratbay.io as io
import pyratbay.tools as pt
import mc3


retrievals = [
    'run_stevenson_mendonca/mcmc_WASP43b_integrated_phase0.25.cfg',
    'run_stevenson_mendonca/mcmc_WASP43b_integrated_phase0.50.cfg',
    'run_stevenson_mendonca/mcmc_WASP43b_integrated_phase0.75.cfg',
    'run_stevenson_mendonca/mcmc_WASP43b_resolved_phase0.25.cfg',
    'run_stevenson_mendonca/mcmc_WASP43b_resolved_phase0.50.cfg',
    'run_stevenson_mendonca/mcmc_WASP43b_resolved_phase0.75.cfg',
    ]


for cfg in retrievals:
    with pt.cd(os.path.dirname(cfg)):
        pyrat = pb.run(os.path.basename(cfg), run_step='init', no_logfile=True)

        mcmc = np.load(pyrat.ret.mcmcfile)
        pyrat.ret.posterior, _, _ = mc3.utils.burn(mcmc)
        pyrat.percentile_spectrum()
        pfile = pyrat.ret.mcmcfile.replace('.npz', '.pickle')
        io.save_pyrat(pyrat, pfile)
        mcmc.close()
