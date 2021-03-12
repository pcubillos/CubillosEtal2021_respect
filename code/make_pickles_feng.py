import os

import numpy as np

import pyratbay as pb
import pyratbay.io as io
import pyratbay.tools as pt
import mc3


retrievals = [
    'run_feng2020/mcmc_WASP43b_integrated_phase0.06.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.12.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.19.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.25.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.31.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.38.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.44.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.50.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.56.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.62.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.69.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.75.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.81.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.88.cfg',
    'run_feng2020/mcmc_WASP43b_integrated_phase0.94.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.06.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.12.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.19.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.25.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.31.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.38.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.44.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.50.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.56.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.62.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.69.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.75.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.81.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.88.cfg',
    'run_feng2020/mcmc_WASP43b_resolved_phase0.94.cfg',
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

