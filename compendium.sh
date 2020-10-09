# Define topdir (in your top working directory) to make your life easier:
topdir=`pwd`

# Installs:
pip install mc3==3.0.2
pip install pyratbay==0.9.1
pip install scipy==1.3.3

cd $topdir
git clone https://github.com/pcubillos/rate

git clone --recursive https://github.com/natashabatalha/pandexo
cd pandexo
git checkout b9f1f06
python setup.py install
# You will need all the pandeia 1.5 business (see pandexo docs)

# Generate filter files:
cd $topdir/code
$topdir/code/make_filters.py  # TBD: make sure synthetic works

# Download Stellar model:
cd $topdir/inputs
wget http://kurucz.harvard.edu/grids/gridp00aodfnew/fp00ak2odfnew.pck


# Download repack-exomol/hitemp data:
cd $topdir/inputs/opacity
wget -i wget_repack-exomol_H2O-CH4.txt
wget -i wget_hitemp_CO-CO2.txt
unzip '*.zip'
rm -f *.zip
bzip2 -d 05_HITEMP2019.par.bz2


# Generate partition-function files:
cd $topdir/run_setup
pbay -pf exomol ../inputs/opacity/1H2-16O__POKAZATEL.pf
pbay -pf exomol ../inputs/opacity/12C-1H4__YT10to10.pf

# Make TLI files:
cd $topdir/run_setup/
pbay -c tli_CO_hitemp_li2019.cfg
pbay -c tli_CO2_hitemp_2010.cfg
pbay -c tli_H2O_exomol_pokazatel.cfg
pbay -c tli_CH4_exomol_yt10to10.cfg

# Make atmospheric files:
cd $topdir/run_setup
pbay -c atm_equilibrium.cfg

# Make opacity files:
cd $topdir/run_setup
pbay -c opacity_H2O_0.8-5.5um_R10000.cfg
pbay -c opacity_CH4_0.8-5.5um_R10000.cfg
pbay -c opacity_CO2_0.8-5.5um_R10000.cfg
pbay -c opacity_CO_0.8-5.5um_R10000.cfg


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Run HST+Spitzer retrieval:
cd $topdir/run_resolved
pbay -c mcmc_WASP43b_day_resolved.cfg
pbay -c mcmc_WASP43b_east_resolved.cfg
pbay -c mcmc_WASP43b_west_resolved.cfg

cd $topdir/run_integrated
pbay -c mcmc_WASP43b_day_integrated.cfg
pbay -c mcmc_WASP43b_east_integrated.cfg
pbay -c mcmc_WASP43b_west_integrated.cfg

# Post analysis/plot:
cd $topdir
python code/make_pickles.py
python code/fig_WASP43b_data_retrieval.py


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Model WASP-43b phase curve:
cd $topdir/run_simulation
python ../code/model_WASP43b.py
python ../code/make_filters.py
python ../code/fig_WASP43b_model_spectra.py

# Retrieve simulated JWST WASP-43b spectra:
cd $topdir
sh inputs/launch_jwst_sim_resolved.sh
sh inputs/launch_jwst_sim_integrated.sh
python ../code/fig_WASP43b_model_retrieval.py


