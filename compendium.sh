# Define topdir (in your top working directory) to make your life easier:
topdir=`pwd`

# Installs:
pip install mc3==3.0.0
pip install pyratbay==0.9.0a5
#pip install pandexo.engine==1.4
#pip install scipy==1.3.3
cd $topdir
git clone https://github.com/pcubillos/rate

git clone --recursive https://github.com/natashabatalha/pandexo
cd pandexo
# either git checkout b9f1f06+pandeia 1.5 or edit
python setup.py install

# Generate filter files:
cd $topdir/code
#$topdir/code/make_filters.py

# Download Stellar model:
cd $topdir/inputs
wget http://kurucz.harvard.edu/grids/gridp00aodfnew/fp00ak2odfnew.pck


# Download repack-exomol/hitemp data:
cd $topdir/inputs/opacity
wget -i wget_repack-exomol_H2O-CH4.txt
wget -i wget_hitemp_CO_CO2.txt
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
pbay -c atm_vp_gcm.cfg

# Make opacity files:
cd $topdir/run_setup
pbay -c opacity_H2O_0.8-5.5um_R10000.cfg
pbay -c opacity_CH4_0.8-5.5um_R10000.cfg
pbay -c opacity_CO2_0.8-5.5um_R10000.cfg
pbay -c opacity_CO_0.8-5.5um_R10000.cfg

pbay -c opacity_H2O_0.6-12.0um_R10000.cfg
pbay -c opacity_CH4_0.6-12.0um_R10000.cfg
pbay -c opacity_CO2_0.6-12.0um_R10000.cfg
pbay -c opacity_CO_0.6-12.0um_R10000.cfg

# Run retrieval:
cd $topdir/run_resolved/
pbay -c mcmc_WASP43b_day_resolved.cfg
pbay -c mcmc_WASP43b_east_resolved.cfg
pbay -c mcmc_WASP43b_west_resolved.cfg

cd $topdir/run_integrated/
pbay -c mcmc_WASP43b_day_integrated.cfg
pbay -c mcmc_WASP43b_east_integrated.cfg
pbay -c mcmc_WASP43b_west_integrated.cfg


# Model WASP-43b phase curve:
cd $topdir/run_simulation
python ../code/model_WASP43b.py
python ../code/make_filters.py
python ../code/fig_WASP43b_model_atmosphere.py
python ../code/fig_WASP43b_model_spectra.py

# Retrieve model WASP-43b spectra:
cd $topdir/run_resolved_sim
pbay -c mcmc_model_WASP43b_hot_resolved.cfg
pbay -c mcmc_model_WASP43b_day_resolved.cfg
pbay -c mcmc_model_WASP43b_east_resolved.cfg
pbay -c mcmc_model_WASP43b_west_resolved.cfg
pbay -c mcmc_model_WASP43b_night_resolved.cfg

cd $topdir/run_integrated_sim
pbay -c mcmc_model_WASP43b_hot_integrated.cfg
pbay -c mcmc_model_WASP43b_day_integrated.cfg
pbay -c mcmc_model_WASP43b_east_integrated.cfg
pbay -c mcmc_model_WASP43b_west_integrated.cfg
pbay -c mcmc_model_WASP43b_night_integrated.cfg



