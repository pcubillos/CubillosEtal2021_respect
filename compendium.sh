# Define topdir (in your top working directory) to make your life easier:
topdir=`pwd`

# Installs:
pip install lbl-repack==1.3.0
pip install mc3==3.0.0
cd $topdir
git clone https://github.com/pcubillos/rate

# Clone (download) the necessary code:
cd $topdir
git clone --recursive https://github.com/pcubillos/pyratbay
cd $topdir/pyratbay
git checkout 4d6a388
python setup.py develop


# Generate filter files:
cd $topdir/code
$topdir/code/make_filters.py

# Download Stellar model:
cd $topdir/inputs
wget http://kurucz.harvard.edu/grids/gridp00aodfnew/fp00ak2odfnew.pck


# Download Exomol data:
cd $topdir/inputs/opacity
wget -i wget_exomol_CH4.txt
wget -i wget_exomol_H2O.txt

# Download CO data:
cd $topdir/inputs/opacity
wget http://iopscience.iop.org/0067-0049/216/1/15/suppdata/apjs504015_data.tar.gz
tar -xvzf apjs504015_data.tar.gz
rm -f apjs504015_data.tar.gz ReadMe Table_S1.txt Table_S2.txt \
      Table_S3.txt Table_S4.txt Table_S6.par


# Generate partition-function files for H2O:
cd $topdir/run01
pbay -pf exomol $topdir/inputs/opacity/1H2-16O__POKAZATEL.pf

# Generate partition-function file for CH4:
cd $topdir/run01
pbay -pf exomol $topdir/inputs/opacity/12C-1H4__YT10to10.pf


# Compress LBL databases:
cd $topdir/run01
repack repack_H2O.cfg
repack repack_CH4.cfg  # TBD


# Make TLI files:
cd $topdir/run01/
pbay -c tli_Li_CO.cfg
pbay -c tli_hitemp_CO2.cfg
pbay -c tli_exomol_H2O.cfg
pbay -c tli_exomol_CH4.cfg


# Make atmospheric files:
cd $topdir/run01/
pbay -c atm_uniform.cfg
pbay -c atm_vp_gcm.cfg

# Make opacity files:
cd $topdir/run01/
pbay -c opacity_H2O_0.6-12.0um_R10000.cfg
pbay -c opacity_CH4_0.6-12.0um_R10000.cfg
pbay -c opacity_CO_0.6-12.0um_R10000.cfg
pbay -c opacity_CO2_0.6-12.0um_R10000.cfg

# Run retrieval:
cd $topdir/run02_resolved/
pbay -c mcmc_WASP43b_day_resolved.cfg
pbay -c mcmc_WASP43b_east_resolved.cfg
pbay -c mcmc_WASP43b_west_resolved.cfg

cd $topdir/run03_integrated/
pbay -c mcmc_WASP43b_day_integrated.cfg   # TBD
pbay -c mcmc_WASP43b_east_integrated.cfg  # TBD
pbay -c mcmc_WASP43b_west_integrated.cfg  # TBD

