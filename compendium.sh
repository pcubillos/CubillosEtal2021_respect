# Define topdir (in your top working directory) to make your life easier:
topdir=`pwd`

# Clone (download) the necessary code:
cd $topdir
git clone --recursive https://github.com/pcubillos/pyratbay
cd $topdir/pyratbay
git checkout 9dc2bd8
cd $topdir/pyratbay/modules/MCcubed
git checkout 9819e4a
# Compile:
cd $topdir/pyratbay
make

# Install repack:
pip install lbl-repack==1.3.0


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
      Table_S3.txt Table_S3.txt Table_S6.par


# Generate partition-function files for H2O:
cd $topdir/run01
python $topdir/pyratbay/pbay.py -pf exomol \
    $topdir/inputs/opacity/1H2-16O__POKAZATEL.pf

# Generate partition-function file for CH4:
cd $topdir/run01
python $topdir/pyratbay/pbay.py -pf exomol \
       $topdir/inputs/opacity/12C-1H4__YT10to10.pf


# Compress LBL databases:  # TBD
cd $topdir/run01
python $topdir/repack/repack.py repack_H2O.cfg
python $topdir/repack/repack.py repack_CH4.cfg


# Make TLI files:  ## OK
cd $topdir/run01/
python $topdir/pyratbay/pbay.py -c tli_Li_CO.cfg
python $topdir/pyratbay/pbay.py -c tli_exomol_H2O.cfg
python $topdir/pyratbay/pbay.py -c tli_exomol_CH4.cfg


# Make atmospheric files:
cd $topdir/run01/
python $topdir/pyratbay/pbay.py -c atm_uniform.cfg

# Make opacity files:
cd $topdir/run01/
python $topdir/pyratbay/pbay.py -c opacity_H2O_1.0-5.5um.cfg
python $topdir/pyratbay/pbay.py -c opacity_CH4_1.0-5.5um.cfg
python $topdir/pyratbay/pbay.py -c opacity_CO_1.0-5.5um.cfg

# Run retrieval:
cd $topdir/run02_resolved/
python $topdir/pyratbay/pbay.py -c mcmc_WASP43b_day_resolved.cfg
python $topdir/pyratbay/pbay.py -c mcmc_WASP43b_east_resolved.cfg
python $topdir/pyratbay/pbay.py -c mcmc_WASP43b_west_resolved.cfg

