# Define topdir (in your top working directory) to make your life easier:
topdir=`pwd`

# Clone (download) the necessary code:
cd $topdir
git clone --recursive https://github.com/pcubillos/pyratbay
cd $topdir/pyratbay
git checkout 36c7d1f # Update when necessary
make
cd $topdir/modules/MCcubed
git checkout ca0fb5d


cd $topdir
git clone https://github.com/pcubillos/repack
cd $topdir/repack
git checkout 4ba3633
make


## Temporary hack:
#cd $topdir/pyratbay/modules
#mv TEA jb_TEA
#git clone https://github.com/pcubillos/TEA
#cd TEA
#git checkout multiproc


# Generate filter files:
#cd $topdir/code
#$topdir/code/make_filters.py > $topdir/code/filter_info.txt


# Download Exomol data:
cd $topdir/inputs/opacity
wget -i wget_exomol_CH4.txt

# Download HITEMP data:
cd $topdir/inputs/opacity
wget --user=HITRAN --password=getdata -N -i wget_hitemp_H2O.txt
unzip '*.zip'
rm -f *.zip

# Download CO data:
cd $topdir/inputs/opacity
wget http://iopscience.iop.org/0067-0049/216/1/15/suppdata/apjs504015_data.tar.gz
tar -xvzf apjs504015_data.tar.gz
rm -f apjs504015_data.tar.gz ReadMe Table_S1.txt Table_S2.txt \
      Table_S3.txt Table_S6.par


# Generate partition-function files for H2O:
cd $topdir/run01
python $topdir/code/pf_tips_H2O.py

# Generate partition-function file for CH4:
cd $topdir/run01
python $topdir/pyratbay/scripts/PFformat_Exomol.py \
       $topdir/inputs/opacity/12C-1H4__YT10to10.pf


# Compress LBL databases:
cd $topdir/run01
python $topdir/repack/repack.py repack_H2O.cfg
python $topdir/repack/repack.py repack_CH4.cfg  # TBD


# Make TLI files:
cd $topdir/run01/
python $topdir/pyratbay/pbay.py -c tli_Li_CO.cfg
python $topdir/pyratbay/pbay.py -c tli_hitemp_H2O.cfg
python $topdir/pyratbay/pbay.py -c tli_exomol_CH4.cfg


# Make atmospheric files:
cd $topdir/run01/
python $topdir/pyratbay/pbay.py -c atm_tea.cfg

# Make nominal opacity file (H2O CO CO2 CH4 HCN NH3):
cd $topdir/run01/
python $topdir/pyratbay/pbay.py -c opacity_nominal_4.8-14um.cfg

# Run retrieval:
cd $topdir/run02_clear/
python $topdir/pyratbay/pbay.py -c mcmc_wasp43b_lon+000_w0-cdm00-c.cfg
