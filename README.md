# Tools for downloading and regridding MEPS Harmonie atm data

## Data source 

Data is downloaded from Met Norway THREDDS server:
http://thredds.met.no/thredds/metno.html

## Routines

### 1. Download Harmonie fields 

See `fetch_harmonie_data.py`

Downloads fields from OPENDAP server, e.g.:
http://thredds.met.no/thredds/dodsC/meps25epsarchive/2017/01/15/meps_subset_2_5km_20170115T00Z.nc.html

Currently supported fields are T2m, U10m, V10m, sfcpres.

Generates monthly files with hourly data with no gaps: `harmonie_T2m_y2017m01.nc`.

### 2. Regrid to Nemo grid and blend with ECMWF fields

See `regrid_and_blend.py`

Assumes monthly input files with 1h time resolution:

 - coarse grid `ecmwf_T2m_y2017m01.nc`
 - fine grid `harmonie_T2m_y2017m01.nc`

Target grid is the NemoNordic 1 nm grid, hard-coded in the script.

Regrids coarse field on target grid. If fine grid is available, regrids it too and
blends on top of the coarse field.

Produces monthly files: `harmonie-blend_T2m_y2017m01.nc`

3. Plot example fields

See `plot_atm_field.py`
