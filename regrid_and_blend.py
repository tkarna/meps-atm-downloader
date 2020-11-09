"""
Regrid Harmonie and ECMWF fields on Nemo grid.

Uses xesmf package.
"""
import numpy
import pandas as pd
import xarray as xr
import xesmf as xe
import glob
import datetime
import dateutil.rrule


def get_nemo_grid():
    """
    Default Nemo grid as a xarray dataset
    """
    nlat = 1046
    nlon = 1238
    lat_min, lat_max = 48.4917, 65.90809
    lon_min, lon_max = -4.15278, 30.207987
    lat = numpy.linspace(lat_min, lat_max, nlat)
    lon = numpy.linspace(lon_min, lon_max, nlon)

    return xr.Dataset({'lat': (['lat'], lat), 'lon': (['lon'], lon)})


def _load_field(date, var, prefix, mode):

    if mode == 'monthly':
        date_str = date.strftime('y%Ym%m')
    elif mode == 'daily':
        date_str = date.strftime('y%Ym%md%d')
    elif mode == 'no-merge':
        date_str = date.strftime('y%Ym%md%d-%H')
    else:
        raise ValueError('Invalid mode: {:}'.format(mode))

    filename = '{prefix:}_{var:}_{date:}.nc'.format(
        prefix=prefix, var=var, date=date_str)

    print('Loading {:}'.format(filename))
    ds = xr.open_dataset(filename)
    return ds


def load_harmonie_field(date, var, mode):
    """
    Load Harmonie atm field from netCDF in to xarray.
    """
    ds = _load_field(date, var, 'harmonie', mode)
    ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    # drop unnecessary coordinates
    keep_coords = ['time', 'lon', 'lat']
    for c in ds.coords:
        if c not in keep_coords:
            ds = ds.drop(c)
    # take the first variable
    variables = [v for v in ds.variables if v not in ['lat', 'lon', 'time']]
    var = variables[0]
    dr = ds[var]
    return dr


def load_ecmwf_field(date, var, mode):
    """
    Load ECMWF atm field from netCDF in to xarray.
    """
    ds = _load_field(date, var, 'ecmwf', mode)
    # take the first variable
    variables = [v for v in ds.variables if v not in ['lat', 'lon', 'time']]
    var = variables[0]
    dr = ds[var]
    return dr


def get_blend_mask(data_array, regridder, buffer=20):
    """
    Construct a blend mask from a data array and regridder.

    First generates a scalar field that has a linear ramp from 0 at the
    boundaries to 1 in the interior. Ramp width in pixels is defined by
    :buffer: argument.

    The scalar field is the mapped onto the target grid using the regridder.
    The resulting blend mask has zeros outside the source data set.
    """
    dr_mask = data_array[0, :, :].copy()
    nx, ny = dr_mask.values.shape
    x = numpy.arange(nx, dtype=float)
    y = numpy.arange(ny, dtype=float)

    X, Y = numpy.meshgrid(y, x)
    mask = numpy.minimum(X, Y)
    mask = numpy.minimum(mask, numpy.flip(mask))
    mask = numpy.minimum((mask + 1)/buffer, 1.0)

    dr_mask.values[:] = mask

    dr_out = regridder(dr_mask)
    blend_mask = dr_out.values.copy()
    return blend_mask


def merge_monthly_files(date, var, grid_target, fileprefix,
                        mode='monthly'):
    """
    Reads two atm fields, regrids them on the model grid, and blends them.

    Reads ecmwf field as a "coarse" field from monthly files:
    ecmwf_T2m_y2016m11.nc ...

    Reads harmonie fields as a "fine" field from files:
    harmonie_T2m_y2016m11.nc ...

    Produces a blended field on Nemo grid:
    {fileprefix}_T2m_y2016m11.nc ...
    """
    field_coarse = load_ecmwf_field(date, var, mode='monthly')
    regridder_coarse = xe.Regridder(field_coarse, grid_target, 'bilinear',
                                    reuse_weights=True)

    field_fine = load_harmonie_field(date, var, mode)
    regridder_fine = xe.Regridder(field_fine, grid_target, 'bilinear', reuse_weights=True)

    blend_mask = get_blend_mask(field_fine, regridder_fine)

    # blend each time slice separately (saves memory)
    slice_list = []
    # NOTE loop over coarse time stamps
    # for sdate, coarse_slice in field_coarse.groupby('time'):
    #     # make new slice by regridding coarse field
    #     new_slice = regridder_coarse(coarse_slice)
    #     source_str = 'coarse'
    #     try:
    #         fine_slice = field_fine.sel(time=sdate)
    #         new_fine_slice = regridder_fine(fine_slice)
    #         new_slice.values[:] = (
    #             blend_mask*new_fine_slice.values +
    #             (1. - blend_mask)*new_slice.values
    #         )
    #         source_str += ' + fine'
    #     except KeyError as e:
    #         fine_slice = None
    #     print('{:}: {:}'.format(sdate, source_str))
    #     slice_list.append(new_slice)

    # NOTE loop over fine time stamps
    for sdate, fine_slice in field_fine.groupby('time'):
        # make new slice by regridding coarse field
        coarse_slice = field_coarse.sel(time=sdate)
        new_slice = regridder_coarse(coarse_slice)
        source_str = 'coarse'

        new_fine_slice = regridder_fine(fine_slice)
        new_slice.values[:] = (
            blend_mask*new_fine_slice.values +
            (1. - blend_mask)*new_slice.values
        )
        source_str += ' + fine'

        print('{:}: {:}'.format(sdate, source_str))
        slice_list.append(new_slice)

    # concatenate time slices
    regridded_field = xr.concat(slice_list, dim='time')

    # make output dataset
    # time_array = field_coarse['time']
    time_array = field_fine['time']
    lat_array = grid_target['lat']
    lon_array = grid_target['lon']
    coordinates = {'time': time_array, 'lat': lat_array, 'lon': lon_array}
    ds_out = xr.Dataset({var: (['time', 'lat', 'lon'], regridded_field)},
                        coords=coordinates)
    print(ds_out)
    # workaround to fix write: RuntimeError: NetCDF: Invalid argument
    # this is related to using unliminted_dims arg
    ds_out.time.encoding.pop('contiguous', None)

    if mode == 'monthly':
        date_str = date.strftime('y%Ym%m')
    elif mode == 'daily':
        date_str = date.strftime('y%Ym%md%d')
    elif mode == 'no-merge':
        date_str = date.strftime('y%Ym%md%d-%H')
    else:
        raise ValueError('Invalid mode: {:}'.format(mode))
    outfile = r'{:}_{:}_{:}.nc'.format(
        fileprefix, var, date_str)
    print('Saving to {:}'.format(outfile))
    ds_out.to_netcdf(path=outfile, mode='w', format='NETCDF4',
                     unlimited_dims={'time': True})


fileprefix = 'harmonie-blend'

grid_target = get_nemo_grid()

# months to process, end inclusive
start_date = datetime.datetime(2017, 9, 1)
end_date = datetime.datetime(2017, 9, 30)

var_list = [
    'T2m',
    'U10m',
    'V10m',
    'sfcpres',
]

# choose to process daily or monthly or 6-hour files
# mode = 'monthly'
# mode = 'daily'
mode = 'no-merge'

rule_dict = {
    'monthly': (dateutil.rrule.MONTHLY, 1),
    'daily': (dateutil.rrule.DAILY, 1),
    'no-merge': (dateutil.rrule.HOURLY, 6),
}

rule, interval = rule_dict[mode]
for date in dateutil.rrule.rrule(rule,
                                 interval=interval,
                                 dtstart=start_date,
                                 until=end_date):
    print('processing {:}'.format(date))
    for var in var_list:
        merge_monthly_files(date, var, grid_target, fileprefix, mode)
