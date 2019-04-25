import iris
import numpy
import datetime
import dateutil.rrule
from dateutil.relativedelta import relativedelta
from iris.experimental.equalise_cubes import equalise_attributes

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

url_pattern = 'http://thredds.met.no/thredds/dodsC/meps25epsarchive/{year:}/{month:02d}/{day:02d}/meps_subset_2_5km_{year:}{month:02d}{day:02d}T{hour:02d}Z.nc'

st_name_map = {
    'T2m': 'air_temperature',
    'U10m': 'x_wind',
    'V10m': 'y_wind',
    'radsw': '',
    'radlw': '',
    'RH2M': 'relative_humidity',
    'Q2': '',
    'precip': 'precipitation_amount',
    'snow': 'snowfall_amount',
    'sfcpres': 'surface_air_pressure',
}

ncvar_name_map = {
    'T2m': 'air_temperature_2m',
    'U10m': 'x_wind_10m',
    'V10m': 'y_wind_10m',
    'radsw': '',
    'radlw': '',
    'RH2M': 'relative_humidity_2m',
    'Q2': '',
    'precip': 'precipitation_amount_acc',
    'snow': 'snowfall_amount_acc',
    'sfcpres': 'surface_air_pressure',
}


def load_cube(url, standard_name, ncvar_name):
    cube_list = iris.load(url, standard_name)
    for c in cube_list:
        if c.var_name == ncvar_name:
            return c
    raise IOError(
        'Could not find field {:} ({:}) in {:}'.format(
            standard_name,
            ncvar_name,
            url)
    )


def get_cube_datetime(cube, index):
    time = cube.coord('time')
    return time.units.num2date(time.points[index])


def constrain_cube_time(cube, start_time=None, end_time=None):
    """
    Constrain time axis between start_time and end_time

    :kwarg datetime start_time: first time stamp to be included
    :kwarg datetime end_time: last time stamp to be included
    :returns: an iris Cube instance
    :raises: AssertionError if requested time period out of range
    """
    if start_time is None:
        start_time = get_cube_datetime(cube, 0)
    if end_time is None:
        end_time = get_cube_datetime(cube, -1)
    # convert to float in cube units
    time_coord = cube.coord('time')
    assert end_time >= get_cube_datetime(cube, 0), \
        'No overlapping time period found. end_time before first time stamp.'
    assert start_time <= get_cube_datetime(cube, -1), \
        'No overlapping time period found. start_time after last time stamp.'
    time_constrain = iris.Constraint(
        coord_values={'time': lambda t: start_time <= t.point <= end_time})
    new_cube = cube.extract(time_constrain)
    return new_cube


def load_harmonie_cube(date, var, start_time=None, end_time=None):
    print('  downloading {:} for date {:}'.format(var, date))
    url = url_pattern.format(year=date.year, month=date.month,
                             day=date.day, hour=date.hour)
    print('  {:}'.format(url))
    st_name = st_name_map[var]
    nc_name = ncvar_name_map[var]
    cube = load_cube(url, st_name, nc_name)

    # remove height
    cube = cube[:, 0, :, :, :]
    # take first realization
    cube = cube[:, 0, :, :]
    cube.remove_coord('projection_x_coordinate')
    cube.remove_coord('projection_y_coordinate')
    cube.remove_coord('realization')

    time_coord = cube.coord('time')
    if time_coord not in cube.dim_coords:
        # time coord is malformed, probably missing time frames
        # try to fix by removing invalid time stamps
        good_time_ix = time_coord.points < 1e20
        cube = cube[good_time_ix, :, :]
        time_coord = iris.coords.DimCoord.from_coord(cube.coord('time'))
        cube.remove_coord('time')
        cube.add_dim_coord(time_coord, 0)

    if start_time is not None or end_time is not None:
        cube = constrain_cube_time(cube, start_time, end_time)
    if start_time is not None and end_time is not None:
        hours = (end_time - start_time).total_seconds()/3600. + 1
        hours = int(numpy.round(hours))
        ntime = len(cube.coord('time').points)
        msg = 'Wrong nb of time steps: {:}, expected {:}'.format(ntime, hours)
        assert ntime == hours, msg

    return cube


def load_harmonie_cube_retry(date, var, start_time=None, end_time=None,
                             nattempts=5):
    """
    Tries to download a field nattempts times.
    """
    cube = None
    error = None
    for i in range(nattempts):
        try:
            cube = load_harmonie_cube(date, var, start_time, end_time)
        except (OSError, AttributeError, AssertionError) as e:
            error = e
            continue
        break
    if cube is None:
        raise error
    return cube


def load_harmonie_cube_recursive(date, var, start_time=None, end_time=None,
                                 offset_hours=0, increment=6,
                                 max_increment=48):
    try_again = False
    try:
        shift_date = date - relativedelta(hours=offset_hours)
        cube = load_harmonie_cube_retry(shift_date, var, start_time, end_time)
    except (OSError, AttributeError, AssertionError) as e:
        if increment >= max_increment:
            print('  ** Raising error: offset {:} >= {:}'.format(increment, max_increment))
            raise e
        else:
            try_again = True

    if try_again:
        new_offset = offset_hours + increment
        print('  ** Failed, reading from previous run: offset = {:} h'.format(new_offset))
        cube = load_harmonie_cube_recursive(date, var,
                                            start_time, end_time,
                                            new_offset, increment,
                                            max_increment)

    return cube


def load_harmonie_month(start_date, var):
    cube_list = iris.cube.CubeList()
    hours = 6
    last_date = start_date + relativedelta(months=1, hours=-hours)
    rule = dateutil.rrule.rrule(freq=dateutil.rrule.HOURLY, interval=hours,
                                dtstart=start_date, until=last_date)
    for date in rule:
        print('Fetching {:} for date {:}'.format(var, date))
        # set time span
        start_time = date
        end_time = date + relativedelta(hours=hours - 1)
        cube = load_harmonie_cube_recursive(date, var, start_time, end_time)
        time_constrain = iris.Constraint(coord_values={'time': lambda t: date <= t.point <= end_time})
        cube = cube.extract(time_constrain)
        print('Obtained data for {:} -> {:}'.format(
            get_cube_datetime(cube, 0), get_cube_datetime(cube, -1)))
        # clean unnecessary metadata
        cube.attributes.pop('min_time')
        cube.attributes.pop('max_time')
        cube.attributes.pop('_ChunkSizes')
        cube_list.append(cube)
    equalise_attributes(cube_list)
    cube = cube_list.concatenate_cube()
    return cube


name = 'harmonie'
# fetch for these months (end inclusive)
start_date = datetime.datetime(2017, 4, 1)
end_date = datetime.datetime(2017, 6, 1)

var_list = [
    'T2m',
    'U10m',
    'V10m',
    #'RH2M',
    #'precip',
    #'snow',
    'sfcpres',
]


for date in dateutil.rrule.rrule(dateutil.rrule.MONTHLY,
                                 dtstart=start_date,
                                 until=end_date):
    for var in var_list:
        #try:
            cube = load_harmonie_month(date, var)
            print(cube)
            print('Time span {:} -> {:}'.format(
                get_cube_datetime(cube, 0), get_cube_datetime(cube, -1)))
            filename = '{name:}_{var:}_y{year:}m{month:02d}.nc'.format(
                name=name, var=var, year=date.year, month=date.month)
            print('Saving {:}'.format(filename))
            iris.save(cube, filename)
        #except Exception as e:
           #print('Failed {:} {:}'.format(var, date))
           #print(e)

