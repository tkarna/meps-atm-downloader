import iris
import numpy


def check_file(f):

    print('Loading file {:}'.format(f))
    cube_list = iris.load(f)

    for cube in cube_list:
        print('Checking field {:}'.format(cube.standard_name))

        ntime = cube.shape[0]
        time_coord = cube.coord('time')

        nbadframes = 0
        for i in range(ntime):
            date = time_coord.units.num2date(time_coord.points[i])
            if not numpy.all(numpy.isfinite(cube[i, :, :].data)):
                print('{:}: bad values'.format(date))
                nbadframes += 1

        if nbadframes > 0:
            print('FAILED: found {:} time frames with invalid data'.format(nbadframes))
        else:
            print('All data is valid.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Check validity of netcdf file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('ncfile',
                        help='netCDF file to check.')
    args = parser.parse_args()

    check_file(args.ncfile)
