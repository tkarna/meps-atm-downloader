from siri_omen import *

cube = iris.load_cube('test_nnordic_air_temperature_2m.nc')
# add coord metadata
cube.coord('lon').standard_name = 'longitude'
cube.coord('lat').standard_name = 'latitude'

nemonordic_extent = [-4.5, 30.5, 48.0, 66.0]
go = plot_map.GeographicPlot(extent=nemonordic_extent)
p = go.add_cube(cube[0, :, :], cmap='magma', vmin=243, vmax=283)
# plt.savefig('testplot.png', dpi=200, bbox_inches='tight')


import netCDF4
bathy = netCDF4.Dataset('bathy_meter_mod_minvalue.nc')
landmask = bathy['mask'][:]
mask_cube = cube[0, :, :].copy()
mask_cube.data[:] = landmask
mask_cube.data[landmask == 0] = numpy.nan

go.add_cube(mask_cube, alpha=0.2, kind='contourf', zorder=3, colors='DodgerBlue')

go.add_colorbar(p)
plt.savefig('testplot.png', dpi=200, bbox_inches='tight')
