import numpy as np
import xarray as xr
import cartopy.crs as ccrs


def bounds_from_centers(coord):

    coord = np.asarray(coord)
    n = coord.size
    steps = coord[1:] - coord[:-1]

    bounds = np.empty(n + 1, dtype=coord.dtype)
    bounds[0] = coord[0] - 0.5*steps[0]
    bounds[1:n] = coord[:-1] + 0.5*steps
    bounds[-1] = coord[-1] + 0.5*steps[-1]

    return bounds


def rotated_bounds(ds_rot, rotpole_crs):

    rlat_1d = ds_rot['rlat'].values
    rlon_1d = ds_rot['rlon'].values

    rlat_b_1d = bounds_from_centers(rlat_1d)
    rlon_b_1d = bounds_from_centers(rlon_1d)

    rlon_b_2d, rlat_b_2d = np.meshgrid(rlon_b_1d, rlat_b_1d)

    plate = ccrs.PlateCarree()

    pts = plate.transform_points(rotpole_crs,
                                 rlon_b_2d,  
                                 rlat_b_2d) 

    lon_b = pts[..., 0]
    lat_b = pts[..., 1]

    return lat_b, lon_b


def grid_with_bounds(ds, rotpole_native=None):

    spatial_dims = [
        d for d in ds.dims
        if d in ('rlat', 'rlon', 'latitude', 'longitude')
    ]

    if 'rlat' in spatial_dims and 'rlon' in spatial_dims:

        dim_lat, dim_lon = 'rlat', 'rlon'
        lat = ds['latitude'].values
        lon = ds['longitude'].values 
        lat_b, lon_b = rotated_bounds(ds, rotpole_native)

    else:
        dim_lat, dim_lon = 'latitude', 'longitude'
        lat1d, lon1d = ds['latitude'].values, ds['longitude'].values
        lat_b1d = bounds_from_centers(lat1d)
        lon_b1d = bounds_from_centers(lon1d)

        lon, lat = np.meshgrid(lon1d, lat1d)
        lon_b, lat_b = np.meshgrid(lon_b1d, lat_b1d)

    grid_bounds =  xr.Dataset(
        {
            'lon':   ((dim_lat, dim_lon), lon),
            'lat':   ((dim_lat, dim_lon), lat),
            'lon_b': ((f'{dim_lat}_b', f'{dim_lon}_b'), lon_b),
            'lat_b': ((f'{dim_lat}_b', f'{dim_lon}_b'), lat_b),
        }
    )

    return grid_bounds
