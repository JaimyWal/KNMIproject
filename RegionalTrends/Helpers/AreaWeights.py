import numpy as np
import xarray as xr
from pyproj import Geod
from importlib import reload
import cartopy.crs as ccrs

from RegionalTrends.Helpers import GridBounds
reload(GridBounds)          
from RegionalTrends.Helpers.GridBounds import bounds_from_centers, rotated_bounds


def cell_areas_from_bounds(lat_b, lon_b, ellps='WGS84'):

    geod = Geod(ellps=ellps)

    lonSW, latSW = lon_b[:-1, :-1], lat_b[:-1, :-1]
    lonSE, latSE = lon_b[:-1,  1:], lat_b[:-1,  1:]
    lonNE, latNE = lon_b[ 1:,  1:], lat_b[ 1:,  1:]
    lonNW, latNW = lon_b[ 1:, :-1], lat_b[ 1:, :-1]

    ny, nx = lonSW.shape
    areas = np.empty((ny, nx), dtype=float)

    for jj in range(ny):
        for ii in range(nx):
            lons = [lonSW[jj, ii], lonSE[jj, ii], lonNE[jj, ii], lonNW[jj, ii]]
            lats = [latSW[jj, ii], latSE[jj, ii], latNE[jj, ii], latNW[jj, ii]]
            grid_area, _ = geod.polygon_area_perimeter(lons, lats)
            areas[jj, ii] = abs(grid_area)

    return areas


def area_weights(ds, rotpole_native=None):

    # Handle station dimension (from station-based grid selection)
    if 'station' in ds.dims and 'station_lat' in ds.coords and 'station_lon' in ds.coords:
        station_lats = ds['station_lat'].values
        station_lons = ds['station_lon'].values
        
        if rotpole_native is not None and not isinstance(rotpole_native, ccrs.PlateCarree):
            # Transform geographic coords to rotated coords, then use cos(rlat)
            plate = ccrs.PlateCarree()
            rotated_pts = rotpole_native.transform_points(plate, station_lons, station_lats)
            rlats = rotated_pts[:, 1]
            cos_weights = np.cos(np.deg2rad(rlats))
        else:
            # Regular lat/lon grid: use cos of geographic latitude
            cos_weights = np.cos(np.deg2rad(station_lats))
        
        return xr.DataArray(
            cos_weights,
            dims=('station',),
            coords={'station': ds['station']},
            name='area_weight'
        )

    spatial_dims = [
        d for d in ds.dims
        if d in ('rlat', 'rlon', 'latitude', 'longitude')
    ]

    if not spatial_dims:
        return None
    
    if 'rlat' in spatial_dims and 'rlon' in spatial_dims:
        dim_lat, dim_lon = 'rlat', 'rlon'
        lat_b, lon_b = rotated_bounds(ds, rotpole_native)
    else:
        dim_lat, dim_lon = 'latitude', 'longitude'
        lat1d, lon1d = ds['latitude'], ds['longitude']
        lat_b1d = bounds_from_centers(lat1d.values)
        lon_b1d = bounds_from_centers(lon1d.values)
        lon_b, lat_b = np.meshgrid(lon_b1d, lat_b1d)

    areas = cell_areas_from_bounds(lat_b, lon_b)

    area_weights = xr.DataArray(
        areas,
        dims=(dim_lat, dim_lon),
        coords={dim_lat: ds[dim_lat], dim_lon: ds[dim_lon]},
        name='cell_area',
        attrs={'units': 'm2'}
    )

    return area_weights


def area_weighted_mean(ds, rotpole_native=None, weights=None):
    
    # Handle station dimension first
    if 'station' in ds.dims:
        if weights is None:
            weights = area_weights(ds, rotpole_native=rotpole_native)
        
        if weights is None:
            return ds.mean(dim='station')
        
        mask = ds.notnull()
        w_masked = weights.where(mask)
        
        weighted_sum = (ds * w_masked).sum(dim='station')
        sum_of_weights = w_masked.sum(dim='station')
        
        area_mean = weighted_sum / sum_of_weights
        area_mean.attrs.update(ds.attrs)
        return area_mean
     
    spatial_dims = [
        d for d in ds.dims
        if d in ('rlat', 'rlon', 'latitude', 'longitude')
    ]
     
    if not spatial_dims:
        return ds

    if weights is None:
        weights = area_weights(ds, rotpole_native=rotpole_native)

    mask = ds.notnull()
    w_masked = weights.where(mask)

    weighted_sum = (ds*w_masked).sum(dim=spatial_dims)
    sum_of_weights = w_masked.sum(dim=spatial_dims)

    area_mean = weighted_sum / sum_of_weights
    area_mean.attrs.update(ds.attrs)

    return area_mean