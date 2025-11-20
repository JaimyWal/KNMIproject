import os
import glob
import numpy as np
import xarray as xr
import pandas as pd

xr.set_options(use_new_combine_kwarg_defaults=True)

def preprocess_racmo_monthly(
    dir_path,
    var_name,
    months=None,
    years=None,
    lats=None,
    lons=None,
    chunks_time=180,
    chunks_lat=200,
    chunks_lon=200,
):
    file_pattern = '*.nc'

    file_list = sorted(glob.glob(os.path.join(dir_path, file_pattern)))

    data_raw = xr.open_mfdataset(
        file_list,
        combine='by_coords',
        chunks='auto'
    )

    # Select variable and cast to float32
    da = data_raw[var_name].astype('float32')

    lat2d = data_raw['lat']   # dims: (rlat, rlon)
    lon2d = data_raw['lon']   # dims: (rlat, rlon)

    # spatial selection
    if isinstance(lats, (list, tuple)) and len(lats) == 2 and \
       isinstance(lons, (list, tuple)) and len(lons) == 2:

        lat_min, lat_max = sorted(lats)
        lon_min, lon_max = sorted(lons)

        mask = (
            (lat2d >= lat_min) & (lat2d <= lat_max) &
            (lon2d >= lon_min) & (lon2d <= lon_max)
        ).compute()

        da = da.where(mask, drop=True)

    # nearest grid point to a single (lat, lon) 
    elif isinstance(lats, (int, float)) and \
         isinstance(lons, (int, float)):

        target_lat = float(lats)
        target_lon = float(lons)

        # squared distance in lat lon space
        dist2 = (lat2d - target_lat)**2 + (lon2d - target_lon)**2

        # find indices of minimum distance
        ii, jj = np.unravel_index(
            np.argmin(dist2.values),
            dist2.shape
        )

        da = da.isel(rlat=ii, rlon=jj)

    if var_name == 't2m':
        da = da - 273.15
    elif var_name == 'tp':
        da = da * 1000.0

    data_monthly_full = (
        da
        .resample(time='MS')
        .mean('time')
    )

    time_sel = data_monthly_full.time

    # Month selection
    if months is not None:
        time_sel = time_sel.where(time_sel.dt.month.isin(months), drop=True)

    # Year selection
    if years is not None and isinstance(years, (list, tuple)):
        if len(years) == 2:
            start, end = years
            if start is not None:
                time_sel = time_sel.where(time_sel.dt.year >= start, drop=True)
            if end is not None:
                time_sel = time_sel.where(time_sel.dt.year <= end, drop=True)
        else:
            time_sel = time_sel.where(time_sel.dt.year.isin(years), drop=True)

    data_monthly = (
        data_monthly_full
        .sel(time=time_sel)
        .chunk({'time': chunks_time,
                'rlat': chunks_lat,
                'rlon': chunks_lon})
        .persist()
    )

    return data_monthly

test = preprocess_racmo_monthly('/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/t2m', 
                                't2m',
                                lats=[50, 60],
                                lons=[0, 10],
                                months=[6,7,8],
                                years=[2000, 2020])









# import cartopy.crs as ccrs

# racmo = xr.open_dataset('/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/t2m/t2m.KNMI-2000.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')

# rp = racmo['rotated_pole']  # or whatever the name is in the file
# pole_lat = rp.grid_north_pole_latitude
# pole_lon = rp.grid_north_pole_longitude
# central_rlon = 18.0


# rotpole = ccrs.RotatedPole(
#     pole_latitude=pole_lat,
#     pole_longitude=pole_lon,
#     central_rotated_longitude=central_rlon,
# )