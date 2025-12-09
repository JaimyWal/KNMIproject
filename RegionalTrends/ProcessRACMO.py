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
    already_monthly=False):
    
    file_pattern = '*.nc'

    file_list = sorted(glob.glob(os.path.join(dir_path, file_pattern)))

    if already_monthly:
        data_raw = xr.open_mfdataset(
            file_list,
            combine='by_coords',
            chunks='auto',
            decode_times=False
        )

        time_var = data_raw['time']
        n_time = time_var.sizes['time']

        time_index = pd.date_range('1979-01-01', periods=n_time, freq='MS')

        data_raw = data_raw.assign_coords(time=time_index)

    else:
        data_raw = xr.open_mfdataset(
            file_list,
            combine='by_coords',
            chunks='auto'
        )

    data_raw = data_raw.rename({'lat': 'latitude', 'lon': 'longitude'})

    # Select variable and cast to float32
    da = data_raw[var_name].astype('float32')

    lat2d = data_raw['latitude']   # dims: (rlat, rlon)
    lon2d = data_raw['longitude']   # dims: (rlat, rlon)

    # spatial selection
    if isinstance(lats, (list, tuple)) and len(lats) == 2 and \
       isinstance(lons, (list, tuple)) and len(lons) == 2:

        lat_min, lat_max = sorted(lats)
        lon_min, lon_max = sorted(lons)

        mask = (
            (lat2d >= lat_min) & (lat2d <= lat_max) &
            (lon2d >= lon_min) & (lon2d <= lon_max)
        ).compute()

        # valid_rlat = mask.any('rlon')
        # valid_rlon = mask.any('rlat')

        # da = da.sel(rlat=valid_rlat, rlon=valid_rlon)

        da = da.where(mask, drop=True)

    # nearest grid point to a single (lat, lon) 
    elif isinstance(lats, (int, float)) and isinstance(lons, (int, float)):

        target_lat = float(lats)
        target_lon = float(lons)

        # convert to radians
        lat_rad = np.deg2rad(lat2d)
        lon_rad = np.deg2rad(lon2d)
        target_lat_rad = np.deg2rad(target_lat)
        target_lon_rad = np.deg2rad(target_lon)

        # haversine great circle distance
        dlat = lat_rad - target_lat_rad
        dlon = lon_rad - target_lon_rad

        a = (
            np.sin(dlat / 2.0)**2
            + np.cos(target_lat_rad)*np.cos(lat_rad)*np.sin(dlon / 2.0)**2
        )

        R = 6371.0  # Earth radius in km
        dist = 2.0*R*np.arcsin(np.sqrt(a))  # distance in km

        # find indices of minimum distance
        ii, jj = np.unravel_index(
            np.nanargmin(dist.values),
            dist.shape
        )

        da = da.isel(rlat=[ii], rlon=[jj])

    if var_name == 't2m':
        da = da - 273.15
        da.attrs['units'] = 'degC'
    elif var_name == 'precip':
        # from kg/m2/s to mm/day
        da = da*86400.0
        da.attrs['units'] = 'mm/day'

    if already_monthly: 
        data_full = da
    else:
        data_full = (
            da
            .resample(time='MS')
            .mean('time')
        )

    time_sel = data_full.time

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
        data_full
        .sel(time=time_sel)
        .chunk({'time': chunks_time,
                'rlat': chunks_lat,
                'rlon': chunks_lon})
        .persist()
    )

    return data_monthly

# test_daily = preprocess_racmo_monthly(
#     dir_path='/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/t2m',
#     var_name='t2m',
#     months=[6,7,8],
#     years=(2000, 2010),
#     lats=(50, 70),
#     lons=(0, 20)
# )

# test_monthly = preprocess_racmo_monthly(
#     dir_path='/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Monthly_data/t2m',
#     var_name='t2m',
#     months=[6,7,8],
#     years=(2000, 2010),
#     lats=(50, 70),
#     lons=(0, 20),
#     already_monthly=True
# )