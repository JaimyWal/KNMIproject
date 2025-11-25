import numpy as np
import xarray as xr
import pandas as pd

def preprocess_eobs_monthly(
    file_path,
    var_name,
    months=None,
    years=None,
    lats=None,
    lons=None,
    chunks_time=180,
    chunks_lat=200,
    chunks_lon=200):

    data_raw = xr.open_dataset(file_path, chunks='auto')

    da = data_raw[var_name].astype('float32')

    lat1d = data_raw['latitude']  
    lon1d = data_raw['longitude']

    lat2d, lon2d = xr.broadcast(lat1d, lon1d)
    
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

        da = da.isel(latitude=[ii], longitude=[jj])

    data_monthly_full = (
        da
        .resample(time='MS')
        .mean('time')
    )

    time_sel = data_monthly_full.time

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
                'latitude': chunks_lat,
                'longitude': chunks_lon})
        .persist()
    )

    return data_monthly