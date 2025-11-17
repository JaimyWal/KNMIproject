import numpy as np
import xarray as xr
import pandas as pd

def preprocess_era5(
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

    latitudes = data_raw['latitude'].values   
    longitudes = data_raw['longitude'].values

    if isinstance(lats, (list, tuple)) and len(lats) == 2:
        lat_min, lat_max = lats # note: descending in ERA5
        lat_slice = slice(lat_max, lat_min)
        da = da.sel(latitude=lat_slice)
    elif isinstance(lats, (int, float, np.floating)):
        lat_idx = np.abs(latitudes - float(lats)).argmin()
        nearest_lat = float(latitudes[lat_idx])
        da = da.sel(latitude=slice(nearest_lat, nearest_lat))

    if isinstance(lons, (list, tuple)) and len(lons) == 2:
        lon_slice = slice(*lons)
        da = da.sel(longitude=lon_slice)
    elif isinstance(lons, (int, float, np.floating)):
        lon_idx = np.abs(longitudes - float(lons)).argmin()
        nearest_lon = float(longitudes[lon_idx])
        da = da.sel(longitude=slice(nearest_lon, nearest_lon))

    time_sel = da['valid_time']

    if months is not None:
        time_sel = time_sel.where(time_sel.dt.month.isin(months), drop=True)

    if years is not None and isinstance(years, (list, tuple)):
        if len(years) == 2:
            start, end = years
            if start is not None:
                time_sel = time_sel.where(time_sel.dt.year >= start, drop=True)
            if end is not None:
                time_sel = time_sel.where(time_sel.dt.year <= end, drop=True)
        else:
            time_sel = time_sel.where(time_sel.dt.year.isin(years), drop=True)

    data = da.sel(valid_time=time_sel)

    data = data.rename({'valid_time': 'time'})

    if var_name == 't2m':
        data = data - 273.15

    elif var_name == 'tp':
        data = data*1000

    dt = pd.DatetimeIndex(data['time'].values)
    days_in_year = np.where(dt.is_leap_year, 366, 365)

    t_years = xr.DataArray(
        dt.year + (dt.dayofyear - 1) / days_in_year,
        coords={'time': data['time']},
    )

    data_processed = (
        data
        .assign_coords(t_years=t_years)
        .chunk({'time': chunks_time,
                'latitude': chunks_lat,
                'longitude': chunks_lon})
        .persist()
    )

    return data_processed