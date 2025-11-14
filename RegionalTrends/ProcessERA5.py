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

    if isinstance(lats, (list, tuple)) and len(lats) == 2:
        lat_slice = slice(*lats[::-1]) # Note: ERA5 latitude is from north to south
    else:
        lat_slice = slice(None)

    if isinstance(lons, (list, tuple)) and len(lons) == 2:
        lon_slice = slice(*lons)
    else:
        lon_slice = slice(None)

    time_sel = data_raw['valid_time']

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

    data = (
        data_raw[var_name]
        .sel(latitude=lat_slice, longitude=lon_slice, valid_time=time_sel)
        .astype('float32')                                            
    )

    dt = pd.DatetimeIndex(data['valid_time'].values)
    days_in_year = np.where(dt.is_leap_year, 366, 365)

    t_years = xr.DataArray(
        dt.year + (dt.dayofyear - 1) / days_in_year,
        coords={'valid_time': data['valid_time']},
    )

    data_processed = (
        data
        .assign_coords(t_years=t_years)
        .chunk({'valid_time': chunks_time,
                'latitude': chunks_lat,
                'longitude': chunks_lon})
        .persist()
    )

    return data_processed