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

    if isinstance(lats, (list, tuple)) and len(lats) == 2:
        lat_slice = slice(*lats)
    elif lats is None:
        lat_slice = slice(None)

    if isinstance(lons, (list, tuple)) and len(lons) == 2:
        lon_slice = slice(*lons)
    elif lons is None:
        lon_slice = slice(None)

    time_sel = data_raw.time

    if months is not None:
        time_sel = time_sel.where(time_sel.dt.month.isin(months), drop=True)

    if years is not None:
        if isinstance(years, (list, tuple)) and len(years) == 2 and any(v is None for v in years):
            if years[0] is not None:
                time_sel = time_sel.where(time_sel.dt.year >= years[0], drop=True)
            if years[1] is not None:
                time_sel = time_sel.where(time_sel.dt.year <= years[1], drop=True)
        else:
            time_sel = time_sel.where(time_sel.dt.year.isin(years), drop=True)

    data_daily = (
        data_raw[var_name]
        .sel(latitude=lat_slice, longitude=lon_slice, time=time_sel)
        .astype('float32')
    )

    data_monthly = data_daily.resample(time='MS').mean('time')

    dt = pd.DatetimeIndex(data_monthly['time'].values)
    days_in_year = np.where(dt.is_leap_year, 366, 365)

    t_years = xr.DataArray(
        dt.year + (dt.dayofyear - 1) / days_in_year,
        coords={'time': data_monthly['time']},
    )

    data_monthly = (
        data_monthly
        .assign_coords(t_years=t_years)
        .chunk({'time': chunks_time,
                'latitude': chunks_lat,
                'longitude': chunks_lon})
        .persist()
    )

    return data_monthly

# To do:
# - Add options for different datasets (ERA5, KNMI)