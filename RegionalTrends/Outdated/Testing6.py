#%% Imports

import glob

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import dask
from pathlib import Path
import sys
from importlib import reload

# Custom libraries
PROJECT_ROOT = Path.home() / 'KNMIproject'
sys.path.insert(0, str(PROJECT_ROOT))

from RegionalTrends.Helpers import ProcessVar
reload(ProcessVar)
from RegionalTrends.Helpers.ProcessVar import load_var

from RegionalTrends.Helpers import AreaWeights
reload(AreaWeights)
from RegionalTrends.Helpers.AreaWeights import area_weights, area_weighted_mean

from RegionalTrends.Helpers import ComputeTendencies
reload(ComputeTendencies)
from RegionalTrends.Helpers.ComputeTendencies import construct_tendency, full_interval_time

import RegionalTrends.Helpers.Config.Constants as Constants
reload(Constants)

plt.rcParams['axes.unicode_minus'] = False
dask.config.set(scheduler='threads', num_workers=12)


#%% 

def tendency_contributions(da, split='Seasonal', weighted=False, start_month=None):

    time_index = pd.DatetimeIndex(da['time'].values)
    month_index = time_index.month.astype(int)

    if split == 'Monthly':
        split_keys = list(range(1, 13))
        split_index = month_index
        start_month = 1 if start_month is None else start_month
    else:
        split_keys = ['DJF', 'MAM', 'JJA', 'SON']
        split_index = np.array([
            'DJF' if month in [12, 1, 2] else
            'MAM' if month in [3, 4, 5] else
            'JJA' if month in [6, 7, 8] else
            'SON'
            for month in month_index
        ])
        start_month = 12 if start_month is None else start_month

    if split == 'Seasonal' and 'season_year' in da.coords:
        year_index = np.asarray(da['season_year'].values, dtype=int)
    elif start_month == 1:
        year_index = time_index.year.astype(int)
    else:
        year_index = (time_index.year + (month_index >= start_month)).astype(int)

    available_years = np.unique(year_index)
    target_years = available_years[:-1] if weighted else available_years

    def select_year(target_year):
        year_mask = xr.DataArray(
            year_index == target_year,
            dims='time',
            coords={'time': da['time']},
        )
        da_year = da.where(year_mask, drop=True)
        split_year = split_index[year_index == target_year]
        return da_year, split_year

    def make_weights(da_year, reverse=False):
        n_time = da_year.sizes['time']
        values = np.arange(n_time - 1, -1, -1) if reverse else np.arange(1, n_time + 1)
        return xr.DataArray(
            values.astype('float32') / n_time,
            dims='time',
            coords={'time': da_year['time']},
        )

    def split_sum(da_year, split_year, key, weights=None):
        time_sel = np.where(split_year == key)[0]
        out = da_year.isel(time=time_sel)
        if weights is not None:
            w = weights.isel(time=time_sel)
            out = out.where(w != 0, 0)
            out = out*w
        return out.sum('time', skipna=False)

    out = {}
    for key in split_keys:
        year_values = []
        for year in target_years:
            da_y, split_y = select_year(year)

            if weighted:
                da_y1, split_y1 = select_year(year + 1)
                value = (
                    split_sum(da_y, split_y, key, make_weights(da_y))
                    + split_sum(da_y1, split_y1, key, make_weights(da_y1, reverse=True))
                )
            else:
                value = split_sum(da_y, split_y, key)

            year_values.append(value)

        out[key] = (
            xr.concat(year_values, dim='year')
            .assign_coords(year=target_years)
            .astype('float32')
        )

    return out


def open_dataset(path):

    files = sorted(glob.glob(path))

    try:
        ds = xr.open_mfdataset(
            files,
            combine='by_coords',
            chunks='auto',
            decode_times=True,
        )

        return ds

    except Exception:

        ds = xr.open_mfdataset(
            files,
            combine='by_coords',
            chunks='auto',
            decode_times=False,
        )

        n_time = ds.sizes['time']
        new_time = pd.date_range(start='1979-01-01', periods=n_time, freq='MS')

        ds = ds.assign_coords(time=new_time)

        ds['time'].attrs = {}

        return ds

tendtot_raw = open_dataset('/nobackup_1/users/walj/racmo24/Threehourly/tendtot*.nc')
tendtot_raw_shifted = tendtot_raw.tendtot.shift(time=-1)
tendtot_cont_loaded = open_dataset('/nobackup_1/users/walj/racmo24/Seasonal/tendtot*cont*.nc')


#%%

tendtot_raw_cont = tendency_contributions(tendtot_raw_shifted, split='Seasonal', weighted=True)
tendtot_loaded_cont = tendency_contributions(tendtot_cont_loaded.tendtot, split='Seasonal', weighted=False)



