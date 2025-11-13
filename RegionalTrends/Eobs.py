#%% Imports

# Standard libraries
import numpy as np
import xarray as xr
import pandas as pd
import colormaps as cmaps
import os
from dask.distributed import Client, get_client
import cmocean
from importlib import reload

# Custom functions
import PlotFigure
reload(PlotFigure)          
from PlotFigure import plot_map 

#%% User inputs

var = 'Tg'
months = None
years = [1970, None]
lats = [35, 72]
lons = [-12, 35]

#%% Data loading and processing

datasets = {
    'Tg': {
        'file': 'tg_ens_mean_0.1deg_reg_v31.0e.nc',
        'variable': 'tg',
        'label_mean': r'Temperature (°C)',
        'label_trend': r'Trend (°C / decade)',
        'cmap_mean': 'Spectral_r',
        'cmap_trend': cmaps.temp_19lev,
        'crange_mean': (-5, 20),
        'crange_trend': (-1, 1),
        'extreme_mean': ('xkcd:purple', 'xkcd:pink'),
        'extreme_trend': ('xkcd:purple', 'xkcd:orange')
    },
    'P': {
        'file': 'rr_ens_mean_0.1deg_reg_v31.0e.nc',
        'variable': 'rr',
        'label_mean': 'Precipitation (mm)',
        'label_trend': 'Trend (mm / decade)',
        'cmap_mean': cmocean.cm.rain,
        'cmap_trend': cmaps.temp_19lev,
        'crange_mean': (0, 5),
        'crange_trend': (-0.5, 0.5),
        'extreme_mean': (None, 'xkcd:blue'),
        'extreme_trend': ('xkcd:purple', 'xkcd:orange')
    }
}

try:
    client = get_client()
except ValueError:
    client = Client(n_workers=1, threads_per_worker=8, processes=False)

input_file_data = os.path.join('/nobackup/users/walj/eobs', datasets[var]['file'])
data_raw = xr.open_dataset(input_file_data, chunks='auto')

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
    if len(years) == 2 and any(v is None for v in years):
        if years[0] is not None:
            time_sel = time_sel.where(time_sel.dt.year >= years[0], drop=True)
        if years[1] is not None:
            time_sel = time_sel.where(time_sel.dt.year <= years[1], drop=True)
    else:
        time_sel = time_sel.where(time_sel.dt.year.isin(years), drop=True)

var_name = datasets[var]['variable']

data = (
    data_raw[var_name]
      .sel(latitude=lat_slice, longitude=lon_slice, time=time_sel)
      .astype('float32')                       
      .chunk({'time': 180, 'latitude': 200, 'longitude': 200})  
      .persist()                                
)

dt = pd.DatetimeIndex(data['time'].values)
days_in_year = np.where(dt.is_leap_year, 366, 365)
t_years = xr.DataArray(
    dt.year + (dt.dayofyear - 1) / days_in_year,
    coords={'time': data['time']}
)

data = data.assign_coords(t_years=t_years)

lat_vals = data['latitude'].values
lon_vals = data['longitude'].values

#%% Calculate mean and plot

data_avg = data.mean(dim='time').compute()

plot_map(data_avg, 
         lon_vals, 
         lat_vals, 
         datasets[var]['crange_mean'], 
         datasets[var]['label_mean'], 
         datasets[var]['cmap_mean'], 
         extreme_colors=datasets[var]['extreme_mean'],
         c_ticks=5,
         show_x_ticks=False,
         show_y_ticks=False
)

#%% Linear trends and plot

data_t = data.swap_dims({'time': 't_years'}).sortby('t_years')
fits = data_t.polyfit(dim='t_years', deg=1, skipna=True)

slope = fits.polyfit_coefficients.sel(degree=1)

trend_decade = (slope*10).astype('float32').compute()

plot_map(
    trend_decade,
    lon_vals,
    lat_vals,
    crange=datasets[var]['crange_trend'],
    label=datasets[var]['label_trend'],
    cmap=datasets[var]['cmap_trend'],
    extreme_colors=datasets[var]['extreme_trend'],
    c_ticks=10,
    show_x_ticks=False,
    show_y_ticks=False
)

#%%

# Figure for specifically de Bilt!