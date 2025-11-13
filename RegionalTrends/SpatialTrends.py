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
import PlotMaps
reload(PlotMaps)          
from PlotMaps import plot_map 

import AggregateMonthly
reload(AggregateMonthly)          
from AggregateMonthly import preprocess_eobs_monthly 

#%% User inputs

var = 'Tg'
data = 'Eobs' # Include options for ERA5 and KNMI (and also aggregate)
months = [12,1,2]
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

data = preprocess_eobs_monthly(
    file_path=input_file_data,
    var_name=datasets[var]['variable'],
    months=months,
    years=years,
    lats=lats,
    lons=lons,
    chunks_time=180,
    chunks_lat=200,
    chunks_lon=200
)

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