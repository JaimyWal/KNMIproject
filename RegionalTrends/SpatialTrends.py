#%% Imports

# Standard libraries
import numpy as np
import xarray as xr
import pandas as pd
import colormaps as cmaps
import matplotlib.pyplot as plt
import os
from dask.distributed import Client, get_client
import cmocean
from importlib import reload

# Custom functions
import PlotMaps
reload(PlotMaps)          
from PlotMaps import plot_map 

import ProcessEobs
reload(ProcessEobs)          
from ProcessEobs import preprocess_eobs_monthly 

import ProcessERA5
reload(ProcessERA5)          
from ProcessERA5 import preprocess_era5 

#%% User inputs

var = 'Tg'
data_source = 'ERA5'
resolution = 'Fine'

months = [10, 11, 12, 1, 2, 3]
years = [1961, 2000]
lats = [35, 72]
lons = [-12, 35]

# RACMO toevoegen?

#%% Dataset configurations

plot_cfg = {
    'Tg': {
        'label_mean': r'Temperature (°C)',
        'label_trend': r'Absolute trend (°C / decade)',
        'cmap_mean': 'Spectral_r',
        'cmap_trend': cmaps.temp_19lev,
        'crange_mean': (-5, 20),
        'crange_trend': (-1, 1),
        'extreme_mean': ('xkcd:purple', 'xkcd:pink'),
        'extreme_trend': ('xkcd:purple', 'xkcd:orange'),
    },
    'P': {
        'label_mean': 'Precipitation (mm)',
        'label_trend': 'Relative trend (% / decade)',
        'cmap_mean': cmocean.cm.rain,
        'cmap_trend': plt.get_cmap('BrBG', 20),
        'crange_mean': (0, 5),
        'crange_trend': (-15, 15),
        'extreme_mean': (None, 'xkcd:purple'),
        'extreme_trend': ('xkcd:orange', 'xkcd:purple'),
    },
}

var_name_cfg = {
    'Eobs': {
        'Tg': 'tg',
        'P': 'rr',
    },
    'ERA5': {
        'Tg': 't2m',
        'P': 'tp',
    },
}

file_cfg = {
    'Eobs': {
        'Fine': {
            'Tg': 'tg_ens_mean_0.1deg_reg_v31.0e.nc',
            'P':  'rr_ens_mean_0.1deg_reg_v31.0e.nc',
        },
        'Coarse': {
            'Tg': 'tg_ens_mean_0.25deg_reg_v31.0e.nc',
            'P':  'rr_ens_mean_0.25deg_reg_v31.0e.nc',
        },
    },
    'ERA5': {
        'Fine': {
            # both variables in one file
            'Tg': 'era5_fine.nc',
            'P':  'era5_fine.nc',
        },
        'Coarse': {
            'Tg': 'era5_coarse_t2m.nc',
            'P':  'era5_coarse_tp.nc',
        },
    },
}

base_dir_cfg = {
    'Eobs': '/nobackup/users/walj/eobs',
    'ERA5': '/nobackup/users/walj/era5',
}

cfg = {
    **plot_cfg[var],
    'variable': var_name_cfg[data_source][var],
    'file': file_cfg[data_source][resolution][var],
    'base_dir': base_dir_cfg[data_source],
}

#%% Data loading and processing

try:
    client = get_client()
except ValueError:
    client = Client(n_workers=1, threads_per_worker=8, processes=False)

input_file_data = os.path.join(cfg['base_dir'], cfg['file'])

if data_source == 'Eobs':
    data = preprocess_eobs_monthly(
        file_path=input_file_data,
        var_name=cfg['variable'],
        months=months,
        years=years,
        lats=lats,
        lons=lons,
        chunks_time=180,
        chunks_lat=200,
        chunks_lon=200,
    )
elif data_source == 'ERA5':
    data = preprocess_era5(
        file_path=input_file_data,
        var_name=cfg['variable'],
        months=months,
        years=years,
        lats=lats,
        lons=lons,
        chunks_time=180,
        chunks_lat=200,
        chunks_lon=200,
    )

lat_vals = data['latitude'].values
lon_vals = data['longitude'].values

#%% Calculate mean and plot

data_avg = data.mean(dim='time').compute()

# plot_map(data_avg, 
#          lon_vals, 
#          lat_vals, 
#          cfg['crange_mean'], 
#          cfg['label_mean'], 
#          cfg['cmap_mean'], 
#          extreme_colors=cfg['extreme_mean'],
#          c_ticks=5,
#          show_x_ticks=False,
#          show_y_ticks=False
# )

#%% Linear trends and plot

data_t = data.swap_dims({'time': 't_years'}).sortby('t_years')

fits = data_t.polyfit(dim='t_years', deg=1, skipna=True)

slope = fits.polyfit_coefficients.sel(degree=1)

trend_decade = (slope*10).astype('float32').compute()

if var == 'Tg':
    trend_decade = trend_decade
elif var == 'P':
    trend_decade = (trend_decade / data_avg)*100

plot_map(
    trend_decade,
    lon_vals,
    lat_vals,
    crange=cfg['crange_trend'],
    label=cfg['label_trend'],
    cmap=cfg['cmap_trend'],
    extreme_colors=cfg['extreme_trend'],
    c_ticks=10,
    show_x_ticks=False,
    show_y_ticks=False
)