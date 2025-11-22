#%% Imports

# Standard libraries
import numpy as np
import xarray as xr
import pandas as pd
import colormaps as cmaps
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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

import ProcessRACMO
reload(ProcessRACMO)          
from ProcessRACMO import preprocess_racmo_monthly 

#%% User inputs

var = 'P'
data_source = 'ERA5'
resolution = 'Coarse'

months = [4,5,6,7,8,9]
years = [1987, 2024]
lats = [35, 72]
lons = [-12, 35]

use_rotpole = True
cut_boundaries = False
plot_lats = lats
plot_lons = lons

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
        'extreme_trend': ("#271500", "#001f1f"),
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
    'RACMO': {
        'Tg': 't2m',
        'P': 'precip',
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
            'Tg': 'era5_fine.nc',
            'P':  'era5_fine.nc',
        },
        'Coarse': {
            'Tg': 'era5_coarse_t2m.nc',
            'P':  'era5_coarse_tp.nc',
        },
    },
    'RACMO': {
        'Any': {
            'Tg': 't2m',      
            'P':  'precip',  
        },
    },
}

base_dir_cfg = {
    'Eobs': '/nobackup/users/walj/eobs',
    'ERA5': '/nobackup/users/walj/era5',
    'RACMO': '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data',
}

if data_source == 'RACMO':
    res_key = 'Any'
else:
    res_key = resolution

cfg = {
    **plot_cfg[var],
    'variable': var_name_cfg[data_source][var],
    'file': file_cfg[data_source][res_key][var],
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

elif data_source == 'RACMO':
    data = preprocess_racmo_monthly(
        dir_path=input_file_data,
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

time = data['time']

t_years = xr.DataArray(
    time.dt.year + time.dt.month / 12.0,
    coords={'time': time},
    dims='time'
)

data = data.assign_coords(t_years=t_years).squeeze()

#%%

proj = ccrs.PlateCarree()

if use_rotpole:

    if hasattr(data, 'coords') and 'rotated_pole' in data.coords:
        rp = data['rotated_pole']
    else:
        rotpole_dir = '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/t2m'
        rotpole_file = 't2m.KNMI-2000.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc'
        data_rotpole = xr.open_dataset(os.path.join(rotpole_dir, rotpole_file))

        rp = data_rotpole['rotated_pole']

    pole_lat = rp.grid_north_pole_latitude
    pole_lon = rp.grid_north_pole_longitude
    central_rlon = 18.0

    rotpole = ccrs.RotatedPole(
        pole_latitude=pole_lat,
        pole_longitude=pole_lon,
        central_rotated_longitude=central_rlon,
    )

    proj = rotpole

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

fits = data.polyfit(dim='t_years', deg=1, skipna=True)

slope = fits.polyfit_coefficients.sel(degree=1)

trend_decade = (slope*10).astype('float32').compute()

if var == 'Tg':
    trend_decade = trend_decade # Absolute trend
elif var == 'P':
    trend_decade = (trend_decade / data_avg)*100 # Relative trend

#%%

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
    show_y_ticks=False,
    figsize=(14, 12),
    extent=[*plot_lons, *plot_lats],
    proj=proj,
    rotated_grid=cut_boundaries
)


#%% Potential future code



# Te doen:
# - Nieuwe ERA5 data en kijken naar E-obs data
# - Trends tegen temperatuur opzetten
# - Regridden misschien

# import xarray as xr
# import numpy as np
# import xesmf as xe

# # source data: single RACMO field on native grid
# da_src = racmo['t2m'].isel(time=0)                  # (rlat, rlon)
# src = xr.Dataset(
#     {
#         't2m': da_src
#     },
#     coords={
#         'lat': (('rlat', 'rlon'), racmo['lat'].values),
#         'lon': (('rlat', 'rlon'), racmo['lon'].values),
#     }
# )

# # target regular lat-lon grid
# lat_out = np.arange(35, 72.01, 0.1)
# lon_out = np.arange(-12, 35.01, 0.1)
# dst = xr.Dataset(
#     {
#         'lat': (('lat',), lat_out),
#         'lon': (('lon',), lon_out),
#     }
# )

# # build and apply regridder
# regridder = xe.Regridder(src, dst, 'bilinear')   # or 'conservative'
# t2m_reg = regridder(src['t2m'])                  # shape (lat, lon)

# # now plot on a truly regular PlateCarree grid
# plot_map(
#     data=t2m_reg.values,
#     lon=t2m_reg['lon'].values,
#     lat=t2m_reg['lat'].values,
#     label='T2m (°C)',
#     proj=ccrs.PlateCarree(),
#     rotated_grid=False,
#     extent=[-12, 35, 35, 72],
#     x_ticks_num=False,
#     x_ticks=15,
#     y_ticks_num=False,
#     y_ticks=5,
#     figsize=(14, 12)
# )
