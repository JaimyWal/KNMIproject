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
data_source = 'RACMO'

months = [10,11,12,1,2,3]
years = [1988, 2020]
lats = [20, 80]
lons = [-40, 60]

use_rotpole = True
cut_boundaries = False
plot_lats = [35, 70]
plot_lons = [-10, 35]

fit_against_gmst = True

#%% Dataset configurations

sources = ['Eobs', 'ERA5', 'RACMO']
file_key = next(src for src in sources if src in data_source)

if fit_against_gmst:
    fit_unit = '°C GMST'
    fit_scaling = 1
else:
    fit_unit = 'decade'
    fit_scaling = 10

plot_cfg = {
    'Tg': {
        'label_mean': r'Temperature (°C)',
        'label_trend': r'Absolute trend (°C / ' + fit_unit + ')',
        'cmap_mean': 'Spectral_r',
        'cmap_trend': cmaps.temp_19lev,
        'crange_mean': (-5, 20),
        'crange_trend': (-3, 3),
        'extreme_mean': ('xkcd:purple', 'xkcd:pink'),
        'extreme_trend': ("#000020", "#350000"),
    },
    'P': {
        'label_mean': 'Precipitation (mm)',
        'label_trend': 'Relative trend (% / ' + fit_unit + ')',
        'cmap_mean': cmocean.cm.rain,
        'cmap_trend': plt.get_cmap('BrBG', 20),
        'crange_mean': (0, 5),
        'crange_trend': (-60, 60),
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
    'Eobs_fine': {
        'Tg': 'tg_ens_mean_0.1deg_reg_v31.0e.nc',
        'P': 'rr_ens_mean_0.1deg_reg_v31.0e.nc',
    },
    'Eobs_coarse': {
        'Tg': 'tg_ens_mean_0.25deg_reg_v31.0e.nc',
        'P': 'rr_ens_mean_0.25deg_reg_v31.0e.nc',
    },

    'ERA5_fine': {
        'Tg': 'era5_fine.nc',
        'P': 'era5_fine.nc',
    },
    'ERA5_coarse': {
        'Tg': 'era5_coarse_full_t2m.nc',
        'P': 'era5_coarse_full_tp.nc',
    },

    'RACMO': {
        'Tg': 't2m',
        'P': 'precip',
    },
}

base_dir_cfg = {
    'Eobs': '/nobackup/users/walj/eobs',
    'ERA5': '/nobackup/users/walj/era5',
    'RACMO': '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Monthly_data',
}

cfg = {
    **plot_cfg[var],
    'variable': var_name_cfg[file_key][var],
    'file': file_cfg[data_source][var],
    'base_dir': base_dir_cfg[file_key],
}

#%% Data loading and processing

if months is None:
    months = np.arange(1, 13)

months = np.asarray(months, dtype=int)
month_start = int(months[0])
month_end   = int(months[-1])

years_req = list(years)

years_load = list(years_req)

# if the season crosses the year boundary (eg DJF)
# need December of the year before the first requested year
if month_start > month_end:
    years_load[0] = years_req[0] - 1

try:
    client = get_client()
except ValueError:
    client = Client(n_workers=1, threads_per_worker=8, processes=False)

input_file_data = os.path.join(cfg['base_dir'], cfg['file'])

if file_key == 'Eobs':
    data = preprocess_eobs_monthly(
        file_path=input_file_data,
        var_name=cfg['variable'],
        months=months,
        years=years_load,
        lats=lats,
        lons=lons,
        chunks_time=180,
        chunks_lat=200,
        chunks_lon=200,
    )

elif file_key == 'ERA5':
    data = preprocess_era5(
        file_path=input_file_data,
        var_name=cfg['variable'],
        months=months,
        years=years_load,
        lats=lats,
        lons=lons,
        chunks_time=180,
        chunks_lat=200,
        chunks_lon=200,
    )

elif file_key == 'RACMO':
    data = preprocess_racmo_monthly(
        dir_path=input_file_data,
        var_name=cfg['variable'],
        months=months,
        years=years_load,
        lats=lats,
        lons=lons,
        chunks_time=180,
        chunks_lat=200,
        chunks_lon=200,
        already_monthly=True,
    ).squeeze()

lat_vals = data['latitude'].values
lon_vals = data['longitude'].values


# Climate year agreggation:
month = data['time'].dt.month
year = data['time'].dt.year

# if season stays within calendar year  clim_year = calendar year
# if season crosses year boundary      months >= month_start belong to next year
if month_start <= month_end:
    clim_year = year
else:
    clim_year = xr.where(month >= month_start, year + 1, year)

data = data.assign_coords(clim_year=clim_year)

data_year = data.groupby('clim_year').mean('time')

data_year = data_year.sel(clim_year=slice(years_req[0], years_req[1]))

data_avg = data_year.mean(dim='clim_year').compute()

time_coord = pd.to_datetime(data_year['clim_year'].values.astype(int).astype(str))
data_year_time = data_year.assign_coords(time=('clim_year', time_coord)).swap_dims({'clim_year': 'time'})

if fit_against_gmst:
    file_GMST = '/nobackup/users/walj/era5/era5_gmst_anom.nc'
    data_GMST = xr.open_dataset(file_GMST)
    gmst_roll = data_GMST.rolling(time=5, center=False).mean()
    gmst_full = gmst_roll['GMST']

    gmst_sel = gmst_full.sel(time=data_year_time['time'])
    fit_coord = gmst_sel.astype(float)

else:
    fit_coord = data_year_time['clim_year'].astype(float)

data_fit = (
    data_year_time
    .rename({'time': 'fit_against'})
    .assign_coords(fit_against=('fit_against', fit_coord.values))
)

# Rotated projection:
proj = ccrs.PlateCarree()

if use_rotpole:

    rotpole_dir = '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/precip'
    rotpole_file = 'precip.KNMI-2000.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc'
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

# data_avg = data.mean(dim='time').compute()

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

fits = data_fit.polyfit(dim='fit_against', deg=1, skipna=True)

slope = fits.polyfit_coefficients.sel(degree=1)

trend = (slope*fit_scaling).astype('float32').compute()

if var == 'Tg':
    trend_plot = trend # Absolute trend
elif var == 'P':
    trend_plot = (trend / data_avg)*100 # Relative trend

plot_map(
    trend_plot,
    lon_vals,
    lat_vals,
    crange=cfg['crange_trend'],
    label=cfg['label_trend'],
    cmap=cfg['cmap_trend'],
    extreme_colors=cfg['extreme_trend'],
    c_ticks=10,
    show_x_ticks=True,
    show_y_ticks=True,
    y_ticks_num=False,
    y_ticks=5,
    x_ticks_num=False,
    x_ticks=10,
    figsize=(14, 12),
    extent=[*plot_lons, *plot_lats],
    proj=proj,
    rotated_grid=cut_boundaries
)


#%% Potential future code



# Te doen:
# - Nieuwe ERA5 data en kijken naar E-obs data
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
