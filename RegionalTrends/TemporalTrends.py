#%% Imports

# Standard libraries
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
from importlib import reload

# Custom functions
import ProcessEobs
reload(ProcessEobs)          
from ProcessEobs import preprocess_eobs_monthly 

import ProcessERA5
reload(ProcessERA5)          
from ProcessERA5 import preprocess_era5 

import ProcessKNMI
reload(ProcessKNMI)          
from ProcessKNMI import preprocess_knmi_monthly

import ProcessRACMO
reload(ProcessRACMO)          
from ProcessRACMO import preprocess_racmo_monthly 

#%% User inputs

var = 'P'
location = 'Cabauw'
data_sources = ['KNMI', 'RACMO_monthly', 'ERA5_coarse']

months = [12, 1, 2]
years = [1988, 2020]

fit_against_gmst = True

#%% Dataset configurations

if fit_against_gmst:
    fit_unit = '°C GMST'
    fit_scaling = 1
    fit_x_label = 'GMST Anomaly (°C)'
else:
    fit_unit = 'decade'
    fit_scaling = 10
    fit_x_label = 'Year'

plot_cfg = {
    'Tg': {
        'ylabel_monthly': 'Temperature (°C)',
        'ylabel_fit': 'Temperature (°C)',
        'trend_unit': '°C / ' + fit_unit,
        'ylim_monthly': None,
        'ylim_fit': None,
    },
    'P': {
        'ylabel_monthly': 'Precipitation (mm)',
        'ylabel_fit': 'Daily precipitation (mm)',
        'trend_unit': 'mm / ' + fit_unit,
        'ylim_monthly': None,
        'ylim_fit': None,
    },
}

# Variable names in each dataset
var_name_cfg = {
    'Eobs_fine': {
        'Tg': 'tg',
        'P': 'rr',
    },
    'Eobs_coarse': {
        'Tg': 'tg',
        'P': 'rr',
    },

    'ERA5_fine': {
        'Tg': 't2m',
        'P': 'tp',
    },
    'ERA5_coarse': {
        'Tg': 't2m',
        'P': 'tp',
    },

    'KNMI': {
        'Tg': 'TG',
        'P': 'RH',
    },

    'RACMO_daily': {
        'Tg': 't2m',
        'P': 'precip',
    },
    'RACMO_monthly': {
        'Tg': 't2m',
        'P': 'precip',
    },
}

# File names per dataset
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
        'Tg': 'era5_coarse_t2m.nc',
        'P': 'era5_coarse_tp.nc',
    },

    'KNMI': {
        'Tg': 'KNMI_' + location + '.txt',
        'P': 'KNMI_' + location + '.txt',
    },

    'RACMO_daily': {
        'Tg': 't2m',
        'P': 'precip',
    },
    'RACMO_monthly': {
        'Tg': 't2m',
        'P': 'precip',
    },
}

# Base directories
base_dir_cfg = {
    'Eobs_fine': '/nobackup/users/walj/eobs',
    'Eobs_coarse': '/nobackup/users/walj/eobs',
    'ERA5_fine': '/nobackup/users/walj/era5',
    'ERA5_coarse': '/nobackup/users/walj/era5',
    'KNMI': '/nobackup/users/walj/knmi',
    'RACMO_daily': '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data',
    'RACMO_monthly': '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Monthly_data',
}

# Coordinates of stations
station_coord_cfg = {
    'Bilt': {
        'lat': 52.098872302947974,
        'lon': 5.179442289152804,
    },
    'Cabauw': {
        'lat': 51.970212171384865,
        'lon': 4.926283190645085,
    },
}

cfg = {}

for src in data_sources:
    cfg[src] = {
        'variable': var_name_cfg[src][var],
        'file': file_cfg[src][var],
        'base_dir': base_dir_cfg[src],
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

lat_station = station_coord_cfg[location]['lat']
lon_station = station_coord_cfg[location]['lon']

data_all = {}

for src in data_sources:
    input_file_data = os.path.join(cfg[src]['base_dir'], cfg[src]['file'])

    if 'Eobs' in src:
        data_all[src] = preprocess_eobs_monthly(
            file_path=input_file_data,
            var_name=cfg[src]['variable'],
            months=months,
            years=years_load,
            lats=lat_station,
            lons=lon_station,
        ).squeeze()

    elif 'ERA5' in src:
        data_all[src] = preprocess_era5(
            file_path=input_file_data,
            var_name=cfg[src]['variable'],
            months=months,
            years=years_load,
            lats=lat_station,
            lons=lon_station,
        ).squeeze()

    elif 'KNMI' in src:
        data_all[src] = preprocess_knmi_monthly(
            file_path=input_file_data,
            var_name=cfg[src]['variable'],
            months=months,
            years=years_load,
        ).squeeze()

    elif 'RACMO' in src:
        data_all[src] = preprocess_racmo_monthly(
            dir_path=input_file_data,
            var_name=cfg[src]['variable'],
            months=months,
            years=years_load,
            lats=lat_station,
            lons=lon_station,
            already_monthly=('monthly' in src),
        ).squeeze()

# Yearly aggregates:
data_year = {}

for src in data_sources:
    da = data_all[src]
    month = da['time'].dt.month
    year = da['time'].dt.year
    
    # if season stays within calendar year: clim_year = calendar year
    # if season crosses year boundary: months >= month_start belong to next year
    if month_start <= month_end:
        clim_year = year
    else:
        clim_year = xr.where(month >= month_start, year + 1, year)

    da = da.assign_coords(clim_year=clim_year)

    da_year = da.groupby('clim_year').mean('time')

    da_year = da_year.sel(clim_year=slice(years_req[0], years_req[1]))

    time_coord = pd.to_datetime(da_year['clim_year'].values.astype(int).astype(str))
    da_year = da_year.assign_coords(time=('clim_year', time_coord)).swap_dims({'clim_year': 'time'})

    if fit_against_gmst:
        file_GMST = '/nobackup/users/walj/era5/era5_gmst_anom.nc'
        data_GMST = xr.open_dataset(file_GMST)
        gmst_roll = data_GMST.rolling(time=10, center=False).mean()
        gmst_full = gmst_roll['GMST']

        gmst_sel = gmst_full.sel(time=da_year['time'])
        fit_coord = gmst_sel.astype(float)

    else:
        fit_coord = da_year['clim_year'].astype(float)

    da_year = da_year.assign_coords(fit_against=fit_coord)
    data_year[src] = da_year

#%% Monthly data

colors = {
    'KNMI': "#000000",
    'ERA5': "#0168DE",
    'Eobs': "#00A236",
    'RACMO': "#DB2525",
}

# fig, ax = plt.subplots(1, figsize=(12, 8))

# for src in data_sources:

#     base_name = next(key for key in colors if key in src)
#     color = next(colors[key] for key in colors if key in src)

#     ax.plot(data_all[src].time, data_all[src].values, 
#             c=color, alpha=0.8, linewidth=2, label=base_name)
    
# ax.grid()
# ax.set_xlabel('Year', fontsize=28)
# ax.set_ylabel(plot_cfg[var]['ylabel_monthly'], fontsize=28)
# ax.tick_params(axis='both', labelsize=20, length=6)

# if plot_cfg[var]['ylim_monthly'] is not None:
#     ax.set_ylim(*plot_cfg[var]['ylim_monthly'])
# # ax.set_xlim(pd.Timestamp(f'{years[0]}-01-01'),
# #             pd.Timestamp(f'{years[1]}-12-31'))

# leg=ax.legend(fontsize=22, handlelength=1.5, handletextpad=0.4, loc='upper left')
# for line in leg.get_lines():
#     line.set_linewidth(4.0)

#%% Yearly trends

trend_stats = {} 

for src in data_sources:

    x_arr = data_year[src]['fit_against'].values
    y_arr = data_year[src].values

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]

    coeffs, cov = np.polyfit(x_clean, y_clean, 1, cov=True)
    slope = coeffs[0]
    intercept = coeffs[1]

    slope_std = np.sqrt(cov[0, 0]) # std of slope parameter
    slope_trend = slope*fit_scaling
    slope_trend_std = slope_std*fit_scaling

    n = len(x_clean)
    x_mean = x_clean.mean()
    Sxx = np.sum((x_clean - x_mean)**2)

    y_fit = slope*x_clean + intercept
    resid = y_clean - y_fit
    s2 = np.sum(resid**2) / (n - 2) # residual variance
    s = np.sqrt(s2) # residual std

    trend_stats[src] = {
        'slope': slope,
        'intercept': intercept,
        'slope_trend': slope_trend,
        'slope_trend_std': slope_trend_std,
        'n': n,
        'x_mean': x_mean,
        'Sxx': Sxx,
        's': s,
    }

fig, ax = plt.subplots(1, figsize=(12, 8))

for src in data_sources:

    stats = trend_stats[src]
    slope = stats['slope']
    intercept = stats['intercept']
    slope_trend = stats['slope_trend']
    slope_trend_std = stats['slope_trend_std']
    n = stats['n']
    x_mean = stats['x_mean']
    Sxx = stats['Sxx']
    s = stats['s']

    color = next(colors[key] for key in colors if key in src)
    base_name = next(key for key in colors if key in src)
    if base_name == 'KNMI':
        base_name = 'Observed'

    label = (
        f'{base_name} (trend: {slope_trend:.2f} ± {slope_trend_std:.2f} '
        f'{plot_cfg[var]["trend_unit"]})'
    )

    x_arr = data_year[src]['fit_against'].values
    y_arr = data_year[src].values
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]

    order = np.argsort(x_clean)
    x_sorted = x_clean[order]
    y_sorted = y_clean[order]

    ax.scatter(
        x_sorted,
        y_sorted,
        c=color,
        s=100,
        alpha=0.7,
        zorder=10
    )

    y_trend = intercept + slope*x_sorted

    se_mean = s*np.sqrt(1.0 / n + (x_sorted - x_mean)**2 / Sxx)
    t_val = 1.96
    y_hi = y_trend + t_val*se_mean
    y_lo = y_trend - t_val*se_mean

    ax.plot(
        x_sorted,
        y_trend,
        c=color,
        linewidth=3,
        alpha=1,
        label=label
    )

    ax.fill_between(
        x_sorted,
        y_lo,
        y_hi,
        color=color,
        alpha=0.15,
        linewidth=0,
    )

ax.grid()
ax.set_xlabel(fit_x_label, fontsize=28)
ax.set_ylabel(plot_cfg[var]['ylabel_fit'], fontsize=28)
ax.tick_params(axis='both', labelsize=20, length=6)

if plot_cfg[var]['ylim_fit'] is not None:
    ax.set_ylim(*plot_cfg[var]['ylim_fit'])

leg = ax.legend(fontsize=18, handlelength=1.5, handletextpad=0.4, loc='best')
for line in leg.get_lines():
    line.set_linewidth(4.0)
leg.set_zorder(20)

# labels for CF en line ook.

# Misschien max en minimum temperatures?


# colors = {
#     'Eobs':  '#4285F4',
#     'ERA5':  '#34A853',
#     'KNMI':  '#FBBC05',
#     'RACMO': '#EA4335',
# }