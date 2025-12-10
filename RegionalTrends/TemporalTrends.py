#%% Imports

# Standard libraries
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from importlib import reload

# Custom functions
import ProcessEobs
reload(ProcessEobs)          
from ProcessEobs import preprocess_eobs_monthly 

import ProcessERA5
reload(ProcessERA5)          
from ProcessERA5 import preprocess_era5 

import ProcessStation as ProcessStation
reload(ProcessStation)          
from ProcessStation import preprocess_station_monthly

import ProcessRACMO
reload(ProcessRACMO)          
from ProcessRACMO import preprocess_racmo_monthly 

#%% User inputs

var = 'Tg'
relative_precip = False
location = 'Bilt'
data_sources = ['KNMI', 'ERA5_coarse', 'Eobs_fine']

months = [12, 1, 2]
years = [2016, 2020]

fit_against_gmst = False
rolling_mean_var = False
rolling_mean_years = 1
min_periods = None

#%% Dataset configurations

if fit_against_gmst:
    fit_unit = '°C GMST'
    fit_scaling = 1
    fit_x_label = 'ΔGMST (°C)'
else:
    fit_unit = 'decade'
    fit_scaling = 10
    fit_x_label = 'Year'

if relative_precip:
    precip_label = 'Precipitation (% of climatology)'
    precip_trend_unit = '% / ' + fit_unit
else:
    precip_label = 'Precipitation (mm)'
    precip_trend_unit = 'mm / ' + fit_unit

plot_cfg = {
    'Tg': {
        'ylabel_monthly': 'Temperature (°C)',
        'ylabel_fit': 'Temperature (°C)',
        'trend_unit': '°C / ' + fit_unit,
        'ylim_monthly': None,
        'ylim_fit': None,
    },
    'P': {
        'ylabel_monthly': precip_label,
        'ylabel_fit': precip_label,
        'trend_unit': precip_trend_unit,
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
        data_all[src] = preprocess_station_monthly(
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

    da_year = da_year.sel(clim_year=slice(years_req[0], years_req[-1]))

    if relative_precip and var == 'P':
        data_avg = da_year.mean(dim='clim_year')
        da_year = 100*da_year / data_avg

    time_coord = pd.to_datetime(da_year['clim_year'].values.astype(int).astype(str))
    da_year = da_year.assign_coords(time=('clim_year', time_coord)).swap_dims({'clim_year': 'time'})

    if rolling_mean_var:
        da_year = da_year.rolling(time=rolling_mean_years, center=True, min_periods=min_periods).mean()

    if fit_against_gmst:
        file_GMST = '/nobackup/users/walj/era5/era5_gmst_anom.nc'
        data_GMST = xr.open_dataset(file_GMST)
        gmst_roll = data_GMST.rolling(time=rolling_mean_years, center=True, min_periods=min_periods).mean()
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

    # Add intercept column
    X = sm.add_constant(x_clean)

    # Fit Ordinary Least Squares
    model = sm.OLS(y_clean, X).fit()

    slope = model.params[1]
    intercept = model.params[0]

    slope_std = model.bse[1]
    slope_trend = slope*fit_scaling
    slope_trend_std = slope_std*fit_scaling

    trend_stats[src] = {
        'model': model,
        'x_clean': x_clean,
        'y_clean': y_clean,
        'slope': slope,
        'intercept': intercept,
        'slope_trend': slope_trend,
        'slope_trend_std': slope_trend_std,
    }

fig, ax = plt.subplots(1, figsize=(12, 8))

for src in data_sources:

    stats = trend_stats[src]
    model = stats['model']
    x_clean = stats['x_clean']
    y_clean = stats['y_clean']

    slope_trend = stats['slope_trend']
    slope_trend_std = stats['slope_trend_std']

    color = next(colors[key] for key in colors if key in src)
    base_name = next(key for key in colors if key in src)
    if base_name == 'KNMI':
        base_name = 'Observed'

    label = (
        f'{base_name} (trend: {slope_trend:.2f} ± {slope_trend_std:.2f} '
        f'{plot_cfg[var]["trend_unit"]})'
    )

    order = np.argsort(x_clean)
    x_sorted = x_clean[order]
    y_sorted = y_clean[order]
    
    X_sorted = sm.add_constant(x_sorted)
    pred = model.get_prediction(X_sorted)
    frame = pred.summary_frame(alpha=0.05)  # 95 percent CI

    y_trend = frame['mean'].values
    y_lo = frame['mean_ci_lower'].values
    y_hi = frame['mean_ci_upper'].values
    
    # ax.scatter(
    #     x_sorted,
    #     y_sorted,
    #     c=color,
    #     s=100,
    #     alpha=0.7,
    #     zorder=10
    # )

    ax.plot(
            x_sorted,
            y_sorted,
            c=color,
            linewidth=2.5,
            zorder=10,
            ms=6,
            marker='o'
    )

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

# Relative precipitation?

# colors = {
#     'Eobs':  '#4285F4',
#     'ERA5':  '#34A853',
#     'KNMI':  '#FBBC05',
#     'RACMO': '#EA4335',
# }

# #%% Yearly series + rolling means per data source (separate figures)
# # and differences relative to observations (KNMI)

# ref_src = 'KNMI'   # reference observations

# for src in data_sources:
#     da_year = data_year[src]

#     # x coordinate in time
#     time_coord = da_year['time']
#     y_yearly = da_year

#     # optional: anomalies relative to some internal ref period
#     # ref = da_year.sel(time=slice('1940', '1970')).mean()
#     # y_anom = da_year - ref
#     # for now: raw yearly data
#     y_anom = da_year

#     roll3      = y_anom.rolling(time=3, center=True).mean()
#     roll5      = y_anom.rolling(time=5, center=True).mean()
#     roll7      = y_anom.rolling(time=7, center=True).mean()
#     roll3_edge = y_anom.rolling(time=3, center=True, min_periods=1).mean()
#     roll5_edge = y_anom.rolling(time=5, center=True, min_periods=1).mean()
#     roll7_edge = y_anom.rolling(time=7, center=True, min_periods=1).mean()

#     # nicer name for panel and legend
#     base_key = next(key for key in colors if key in src)
#     name = 'Observed' if base_key == 'KNMI' else base_key

#     # ========== 1) Absolute yearly series + rolling means ==========
#     fig, ax = plt.subplots(1, figsize=(12, 8))

#     # yearly series
#     ax.plot(
#         time_coord,
#         y_anom.values,
#         c='xkcd:black',
#         linewidth=2,
#         label=f'{name} yearly'
#     )

#     # 3 year rolling mean
#     ax.plot(
#         roll3['time'],
#         roll3.values,
#         c='xkcd:red',
#         linewidth=3,
#         label='3 year rolling mean'
#     )
#     ax.plot(
#         roll3_edge['time'],
#         roll3_edge.values,
#         c='xkcd:red',
#         linewidth=3,
#         linestyle='--'
#     )

#     # 5 year rolling mean
#     ax.plot(
#         roll5['time'],
#         roll5.values,
#         c='xkcd:green',
#         linewidth=3,
#         label='5 year rolling mean'
#     )
#     ax.plot(
#         roll5_edge['time'],
#         roll5_edge.values,
#         c='xkcd:green',
#         linewidth=3,
#         linestyle='--'
#     )

#     # 7 year rolling mean
#     ax.plot(
#         roll7['time'],
#         roll7.values,
#         c='xkcd:blue',
#         linewidth=3,
#         label='7 year rolling mean'
#     )
#     ax.plot(
#         roll7_edge['time'],
#         roll7_edge.values,
#         c='xkcd:blue',
#         linewidth=3,
#         linestyle='--'
#     )

#     ax.set_xlabel('Year', fontsize=28)
#     ax.set_ylabel(plot_cfg[var]['ylabel_fit'], fontsize=28)
#     ax.tick_params(axis='both', labelsize=20, length=6)
#     ax.grid()
#     ax.set_title(name, fontsize=24)

#     leg = ax.legend(
#         fontsize=18,
#         handlelength=1.5,
#         handletextpad=0.4,
#         loc='best'
#     )
#     for line in leg.get_lines():
#         line.set_linewidth(4.0)

#     # ========== 2) Differences relative to observations ==========
#     # skip for the reference dataset itself
#     if ref_src in src:
#         continue

#     da_ref = data_year[ref_src]

#     # align in time, just in case there is a mismatch
#     da_model_aligned, da_ref_aligned = xr.align(
#         da_year,
#         da_ref,
#         join='inner'
#     )

#     diff = da_model_aligned - da_ref_aligned

#     diff_roll3      = diff.rolling(time=3, center=True).mean()
#     diff_roll5      = diff.rolling(time=5, center=True).mean()
#     diff_roll7      = diff.rolling(time=7, center=True).mean()
#     diff_roll3_edge = diff.rolling(time=3, center=True, min_periods=1).mean()
#     diff_roll5_edge = diff.rolling(time=5, center=True, min_periods=1).mean()
#     diff_roll7_edge = diff.rolling(time=7, center=True, min_periods=1).mean()

#     fig, ax = plt.subplots(1, figsize=(12, 8))

#     # yearly difference
#     ax.plot(
#         diff['time'],
#         diff.values,
#         c='xkcd:black',
#         linewidth=2,
#         label=f'{name} - Observed yearly'
#     )

#     # 3 year rolling mean of difference
#     ax.plot(
#         diff_roll3['time'],
#         diff_roll3.values,
#         c='xkcd:red',
#         linewidth=3,
#         label='3 year rolling mean difference'
#     )
#     ax.plot(
#         diff_roll3_edge['time'],
#         diff_roll3_edge.values,
#         c='xkcd:red',
#         linewidth=3,
#         linestyle='--'
#     )

#     # 5 year rolling mean of difference
#     ax.plot(
#         diff_roll5['time'],
#         diff_roll5.values,
#         c='xkcd:green',
#         linewidth=3,
#         label='5 year rolling mean difference'
#     )
#     ax.plot(
#         diff_roll5_edge['time'],
#         diff_roll5_edge.values,
#         c='xkcd:green',
#         linewidth=3,
#         linestyle='--'
#     )

#     # 7 year rolling mean of difference
#     ax.plot(
#         diff_roll7['time'],
#         diff_roll7.values,
#         c='xkcd:blue',
#         linewidth=3,
#         label='7 year rolling mean difference'
#     )
#     ax.plot(
#         diff_roll7_edge['time'],
#         diff_roll7_edge.values,
#         c='xkcd:blue',
#         linewidth=3,
#         linestyle='--'
#     )

#     ax.set_xlabel('Year', fontsize=28)
#     ax.set_ylabel(f'{plot_cfg[var]["ylabel_fit"]} difference', fontsize=28)
#     ax.tick_params(axis='both', labelsize=20, length=6)
#     ax.grid()
#     ax.set_title(f'{name} minus Observed', fontsize=24)

#     leg = ax.legend(
#         fontsize=16,
#         handlelength=1.5,
#         handletextpad=0.4,
#         loc='best'
#     )
#     for line in leg.get_lines():
#         line.set_linewidth(4.0)

