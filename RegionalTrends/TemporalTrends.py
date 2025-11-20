#%% Imports

# Standard libraries
import numpy as np
import pandas as pd
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

#%% User inputs

var = 'Tg'
location = 'Cabauw'
resolution = 'Fine'

months = None
years = [1970, 2024]

#%% Dataset configurations

plot_cfg = {
    'Tg': {
        'ylabel_monthly': 'Temperature (°C)',
        'ylabel_yearly':  'Temperature (°C)',
        'trend_unit':     '°C / decade',
        'ylim_monthly':   None,
        'ylim_yearly':    None,
    },
    'P': {
        'ylabel_monthly': 'Precipitation (mm)',
        'ylabel_yearly':  'Daily precipitation (mm)',
        'trend_unit':     'mm / decade',
        'ylim_monthly':   None,
        'ylim_yearly':    None,
    },
}


# Variable names in each dataset
var_name_cfg = {
    'Eobs': {
        'Tg': 'tg',
        'P': 'rr',
    },
    'ERA5': {
        'Tg': 't2m',
        'P': 'tp',
    },
    'KNMI': {
        # adjust to actual column names in the txt if needed
        'Tg': 'TG',
        'P': 'RH',
    },
}

# File names per dataset
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
    'KNMI': {
        # here the key is the station location instead of resolution
        'Bilt': {
            'Tg': 'KNMI_deBilt.txt',
            'P':  'KNMI_deBilt.txt',
        },
        'Cabauw': {
            'Tg': 'KNMI_Cabauw.txt',
            'P':  'KNMI_Cabauw.txt',
        },
    },
}

# Base directories
base_dir_cfg = {
    'Eobs': '/nobackup/users/walj/eobs',
    'ERA5': '/nobackup/users/walj/era5',
    'KNMI': '/nobackup/users/walj/knmi',
}

# Optional coordinates for KNMI stations
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

for src in ['Eobs', 'ERA5', 'KNMI']:

    # For EOBS and ERA5 the second key is resolution
    if src in ['Eobs', 'ERA5']:
        file_key = resolution
    # For KNMI the second key is the station name
    else:
        file_key = location

    cfg[src] = {
        'variable': var_name_cfg[src][var],
        'file':     file_cfg[src][file_key][var],
        'base_dir': base_dir_cfg[src],
    }

#%% Data loading and processing

lat_station = station_coord_cfg[location]['lat']
lon_station = station_coord_cfg[location]['lon']

data_all = {}

for src, c in cfg.items():
    input_file_data = os.path.join(c['base_dir'], c['file'])

    if src == 'Eobs':
        data_all['Eobs'] = preprocess_eobs_monthly(
            file_path=input_file_data,
            var_name=c['variable'],
            months=months,
            years=years,
            lats=lat_station,
            lons=lon_station,
        ).squeeze()

    elif src == 'ERA5':
        data_all['ERA5'] = preprocess_era5(
            file_path=input_file_data,
            var_name=c['variable'],
            months=months,
            years=years,
            lats=lat_station,
            lons=lon_station,
        ).squeeze()

    elif src == 'KNMI':
        data_all['KNMI'] = preprocess_knmi_monthly(
            file_path=input_file_data,
            var_name=c['variable'],
            months=months,
            years=years,
        ).squeeze()

# Yearly aggregates:
data_year = {}

for src in data_all.keys():
    da_year = data_all[src].resample(time='YS').mean()
    year_coord = da_year['time'].dt.year.astype(float)
    data_year[src] = (da_year
                      .assign_coords(year=year_coord)
                      .sortby('year'))

#%% Monthly data

colors = {
    'Eobs': 'xkcd:red',
    'ERA5': 'xkcd:blue',
    'KNMI': 'xkcd:green',
}

# fig, ax = plt.subplots(1, figsize=(12, 8))

# for src in ['Eobs', 'ERA5', 'KNMI']:

#     ax.plot(data_all[src].time, data_all[src].values, 
#             c=colors[src], alpha=0.8, linewidth=2, label=src)
    
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

data_fits = {}

for src in data_year.keys():

    fit = data_year[src].polyfit(dim='year', deg=1, skipna=True)
    data_fits[src] = fit


fig, ax = plt.subplots(1, figsize=(12, 8))

for src in ['Eobs', 'ERA5', 'KNMI']:

    slope = data_fits[src].polyfit_coefficients.sel(degree=1)
    intercept = data_fits[src].polyfit_coefficients.sel(degree=0)

    slope_decade = float(slope*10)

    label = f'{src} (trend {slope_decade:.2f} {plot_cfg[var]["trend_unit"]})'

    ax.plot(
        data_year[src].time,
        data_year[src].values,
        c=colors[src],
        linewidth=3,
        alpha=1,
        label=label,
    )

    years_plot = data_year[src]['year']
    trend_vals = intercept + slope*years_plot

    ax.plot(
        data_year[src].time,
        trend_vals,
        c=colors[src],
        linewidth=3,
        alpha=1,
        linestyle='--'
    )

ax.grid()
ax.set_xlabel('Year', fontsize=28)
ax.set_ylabel(plot_cfg[var]['ylabel_yearly'], fontsize=28)
ax.tick_params(axis='both', labelsize=20, length=6)

if plot_cfg[var]['ylim_yearly'] is not None:
    ax.set_ylim(*plot_cfg[var]['ylim_yearly'])
# ax.set_xlim(pd.Timestamp(f'{years[0]}-01-01'),
#             pd.Timestamp(f'{years[1]}-12-31'))

leg=ax.legend(fontsize=18, handlelength=1.5, handletextpad=0.4, loc='best')
for line in leg.get_lines():
    line.set_linewidth(4.0)



# Ik zie nauwelijks een trend voor P? Maar op deze website wel... https://www.knmi.nl/klimaat
# Enorm grote waarde aan het begin van KNMI cabauw?? Komt doordat het eerste jaar geen wintertemperaturen heeft.
# Maybe yearly sums and not daily averages for precipitation?
# Cabauw is missing data before 1986....
# Misschien max en minimum temperatures?
# Eobs best slecht voor coarse
