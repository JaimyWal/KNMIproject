#%% Imports

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

import ProcessKNMI
reload(ProcessKNMI)
from ProcessKNMI import preprocess_knmi_monthly

import ProcessRACMO
reload(ProcessRACMO)
from ProcessRACMO import preprocess_racmo_monthly


#%%

plot_cfg = {
    'Tg': {
        'ylabel_monthly': 'Temperature (°C)',
        'ylabel_fit':     'Temperature (°C)',
        'trend_unit':     '°C / decade',
        'ylim_monthly':   None,
        'ylim_fit':       None,
    },
    'P': {
        'ylabel_monthly': 'Precipitation (mm)',
        'ylabel_fit':     'Daily precipitation (mm)',
        'trend_unit':     'mm / decade',
        'ylim_monthly':   None,
        'ylim_fit':       None,
    },
}

var_name_cfg = {
    'Eobs_fine': {
        'Tg': 'tg',
        'P':  'rr',
    },
    'Eobs_coarse': {
        'Tg': 'tg',
        'P':  'rr',
    },
    'ERA5_fine': {
        'Tg': 't2m',
        'P':  'tp',
    },
    'ERA5_coarse': {
        'Tg': 't2m',
        'P':  'tp',
    },
    'KNMI': {
        'Tg': 'TG',
        'P':  'RH',
    },
    'RACMO_daily': {
        'Tg': 't2m',
        'P':  'precip',
    },
    'RACMO_monthly': {
        'Tg': 't2m',
        'P':  'precip',
    },
}

file_cfg = {
    'Eobs_fine': {
        'Tg': 'tg_ens_mean_0.1deg_reg_v31.0e.nc',
        'P':  'rr_ens_mean_0.1deg_reg_v31.0e.nc',
    },
    'Eobs_coarse': {
        'Tg': 'tg_ens_mean_0.25deg_reg_v31.0e.nc',
        'P':  'rr_ens_mean_0.25deg_reg_v31.0e.nc',
    },
    'ERA5_fine': {
        'Tg': 'era5_fine.nc',
        'P':  'era5_fine.nc',
    },
    'ERA5_coarse': {
        'Tg': 'era5_coarse_t2m.nc',
        'P':  'era5_coarse_tp.nc',
    },
    'KNMI': {
        'Tg': 'KNMI_Cabauw.txt',      # will overwrite below for other locations
        'P':  'KNMI_Cabauw.txt',
    },
    'RACMO_daily': {
        'Tg': 't2m',
        'P':  'precip',
    },
    'RACMO_monthly': {
        'Tg': 't2m',
        'P':  'precip',
    },
}

base_dir_cfg = {
    'Eobs_fine':    '/nobackup/users/walj/eobs',
    'Eobs_coarse':  '/nobackup/users/walj/eobs',
    'ERA5_fine':    '/nobackup/users/walj/era5',
    'ERA5_coarse':  '/nobackup/users/walj/era5',
    'KNMI':         '/nobackup/users/walj/knmi',
    'RACMO_daily':  '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data',
    'RACMO_monthly':'/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Monthly_data',
}

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

colors = {
    'KNMI':  '#000000',
    'ERA5':  '#0168DE',
    'Eobs':  '#00A236',
    'RACMO': '#DB2525',
}


#%% 

def plot_trend_panel(ax,
                     var,
                     months,
                     years,
                     location='Cabauw',
                     data_sources=None,
                     relative_precip=False,
                     fit_against_gmst=False,
                     rolling_mean_var=False,
                     rolling_mean_years=1,
                     min_periods=None):

    if data_sources is None:
        data_sources = ['KNMI', 'RACMO_monthly', 'ERA5_coarse']

    years_req = list(years)
    years_load = list(years_req)

    months = np.asarray(months, dtype=int)
    month_start = int(months[0])
    month_end   = int(months[-1])

    if month_start > month_end:
        years_load[0] = years_req[0] - 1

    lat_station = station_coord_cfg[location]['lat']
    lon_station = station_coord_cfg[location]['lon']

    if fit_against_gmst:
        fit_unit = '°C GMST'
        fit_scaling = 1
        fit_x_label = 'ΔGMST (°C)'
    else:
        fit_unit = 'decade'
        fit_scaling = 10
        fit_x_label = 'Year'

    local_plot_cfg = {k: v.copy() for k, v in plot_cfg.items()}
    if relative_precip and var == 'P':
        local_plot_cfg['P']['ylabel_monthly'] = 'Precipitation (% of climatology)'
        local_plot_cfg['P']['ylabel_fit']     = 'Precipitation (% of climatology)'
        local_plot_cfg['P']['trend_unit']     = '% / ' + fit_unit
    else:
        local_plot_cfg['P']['ylabel_monthly'] = 'Precipitation (mm)'
        local_plot_cfg['P']['ylabel_fit']     = 'Daily precipitation (mm)'
        local_plot_cfg['P']['trend_unit']     = 'mm / ' + fit_unit

    data_all = {}
    for src in data_sources:
        cfg_src = {
            'variable': var_name_cfg[src][var],
            'file':     file_cfg[src][var] if src != 'KNMI' else f'KNMI_{location}.txt',
            'base_dir': base_dir_cfg[src],
        }
        input_file_data = os.path.join(cfg_src['base_dir'], cfg_src['file'])

        if 'Eobs' in src:
            data_all[src] = preprocess_eobs_monthly(
                file_path=input_file_data,
                var_name=cfg_src['variable'],
                months=months,
                years=years_load,
                lats=lat_station,
                lons=lon_station,
            ).squeeze()

        elif 'ERA5' in src:
            data_all[src] = preprocess_era5(
                file_path=input_file_data,
                var_name=cfg_src['variable'],
                months=months,
                years=years_load,
                lats=lat_station,
                lons=lon_station,
            ).squeeze()

        elif 'KNMI' in src:
            data_all[src] = preprocess_knmi_monthly(
                file_path=input_file_data,
                var_name=cfg_src['variable'],
                months=months,
                years=years_load,
            ).squeeze()

        elif 'RACMO' in src:
            data_all[src] = preprocess_racmo_monthly(
                dir_path=input_file_data,
                var_name=cfg_src['variable'],
                months=months,
                years=years_load,
                lats=lat_station,
                lons=lon_station,
                already_monthly=('monthly' in src),
            ).squeeze()

    data_year = {}
    for src in data_sources:
        da = data_all[src]
        month = da['time'].dt.month
        year  = da['time'].dt.year

        if month_start <= month_end:
            clim_year = year
        else:
            clim_year = xr.where(month >= month_start, year + 1, year)

        da = da.assign_coords(clim_year=clim_year)
        da_year = da.groupby('clim_year').mean('time')
        da_year = da_year.sel(clim_year=slice(years_req[0], years_req[-1]))

        if relative_precip and var == 'P':
            data_avg = da_year.mean(dim='clim_year')
            da_year = 100 * da_year / data_avg

        time_coord = pd.to_datetime(da_year['clim_year'].values.astype(int).astype(str))
        da_year = da_year.assign_coords(time=('clim_year', time_coord)).swap_dims({'clim_year': 'time'})

        if rolling_mean_var:
            da_year = da_year.rolling(time=rolling_mean_years,
                                      center=True,
                                      min_periods=min_periods).mean()

        if fit_against_gmst:
            file_GMST = '/nobackup/users/walj/era5/era5_gmst_anom.nc'
            data_GMST = xr.open_dataset(file_GMST)
            gmst_roll = data_GMST.rolling(time=rolling_mean_years,
                                          center=True,
                                          min_periods=min_periods).mean()
            gmst_full = gmst_roll['GMST']
            gmst_sel = gmst_full.sel(time=da_year['time'])
            fit_coord = gmst_sel.astype(float)
        else:
            fit_coord = da_year['clim_year'].astype(float)

        da_year = da_year.assign_coords(fit_against=fit_coord)
        data_year[src] = da_year

    for src in data_sources:
        x_arr = data_year[src]['fit_against'].values
        y_arr = data_year[src].values

        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_clean = x_arr[mask]
        y_clean = y_arr[mask]

        X = sm.add_constant(x_clean)
        model = sm.OLS(y_clean, X).fit()

        slope = model.params[1]
        slope_std = model.bse[1]
        slope_trend = slope * fit_scaling
        slope_trend_std = slope_std * fit_scaling

        color = next(colors[key] for key in colors if key in src)
        base_name = next(key for key in colors if key in src)
        if base_name == 'KNMI':
            base_name = 'Observed'

        order = np.argsort(x_clean)
        x_sorted = x_clean[order]
        y_sorted = y_clean[order]

        X_sorted = sm.add_constant(x_sorted)
        pred = model.get_prediction(X_sorted)
        frame = pred.summary_frame(alpha=0.05)

        y_trend = frame['mean'].values
        y_lo = frame['mean_ci_lower'].values
        y_hi = frame['mean_ci_upper'].values

        label = (
            f'{base_name} (trend: {slope_trend:.2f} ± {slope_trend_std:.2f} '
            f'{local_plot_cfg[var]["trend_unit"]})'
        )

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
            linewidth=2.5,
            alpha=1,
            label=label,
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
    ax.set_xlabel(fit_x_label, fontsize=20)
    ax.set_ylabel(local_plot_cfg[var]['ylabel_fit'], fontsize=20)
    ax.tick_params(axis='both', labelsize=15, length=4)
    leg = ax.legend(fontsize=15, handlelength=1.5, handletextpad=0.4, loc='best')
    for line in leg.get_lines():
        line.set_linewidth(3.0)
    leg.set_zorder(20)


#%%

location = 'Cabauw'
variable = 'P'
years = [1987, 2020]
months = [6, 7, 8] 
data_sources = ['KNMI', 'RACMO_monthly', 'ERA5_coarse']

fit_against_gmst = False
rolling_mean_var = True
min_periods = 1
relative_precip = True


fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True)

plot_trend_panel(
    ax=axes[0, 0],
    var=variable,
    months=months,
    years=years,
    location=location,
    fit_against_gmst=fit_against_gmst,
    rolling_mean_var=rolling_mean_var,
    rolling_mean_years=1,
    min_periods=min_periods,
    relative_precip=relative_precip
)

plot_trend_panel(
    ax=axes[0, 1],
    var=variable,
    months=months,
    years=years,
    location=location,
    fit_against_gmst=fit_against_gmst,
    rolling_mean_var=rolling_mean_var,
    rolling_mean_years=3,
    min_periods=min_periods,
    relative_precip=relative_precip
)

plot_trend_panel(
    ax=axes[1, 0],
    var=variable,
    months=months,
    years=years,
    location=location,
    fit_against_gmst=fit_against_gmst,
    rolling_mean_var=rolling_mean_var,
    rolling_mean_years=5,
    min_periods=min_periods,
    relative_precip=relative_precip
)

plot_trend_panel(
    ax=axes[1, 1],
    var=variable,
    months=months,
    years=years,
    location=location,
    fit_against_gmst=fit_against_gmst,
    rolling_mean_var=rolling_mean_var,
    rolling_mean_years=7,
    min_periods=min_periods,
    relative_precip=relative_precip
)

axes[0, 0].set_title('No rolling mean', fontsize=20, fontweight='bold')
axes[0, 1].set_title('3-year rolling mean', fontsize=20, fontweight='bold')
axes[1, 0].set_title('5-year rolling mean', fontsize=20, fontweight='bold')
axes[1, 1].set_title('7-year rolling mean', fontsize=20, fontweight='bold')

axes[0, 0].set_xlabel('')
axes[0, 1].set_xlabel('')
axes[1, 1].set_ylabel('')
axes[0, 1].set_ylabel('')

plt.draw()  

xmins, xmaxs, ymins, ymaxs = [], [], [], []

for ax in axes.flat:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xmins.append(xmin)
    xmaxs.append(xmax)
    ymins.append(ymin)
    ymaxs.append(ymax)

x_low  = min(xmins)
x_high = max(xmaxs)
y_low  = min(ymins)
y_high = max(ymaxs)

for ax in axes.flat:
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)

if relative_precip:
    fig.text(
        -0.02, 0.5,
        'Precipitation (% of climatology)',
        va='center',
        rotation='vertical',
        fontsize=24
    )
    axes[1, 0].set_ylabel('')
    axes[0, 0].set_ylabel('')

fig.tight_layout()
plt.show()
