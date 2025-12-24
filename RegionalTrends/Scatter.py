#%% Imports

# Standard libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import os
from dask.distributed import Client, get_client
from importlib import reload

# Custom libraries
import ProcessNetCDF
reload(ProcessNetCDF)          
from ProcessNetCDF import preprocess_netcdf

import ProcessStation
reload(ProcessStation)          
from ProcessStation import preprocess_station

import AreaWeights
reload(AreaWeights)
from AreaWeights import area_weights, area_weighted_mean

plt.rcParams['axes.unicode_minus'] = False

#%% User inputs

# Main arguments
n_runs = 3
var = 'SW' #
data_base = ['L5', 'L5', 'L5'] # 
data_compare = ['Eobs_fine', 'RACMO2.4', 'ERA5_coarse'] # 

# Data selection arguments
freq_sel = 'Monthly' #
months = None # 
years = [2016, 2024]
lats = [50.7, 53.6]
lons = [3.25, 7.35]
proj_sel = 'RACMO2.4'
land_only = False
trim_border = None

# Plotting arguments
share_labels = True

# Other arguments
rolling_mean_var = False
rolling_mean_years = 7
min_periods = 1

#%% Obtain rotated grid

def load_rotpole(rotpole_dir, rotpole_file):

    ds = xr.open_dataset(os.path.join(rotpole_dir, rotpole_file))

    rp = ds['rotated_pole']
    pole_lat = rp.grid_north_pole_latitude
    pole_lon = rp.grid_north_pole_longitude

    rotpole = ccrs.RotatedPole(
        pole_latitude=pole_lat,
        pole_longitude=pole_lon,
    )

    return rotpole

rotpole23 = load_rotpole(
    '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/precip',
    'precip.KNMI-2000.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc'
)

rotpole24 = load_rotpole(
    '/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly',
    'pr_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc'
)

#%% Dataset configurations

try:
    client = get_client()
except ValueError:
    client = Client(n_workers=1, threads_per_worker=8, processes=False)

data_sources = ['Eobs', 'ERA5', 'RACMO2.3', 'RACMO2.4']
station_sources = ['Bilt', 'Cabauw', 'Eelde', 'Maastricht', 'Vlissingen', 'Kooy']

var_name_cfg = {
    'Eobs': {
        'Tg': 'tg',
        'P': 'rr',
        'SW': 'qq',
    },
    'ERA5': {
        'Tg': 't2m',
        'P': 'tp',
        'SW': 'ssrd',
    },
    'RACMO2.3': {
        'Tg': 't2m',
        'P': 'precip',
        'Sq': 'sund'
    },
    'RACMO2.4': {
        'Tg': 'tas',
        'P': 'pr',
        'Sq': 'sund',
        'SW': 'rsds',
    },
    'Station': {
        'Tg': 'TG',
        'P': 'RH',
        'Sq': 'SQ',
        'SW': 'Q'
    },
}

station_coord_cfg = {
    'Bilt': {
        'latitude': 52.098872302947974,
        'longitude': 5.179442289152804,
    },
    'Cabauw': {
        'latitude': 51.970212171384865,
        'longitude': 4.926283190645085,
    },
    'Eelde': {
        'latitude': 53.12385846866912,
        'longitude': 6.584799434350561
    },
    'Maastricht': {
        'latitude': 50.90548320406765,
        'longitude': 5.761839846736004
    },
    'Vlissingen': {
        'latitude': 51.441328455552586,
        'longitude': 3.5958610840956884
    },
    'Kooy': {
        'latitude': 52.924172463538795,
        'longitude': 4.779336630180403
    },
}

file_cfg = {
    'Eobs_fine': {
        'Tg': 'tg_ens_mean_0.1deg_reg_v31.0e.nc',
        'P': 'rr_ens_mean_0.1deg_reg_v31.0e.nc',
        'SW': 'qq_ens_mean_0.1deg_reg_v31.0e.nc',
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
        'SW': 'era5_rsds.nc',
    },

    'RACMO2.3': {
        'Tg': 't2m/*.nc',
        'P': 'precip/*.nc',
        'Sq': 'sund/*.nc',
    },
    'RACMO2.4': {
        'Tg': 'Daily/tas.*.nc',
        'P': 'Daily/pr.*.nc',
        'Sq': 'Daily/sund.*.nc',
        'SW': 'Daily/rsds.*.nc',
    },

    'Station': {
        'Bilt': 'KNMI_Bilt.txt',
        'Cabauw': 'KNMI_Cabauw.txt',
        'Eelde': 'KNMI_Eelde.txt',
        'Maastricht': 'KNMI_Maastricht.txt',
        'Vlissingen': 'KNMI_Vlissingen.txt',
        'Kooy': 'KNMI_Kooy.txt'
    },
}

base_dir_cfg = {
    'Eobs': '/nobackup/users/walj/eobs',
    'ERA5': '/nobackup/users/walj/era5',
    'RACMO2.3': f'/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data',
    'RACMO2.4': '/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning',
    'Station': '/nobackup/users/walj/knmi',
}

proj_cfg = {
    'RACMO2.3': rotpole23,
    'RACMO2.4': rotpole24,
}

proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())

#%% Loading and processing data

def make_cfg(data_source, var):

    if any(src in data_source for src in data_sources):
        file_key = next(src for src in data_sources if src in data_source)
        cfg = {
            'variable': var_name_cfg[file_key][var],
            'file': file_cfg[data_source][var],
            'base_dir': base_dir_cfg[file_key],
            'file_key': file_key,
            'datatype': 'netcdf',
            'proj': proj_cfg.get(file_key, ccrs.PlateCarree()),
        }
        return cfg
    
    elif data_source in station_sources:
        cfg = {
            'variable': var_name_cfg['Station'][var],
            'file': file_cfg['Station'][data_source],
            'base_dir': base_dir_cfg['Station'],
            'file_key': 'Station',
            'datatype': 'station',
            'proj': ccrs.PlateCarree(),
        }
        return cfg


def process_source(data_source,
                   var,
                   months=None,
                   years=None,
                   lats=None,
                   lons=None,
                   land_only=False,
                   trim_border=None,
                   rotpole_sel=ccrs.PlateCarree(),
                   rolling_mean_var=False,
                   rolling_mean_years=1,
                   min_periods=1):
    
    cfg = make_cfg(data_source, var)

    if months is None:
        months_local = np.arange(1, 13)
    else:
        months_local = np.asarray(months, dtype=int)

    if years is None:
        years_req = None
        years_load = None
    else:
        years_req = list(years)
        years_load = list(years_req)

        month_start = months_local[0]
        month_end = months_local[-1]

        if month_start > month_end:
            years_load[0] = years_req[0] - 1

    input_file_data = os.path.join(cfg['base_dir'], cfg['file'])

    trim_local = trim_border
    if data_source == 'RACMO2.4' and trim_border is None:
        trim_local = 8

    if cfg['datatype'] == 'netcdf':
        data = preprocess_netcdf(
            source=cfg['file_key'],
            file_path=input_file_data,
            var_name=cfg['variable'],
            months=months_local,
            years=years_load,
            lats=lats,
            lons=lons,
            land_only=land_only,
            trim_border=trim_local,
            rotpole_sel=rotpole_sel,
            rotpole_native=cfg['proj']
        ).squeeze()

    elif cfg['datatype'] == 'station':
        data = preprocess_station(
            file_path=input_file_data,
            var_name=cfg['variable'],
            months=months,
            years=years_load,
        ).squeeze()

    month_d = data['time'].dt.month
    year_d = data['time'].dt.year

    month_start = months_local[0]
    month_end = months_local[-1]

    if month_start <= month_end:
        clim_year_d = year_d
    else:
        clim_year_d = xr.where(month_d >= month_start, year_d + 1, year_d)

    data = data.assign_coords(clim_year=clim_year_d)

    data_monthly = data.resample(time='MS').mean('time')
    existing = pd.DatetimeIndex(data['time'].values).to_period('M').unique().to_timestamp()
    data_monthly = data_monthly.sel(time=existing)

    month_m = data_monthly['time'].dt.month
    year_m = data_monthly['time'].dt.year

    if month_start <= month_end:
        clim_year_m = year_m
    else:
        clim_year_m = xr.where(month_m >= month_start, year_m + 1, year_m)

    data_monthly = data_monthly.assign_coords(clim_year=clim_year_m)

    data_year = data_monthly.groupby('clim_year').mean('time')

    if years_req is not None:
        y0, y1 = years_req[0], years_req[-1]

        data = data.where((data['clim_year'] >= y0) & (data['clim_year'] <= y1), drop=True)
        data_monthly = data_monthly.where(
            (data_monthly['clim_year'] >= y0) & (data_monthly['clim_year'] <= y1),
            drop=True
        )

        data_year = data_year.sel(clim_year=slice(y0, y1))

    data_avg = data_year.mean(dim='clim_year').astype('float32')

    time_coord = pd.to_datetime(data_year['clim_year'].astype(str))
    data_yearly = data_year.assign_coords(time=('clim_year', time_coord)).swap_dims({'clim_year': 'time'})

    if rolling_mean_var:
        data_yearly = data_yearly.rolling(
            time=rolling_mean_years,
            center=True,
            min_periods=min_periods
        ).mean()

    return data, data_monthly, data_yearly, data_avg

#%% Ensure lists for looped parameters

def ensure_list(param, n, nested=False):

    if not isinstance(param, list):
        return [param]*n

    if not nested:
        return param

    if any(isinstance(p, list) for p in param):
        return param

    return [param]*n

var_list = ensure_list(var, n_runs)
data_base_list = ensure_list(data_base, n_runs)
data_compare_list = ensure_list(data_compare, n_runs)
months_list = ensure_list(months, n_runs, nested=True)
freq_sel_list = ensure_list(freq_sel, n_runs)

#%% Further processing of base and comparison data

def read_data(
    frequency,
    data_source,
    var,
    months=None,
    years=None,
    lats=None,
    lons=None,
    land_only=False,
    trim_border=None,
    rotpole_sel=ccrs.PlateCarree(),
    rolling_mean_var=False,
    rolling_mean_years=1,
    min_periods=1
):

    if data_source == 'L5':
        l5_stations = ['Kooy', 'Bilt', 'Vlissingen', 'Eelde', 'Maastricht']

        sels = []

        for st in l5_stations:
            d_raw, d_mm, d_yy, _ = process_source(
                st,
                var,
                months=months,
                years=years,
                lats=None,
                lons=None,
                land_only=False,
                trim_border=None,
                rotpole_sel=ccrs.PlateCarree(),
                rolling_mean_var=rolling_mean_var,
                rolling_mean_years=rolling_mean_years,
                min_periods=min_periods
            )

            if frequency == 'Daily':
                d_sel = d_raw
            elif frequency == 'Monthly':
                d_sel = d_mm
            elif frequency == 'Yearly':
                d_sel = d_yy

            sels.append(d_sel)

        sels = xr.align(*sels, join='inner')

        data_sel = xr.concat(sels, dim='station').mean('station')

        return data_sel

    is_station = data_source in station_sources

    if not is_station:
        if isinstance(lats, str) or isinstance(lons, str):
            station_name = lats if isinstance(lats, str) else lons
            lat_sel = station_coord_cfg[station_name]['latitude']
            lon_sel = station_coord_cfg[station_name]['longitude']
        else:
            lat_sel = lats
            lon_sel = lons
    else:
        lat_sel = None
        lon_sel = None

    data_raw, data_mm, data_yy, data_avg = process_source(
        data_source,
        var,
        months=months,
        years=years,
        lats=lat_sel,
        lons=lon_sel,
        land_only=land_only,
        trim_border=trim_border,
        rotpole_sel=rotpole_sel,
        rolling_mean_var=rolling_mean_var,
        rolling_mean_years=rolling_mean_years,
        min_periods=min_periods
    )

    if frequency == 'Daily':
        data_sel = data_raw
    elif frequency == 'Monthly':
        data_sel = data_mm
    elif frequency == 'Yearly':
        data_sel = data_yy

    weights = area_weights(data_avg, rotpole_native=proj_cfg.get(data_source, ccrs.PlateCarree()))
    data_sel = area_weighted_mean(data_sel, weights=weights)

    return data_sel


results_base = []
results_compare = []

for ii in range(n_runs):

    data_sel_base = read_data(
        freq_sel_list[ii],
        data_base_list[ii], 
        var_list[ii],
        months=months_list[ii],
        years=years,
        lats=lats,
        lons=lons,
        land_only=land_only,
        trim_border=trim_border, 
        rotpole_sel=proj_sel,
        rolling_mean_var=rolling_mean_var,
        rolling_mean_years=rolling_mean_years,
        min_periods=min_periods
    )

    results_base.append(data_sel_base)

    data_sel_comp = read_data(
        freq_sel_list[ii],
        data_compare_list[ii], 
        var_list[ii],
        months=months_list[ii],
        years=years,
        lats=lats,
        lons=lons,
        land_only=land_only,
        trim_border=trim_border, 
        rotpole_sel=proj_sel,
        rolling_mean_var=rolling_mean_var,
        rolling_mean_years=rolling_mean_years,
        min_periods=min_periods
    )

    results_compare.append(data_sel_comp)

#%% Obtain plotting data

normal_data = []
diff_data = []
all_diffs = []
all_x = []
all_y = []

for ii in range(n_runs):
    xb = results_base[ii]
    yc = results_compare[ii]

    xb, yc = xr.align(xb, yc, join='inner')

    x = np.asarray(xb)
    y = np.asarray(yc)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    d = y - x

    normal_data.append((x, y))
    diff_data.append((x, d))
    all_diffs.append(d)
    all_x.append(x)
    all_y.append(y)


#%% Plot results

source_labels = {
    'Eobs_fine': 'E-OBS',
    'Eobs_coarse': 'E-OBS',
    'ERA5_fine': 'ERA5',
    'ERA5_coarse': 'ERA5',
    'RACMO2.3': 'R2.3',
    'RACMO2.4': 'R2.4',
    'L5': 'L5',
}

var_labels = {
    'Tg': 'Temperature (°C)',
    'P': 'Precipitation (mm)',
    'Sq': 'Sund. (hours/day)',
    'SW': r'SW$_{\text{in}}$ (W/m$^2$)',
}

def make_axis_label(source, var):
    src = source_labels.get(source, source)
    varlab = var_labels.get(var, var)
    return f'{src} {varlab}'

def pad_limits(data, pad_frac=0.02):
    lo = np.nanmin(data)
    hi = np.nanmax(data)
    pad = pad_frac*(hi - lo)
    return lo - pad, hi + pad


same_base = len(set(data_base_list)) == 1
same_comp = len(set(data_compare_list)) == 1
same_var = len(set(var_list)) == 1

share_x = share_labels and same_base and same_var
share_y = share_labels and same_var

fig, axes = plt.subplots(
    1, n_runs,
    figsize=(4*n_runs, 4),
    sharex=share_x,
    sharey=share_y
)

if share_x and (not share_y):
    wspace = 0.3
elif share_x and share_y:
    wspace = 0.06
else:
    wspace = 0.10

if share_x or share_y:
    bottom_margin = 0.14
    label_offset = 0.18

    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        bottom=0.08,
        top=0.995,
        wspace=wspace
    )

if n_runs == 1:
    axes = [axes]

for ii, ax in enumerate(axes):

    x, y = normal_data[ii]

    ax.scatter(x, y, s=12, color='k', alpha=0.4)

    if share_y or share_x:
        y_all = np.concatenate(all_y)
        ylo, yhi = pad_limits(y_all, pad_frac=0.02)
        x_all = np.concatenate(all_x)
        xlo, xhi = pad_limits(x_all, pad_frac=0.02)
    else:
        ylo, yhi = pad_limits(y, pad_frac=0.02)
        xlo, xhi = pad_limits(x, pad_frac=0.02)

    lo, hi = min(xlo, ylo), max(xhi, yhi)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.plot([lo*2 - 2, hi*2 + 2], [lo*2 - 2, hi*2 + 2], lw=1, color='k', ls='--')

    ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)

    if not share_x:
        ax.set_xlabel(make_axis_label(data_base_list[ii], var), fontsize=18)
    else:
        ax.set_xlabel('') 

    if not share_y:
        ax.set_ylabel(make_axis_label(data_compare_list[ii], var), fontsize=18)
    else:
        ax.set_ylabel('')

    ticks = ax.get_xticks()
    ticks = ticks[(ticks >= lo) & (ticks <= hi)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis='both', labelsize=14, length=6)

    comp_lab = source_labels.get(data_compare_list[ii], data_compare_list[ii])
    ax.set_title(f'{comp_lab}', fontsize=26, fontweight='bold')

    ax.grid()

if share_x:
    shared_xlabel = make_axis_label(data_base_list[0], var)
    left = axes[0].get_position().x0
    right = axes[-1].get_position().x1
    x_center = 0.5*(left + right)
    fig.text(
        x_center, bottom_margin - label_offset, shared_xlabel,
        ha='center',
        va='center',
        fontsize=28
    )

if share_y:
    shared_ylabel = f'{var_labels.get(var, var)}'
    pos0 = axes[0].get_position()
    y_center = 0.5*(pos0.y0 + pos0.y1)
    fig.text(
        pos0.x0 - 0.05, y_center, shared_ylabel,
        rotation=90,
        ha='center',
        va='center',
        fontsize=22
    )

plt.show()


# Colorbar for time # Options for colorbar coloring (season or over time) 
# or maybe shape for season
# horizontal colorbar below plots
# Colorbar op basis van jaar!

#%% Difference scatter plots

share_x = share_labels and same_base and same_var
share_y = share_labels and same_var

fig, axes = plt.subplots(
    1, n_runs,
    figsize=(4*n_runs, 4),
    sharex=share_x,
    sharey=share_y
)

if share_x or share_y:
    bottom_margin = 0.14
    label_offset = 0.17

    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        bottom=bottom_margin,
        top=0.995,
        wspace=0.06
    )

if n_runs == 1:
    axes = [axes]

if share_y:
    d_all = np.concatenate(all_diffs)
    dlo, dhi = pad_limits(d_all, pad_frac=0.02)

if share_x:
    x_all = np.concatenate(all_x)
    xlo, xhi = pad_limits(x_all, pad_frac=0.02)

for ii, ax in enumerate(axes):

    x, d = diff_data[ii]

    ax.scatter(x, d, s=12, color='k', alpha=0.4)
    ax.axhline(0, lw=1, color='k', ls='--')

    if share_y:
        d_all = np.concatenate(all_diffs)
        dlo, dhi = pad_limits(d_all, pad_frac=0.02)
    else:
        dlo, dhi = pad_limits(d, pad_frac=0.02)
    ax.set_ylim(dlo, dhi)

    if share_x:
        x_all = np.concatenate(all_x)
        xlo, xhi = pad_limits(x_all, pad_frac=0.02)
    else:
        xlo, xhi = pad_limits(x, pad_frac=0.02)
    ax.set_xlim(xlo, xhi)

    ax.set_aspect('auto')
    ax.set_box_aspect(1)   

    if not share_x:
        ax.set_xlabel(make_axis_label(data_base_list[ii], var), fontsize=16)
    else:
        ax.set_xlabel('')

    if not share_y:
        ax.set_ylabel(fr'$\Delta${var_labels.get(var, var)}', fontsize=16)
    else:
        ax.set_ylabel('')

    ax.tick_params(axis='both', labelsize=14, length=6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    base_lab = source_labels.get(data_base_list[ii], data_base_list[ii])
    comp_lab = source_labels.get(data_compare_list[ii], data_compare_list[ii])

    ax.set_title(f'{comp_lab} - {base_lab}', fontsize=26, fontweight='bold')

    ax.grid()

if share_x:
    shared_xlabel = make_axis_label(data_base_list[0], var)
    left = axes[0].get_position().x0
    right = axes[-1].get_position().x1
    x_center = 0.5*(left + right)
    fig.text(
        x_center, bottom_margin - label_offset, shared_xlabel,
        ha='center',
        va='center',
        fontsize=28
    )

if share_y:
    shared_ylabel = fr'$\Delta${var_labels.get(var, var)}'
    pos0 = axes[0].get_position()
    y_center = 0.5*(pos0.y0 + pos0.y1)
    fig.text(
        pos0.x0 - 0.06, y_center, shared_ylabel,
        rotation=90,
        ha='center',
        va='center',
        fontsize=22
    )

plt.show()

