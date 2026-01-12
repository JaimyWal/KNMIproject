#%% Imports

# Standard libraries
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from pathlib import Path
import sys
from importlib import reload

# Custom libraries
PROJECT_ROOT = Path.home() / 'KNMIproject'
sys.path.insert(0, str(PROJECT_ROOT))

from RegionalTrends.Helpers import ProcessSource
reload(ProcessSource)
from RegionalTrends.Helpers.ProcessSource import process_source

from RegionalTrends.Helpers import AreaWeights
reload(AreaWeights)
from RegionalTrends.Helpers.AreaWeights import area_weights, area_weighted_mean

# Data config custom libraries
import RegionalTrends.Helpers.Config.Constants as Constants
reload(Constants)

import RegionalTrends.Helpers.Config.Paths as Paths
reload(Paths)
from RegionalTrends.Helpers.Config.Paths import build_file_cfg, freq_tags


plt.rcParams['axes.unicode_minus'] = False
dask.config.set(scheduler='threads', num_workers=12)

#%% User inputs

# Main arguments
n_runs = 3
var = 'IWP' #
data_base = ['ERA5_coarse', 'ERA5_coarse', 'RACMO2.3'] # 
data_compare = ['RACMO2.3', 'RACMO2.4', 'RACMO2.4'] # 

# Data selection arguments
freq_sel = 'Monthly' #
months = None # 
years = [2015, 2020]
lats = [50.7, 53.6]
lons = [3.25, 7.35]
proj_sel = 'RACMO2.4'
land_only = True
trim_border = None

# Plotting arguments
share_labels = True

# Other arguments
rolling_mean_var = False
rolling_mean_years = 7
min_periods = 1

#%% Dataset configurations

data_sources = Constants.DATA_SOURCES
station_sources = Constants.STATION_SOURCES
var_name_cfg = Constants.VAR_NAME_CFG
station_coord_cfg = Constants.STATION_COORD_CFG
proj_cfg = Constants.PROJ_CFG
var_labels_cfg = Constants.LABEL_PLOT_CFG

proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())

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
    data_sources,
    station_sources,
    var_name_cfg,
    proj_cfg,
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
    
    if frequency == 'Daily':
        monthly_or_daily = 'Daily'
    else:
        monthly_or_daily = 'Monthly'
    freq_str, racmo24_sep = freq_tags(monthly_or_daily)

    if frequency == 'Daily':
        freq_source = 'raw'
    elif frequency == 'Monthly':
        freq_source = 'monthly'
    elif frequency == 'Yearly':
        freq_source = 'yearly'

    file_cfg = build_file_cfg(freq_str, racmo24_sep)

    if data_source == 'L5':

        l5_stations = ['Kooy', 'Bilt', 'Vlissingen', 'Eelde', 'Maastricht']

        sels = []

        for st in l5_stations:
            d_sel = process_source(
                st,
                var,
                data_sources,
                station_sources,
                var_name_cfg,
                file_cfg,
                proj_cfg,
                months=months,
                years=years,
                lats=None,
                lons=None,
                land_only=False,
                trim_border=None,
                rotpole_sel=ccrs.PlateCarree(),
                rolling_mean_var=rolling_mean_var,
                rolling_mean_years=rolling_mean_years,
                min_periods=min_periods,
                return_items=(freq_source)
            )

            sels.append(d_sel[freq_source])

        sels = xr.align(*sels, join='inner')

        data_proc = xr.concat(sels, dim='station').mean('station')

        return data_proc

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

    data_sel = process_source(
        data_source,
        var,
        data_sources,
        station_sources,
        var_name_cfg,
        file_cfg,
        proj_cfg,
        months=months,
        years=years,
        lats=lat_sel,
        lons=lon_sel,
        land_only=land_only,
        trim_border=trim_border,
        rotpole_sel=rotpole_sel,
        rolling_mean_var=rolling_mean_var,
        rolling_mean_years=rolling_mean_years,
        min_periods=min_periods,
        return_items=(freq_source, 'avg')
    )

    weights = area_weights(data_sel['avg'], rotpole_native=proj_cfg.get(data_source, ccrs.PlateCarree()))
    data_proc = area_weighted_mean(data_sel[freq_source], weights=weights)

    return data_proc


results_base = []
results_compare = []

for ii in range(n_runs):

    data_sel_base = read_data(
        freq_sel_list[ii],
        data_base_list[ii], 
        var_list[ii],
        data_sources,
        station_sources,
        var_name_cfg,
        proj_cfg,
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
        data_sources,
        station_sources,
        var_name_cfg,
        proj_cfg,
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

def make_axis_label(source, var):
    src = source_labels.get(source, source)
    varlab = var_labels_cfg.get(var, var)
    return f'{src} {varlab}'

def pad_limits(data, pad_frac=0.02):
    lo = np.nanmin(data)
    hi = np.nanmax(data)
    pad = pad_frac*(hi - lo)
    return lo - pad, hi + pad


same_base = len(set(data_base_list)) == 1
same_comp = len(set(data_compare_list)) == 1
same_var = len(set(var_list)) == 1

share_x = share_labels and same_base and same_var and n_runs > 1
share_y = share_labels and same_var and n_runs > 1

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

    if freq_sel_list[ii] == 'Daily':
        scatter_size = 12
        scatter_alpha = 0.4
    elif freq_sel_list[ii] == 'Monthly':
        scatter_size = 30
        scatter_alpha = 0.8
    elif freq_sel_list[ii] == 'Yearly':
        scatter_size = 40
        scatter_alpha = 1

    ax.scatter(x, y, s=scatter_size, color='k', alpha=scatter_alpha)

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

    ax.plot([0, 1], [0, 1],
        transform=ax.transAxes,
        lw=1.5,
        color='xkcd:brick red',
        ls='--')

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

    if share_y is True or n_runs > 1:
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
    # shared_ylabel = f'{var_labels_cfg.get(var, var)}'
    # pos0 = axes[0].get_position()
    # y_center = 0.5*(pos0.y0 + pos0.y1)
    # fig.text(
    #     pos0.x0 - 0.06, y_center, shared_ylabel,
    #     rotation=90,
    #     ha='center',
    #     va='center',
    #     fontsize=22
    # )
    axes[0].set_ylabel(var_labels_cfg.get(var, var), fontsize=22)

plt.show()


# Colorbar for time # Options for colorbar coloring (season or over time) 
# or maybe shape for season
# horizontal colorbar below plots
# Colorbar op basis van jaar!

#%% Difference scatter plots

share_x = share_labels and same_base and same_var and n_runs > 1
share_y = share_labels and same_var and n_runs > 1

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

    if freq_sel_list[ii] == 'Daily':
        scatter_size = 12
        scatter_alpha = 0.4
    elif freq_sel_list[ii] == 'Monthly':
        scatter_size = 30
        scatter_alpha = 0.8
    elif freq_sel_list[ii] == 'Yearly':
        scatter_size = 40
        scatter_alpha = 1

    ax.scatter(x, d, s=scatter_size, color='k', alpha=scatter_alpha)
    ax.axhline(0, lw=1.5, color='xkcd:brick red', ls='--')

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
        ax.set_ylabel(fr'$\Delta${var_labels_cfg.get(var, var)}', fontsize=16)
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
    # shared_ylabel = fr'$\Delta${var_labels_cfg.get(var, var)}'
    # pos0 = axes[0].get_position()
    # y_center = 0.5*(pos0.y0 + pos0.y1)
    # fig.text(
    #     pos0.x0 - 0.06, y_center, shared_ylabel,
    #     rotation=90,
    #     ha='center',
    #     va='center',
    #     fontsize=22
    # )
    axes[0].set_ylabel(fr'$\Delta${var_labels_cfg.get(var, var)}', fontsize=22)

plt.show()