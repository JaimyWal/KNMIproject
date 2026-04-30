#%% Imports

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import dask
from pathlib import Path
import sys
from importlib import reload

# Custom libraries
PROJECT_ROOT = Path.home() / 'KNMIproject'
sys.path.insert(0, str(PROJECT_ROOT))

from RegionalTrends.Helpers import ProcessVar
reload(ProcessVar)
from RegionalTrends.Helpers.ProcessVar import load_var

from RegionalTrends.Helpers import AreaWeights
reload(AreaWeights)
from RegionalTrends.Helpers.AreaWeights import area_weights, area_weighted_mean

from RegionalTrends.Helpers import ComputeTendencies
reload(ComputeTendencies)
from RegionalTrends.Helpers.ComputeTendencies import construct_tendency, full_interval_time

import RegionalTrends.Helpers.Config.Constants as Constants
reload(Constants)

plt.rcParams['axes.unicode_minus'] = False
dask.config.set(scheduler='threads', num_workers=12)

#%% User inputs

# Main arguments
var_tend_main = {
    'Total Tendency': 'tendtot',
    'Total Radiative Tendency': 'radtot',
    'Shortwave Radiative Tendency': 'swnet',
    'Longwave Radiative Tendency': 'lwnet',
    'Total Dynamical Tendency': 'dyntot',
}
var_tend_sums = {
    'Total Tendency': ['dyntot', 'radtot', 'senstot', 'phasetot', 'frictot', 'numtot'],
    'Total Radiative Tendency': ['swnet', 'lwnet'],
    'Shortwave Radiative Tendency': ['swtopdn', 'swtopup', 'swbotdn', 'swbotup'],
    'Longwave Radiative Tendency': ['lwtopdn', 'lwtopup', 'lwbotdn', 'lwbotup'],
    'Total Dynamical Tendency': ['horadv', 'vertadv', 'adicomp', 'orography'],
}
var_tend_close_pr = ['dyntot', 'radtot', 'senstot', 'phasetot', 'frictot', 'numtot']
var_tend_close_true = 'tendtot'
var_temp = 'templ1'
file_freq = 'Seasonal'
proc_freq = 'Seasonal'
relation = 'Adjacent'
save_name_base = None

# Data selection arguments
years = np.arange(1961, 2023 + 1)
months_dict = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
# months_dict = {'Jan': [1], 'Feb': [2], 'Mar': [3], 'Apr': [4], 'May': [5], 'Jun': [6], 'Jul': [7], 'Aug': [8], 'Sep': [9], 'Oct': [10], 'Nov': [11], 'Dec': [12]}
# months_dict = None
lats = [50.7, 53.6]
lons = [3.25, 7.35]
proj_sel = 'RACMO2.4'
land_only = True
trim_border = None

# Group time series plot arguments
group_layout = (2, 2)
group_panel_width = 6
group_panel_height = 4
group_y_range = None
group_shared_labels = True
group_mirror_y_axes = True
group_x_year = True
group_yearly = True

# Closure time series plot arguments
close_layout = (2, 2)
close_panel_width = 6
close_panel_height = 4
close_y_range = None
close_shared_labels = True
close_mirror_y_axes = True
close_x_year = True
close_yearly = True

# Reference
# Optie voor year round plot (als relatie adjacent is)

#%% Dataset configurations

data_sources = Constants.DATA_SOURCES
station_sources = Constants.STATION_SOURCES
station_coord_cfg = Constants.STATION_COORD_CFG
enbud_vars = Constants.ENBUD_VARS
tendency_vars = Constants.TENDENCY_VARS
var_file_cfg = Constants.VAR_FILE_CFG
var_name_cfg = Constants.VAR_NAME_CFG
var_units_cfg = Constants.VAR_UNIT_CFG
var_symbol_cfg = Constants.VAR_SYMBOL_CFG
proj_cfg = Constants.PROJ_CFG

# Setup projections
proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())

SEASONS = (
    ((12, 1, 2), 1),
    ((3, 4, 5), 4),
    ((6, 7, 8), 7),
    ((9, 10, 11), 10),
)

COLORS = ['#000000', '#DB2525', '#0168DE', '#00A236', '#CA721B', '#7B2CBF', '#E91E8C', '#808080']

PANEL_KEYS = list(months_dict.keys()) if months_dict is not None else ['All']

#%% ============================================================================
#   HELPER FUNCTIONS
#   ============================================================================

def months_span_year_boundary(months):
    if months is None or len(months) == 0:
        return False
    months_arr = np.asarray(months, dtype=int)
    return int(months_arr[0]) > int(months_arr[-1])

def needs_extended_years(months_dict=None):
    if months_dict:
        for months in months_dict.values():
            if months_span_year_boundary(months):
                return True
    return False

def assign_clim_year(data, months):

    months_arr = np.asarray(months, dtype=int)
    month_start = int(months_arr[0])
    month_end = int(months_arr[-1])
    spans_year_boundary = month_start > month_end  # e.g., [12, 1, 2]
    
    month_vals = data['time'].dt.month
    year_vals = data['time'].dt.year
    
    if spans_year_boundary:
        # December gets next year as climate year
        clim_year = xr.where(month_vals >= month_start, year_vals + 1, year_vals)
    else:
        clim_year = year_vals
    
    return data.assign_coords(clim_year=('time', clim_year.values))

def filter_by_season(data, months, years):
    
    # Handle 2D data (no time dimension)
    if 'time' not in data.dims:
        return data
    
    months_arr = np.asarray(months, dtype=int)
    
    # Filter by months first
    month_vals = data['time'].dt.month
    in_season = month_vals.isin(months_arr)
    data_season = data.where(in_season, drop=True)
    
    # Assign climate year for this specific season
    data_season = assign_clim_year(data_season, months)
    
    # Filter by exact climate years
    years_arr = np.asarray(years)
    return data_season.where(data_season['clim_year'].isin(years_arr), drop=True)

def compute_seasonal_yearly(data, months, years, proc_type='Mean'):
    
    filtered = filter_by_season(data, months, years)    

    # Group by climate year to get seasonal yearly means
    if proc_type == 'Max':
        yearly = filtered.groupby('clim_year').max('time')
    elif proc_type == 'Min':
        yearly = filtered.groupby('clim_year').min('time')
    else:
        month_weights = filtered['time'].dt.days_in_month.astype('float32')
        month_weights = month_weights.assign_coords(clim_year=filtered['clim_year'])

        weighted_sum = (filtered*month_weights).groupby('clim_year').sum('time')
        weight_sum = month_weights.where(filtered.notnull()).groupby('clim_year').sum('time')
        yearly = weighted_sum / weight_sum
        # yearly = filtered.groupby('clim_year').mean('time')
    
    return yearly.astype('float32')

def compute_yearly_mean(data, interval):
    if interval in {'Raw', 'Daily'}:
        yearly = data.groupby('time.year').mean('time')
    elif interval == 'Monthly':
        weights = data['time'].dt.days_in_month.astype('float32')
        weighted_sum = (data*weights).groupby('time.year').sum('time')
        weight_sum = weights.where(data.notnull()).groupby('time.year').sum('time')
        yearly = weighted_sum / weight_sum
    elif interval == 'Seasonal':
        time_index = pd.DatetimeIndex(data['time'].values)
        season_days = (
            (time_index - pd.DateOffset(months=1)).days_in_month
            + time_index.days_in_month
            + (time_index + pd.DateOffset(months=1)).days_in_month
        ).astype('float32')
        weights = xr.DataArray(season_days, dims=['time'], coords={'time': data['time']})
        weighted_sum = (data*weights).groupby('time.year').sum('time')
        weight_sum = weights.where(data.notnull()).groupby('time.year').sum('time')
        yearly = weighted_sum / weight_sum

    year_vals = yearly['year'].values
    full_time = full_interval_time(data['time'], interval='Yearly')
    return yearly.rename({'year': 'time'}).assign_coords(
        time=('time', pd.to_datetime(dict(year=year_vals, month=1, day=1)))
    ).reindex(time=full_time).astype('float32')

def sum_tendency_terms(data_dict, vars_to_sum):
    total = 0
    for var in vars_to_sum:
        total = total + data_dict[var]
    return total

def build_group_panel_data(group_name, tendency_data):
    main_var = var_tend_main[group_name]
    group_vars = var_tend_sums.get(group_name, [])

    return {
        'main': tendency_data.get(main_var),
        'components': {var: tendency_data.get(var) for var in group_vars},
    }

def build_closure_panel_data(tendency_data, temp_close):
    summed_processes = sum_tendency_terms(tendency_data, var_tend_close_pr)
    true_tendency = tendency_data.get(var_tend_close_true)
    diff_true = None if temp_close is None or true_tendency is None else temp_close - true_tendency
    diff_sum = None if temp_close is None or summed_processes is None else temp_close - summed_processes

    return {
        'summed_processes': summed_processes,
        'true_tendency': true_tendency,
        'temp_close': temp_close,
        'diff_true': diff_true,
        'diff_sum': diff_sum,
    }

#%% ============================================================================
#   PHASE 1: LOAD ALL DATA ONCE
#   ============================================================================

print('='*60)
print('PHASE 1: Loading all data')
print('='*60)

var_tend_main_vars = list(var_tend_main.values())
var_tend_sum_vars = [var for vars_group in var_tend_sums.values() for var in vars_group]
vars_all = set(var_tend_main_vars + var_tend_sum_vars + var_tend_close_pr + [var_tend_close_true] +
               ([var_temp] if var_temp is not None else []))
vars_tendencies = set(var_tend_main_vars + var_tend_sum_vars + var_tend_close_pr + [var_tend_close_true])

# Extend years if needed for DJF-style seasons
years_load = list(years)
years_load.append(years_load[-1] + 1) 
if (needs_extended_years(months_dict=months_dict) or proc_freq == 'Seasonal') and file_freq != 'Seasonal':
    extra_years = {y - 1 for y in years} - set(years)
    years_load = sorted(set(years_load) | extra_years)

# Load all var data
var_data = {}
weights = None

for var in vars_all:
    print(f'  Loading: {var}')
    
    # Load raw data
    data_raw = load_var(
        var=var,
        data_source='RACMO2.4A',
        data_sources=data_sources,
        station_sources=station_sources,
        file_freq=file_freq,
        var_file_cfg=var_file_cfg,
        proj_cfg=proj_cfg,
        months=None,
        years=years_load,
        lats=lats,
        lons=lons,
        land_only=land_only,
        trim_border=trim_border,
        rotpole_sel=proj_sel,
        station_coords=station_coord_cfg,
    )
    
    if weights is None:
        weights = area_weights(data_raw.isel(time=0), 
                                rotpole_native=proj_cfg.get('RACMO2.4A', ccrs.PlateCarree()))
    
    var_data[var] = data_raw.astype('float32')

print(f'Loaded {len(var_data)} datasets\n')

#%% ============================================================================
#   PHASE 2: DOMAIN AVERAGING OF RAW DATA
#   ============================================================================

print('='*60)
print('PHASE 2: Domain averaging of raw data')
print('='*60)

var_weighted = {}

for var in vars_all:
    print(f'  Area-weighting: {var}')
    var_weighted[var] = area_weighted_mean(var_data[var], weights=weights).compute().astype('float32')
    del var_data[var]

var_data = None

print(f'Area-weighted {len(var_weighted)} datasets\n')

#%% ============================================================================
#   PHASE 3: COMPUTING TENDENCIES / TEMPERATURE AVERAGES
#   ============================================================================

print('='*60)
print('PHASE 3: Computing tendencies / temperature averages')
print('='*60)

var_tendencies = {}

for var in vars_tendencies:
    print(f'  Computing tendency: {var}')

    if file_freq == proc_freq and relation == 'Adjacent':
        var_tendencies[var] = var_weighted[var]
    else:
        var_tendencies[var] = construct_tendency(
            var_weighted[var],
            interval=proc_freq,
            relation=relation,
            return_intermediates=False
        )

var_temperature = None
if var_temp is not None:
    temp_da = var_weighted[var_temp]

    if file_freq == proc_freq:
        var_temperature = temp_da.astype('float32')
    elif proc_freq == 'Raw':
        var_temperature = temp_da.astype('float32')
    elif proc_freq == 'Daily':
        var_temperature = temp_da.resample(time='1D').mean(skipna=False).reindex(
            time=full_interval_time(temp_da['time'], interval='Daily')
        ).astype('float32')
    elif proc_freq == 'Monthly':
        var_temperature = temp_da.resample(time='MS').mean(skipna=False).reindex(
            time=full_interval_time(temp_da['time'], interval='Monthly')
        ).astype('float32')
    elif proc_freq == 'Seasonal':
        if file_freq == 'Monthly':
            monthly_temperature = temp_da
        else:
            monthly_temperature = temp_da.resample(time='MS').mean(skipna=False).reindex(
                time=full_interval_time(temp_da['time'], interval='Monthly')
            ).astype('float32')
        seasonal_parts = []
        years_temp = np.append(years, years[-1] + 1)
        for months, time_month in SEASONS:
            seasonal = compute_seasonal_yearly(monthly_temperature, months, years_temp, proc_type='Mean')
            season_year = seasonal['clim_year'].values
            seasonal = seasonal.rename({'clim_year': 'time'}).assign_coords(
                season_year=('time', season_year),
                time=('time', pd.to_datetime(dict(year=season_year, month=time_month, day=1))),
            )
            seasonal_parts.append(seasonal)
        var_temperature = xr.concat(seasonal_parts, dim='time').sortby('time').astype('float32')

var_temp_close = None
if var_temperature is not None:
    if relation == 'Adjacent':
        var_temp_close = var_temperature.shift(time=-1) - var_temperature
    elif relation == 'Yearly':
        if proc_freq == 'Seasonal':
            var_temp_close = var_temperature.shift(time=-4) - var_temperature
        elif proc_freq == 'Monthly':
            var_temp_close = var_temperature.shift(time=-12) - var_temperature
        else:
            print('Yearly relation not implemented for this proc_freq, because of leap years')

var_tend_yearly = {}
var_temp_yearly = None
var_temp_close_yearly = None
if group_yearly or close_yearly:
    for var in vars_tendencies:
        var_tend_yearly[var] = construct_tendency(
            var_tendencies[var],
            interval='Yearly',
            relation='Adjacent',
            return_intermediates=False
        )
    if var_temperature is not None:
        var_temp_yearly = compute_yearly_mean(var_temperature, proc_freq)
        var_temp_close_yearly = var_temp_yearly.shift(time=-1) - var_temp_yearly

# Filter for selected months and years
for var, data in var_tendencies.items():
    var_tendencies[var] = data.where(data['time'].dt.year.isin(years), drop=True)

for var, data in var_tend_yearly.items():
    var_tend_yearly[var] = data.where(data['time'].dt.year.isin(years), drop=True)

if var_temperature is not None:
    # var_temperature = var_temperature.where(var_temperature['time'].dt.year.isin(years), drop=True)
    var_temp_close = var_temp_close.where(var_temp_close['time'].dt.year.isin(years), drop=True)

if var_temp_yearly is not None:
    # var_temp_yearly = var_temp_yearly.where(var_temp_yearly['time'].dt.year.isin(years), drop=True)
    var_temp_close_yearly = var_temp_close_yearly.where(var_temp_close_yearly['time'].dt.year.isin(years), drop=True)

var_tend_proc = {}
var_temp_proc = {}
var_temp_close_proc = {}
if months_dict is not None:
    for month_key, months in months_dict.items():

        var_tend_proc[month_key] = {}
        for var, data in var_tendencies.items():
            var_tend_proc[month_key][var] = data.where(data['time'].dt.month.isin(months), drop=True)

        if var_temperature is not None:
            var_temp_proc[month_key] = var_temperature.where(
                var_temperature['time'].dt.month.isin(months),
                drop=True
            )

        if var_temp_close is not None:
            var_temp_close_proc[month_key] = var_temp_close.where(
                var_temp_close['time'].dt.month.isin(months),
                drop=True
            )
else:
    var_tend_proc['All'] = var_tendencies
    var_temp_proc['All'] = var_temperature
    var_temp_close_proc['All'] = var_temp_close

group_panel_data = {
    group_name: {month_key: build_group_panel_data(group_name, var_tend_proc[month_key]) for month_key in PANEL_KEYS}
    for group_name in var_tend_main
}
group_yearly_data = {
    group_name: build_group_panel_data(group_name, var_tend_yearly)
    for group_name in var_tend_main
}

closure_panel_data = {
    month_key: build_closure_panel_data(var_tend_proc[month_key], var_temp_close_proc.get(month_key))
    for month_key in PANEL_KEYS
}
closure_yearly_data = build_closure_panel_data(var_tend_yearly, var_temp_close_yearly) if close_yearly else None

#%% ============================================================================
#   PLOTTING HELPER FUNCTIONS
#   ============================================================================

def format_panel_title(month_key):
    if relation == 'Adjacent' and months_dict is not None:
        panel_keys = list(months_dict.keys())
        if month_key in panel_keys:
            next_key = panel_keys[(panel_keys.index(month_key) + 1) % len(panel_keys)]
            return f'{month_key} $\\rightarrow$ {next_key}'

    return month_key

def plot_time_values(data, use_year_axis=False):
    if data is None or not use_year_axis:
        return None if data is None else data['time'].values

    year_vals = data['time'].dt.year.values
    return pd.to_datetime(dict(year=year_vals, month=1, day=1))

def group_plot_series(group_name, panel_data):
    group_vars = var_tend_sums.get(group_name, [])
    group_colors = COLORS[1:]

    panel_series = [(group_name, panel_data['main'], COLORS[0], '-')]
    for ii, var in enumerate(group_vars):
        panel_series.append((var_name_cfg.get(var, var), panel_data['components'].get(var), group_colors[ii % len(group_colors)], '-'))

    return panel_series

def closure_actual_plot_series(panel_data):
    return [
        ('Summed processes', panel_data['summed_processes'], 'tab:blue', '-'),
        (var_name_cfg.get(var_tend_close_true, var_tend_close_true), panel_data['true_tendency'], 'tab:orange', '-'),
        (r'$\Delta T$', panel_data['temp_close'], 'tab:green', '-'),
    ]

def closure_difference_plot_series(panel_data):
    return [
        (r'$\Delta T$ - tendtot', panel_data['diff_true'], 'tab:red', '-'),
        (r'$\Delta T$ - summed processes', panel_data['diff_sum'], 'tab:purple', '-'),
    ]

def plot_series_list(ax, panel_series, use_year_axis=False):
    for label, data, color, linestyle in panel_series:
        if data is None:
            continue
        ax.plot(
            plot_time_values(data, use_year_axis=use_year_axis),
            data.values,
            c=color,
            lw=2.5,
            ms=4,
            marker='o',
            ls=linestyle,
            label=label,
        )

def format_time_axis(ax, title, title_fs, tick_fs, zero_line=False):
    if zero_line:
        ax.axhline(0.0, color='k', lw=1.0, alpha=0.6)

    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=title_fs, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_fs)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

def style_legend(ax, legend_fs):
    leg = ax.legend(fontsize=legend_fs, handlelength=1.5, loc='best')
    leg.set_zorder(20)
    for line in leg.get_lines():
        line.set_linewidth(3.0)

def normalize_axes(axes, n_rows, n_cols):
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    elif n_rows == 1:
        return axes.reshape(1, -1)
    elif n_cols == 1:
        return axes.reshape(-1, 1)
    return axes

def hide_unused_axes(axes, n_used, n_rows, n_cols):
    for idx in range(n_used, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

def add_shared_axis_labels(fig, axes, x_label, y_label, fontsize, pad=0.01):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    # Get figure dimensions for coordinate conversion
    fig_width, fig_height = fig.get_size_inches() * fig.dpi
    n_cols = axes.shape[1]
    
    # Find leftmost extent of y-axis tick labels (left column)
    min_x_fig = 1.0  # Start with rightmost possible
    for row in range(axes.shape[0]):
        ax = axes[row, 0]
        for label in ax.get_yticklabels():
            bbox = label.get_window_extent(renderer=renderer)
            # Convert to figure coordinates
            x_left_fig = bbox.x0 / fig_width
            min_x_fig = min(min_x_fig, x_left_fig)
    
    # Find bottommost extent of x-axis tick labels (bottom row)
    min_y_fig = 1.0  # Start with topmost possible
    for col in range(n_cols):
        ax = axes[-1, col]
        for label in ax.get_xticklabels():
            bbox = label.get_window_extent(renderer=renderer)
            # Convert to figure coordinates
            y_bottom_fig = bbox.y0 / fig_height
            min_y_fig = min(min_y_fig, y_bottom_fig)
    
    # X label centered below bottom row, positioned relative to tick labels
    x_center = 0.5 * (axes[-1, 0].get_position().x0 + axes[-1, -1].get_position().x1)
    y_bottom = min_y_fig - pad - 0.01
    fig.text(x_center, y_bottom, x_label, ha='center', va='top', fontsize=fontsize*1.2)
    
    # Y label on left side only
    y_center = 0.5 * (axes[0, 0].get_position().y1 + axes[-1, 0].get_position().y0)
    x_left = min_x_fig - pad
    fig.text(x_left, y_center, y_label, ha='right', va='center', rotation=90, fontsize=fontsize)

def save_figure(save_name_base, var, suffix):
    if save_name_base is None:
        return
    
    save_name = f'{save_name_base}_{var}_{suffix}'
    pdf_dir = Path(f'/nobackup/users/walj/Figures_tendencies/pdf/{var}')
    jpg_dir = Path(f'/nobackup/users/walj/Figures_tendencies/jpg/{var}')
    pdf_dir.mkdir(parents=True, exist_ok=True)
    jpg_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(str(pdf_dir / f'{save_name}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(str(jpg_dir / f'{save_name}.jpg'), format='jpg', dpi=300, bbox_inches='tight')

def plot_panel_grid(panel_series_dict, layout, panel_width, panel_height, y_range, shared_labels,
                    mirror_y_axes, use_year_axis, y_label, save_var, suffix, zero_line=False):
    n_panels = len(PANEL_KEYS)
    n_rows, n_cols = layout

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        sharex=True, sharey=False
    )
    wspace = 0.06 if mirror_y_axes else 0.16
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.95, wspace=wspace, hspace=0.18)
    axes = normalize_axes(axes, n_rows, n_cols)

    title_fs = max(18, int(panel_height * 5))
    tick_fs = max(12, int(panel_height * 3))
    legend_fs = max(10, int(panel_height * 2.5))

    for idx, month_key in enumerate(PANEL_KEYS):
        ax = axes[idx // n_cols, idx % n_cols]
        plot_series_list(ax, panel_series_dict[month_key], use_year_axis=use_year_axis)
        format_time_axis(ax, format_panel_title(month_key), title_fs, tick_fs, zero_line=zero_line)

        is_right_col = (idx % n_cols == n_cols - 1) and n_cols > 1
        if mirror_y_axes and is_right_col:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        panel_y_range = y_range.get(month_key) if isinstance(y_range, dict) else y_range
        if panel_y_range is not None:
            ax.set_ylim(*panel_y_range)

        style_legend(ax, legend_fs)

    hide_unused_axes(axes, n_panels, n_rows, n_cols)

    if shared_labels:
        add_shared_axis_labels(
            fig,
            axes,
            'Time',
            y_label,
            fontsize=max(28, int(panel_height * 7))
        )

    save_figure(save_name_base, save_var, suffix)
    plt.show()

def plot_single_panel(panel_series, panel_width, panel_height, y_range, shared_labels, y_label,
                      save_var, suffix, title='Yearly', zero_line=False):
    fig, ax = plt.subplots(1, 1, figsize=(panel_width * 1.4, panel_height * 1.1))

    title_fs = max(18, int(panel_height * 5))
    tick_fs = max(12, int(panel_height * 3))
    legend_fs = max(10, int(panel_height * 2.5))

    plot_series_list(ax, panel_series)
    format_time_axis(ax, title, title_fs, tick_fs, zero_line=zero_line)

    if y_range is not None and not isinstance(y_range, dict):
        ax.set_ylim(*y_range)

    if shared_labels:
        ax.set_xlabel('Time', fontsize=max(18, int(panel_height * 4)))
        ax.set_ylabel(y_label, fontsize=max(18, int(panel_height * 4)))

    style_legend(ax, legend_fs)
    save_figure(save_name_base, save_var, suffix)
    plt.show()

#%% ============================================================================
#   PHASE 4: PLOTTING TENDENCY GROUPS
#   ============================================================================

for group_name in var_tend_main:
    main_var = var_tend_main[group_name]
    group_unit = var_units_cfg.get(main_var, '')
    group_label = f'{group_name} ({group_unit})' if group_unit else group_name
    group_panel_series = {
        month_key: group_plot_series(group_name, group_panel_data[group_name][month_key])
        for month_key in PANEL_KEYS
    }
    plot_panel_grid(
        group_panel_series,
        layout=group_layout,
        panel_width=group_panel_width,
        panel_height=group_panel_height,
        y_range=group_y_range,
        shared_labels=group_shared_labels,
        mirror_y_axes=group_mirror_y_axes,
        use_year_axis=group_x_year,
        y_label=group_label,
        save_var=main_var,
        suffix='group',
    )
if group_yearly:
    for group_name in var_tend_main:
        main_var = var_tend_main[group_name]
        group_unit = var_units_cfg.get(main_var, '')
        group_label = f'{group_name} ({group_unit})' if group_unit else group_name
        group_yearly_series = group_plot_series(group_name, group_yearly_data[group_name])
        plot_single_panel(
            group_yearly_series,
            panel_width=group_panel_width,
            panel_height=group_panel_height,
            y_range=group_y_range,
            shared_labels=group_shared_labels,
            y_label=group_label,
            save_var=main_var,
            suffix='group_yearly',
        )

#%% ============================================================================
#   PHASE 5: PLOTTING / CHECKING CLOSURE
#   ============================================================================

close_unit = var_units_cfg.get(var_tend_close_true, '')
close_label = f'Closure ({close_unit})' if close_unit else 'Closure'
close_diff_label = f'Closure Difference ({close_unit})' if close_unit else 'Closure Difference'
closure_actual_series = {
    month_key: closure_actual_plot_series(closure_panel_data[month_key])
    for month_key in PANEL_KEYS
}
closure_diff_series = {
    month_key: closure_difference_plot_series(closure_panel_data[month_key])
    for month_key in PANEL_KEYS
}

plot_panel_grid(
    closure_actual_series,
    layout=close_layout,
    panel_width=close_panel_width,
    panel_height=close_panel_height,
    y_range=close_y_range,
    shared_labels=close_shared_labels,
    mirror_y_axes=close_mirror_y_axes,
    use_year_axis=close_x_year,
    y_label=close_label,
    save_var=var_tend_close_true,
    suffix='closure_actual',
)
plot_panel_grid(
    closure_diff_series,
    layout=close_layout,
    panel_width=close_panel_width,
    panel_height=close_panel_height,
    y_range=close_y_range,
    shared_labels=close_shared_labels,
    mirror_y_axes=close_mirror_y_axes,
    use_year_axis=close_x_year,
    y_label=close_diff_label,
    save_var=var_tend_close_true,
    suffix='closure_diff',
    zero_line=True,
)

if close_yearly:
    closure_actual_yearly_series = closure_actual_plot_series(closure_yearly_data)
    closure_diff_yearly_series = closure_difference_plot_series(closure_yearly_data)
    plot_single_panel(
        closure_actual_yearly_series,
        panel_width=close_panel_width,
        panel_height=close_panel_height,
        y_range=close_y_range,
        shared_labels=close_shared_labels,
        y_label=close_label,
        save_var=var_tend_close_true,
        suffix='closure_actual_yearly',
    )
    plot_single_panel(
        closure_diff_yearly_series,
        panel_width=close_panel_width,
        panel_height=close_panel_height,
        y_range=close_y_range,
        shared_labels=close_shared_labels,
        y_label=close_diff_label,
        save_var=var_tend_close_true,
        suffix='closure_diff_yearly',
        zero_line=True,
    )

#%%

# Alles converteren naar rates in K/day?
# geen trends maar 11-jarige blokken. Dus vergelijk temperatuur gemiddelde blok [1965, 1975] met [1975, 1985], etc.
# Bepaal ook tendens die verschil in deze blokken verklaart. 
# Kijk hoe we dit kunnen opsplitsen in seizoenaal.

# Nieuw script maken voor trends


# Kijken naar aftrekken van seizoenale cyclus / percentages!!!
# Shapefile voor Nederland
# Sum over Ap and introduce zero
# Float64 for non-net fluxes (huge numbers)
