#%% Imports

# Standard libraries
import numpy as np
import pandas as pd
import xarray as xr
import dask
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import statsmodels.api as sm
import cartopy.crs as ccrs
from pathlib import Path
import sys
from importlib import reload

# Custom libraries
PROJECT_ROOT = Path.home() / 'KNMIproject'
sys.path.insert(0, str(PROJECT_ROOT))

from RegionalTrends.Helpers import PlotMaps
reload(PlotMaps)
from RegionalTrends.Helpers.PlotMaps import plot_map, shared_colorbar

from RegionalTrends.Helpers import ProcessNetCDF
reload(ProcessNetCDF)
from RegionalTrends.Helpers.ProcessNetCDF import is_monthly_time

from RegionalTrends.Helpers import ProcessVar
reload(ProcessVar)
from RegionalTrends.Helpers.ProcessVar import load_var

from RegionalTrends.Helpers import AreaWeights
reload(AreaWeights)
from RegionalTrends.Helpers.AreaWeights import area_weights, area_weighted_mean

# Config libraries
import RegionalTrends.Helpers.Config.Constants as Constants
reload(Constants)
import RegionalTrends.Helpers.Config.Plotting as Plotting
reload(Plotting)
from RegionalTrends.Helpers.Config.Plotting import convert_cmap

plt.rcParams['axes.unicode_minus'] = False
dask.config.set(scheduler='threads', num_workers=12)

#%% User Inputs

# Main arguments
var = 'Tg'
file_freq = 'Monthly'
proc_type = 'Mean'
save_name_base = None#'NLAllSeasons19802020'

# Common data selection arguments
years = [1980, 2020]
lats = [50.7, 53.6]
lons = [3.25, 7.35]
# lats = ['Bilt', 'Eelde', 'Vlissingen', 'Maastricht', 'Kooy']
# lons = ['Bilt', 'Eelde', 'Vlissingen', 'Maastricht', 'Kooy']
proj_sel = 'RACMO2.4'
land_only = True  #!!!!!!!!
trim_border = None

# Trend time series plot arguments
plot_trends = True
data_sources_trend = ['Stations', 'Eobs', 'ERA5', 'ERA5L', 'RACMO2.3', 'RACMO2.4']
stations_trend = ['Bilt', 'Eelde', 'Vlissingen', 'Maastricht', 'Kooy']
months_dict = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
trend_layout = (2, 2)
trend_panel_width = 6
trend_panel_height = 4
trend_y_range = None
trend_shared_labels = True
trend_uncertainty_band = False
trend_mirror_y_axes = True

# Scatter plot arguments
plot_scatter = False
data_sources_scatter = ['Stations', 'ERA5']
data_compare_scatter = ['RACMO2.3', 'RACMO2.4']
stations_scatter = ['Bilt', 'Eelde', 'Vlissingen', 'Maastricht', 'Kooy']
months_scatter = None
scatter_freq = 'monthly'
scatter_layout = (1,2)
scatter_panel_size = 4
scatter_shared_axes = True
scatter_diff_mode = True

# Raw time series plot arguments
plot_raw_time = False
data_sources_raw = ['Stations', 'Eobs', 'ERA5', 'RACMO2.3', 'RACMO2.4']
stations_raw = ['Bilt', 'Eelde', 'Vlissingen', 'Maastricht', 'Kooy']
months_raw = None
raw_time_figsize = (12, 6)

# Zoomed spatial trend map arguments
plot_spatial_trends = False
data_sources_spatial = ['ERA5', 'RACMO2.3', 'RACMO2.4']
months_spatial = None
spatial_layout = (1,3)
spatial_panel_width = 5
spatial_panel_height = 5
spatial_crange = [-1, 1]
spatial_plot_lats = [50.2, 54.1]
spatial_plot_lons = [2.5, 8.1]
spatial_cmap_type = None
spatial_n_colors = 20
spatial_proj_plot = 'RACMO2.4'

# Fit arguments
fit_against_gmst = False
rolling_mean_var = False
rolling_mean_years = 3
min_periods = 1

#%% ============================================================================
#   CONFIGURATION
#   ============================================================================

data_sources = Constants.DATA_SOURCES
station_sources = Constants.STATION_SOURCES
station_coord_cfg = Constants.STATION_COORD_CFG
var_file_cfg = Constants.VAR_FILE_CFG
var_name_cfg = Constants.VAR_NAME_CFG
var_units_cfg = Constants.VAR_UNIT_CFG
var_symbol_cfg = Constants.VAR_SYMBOL_CFG
proj_cfg = Constants.PROJ_CFG
var_colors_cfg = Plotting.VAR_COLORS_CFG

# Fit configuration
if fit_against_gmst:
    fit_unit = '°C GMST'
    fit_scaling = 1
    fit_x_label = r'$\Delta$GMST (°C)'
else:
    fit_unit = 'decade'
    fit_scaling = 10
    fit_x_label = 'Year'

# Trend unit label
if var_units_cfg.get(var, '') == '':
    trend_unit = 'per ' + fit_unit
else:
    trend_unit = var_units_cfg[var] + ' / ' + fit_unit

# Variable labels
var_label = var_name_cfg.get(var, var) + ' (' + var_units_cfg.get(var, '') + ')'

# Projection
proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())
proj_plot = proj_cfg.get(spatial_proj_plot, ccrs.PlateCarree())

# Source labels for plots
SOURCE_LABELS = {
    'Eobs': 'E-OBS', 'Eobs_fine': 'E-OBS', 'Eobs_coarse': 'E-OBS',
    'ERA5': 'ERA5', 'ERA5L': 'ERA5L', 'ERA5_land': 'ERA5L', 'ERA5_coarse': 'ERA5',
    'RACMO2.3': 'R2.3', 'RACMO2.4': 'R2.4', 'RACMO2.4_KEXT06': 'R2.4',
    'Stations': 'Stations',
}

COLORS = ['#000000', '#DB2525', '#0168DE', '#00A236', '#CA721B', '#7B2CBF', '#E91E8C', '#808080']

# Detect if using station-based grid selection (sparse grid)
is_station_grid_selection = (
    isinstance(lats, (list, tuple)) and len(lats) > 0 and isinstance(lats[0], str)
)

#%% ============================================================================
#   HELPER FUNCTIONS
#   ============================================================================

def months_span_year_boundary(months):

    if months is None or len(months) == 0:
        return False
    months_arr = np.asarray(months, dtype=int)
    return int(months_arr[0]) > int(months_arr[-1])

def needs_extended_years(months_dict=None, months_lists=None):

    # Check months_dict values
    if months_dict:
        for months in months_dict.values():
            if months_span_year_boundary(months):
                return True
    # Check additional months lists (scatter, raw, spatial)
    if months_lists:
        for months in months_lists:
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
    
    # Filter by climate year range
    y0, y1 = years[0], years[-1]
    return data_season.where((data_season['clim_year'] >= y0) & 
                              (data_season['clim_year'] <= y1), drop=True)

def compute_seasonal_yearly(data, months, years, proc_type='Mean'):
    
    filtered = filter_by_season(data, months, years)    

    # Group by climate year to get seasonal yearly means
    if proc_type == 'Max':
        yearly = filtered.groupby('clim_year').max('time')
    elif proc_type == 'Min':
        yearly = filtered.groupby('clim_year').min('time')
    else:
        yearly = filtered.groupby('clim_year').mean('time')
    
    return yearly.astype('float32')

def compute_fit_data(yearly, fit_against_gmst, rolling_mean_var, rolling_mean_years, min_periods):

    if rolling_mean_var:
        yearly = yearly.rolling(clim_year=rolling_mean_years, center=True, 
                                min_periods=min_periods).mean()
    
    if fit_against_gmst:
        file_GMST = '/nobackup/users/walj/era5/era5_gmst_anom.nc'
        gmst = xr.open_dataset(file_GMST)['__xarray_dataarray_variable__']
        gmst = gmst.rolling(time=rolling_mean_years, center=True, 
                            min_periods=min_periods).mean()
        time_coord = pd.to_datetime(yearly['clim_year'].values.astype(int).astype(str))
        fit_coord = gmst.sel(time=time_coord).values.astype(float)
    else:
        fit_coord = yearly['clim_year'].values.astype(float)
    
    return yearly.rename({'clim_year': 'fit_against'}).assign_coords(
        fit_against=('fit_against', fit_coord)).astype('float32')

def compute_trend_stats(fit_data, fit_scaling):

    x_arr = fit_data['fit_against'].values
    y_arr = fit_data.values
    
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    
    # Check for insufficient data
    if len(x_clean) < 3:
        print(f'    Warning: Insufficient data points ({len(x_clean)}) for trend calculation')
        return None
    
    X = sm.add_constant(x_clean)

    # lags = np.ceil(len(x_clean)**(1/4)).astype(int)
    # model = sm.OLS(y_clean, X).fit(cov_type='HAC', cov_kwds={'maxlags':3})
    model = sm.OLS(y_clean, X).fit()
    
    return {
        'model': model,
        'x_clean': x_clean,
        'y_clean': y_clean,
        'slope': model.params[1],
        'intercept': model.params[0],
        'slope_trend': model.params[1]*fit_scaling,
        'slope_trend_std': model.bse[1]*fit_scaling,
    }

def get_source_label(src):
    return SOURCE_LABELS.get(src, src)

#%% ============================================================================
#   PHASE 1: LOAD ALL DATA ONCE
#   ============================================================================

print('=' * 60)
print('PHASE 1: Loading all data sources')
print('=' * 60)

# Collect all unique gridded sources and stations needed across all plot types
all_gridded_sources = set()
all_stations = set()

# From trend sources
if plot_trends and data_sources_trend:
    for src in data_sources_trend:
        if src == 'Stations':
            all_stations.update(stations_trend or [])
        else:
            all_gridded_sources.add(src)

# From scatter sources (both base and compare)
if plot_scatter:
    for src in (data_sources_scatter or []):
        if src == 'Stations':
            all_stations.update(stations_scatter or [])
        else:
            all_gridded_sources.add(src)
    for src in (data_compare_scatter or []):
        if src == 'Stations':
            all_stations.update(stations_scatter or [])
        else:
            all_gridded_sources.add(src)

# From raw time series sources
if plot_raw_time and data_sources_raw:
    for src in data_sources_raw:
        if src == 'Stations':
            all_stations.update(stations_raw or [])
        else:
            all_gridded_sources.add(src)

# From spatial sources
if plot_spatial_trends and data_sources_spatial:
    for src in data_sources_spatial:
        all_gridded_sources.add(src)

# Combine all sources to load (gridded + individual stations)
all_sources_to_load = list(all_gridded_sources) + list(all_stations)

# Extend years if needed for DJF-style seasons
years_load = list(years)
if needs_extended_years(months_dict=months_dict, 
                        months_lists=[months_scatter, months_raw, months_spatial]):
    years_load[0] = years[0] - 1

# Load all source data
source_data = {}
weights_cache = {}

for src in all_sources_to_load:
    print(f'  Loading: {src}')
    
    is_station = src in station_sources
    
    # Load raw data
    data_raw = load_var(
        var=var,
        data_source=src,
        data_sources=data_sources,
        station_sources=station_sources,
        file_freq=file_freq,
        var_file_cfg=var_file_cfg,
        proj_cfg=proj_cfg,
        months=None,
        years=years_load,
        lats=None if is_station else lats,
        lons=None if is_station else lons,
        land_only=False if is_station else land_only,
        trim_border=trim_border,
        rotpole_sel=proj_sel,
        station_coords=station_coord_cfg,
    )
    
    # Compute monthly aggregates
    if file_freq == 'Monthly' or is_monthly_time(data_raw['time']):
        data_monthly = data_raw
    else:
        if proc_type == 'Max':
            data_monthly = data_raw.resample(time='MS').max('time')
        elif proc_type == 'Min':
            data_monthly = data_raw.resample(time='MS').min('time')
        else:
            data_monthly = data_raw.resample(time='MS').mean('time')
    
    weights = area_weights(data_monthly.isel(time=0), 
                            rotpole_native=proj_cfg.get(src, ccrs.PlateCarree()))
    weights_cache[src] = weights.compute() if hasattr(weights, 'compute') else weights
    
    source_data[src] = {
        'raw': data_raw,
        'monthly': data_monthly.astype('float32'),
    }

print(f'Loaded {len(source_data)} data sources\n')

#%% ============================================================================
#   PHASE 2: HELPER TO COMPUTE RESULTS FOR A GIVEN MONTHS/SOURCES
#   ============================================================================

def compute_source_results(src, months, years, proc_type, source_data, weights_cache,
                           fit_against_gmst, rolling_mean_var, 
                           rolling_mean_years, min_periods):

    cached = source_data[src]
    
    # Filter to season/months
    monthly_filtered = filter_by_season(cached['monthly'], months, years)
    
    # Compute yearly aggregates
    yearly = compute_seasonal_yearly(cached['monthly'], months, years, proc_type)
    
    weights = weights_cache.get(src)
    monthly_mean = area_weighted_mean(monthly_filtered, weights=weights)
    yearly_mean = area_weighted_mean(yearly, weights=weights)
    avg_val = area_weighted_mean(cached['monthly'].mean('time'), weights=weights)
    
    # Compute fit data with GMST if requested
    fit_data = compute_fit_data(yearly_mean, fit_against_gmst, 
                               rolling_mean_var, rolling_mean_years, min_periods)
    
    return {
        'monthly': monthly_mean.compute() if hasattr(monthly_mean, 'compute') else monthly_mean,
        'yearly': yearly_mean.compute() if hasattr(yearly_mean, 'compute') else yearly_mean,
        'fit': fit_data.compute() if hasattr(fit_data, 'compute') else fit_data,
        'avg': avg_val.compute() if hasattr(avg_val, 'compute') else avg_val,
    }

def compute_stations_aggregate(station_list, results_dict):
    valid_stations = [k for k in station_list if k in results_dict]
    if not valid_stations:
        return None
    return {
        'monthly': xr.concat([results_dict[k]['monthly'] for k in valid_stations], dim='station').mean('station'),
        'yearly': xr.concat([results_dict[k]['yearly'] for k in valid_stations], dim='station').mean('station'),
        'fit': xr.concat([results_dict[k]['fit'] for k in valid_stations], dim='station').mean('station'),
        'avg': xr.concat([results_dict[k]['avg'] for k in valid_stations], dim='station').mean('station'),
    }

def get_expanded_sources(source_list, station_list):
    expanded = []
    has_stations = False
    for src in (source_list or []):
        if src == 'Stations':
            has_stations = True
            expanded.extend(station_list or [])
        else:
            expanded.append(src)
    return expanded, has_stations

print('=' * 60)
print('PHASE 2: Computing area-weighted time series')
print('=' * 60)

# ============================================================================
# TREND RESULTS (uses months_dict for seasonal breakdown)
# ============================================================================
trend_results = {}
if plot_trends and data_sources_trend:
    print('  Computing trend results (seasonal)...')
    
    # Get expanded sources for trends
    trend_sources_expanded, trend_has_stations = get_expanded_sources(
        data_sources_trend, stations_trend)
    
    for month_key, months in months_dict.items():
        results = {}
        for src in trend_sources_expanded:
            results[src] = compute_source_results(
                src, months, years, proc_type, source_data, weights_cache,
                fit_against_gmst, rolling_mean_var,
                rolling_mean_years, min_periods)
        
        # Aggregate stations if 'Stations' was in original list
        if trend_has_stations and stations_trend:
            stations_agg = compute_stations_aggregate(stations_trend, results)
            results['Stations'] = stations_agg
        
        trend_results[month_key] = results

# ============================================================================
# SCATTER RESULTS (uses months_scatter for single period)
# ============================================================================
scatter_results = {}
if plot_scatter and (data_sources_scatter or data_compare_scatter):
    print('  Computing scatter results...')
    
    # Combine base and compare sources
    scatter_all_sources = list(set((data_sources_scatter or []) + (data_compare_scatter or [])))
    scatter_sources_expanded, scatter_has_stations = get_expanded_sources(
        scatter_all_sources, stations_scatter)
    
    # Use months_scatter (defaults to all months if None)
    months_for_scatter = months_scatter if months_scatter else list(range(1, 13))
    
    results = {}
    for src in scatter_sources_expanded:
        results[src] = compute_source_results(
            src, months_for_scatter, years, proc_type, source_data, weights_cache,
            fit_against_gmst, rolling_mean_var,
            rolling_mean_years, min_periods)
    
    # Aggregate stations if 'Stations' was in original list
    if scatter_has_stations and stations_scatter:
        stations_agg = compute_stations_aggregate(stations_scatter, results)
        results['Stations'] = stations_agg
    
    scatter_results = results

# ============================================================================
# RAW TIME SERIES RESULTS (uses months_raw for single period)
# ============================================================================
raw_results = {}
if plot_raw_time and data_sources_raw:
    print('  Computing raw time series results...')
    
    raw_sources_expanded, raw_has_stations = get_expanded_sources(
        data_sources_raw, stations_raw)
    
    # Use months_raw (defaults to all months if None)
    months_for_raw = months_raw if months_raw else list(range(1, 13))
    
    results = {}
    for src in raw_sources_expanded:
        results[src] = compute_source_results(
            src, months_for_raw, years, proc_type, source_data, weights_cache,
            fit_against_gmst, rolling_mean_var,
            rolling_mean_years, min_periods)
    
    # Aggregate stations if 'Stations' was in original list
    if raw_has_stations and stations_raw:
        stations_agg = compute_stations_aggregate(stations_raw, results)
        results['Stations'] = stations_agg
    
    raw_results = results

print('Processing complete\n')

#%% ============================================================================
#   PHASE 3: COMPUTE TREND STATISTICS
#   ============================================================================

print('=' * 60)
print('PHASE 3: Computing trend statistics')
print('=' * 60)

all_trend_stats = {}

if plot_trends and trend_results:
    for month_key in months_dict.keys():
        trend_stats = {}
        for src in data_sources_trend:
            fit_data = trend_results[month_key][src]['fit']
            stats = compute_trend_stats(fit_data, fit_scaling)
            trend_stats[src] = stats
        all_trend_stats[month_key] = trend_stats

print('Trend statistics complete\n')

#%% ============================================================================
#   PLOTTING HELPER FUNCTIONS
#   ============================================================================

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
    pdf_dir = Path(f'/nobackup/users/walj/Figures/pdf/{var}')
    jpg_dir = Path(f'/nobackup/users/walj/Figures/jpg/{var}')
    pdf_dir.mkdir(parents=True, exist_ok=True)
    jpg_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(str(pdf_dir / f'{save_name}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(str(jpg_dir / f'{save_name}.jpg'), format='jpg', dpi=300, bbox_inches='tight')

def get_scatter_config(scatter_freq):
    configs = {
        'yearly': {'data_key': 'fit', 'size': 40, 'alpha': 1.0},
        'monthly': {'data_key': 'monthly', 'size': 30, 'alpha': 0.8},
    }
    return configs.get(scatter_freq, {'data_key': 'raw', 'size': 12, 'alpha': 0.4})

def make_axis_label(source, var, var_name_cfg, var_units_cfg, prefix=''):
    src_label = get_source_label(source)
    var_name_str = var_name_cfg.get(var, var)
    unit_str = var_units_cfg.get(var, '')
    
    label = f'{prefix}{src_label} {var_name_str}' if src_label else f'{prefix}{var_name_str}'
    return f'{label} ({unit_str})' if unit_str else label

def make_diff_label(var, var_name_cfg, var_units_cfg):
    var_name_str = var_name_cfg.get(var, var)
    unit_str = var_units_cfg.get(var, '')
    return fr'$\Delta${var_name_str} ({unit_str})' if unit_str else fr'$\Delta${var_name_str}'

#%% ============================================================================
#   PLOT 1: TREND TIME SERIES (SEASONAL)
#   ============================================================================

if plot_trends:
    print('Plotting trend time series...')
    
    month_keys = list(months_dict.keys())
    n_panels = len(month_keys)
    n_rows, n_cols = trend_layout
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(trend_panel_width * n_cols, trend_panel_height * n_rows),
        sharex=True, sharey=False
    )
    wspace = 0.02 if trend_mirror_y_axes else 0.16
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.95, wspace=wspace, hspace=0.18)
    axes = normalize_axes(axes, n_rows, n_cols)
    
    title_fs = max(18, int(trend_panel_height * 5))
    tick_fs = max(12, int(trend_panel_height * 3))
    legend_fs = max(10, int(trend_panel_height * 2.5))
    
    for idx, month_key in enumerate(month_keys):
        ax = axes[idx // n_cols, idx % n_cols]
        trend_stats = all_trend_stats.get(month_key, {})
        
        for ii, src in enumerate(data_sources_trend):
            if src not in trend_stats:
                continue
            
            stats = trend_stats[src]
            color = COLORS[ii % len(COLORS)]
            label = f'{get_source_label(src)} ({stats["slope_trend"]:.2f} ± {stats["slope_trend_std"]:.2f} {trend_unit})'
            
            order = np.argsort(stats['x_clean'])
            x_sorted, y_sorted = stats['x_clean'][order], stats['y_clean'][order]
            
            # Data points and trend line
            ax.plot(x_sorted, y_sorted, c=color, lw=2, ms=6, marker='o', ls='--')
            
            X_sorted = sm.add_constant(x_sorted)
            frame = stats['model'].get_prediction(X_sorted).summary_frame(alpha=0.05)
            ax.plot(x_sorted, frame['mean'].values, c=color, lw=2.5, label=label, zorder=15)
            
            if trend_uncertainty_band:
                ax.fill_between(x_sorted, frame['mean_ci_lower'].values,
                               frame['mean_ci_upper'].values, color=color, alpha=0.15)
        
        ax.grid(True, alpha=0.3)
        ax.set_title(month_key, fontsize=title_fs, fontweight='bold')
        ax.tick_params(axis='both', labelsize=tick_fs)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        
        # Mirror y-axis to right side for right column
        is_right_col = (idx % n_cols == n_cols - 1) and n_cols > 1
        if trend_mirror_y_axes and is_right_col:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
        
        # Y-range
        y_range = trend_y_range.get(month_key) if isinstance(trend_y_range, dict) else trend_y_range
        if y_range is not None:
            ax.set_ylim(*y_range)
        
        leg = ax.legend(fontsize=legend_fs, handlelength=1.5, loc='best')
        leg.set_zorder(20)
        for line in leg.get_lines():
            line.set_linewidth(3.0)
    
    hide_unused_axes(axes, n_panels, n_rows, n_cols)
    
    if trend_shared_labels:
        add_shared_axis_labels(fig, axes, fit_x_label, var_label, 
                               fontsize=max(28, int(trend_panel_height * 7)))
    
    save_figure(save_name_base, var, 'trend')
    plt.show()

#%% ============================================================================
#   PLOT 2: SCATTER PLOTS (SINGLE PERIOD - NO SEASONAL SEPARATION)
#   ============================================================================

if plot_scatter and data_sources_scatter and data_compare_scatter and scatter_results:
    print('Plotting scatter plots...')
    
    n_panels = len(data_sources_scatter)  # One panel per base-compare pair
    n_rows, n_cols = scatter_layout
    
    # Get scatter configuration
    scatter_cfg = get_scatter_config(scatter_freq)
    
    # Collect data for all panels
    panel_data = []
    all_x, all_y = [], []
    
    for base_src, comp_src in zip(data_sources_scatter, data_compare_scatter):
        if base_src not in scatter_results or comp_src not in scatter_results:
            continue
        
        x_data = scatter_results[base_src][scatter_cfg['data_key']]
        y_data = scatter_results[comp_src][scatter_cfg['data_key']]
        
        x = np.asarray(x_data.values if hasattr(x_data, 'values') else x_data)
        y = np.asarray(y_data.values if hasattr(y_data, 'values') else y_data)
        
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        
        all_x.extend(x)
        all_y.extend(y if not scatter_diff_mode else y - x)
        panel_data.append({'base': base_src, 'comp': comp_src, 'x': x, 'y': y})
    
    if not panel_data:
        print('  No valid scatter data found.')
    else:
        # Check if all panels have same base source
        same_base = len(set(p['base'] for p in panel_data)) == 1
        share_x = scatter_shared_axes and same_base and len(panel_data) > 1
        share_y = scatter_shared_axes and len(panel_data) > 1
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(scatter_panel_size * n_cols, scatter_panel_size * n_rows),
            sharex=share_x, sharey=share_y
        )
        
        wspace = 0.03 if (share_x and share_y) else (0.12 if (share_x or share_y) else 0.25)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, 
                           top=0.92, wspace=wspace, hspace=0.16 if n_rows > 1 else 0.08)
        axes = normalize_axes(axes, n_rows, n_cols)
        
        # Compute shared limits
        if scatter_shared_axes and all_x:
            if scatter_diff_mode:
                xlim = (np.nanmin(all_x), np.nanmax(all_x))
                ylim = (np.nanmin(all_y), np.nanmax(all_y))
            else:
                lim = (min(np.nanmin(all_x), np.nanmin(all_y)), max(np.nanmax(all_x), np.nanmax(all_y)))
                xlim = ylim = lim
            pad_x = 0.02 * (xlim[1] - xlim[0])
            pad_y = 0.02 * (ylim[1] - ylim[0])
            xlim = (xlim[0] - pad_x, xlim[1] + pad_x)
            ylim = (ylim[0] - pad_y, ylim[1] + pad_y)
        
        # Plot each panel
        for idx, pdata in enumerate(panel_data):
            if idx >= n_rows * n_cols:
                break
            
            ax = axes[idx // n_cols, idx % n_cols]
            x, y = pdata['x'], pdata['y']
            y_plot = (y - x) if scatter_diff_mode else y
            
            ax.scatter(x, y_plot, s=scatter_cfg['size'], c='k', alpha=scatter_cfg['alpha'])
            
            # Reference line
            if scatter_diff_mode:
                ax.axhline(0, lw=1.5, color='xkcd:brick red', ls='--', zorder=0)
            else:
                ax.plot([0, 1], [0, 1], transform=ax.transAxes, lw=1.5, 
                       color='xkcd:brick red', ls='--', zorder=0)
                ax.set_aspect('equal', adjustable='box')
            ax.set_box_aspect(1)
            
            # Limits
            if scatter_shared_axes:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
            else:
                local_pad = 0.02 * (np.nanmax(x) - np.nanmin(x))
                if scatter_diff_mode:
                    ax.set_xlim(np.nanmin(x) - local_pad, np.nanmax(x) + local_pad)
                    local_pad_y = 0.02 * (np.nanmax(y_plot) - np.nanmin(y_plot))
                    ax.set_ylim(np.nanmin(y_plot) - local_pad_y, np.nanmax(y_plot) + local_pad_y)
                else:
                    lim_local = (min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y)))
                    ax.set_xlim(lim_local[0] - local_pad, lim_local[1] + local_pad)
                    ax.set_ylim(lim_local[0] - local_pad, lim_local[1] + local_pad)
            
            # Title
            base_lbl, comp_lbl = get_source_label(pdata['base']), get_source_label(pdata['comp'])
            title = f'{comp_lbl} - {base_lbl}' if scatter_diff_mode else comp_lbl
            ax.set_title(title, fontsize=20, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=12, length=6)
            
            # Individual x-axis label for each panel (different base sources)
            x_label = make_axis_label(pdata['base'], var, var_name_cfg, var_units_cfg)
            ax.set_xlabel(x_label, fontsize=14)
        
        hide_unused_axes(axes, len(panel_data), n_rows, n_cols)
        
        # Shared y-axis label only
        if share_y:
            y_label = make_diff_label(var, var_name_cfg, var_units_cfg) if scatter_diff_mode else var_label
            fig.canvas.draw()
            y_center = 0.5 * (axes[0, 0].get_position().y1 + axes[-1, 0].get_position().y0)
            x_left = axes[0, 0].get_position().x0 - 0.05
            fig.text(x_left, y_center, y_label, ha='right', va='center', rotation=90, fontsize=18)
        
        suffix = 'scatter_diff' if scatter_diff_mode else 'scatter'
        save_figure(save_name_base, var, suffix)
        plt.show()

#%% ============================================================================
#   PLOT 3: RAW TIME SERIES
#   ============================================================================

if plot_raw_time and data_sources_raw and raw_results:
    print('Plotting raw time series...')
    
    fig, ax = plt.subplots(1, figsize=raw_time_figsize)
    
    for ii, src in enumerate(data_sources_raw):
        if src not in raw_results:
            continue
        
        data = raw_results[src]['monthly']
        if 'time' in data.dims:
            ax.plot(data['time'].values, data.values, c=COLORS[ii % len(COLORS)], 
                   lw=2, marker='o', ms=4, ls='-', alpha=0.8, label=get_source_label(src))
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel(var_label, fontsize=20)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    leg = ax.legend(fontsize=14, loc='best')
    for h in leg.legend_handles:
        h.set_linestyle('-')
        h.set_marker('')
        h.set_linewidth(3.0)
    
    save_figure(save_name_base, var, 'rawtime')
    plt.show()

#%% ============================================================================
#   PLOT 4: SPATIAL TREND MAPS (ZOOMED REGION)
#   ============================================================================

if plot_spatial_trends and data_sources_spatial:
    print('Plotting spatial trend maps...')
    
    # Setup colormap
    if not var_colors_cfg.get(var):
        var_colors_cfg[var] = var_colors_cfg['Default'].copy()
    trend_cmap = convert_cmap(var_colors_cfg[var].get('cmap_trend', var_colors_cfg['Default']['cmap_trend']),
                             spatial_cmap_type, spatial_n_colors)
    trend_extreme = var_colors_cfg[var].get('extreme_trend', var_colors_cfg['Default']['extreme_trend'])
    
    months_for_spatial = months_spatial if months_spatial else list(range(1, 13))
    n_rows, n_cols = spatial_layout
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(spatial_panel_width * n_cols, spatial_panel_height * n_rows),
        constrained_layout=True,
        subplot_kw={'projection': proj_plot},
        sharex=True, sharey=True
    )
    axes = normalize_axes(axes, n_rows, n_cols)
    
    meshes = []
    trend_fields = []
    
    for idx, spatial_src in enumerate(data_sources_spatial):
        if idx >= n_rows * n_cols or spatial_src not in source_data:
            continue
        
        ax = axes[idx // n_cols, idx % n_cols]
        monthly_data = source_data[spatial_src]['monthly']
        is_station_data = 'station' in monthly_data.dims
        
        # Compute spatial trend
        yearly_spatial = compute_seasonal_yearly(monthly_data, months_for_spatial, years, proc_type)
        fit_spatial = compute_fit_data(yearly_spatial, fit_against_gmst,
                                       rolling_mean_var, rolling_mean_years, min_periods)
        
        fits = fit_spatial.polyfit(dim='fit_against', deg=1, skipna=True)
        trend_field = (fits.polyfit_coefficients.sel(degree=1) * fit_scaling).astype('float32').compute()
        trend_fields.append(trend_field)
        
        show_x = (idx // n_cols == n_rows - 1)
        show_y = (idx % n_cols == 0)
        
        # Get coordinates for plotting
        if is_station_data:
            # Load reference grid for station-based data
            ref_data = load_var(
                var=var, data_source=spatial_src, data_sources=data_sources,
                station_sources=station_sources, file_freq=file_freq,
                var_file_cfg=var_file_cfg, proj_cfg=proj_cfg,
                months=None, years=years, lats=spatial_plot_lats, lons=spatial_plot_lons,
                land_only=False, trim_border=None, rotpole_sel=proj_sel, station_coords=None,
            )
            
            if 'lon' in ref_data.coords:
                lons_plot = ref_data['lon'].values
                lats_plot = ref_data['lat'].values
            elif 'longitude' in ref_data.coords:
                lons_plot = ref_data['longitude'].values
                lats_plot = ref_data['latitude'].values
            else:
                coord_names = list(ref_data.coords)
                lons_plot = ref_data.coords[coord_names[-1]].values
                lats_plot = ref_data.coords[coord_names[-2]].values
            
            # Create 2D grid with station values
            station_lons = monthly_data['station_lon'].values
            station_lats = monthly_data['station_lat'].values
            
            if lons_plot.ndim == 1:
                trend_2d = np.full((len(lats_plot), len(lons_plot)), np.nan, dtype='float32')
                for slon, slat, val in zip(station_lons, station_lats, trend_field.values):
                    trend_2d[np.argmin(np.abs(lats_plot - slat)), np.argmin(np.abs(lons_plot - slon))] = val
            else:
                trend_2d = np.full(lons_plot.shape, np.nan, dtype='float32')
                for slon, slat, val in zip(station_lons, station_lats, trend_field.values):
                    min_idx = np.unravel_index(np.argmin((lons_plot - slon)**2 + (lats_plot - slat)**2), lons_plot.shape)
                    trend_2d[min_idx] = val
            
            trend_field = xr.DataArray(trend_2d, dims=['y', 'x'])
        else:
            # Regular grid
            if 'lon' in monthly_data.coords:
                trend_field = trend_field.assign_coords(latitude=monthly_data['lat'], longitude=monthly_data['lon'])
            elif 'latitude' in monthly_data.coords:
                trend_field = trend_field.assign_coords(latitude=monthly_data['latitude'], longitude=monthly_data['longitude'])
            
            if 'longitude' in trend_field.coords:
                lons_plot = trend_field['longitude'].values
                lats_plot = trend_field['latitude'].values
            else:
                coord_names = list(trend_field.coords)
                lons_plot = trend_field.coords[coord_names[-1]].values
                lats_plot = trend_field.coords[coord_names[-2]].values
        
        mesh, _ = plot_map(
            fig, ax, trend_field, lons_plot, lats_plot,
            crange=spatial_crange, cmap=trend_cmap, extreme_colors=trend_extreme,
            show_x_ticks=True, show_y_ticks=True, show_x_labels=show_x, show_y_labels=show_y,
            x_ticks=2, y_ticks=1, x_ticks_num=False, y_ticks_num=False, tick_size=12,
            extent=[*spatial_plot_lons, *spatial_plot_lats], proj=proj_plot,
            add_colorbar=False, title=get_source_label(spatial_src), title_size=18, show_plot=False,
            lats_area=lats if not is_station_data and isinstance(lats[0], float) else None,
            lons_area=lons if not is_station_data and isinstance(lons[0], float) else None,
            proj_area=proj_plot,
        )
        meshes.append(mesh)
    
    hide_unused_axes(axes, len(data_sources_spatial), n_rows, n_cols)
    
    if meshes:
        shared_colorbar(
            fig=fig, axes=axes.ravel()[:len(meshes)], mesh=meshes[0],
            datasets=trend_fields, crange=spatial_crange,
            label=f'{var_name_cfg.get(var, var)} Trend ({trend_unit})',
            orientation='horizontal', c_ticks=5, c_ticks_num=True,
            tick_labelsize=14, labelsize=18, pad=0.06, thickness=0.03, extend_into_axes=True
        )
    
    save_figure(save_name_base, var, 'spatialtrend')
    plt.show()

print('=' * 60)
print('All plots complete!')
print('=' * 60)

#%%