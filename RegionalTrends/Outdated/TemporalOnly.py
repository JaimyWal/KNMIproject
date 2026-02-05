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

#%% ============================================================================
#   USER INPUTS
#   ============================================================================

# Main arguments
var = 'Tg'
file_freq = 'Monthly'
proc_type = 'Mean'
save_name_base = None # 'NLAllSeasons19802020'

# Common data selection arguments
years = [1980, 2020]
lats = [50.7, 53.6]
lons = [3.25, 7.35]
proj_sel = 'RACMO2.4'
land_only = True
trim_border = None

# Trend time series plot arguments (uses months_dict for seasonal breakdown)
plot_trends = True
data_sources_trend = ['Stations', 'Eobs', 'ERA5', 'RACMO2.3', 'RACMO2.4']  # 'Stations' uses stations_trend
stations_trend = ['Bilt', 'Eelde', 'Vlissingen', 'Maastricht', 'Kooy']
months_dict = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
trend_layout = (2, 2)
trend_panel_width = 6
trend_panel_height = 4
trend_y_range = None  # Can be dict by month_key or single [lo, hi]
trend_shared_labels = True
trend_uncertainty_band = False

# Scatter plot arguments (uses months for single period)
plot_scatter = True
data_sources_scatter = ['Stations', 'ERA5']  # Base sources (x-axis)
data_compare_scatter = ['RACMO2.3', 'RACMO2.4']  # Comparison sources (y-axis)
stations_scatter = ['Bilt', 'Eelde', 'Vlissingen', 'Maastricht', 'Kooy']
months_scatter = [6, 7, 8]  # Regular months (not dict)
scatter_layout = (1, 2)
scatter_panel_size = 4
scatter_shared_axes = True

# Raw time series plot arguments (uses months for single period)
plot_raw_time = True
data_sources_raw = ['Stations', 'Eobs', 'ERA5', 'RACMO2.3', 'RACMO2.4']
stations_raw = ['Bilt', 'Eelde', 'Vlissingen', 'Maastricht', 'Kooy']
months_raw = None  # None for all months, or list like [6, 7, 8]
raw_time_figsize = (12, 6)

# Zoomed spatial trend map arguments (uses months_spatial for single period)
plot_spatial_trends = True
data_sources_spatial = ['Eobs', 'ERA5', 'RACMO2.3', 'RACMO2.4']  # List of gridded sources for spatial maps
months_spatial = [6, 7, 8]  # Regular months (not dict)
spatial_layout = (2, 2)
spatial_panel_width = 5
spatial_panel_height = 5
spatial_crange = [-0.5, 0.5]
spatial_plot_lats = [50.2, 54.1]
spatial_plot_lons = [2.5, 8.1]
spatial_cmap_type = None
spatial_n_colors = 20

# Fit arguments
fit_against_gmst = False
rolling_mean_var = False
rolling_mean_years = 3
min_periods = 1
relative_precip = False

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
if var == 'P' and relative_precip:
    trend_unit = '% / ' + fit_unit
elif var_units_cfg.get(var, '') == '':
    trend_unit = 'per ' + fit_unit
else:
    trend_unit = var_units_cfg[var] + ' / ' + fit_unit

# Variable labels
var_label = var_name_cfg.get(var, var) + ' (' + var_units_cfg.get(var, '') + ')'

# Projection
proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())

# Source labels for plots
SOURCE_LABELS = {
    'Eobs': 'E-OBS', 'Eobs_fine': 'E-OBS', 'Eobs_coarse': 'E-OBS',
    'ERA5': 'ERA5', 'ERA5_fine': 'ERA5', 'ERA5_coarse': 'ERA5',
    'RACMO2.3': 'R2.3', 'RACMO2.4': 'R2.4', 'RACMO2.4_KEXT06': 'R2.4',
    'Stations': 'Stations',
}

COLORS = ['#000000', '#DB2525', '#0168DE', '#00A236', '#CA721B', '#7B2CBF', '#E91E8C', '#808080']

#%% ============================================================================
#   HELPER FUNCTIONS
#   ============================================================================

def needs_extended_years(months_dict):
    """Check if any season spans year boundary (e.g., DJF)."""
    for months in months_dict.values():
        if int(months[0]) > int(months[-1]):
            return True
    return False

def assign_clim_year(data, months):
    """Assign climate year coordinate based on season definition."""
    months_arr = np.asarray(months, dtype=int)
    spans_boundary = int(months_arr[0]) > int(months_arr[-1])
    
    month_vals = data['time'].dt.month
    year_vals = data['time'].dt.year
    
    if spans_boundary:
        clim_year = xr.where(month_vals >= months_arr[0], year_vals + 1, year_vals)
    else:
        clim_year = year_vals
    
    return data.assign_coords(clim_year=('time', clim_year.values))

def filter_by_season(data, months, years):
    """Filter data by season and climate year range."""
    if 'time' not in data.dims:
        return data
    
    months_arr = np.asarray(months, dtype=int)
    in_season = data['time'].dt.month.isin(months_arr)
    data_season = data.where(in_season, drop=True)
    data_season = assign_clim_year(data_season, months)
    
    y0, y1 = years[0], years[-1]
    return data_season.where(
        (data_season['clim_year'] >= y0) & (data_season['clim_year'] <= y1), 
        drop=True
    )

def compute_seasonal_yearly(data, months, years, proc_type='Mean'):
    """Compute yearly seasonal aggregates."""
    filtered = filter_by_season(data, months, years)
    
    if proc_type == 'Max':
        yearly = filtered.groupby('clim_year').max('time')
    elif proc_type == 'Min':
        yearly = filtered.groupby('clim_year').min('time')
    else:
        yearly = filtered.groupby('clim_year').mean('time')
    
    return yearly.astype('float32')

def compute_fit_data(yearly, fit_against_gmst, rolling_mean_var, rolling_mean_years, min_periods):
    """Prepare data for trend fitting with optional GMST coordinate."""
    if rolling_mean_var:
        yearly = yearly.rolling(clim_year=rolling_mean_years, center=True, 
                                min_periods=min_periods).mean()
    
    if fit_against_gmst:
        file_GMST = '/nobackup/users/walj/era5/era5_gmst_anom.nc'
        gmst = xr.open_dataset(file_GMST)['__xarray_dataarray_variable__']
        if rolling_mean_var:
            gmst = gmst.rolling(time=rolling_mean_years, center=True, 
                               min_periods=min_periods).mean()
        time_coord = pd.to_datetime(yearly['clim_year'].values.astype(int).astype(str))
        fit_coord = gmst.sel(time=time_coord).values.astype(float)
    else:
        fit_coord = yearly['clim_year'].values.astype(float)
    
    return yearly.rename({'clim_year': 'fit_against'}).assign_coords(
        fit_against=('fit_against', fit_coord)).astype('float32')

def compute_trend_stats(fit_data, fit_scaling):
    """Compute trend statistics using OLS regression."""
    x_arr = fit_data['fit_against'].values
    y_arr = fit_data.values
    
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    
    if len(x_clean) < 3:
        return None
    
    X = sm.add_constant(x_clean)
    model = sm.OLS(y_clean, X).fit()
    
    return {
        'model': model,
        'x_clean': x_clean,
        'y_clean': y_clean,
        'slope': model.params[1],
        'intercept': model.params[0],
        'slope_trend': model.params[1] * fit_scaling,
        'slope_trend_std': model.bse[1] * fit_scaling,
    }

def get_source_label(src):
    """Get display label for a data source."""
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
if needs_extended_years(months_dict):
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
    
    # Compute weights for gridded data
    if not is_station:
        weights = area_weights(data_monthly.isel(time=0), 
                              rotpole_native=proj_cfg.get(src, ccrs.PlateCarree()))
        if weights is not None:
            weights_cache[src] = weights.compute() if hasattr(weights, 'compute') else weights
    
    source_data[src] = {
        'raw': data_raw,
        'monthly': data_monthly.astype('float32'),
        'is_station': is_station,
    }

print(f'Loaded {len(source_data)} data sources\n')

#%% ============================================================================
#   PHASE 2: HELPER TO COMPUTE RESULTS FOR A GIVEN MONTHS/SOURCES
#   ============================================================================

def compute_source_results(src, months, years, proc_type, source_data, weights_cache,
                           station_sources, fit_against_gmst, rolling_mean_var, 
                           rolling_mean_years, min_periods, var, relative_precip):
    """Compute area-weighted time series results for a single source."""
    cached = source_data[src]
    is_station = cached['is_station']
    
    # Filter to season/months
    monthly_filtered = filter_by_season(cached['monthly'], months, years)
    
    # Compute yearly aggregates
    yearly = compute_seasonal_yearly(cached['monthly'], months, years, proc_type)
    
    # Area-weighted mean for gridded data, direct for stations
    if is_station:
        monthly_mean = monthly_filtered
        yearly_mean = yearly
        avg_val = cached['monthly'].mean('time')
    else:
        weights = weights_cache.get(src)
        monthly_mean = area_weighted_mean(monthly_filtered, weights=weights)
        yearly_mean = area_weighted_mean(yearly, weights=weights)
        avg_val = area_weighted_mean(cached['monthly'].mean('time'), weights=weights)
    
    # Compute fit data with GMST if requested
    fit_data = compute_fit_data(yearly_mean, fit_against_gmst, 
                               rolling_mean_var, rolling_mean_years, min_periods)
    
    # Apply relative precip if needed
    if var == 'P' and relative_precip:
        monthly_mean = 100 * monthly_mean / avg_val
        fit_data = 100 * fit_data / avg_val
    
    return {
        'monthly': monthly_mean.compute() if hasattr(monthly_mean, 'compute') else monthly_mean,
        'yearly': yearly_mean.compute() if hasattr(yearly_mean, 'compute') else yearly_mean,
        'fit': fit_data.compute() if hasattr(fit_data, 'compute') else fit_data,
        'avg': avg_val.compute() if hasattr(avg_val, 'compute') else avg_val,
    }

def compute_stations_aggregate(station_list, results_dict):
    """Aggregate results from multiple stations into 'Stations' entry."""
    valid_stations = [k for k in station_list if k in results_dict]
    if not valid_stations:
        return None
    return {
        'monthly': xr.concat([results_dict[k]['monthly'] for k in valid_stations], dim='station').mean('station'),
        'yearly': xr.concat([results_dict[k]['yearly'] for k in valid_stations], dim='station').mean('station'),
        'fit': xr.concat([results_dict[k]['fit'] for k in valid_stations], dim='station').mean('station'),
        'avg': xr.concat([results_dict[k]['avg'] for k in valid_stations], dim='station').mean('station'),
    }

def get_expanded_sources(source_list, station_list, station_sources):
    """Expand 'Stations' in source_list to actual station names, return (expanded, has_stations)."""
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
        data_sources_trend, stations_trend, station_sources)
    
    for month_key, months in months_dict.items():
        results = {}
        for src in trend_sources_expanded:
            if src in source_data:
                results[src] = compute_source_results(
                    src, months, years, proc_type, source_data, weights_cache,
                    station_sources, fit_against_gmst, rolling_mean_var,
                    rolling_mean_years, min_periods, var, relative_precip)
        
        # Aggregate stations if 'Stations' was in original list
        if trend_has_stations and stations_trend:
            stations_agg = compute_stations_aggregate(stations_trend, results)
            if stations_agg:
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
        scatter_all_sources, stations_scatter, station_sources)
    
    # Use months_scatter (defaults to all months if None)
    months_for_scatter = months_scatter if months_scatter else list(range(1, 13))
    
    results = {}
    for src in scatter_sources_expanded:
        if src in source_data:
            results[src] = compute_source_results(
                src, months_for_scatter, years, proc_type, source_data, weights_cache,
                station_sources, fit_against_gmst, rolling_mean_var,
                rolling_mean_years, min_periods, var, relative_precip)
    
    # Aggregate stations if 'Stations' was in original list
    if scatter_has_stations and stations_scatter:
        stations_agg = compute_stations_aggregate(stations_scatter, results)
        if stations_agg:
            results['Stations'] = stations_agg
    
    scatter_results = results

# ============================================================================
# RAW TIME SERIES RESULTS (uses months_raw for single period)
# ============================================================================
raw_results = {}
if plot_raw_time and data_sources_raw:
    print('  Computing raw time series results...')
    
    raw_sources_expanded, raw_has_stations = get_expanded_sources(
        data_sources_raw, stations_raw, station_sources)
    
    # Use months_raw (defaults to all months if None)
    months_for_raw = months_raw if months_raw else list(range(1, 13))
    
    results = {}
    for src in raw_sources_expanded:
        if src in source_data:
            results[src] = compute_source_results(
                src, months_for_raw, years, proc_type, source_data, weights_cache,
                station_sources, fit_against_gmst, rolling_mean_var,
                rolling_mean_years, min_periods, var, relative_precip)
    
    # Aggregate stations if 'Stations' was in original list
    if raw_has_stations and stations_raw:
        stations_agg = compute_stations_aggregate(stations_raw, results)
        if stations_agg:
            results['Stations'] = stations_agg
    
    raw_results = results

print('Processing complete\n')

#%% ============================================================================
#   PHASE 3: COMPUTE TREND STATISTICS (for trend plots only)
#   ============================================================================

print('=' * 60)
print('PHASE 3: Computing trend statistics')
print('=' * 60)

all_trend_stats = {}

if plot_trends and trend_results:
    for month_key in months_dict.keys():
        trend_stats = {}
        
        for src in data_sources_trend:
            if src in trend_results[month_key]:
                fit_data = trend_results[month_key][src]['fit']
                stats = compute_trend_stats(fit_data, fit_scaling)
                if stats is not None:
                    trend_stats[src] = stats
        
        all_trend_stats[month_key] = trend_stats

print('Trend statistics complete\n')

#%% Temporal plotting for area

#%% ============================================================================
#   PLOT 1: TREND TIME SERIES
#   ============================================================================

if plot_trends:
    print('Plotting trend time series...')
    
    month_keys = list(months_dict.keys())
    n_months = len(month_keys)
    n_rows, n_cols = trend_layout
    
    fig_width = trend_panel_width * n_cols
    fig_height = trend_panel_height * n_rows
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        sharex=True,
        sharey=False
    )
    
    # Normalize axes array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    y_range_is_dict = isinstance(trend_y_range, dict)
    
    for idx, month_key in enumerate(month_keys):
        row_idx, col_idx = idx // n_cols, idx % n_cols
        ax = axes[row_idx, col_idx]
        
        trend_stats = all_trend_stats.get(month_key, {})
        
        for ii, src in enumerate(data_sources_trend):
            if src not in trend_stats:
                continue
            
            stats = trend_stats[src]
            color = COLORS[ii % len(COLORS)]
            label_src = get_source_label(src)
            
            label = (f'{label_src} ({stats["slope_trend"]:.2f} ± '
                    f'{stats["slope_trend_std"]:.2f} {trend_unit})')
            
            order = np.argsort(stats['x_clean'])
            x_sorted = stats['x_clean'][order]
            y_sorted = stats['y_clean'][order]
            
            # Plot data points
            ax.plot(x_sorted, y_sorted, c=color, lw=2, ms=6, marker='o', ls='--')
            
            # Plot trend line
            X_sorted = sm.add_constant(x_sorted)
            pred = stats['model'].get_prediction(X_sorted)
            frame = pred.summary_frame(alpha=0.05)
            
            ax.plot(x_sorted, frame['mean'].values, c=color, lw=2.5, label=label, zorder=15)
            
            if trend_uncertainty_band:
                ax.fill_between(x_sorted, frame['mean_ci_lower'].values,
                               frame['mean_ci_upper'].values, color=color, alpha=0.15)
        
        ax.grid(True, alpha=0.3)
        ax.set_title(month_key, fontsize=max(18, int(trend_panel_height * 5)), fontweight='bold')
        ax.tick_params(axis='both', labelsize=max(12, int(trend_panel_height * 3)))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        
        # Y-range
        if y_range_is_dict:
            range_this = trend_y_range.get(month_key)
        else:
            range_this = trend_y_range
        if range_this is not None:
            ax.set_ylim(*range_this)
        
        leg = ax.legend(fontsize=max(10, int(trend_panel_height * 2.5)), 
                       handlelength=1.5, loc='best')
        for line in leg.get_lines():
            line.set_linewidth(3.0)
    
    # Shared labels using fig.text for proper positioning
    if trend_shared_labels:
        label_fs = max(22, int(trend_panel_height * 6))
        # Get positions after layout is computed
        fig.canvas.draw()
        left_ax = axes[-1, 0]
        right_ax = axes[-1, -1]
        bottom_ax = axes[-1, 0]
        
        # X label centered below bottom row
        x_center = 0.5 * (left_ax.get_position().x0 + right_ax.get_position().x1)
        y_bottom = bottom_ax.get_position().y0 - 0.06
        fig.text(x_center, y_bottom, fit_x_label, ha='center', va='top', fontsize=label_fs)
        
        # Y label centered to left of left column
        y_center = 0.5 * (axes[0, 0].get_position().y1 + axes[-1, 0].get_position().y0)
        x_left = axes[0, 0].get_position().x0 - 0.06
        fig.text(x_left, y_center, var_label, ha='right', va='center', rotation=90, fontsize=label_fs)
    
    # Hide unused subplots
    for idx in range(n_months, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    
    if save_name_base is not None:
        save_name_trend = f'{save_name_base}_{var}_trend'
        
        # Create output directories if they don't exist
        pdf_dir = Path(f'/nobackup/users/walj/Figures/pdf/{var}')
        jpg_dir = Path(f'/nobackup/users/walj/Figures/jpg/{var}')
        pdf_dir.mkdir(parents=True, exist_ok=True)
        jpg_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PDF
        pdf_out = pdf_dir / f'{save_name_trend}.pdf'
        plt.savefig(str(pdf_out), format='pdf', bbox_inches='tight')
        
        # Save JPG
        jpg_out = jpg_dir / f'{save_name_trend}.jpg'
        plt.savefig(str(jpg_out), format='jpg', dpi=300, bbox_inches='tight')
    
    plt.show()

#%% ============================================================================
#   PLOT 2: SCATTER PLOTS
#   ============================================================================

if plot_scatter and data_sources_scatter and data_compare_scatter and scatter_results:
    print('Plotting scatter plots...')
    
    n_base = len(data_sources_scatter)
    n_comp = len(data_compare_scatter)
    n_rows, n_cols = scatter_layout
    
    # Create one panel per base-compare pair
    n_panels = n_base * n_comp
    
    fig_width = scatter_panel_size * n_cols
    fig_height = scatter_panel_size * n_rows
    
    # Use subplots_adjust for tighter control instead of constrained_layout
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        sharex=scatter_shared_axes,
        sharey=scatter_shared_axes
    )
    
    # Adjust spacing for tighter layout
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.16,
        top=0.90,
        wspace=0.08,
        hspace=0.15
    )
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Collect all data for shared limits
    all_vals = []
    
    # Helper function to make axis labels like "ERA5 Temperature"
    def make_scatter_axis_label(source, var):
        src_label = get_source_label(source)
        var_name_str = var_name_cfg.get(var, var)
        return f'{src_label} {var_name_str}'
    
    panel_idx = 0
    for base_src in data_sources_scatter:
        for comp_src in data_compare_scatter:
            if panel_idx >= n_rows * n_cols:
                break
            
            row_idx, col_idx = panel_idx // n_cols, panel_idx % n_cols
            ax = axes[row_idx, col_idx]
            
            # Get data from scatter_results
            if base_src not in scatter_results or comp_src not in scatter_results:
                panel_idx += 1
                continue
            
            x_base = scatter_results[base_src]['fit'].values
            y_comp = scatter_results[comp_src]['fit'].values
            
            # Align data
            mask = np.isfinite(x_base) & np.isfinite(y_comp)
            x = x_base[mask]
            y = y_comp[mask]
            
            all_vals.extend(x)
            all_vals.extend(y)
            
            # All dots are black
            ax.scatter(x, y, s=40, c='k', alpha=0.8)
            
            # 1:1 line
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, lw=1.5, 
                   color='xkcd:brick red', ls='--', zorder=0)
            
            ax.set_aspect('equal', adjustable='box')
            ax.set_box_aspect(1)
            
            # Title is just the comparison source label (e.g., "R2.4")
            comp_label = get_source_label(comp_src)
            ax.set_title(comp_label, fontsize=20, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=12, length=6)
            
            panel_idx += 1
    
    # Set shared limits
    if scatter_shared_axes and all_vals:
        lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
        pad = 0.02 * (hi - lo)
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
    
    # Shared x-axis label: "[Base Source] [Variable Name]" (e.g., "Stations Temperature")
    base_src_first = data_sources_scatter[0]
    shared_xlabel = make_scatter_axis_label(base_src_first, var)
    
    # Get center position for x label using fig.text
    left_pos = axes[-1, 0].get_position().x0
    right_pos = axes[-1, -1].get_position().x1
    x_center = 0.5 * (left_pos + right_pos)
    fig.text(x_center, 0.03, shared_xlabel, ha='center', va='bottom', fontsize=18)
    
    # Y-axis label is just the variable name
    axes[0, 0].set_ylabel(var_name_cfg.get(var, var), fontsize=16)
    
    # Hide unused
    for idx in range(panel_idx, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    
    if save_name_base is not None:
        save_name_scatter = f'{save_name_base}_{var}_scatter'
        
        # Create output directories if they don't exist
        pdf_dir = Path(f'/nobackup/users/walj/Figures/pdf/{var}')
        jpg_dir = Path(f'/nobackup/users/walj/Figures/jpg/{var}')
        pdf_dir.mkdir(parents=True, exist_ok=True)
        jpg_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PDF
        pdf_out = pdf_dir / f'{save_name_scatter}.pdf'
        plt.savefig(str(pdf_out), format='pdf', bbox_inches='tight')
        
        # Save JPG
        jpg_out = jpg_dir / f'{save_name_scatter}.jpg'
        plt.savefig(str(jpg_out), format='jpg', dpi=300, bbox_inches='tight')
    
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
        
        color = COLORS[ii % len(COLORS)]
        label = get_source_label(src)
        
        data = raw_results[src]['monthly']
        
        if 'time' in data.dims:
            ax.plot(data['time'].values, data.values, c=color, lw=2, 
                   marker='o', ms=4, ls='-', alpha=0.8, label=label)
    
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
    
    if save_name_base is not None:
        save_name_raw = f'{save_name_base}_{var}_rawtime'
        
        # Create output directories if they don't exist
        pdf_dir = Path(f'/nobackup/users/walj/Figures/pdf/{var}')
        jpg_dir = Path(f'/nobackup/users/walj/Figures/jpg/{var}')
        pdf_dir.mkdir(parents=True, exist_ok=True)
        jpg_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PDF
        pdf_out = pdf_dir / f'{save_name_raw}.pdf'
        plt.savefig(str(pdf_out), format='pdf', bbox_inches='tight')
        
        # Save JPG
        jpg_out = jpg_dir / f'{save_name_raw}.jpg'
        plt.savefig(str(jpg_out), format='jpg', dpi=300, bbox_inches='tight')
    
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
    
    # Use months_spatial (defaults to all months if None)
    months_for_spatial = months_spatial if months_spatial else list(range(1, 13))
    
    n_sources = len(data_sources_spatial)
    n_rows, n_cols = spatial_layout
    
    fig_width = spatial_panel_width * n_cols
    fig_height = spatial_panel_height * n_rows
    
    proj_plot = proj_cfg.get(proj_sel, ccrs.PlateCarree())
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        subplot_kw={'projection': proj_plot},
        sharex=True,
        sharey=True
    )
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    meshes = []
    trend_fields = []
    
    for idx, spatial_src in enumerate(data_sources_spatial):
        if idx >= n_rows * n_cols:
            break
        
        if spatial_src not in source_data:
            continue
        
        row_idx, col_idx = idx // n_cols, idx % n_cols
        ax = axes[row_idx, col_idx]
        
        # Compute spatial trend for this source
        yearly_spatial = compute_seasonal_yearly(source_data[spatial_src]['monthly'], 
                                                  months_for_spatial, years, proc_type)
        fit_spatial = compute_fit_data(yearly_spatial, fit_against_gmst,
                                       rolling_mean_var, rolling_mean_years, min_periods)
        
        # Compute trend field
        fits = fit_spatial.polyfit(dim='fit_against', deg=1, skipna=True)
        slope = fits.polyfit_coefficients.sel(degree=1)
        trend_field = (slope * fit_scaling).astype('float32').compute()
        
        if var == 'P' and relative_precip:
            avg_spatial = source_data[spatial_src]['monthly'].mean('time').compute()
            trend_field = (trend_field / avg_spatial) * 100.0
        
        trend_fields.append(trend_field)
        
        show_x = (row_idx == n_rows - 1)
        show_y = (col_idx == 0)
        
        # Handle coordinate extraction for different data sources
        # RACMO uses rlon/rlat, E-OBS/ERA5 use longitude/latitude
        if 'longitude' in trend_field.coords:
            lons_plot = trend_field['longitude'].values
            lats_plot = trend_field['latitude'].values
        elif 'rlon' in trend_field.coords:
            lons_plot = trend_field['rlon'].values
            lats_plot = trend_field['rlat'].values
        else:
            # Fallback: try to get last two coords
            coord_names = list(trend_field.coords)
            lons_plot = trend_field.coords[coord_names[-1]].values
            lats_plot = trend_field.coords[coord_names[-2]].values
        
        mesh, _ = plot_map(
            fig, ax,
            trend_field,
            lons_plot,
            lats_plot,
            crange=spatial_crange,
            cmap=trend_cmap,
            extreme_colors=trend_extreme,
            show_x_ticks=True,
            show_y_ticks=True,
            show_x_labels=show_x,
            show_y_labels=show_y,
            x_ticks=2,
            y_ticks=1,
            x_ticks_num=False,
            y_ticks_num=False,
            tick_size=12,
            extent=[*spatial_plot_lons, *spatial_plot_lats],
            proj=proj_plot,
            add_colorbar=False,
            title=get_source_label(spatial_src),
            title_size=18,
            show_plot=False,
            lats_area=lats,
            lons_area=lons,
            proj_area=proj_plot,
        )
        
        meshes.append(mesh)
    
    # Hide unused
    for idx in range(len(data_sources_spatial), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    
    # Shared colorbar - closer to plot and thinner
    if meshes:
        cbar = shared_colorbar(
            fig=fig,
            axes=axes.ravel()[:len(meshes)],
            mesh=meshes[0],
            datasets=trend_fields,
            crange=spatial_crange,
            label=f'{var_name_cfg.get(var, var)} Trend ({trend_unit})',
            orientation='horizontal',
            c_ticks=5,
            c_ticks_num=True,
            tick_labelsize=12,
            labelsize=16,
            pad=0.04,
            thickness=0.025,
            extendfrac=0.03
        )
    
    if save_name_base is not None:
        save_name_spatial = f'{save_name_base}_{var}_spatialtrend'
        
        # Create output directories if they don't exist
        pdf_dir = Path(f'/nobackup/users/walj/Figures/pdf/{var}')
        jpg_dir = Path(f'/nobackup/users/walj/Figures/jpg/{var}')
        pdf_dir.mkdir(parents=True, exist_ok=True)
        jpg_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PDF
        pdf_out = pdf_dir / f'{save_name_spatial}.pdf'
        plt.savefig(str(pdf_out), format='pdf', bbox_inches='tight')
        
        # Save JPG
        jpg_out = jpg_dir / f'{save_name_spatial}.jpg'
        plt.savefig(str(jpg_out), format='jpg', dpi=300, bbox_inches='tight')
    
    plt.show()

print('=' * 60)
print('All plots complete!')
print('=' * 60)

#%%


