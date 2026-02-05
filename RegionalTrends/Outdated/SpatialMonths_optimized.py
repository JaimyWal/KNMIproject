#%% Imports
"""
SpatialMonths_optimized.py - Optimized multi-panel spatial analysis

OPTIMIZATION STRATEGY:
======================
This script is optimized for efficiency through a 3-phase approach:

PHASE 1: Load data once per unique source
    - Load each data source ONCE with months=None (all 12 months)
    - Uses extended years (years[0]-1) to capture December for DJF
    - Only requests 'raw' and 'monthly' from process_source
    
PHASE 2: Build regridders and regrid once
    - Build ONE regridder per unique (base, compare) source pair
    - Regrid the FULL monthly time series ONCE
    - Optional: regrid raw data if daily correlation/RMSE is needed
    
PHASE 3: Slice by season and compute metrics
    - For each season (DJF, MAM, etc.), filter the pre-loaded/regridded data
    - Apply climate year logic: December belongs to NEXT year for DJF
    - Compute seasonal averages, trends, correlations, RMSE from sliced data

CLIMATE YEAR HANDLING:
=====================
For seasons spanning year boundary (e.g., DJF = [12, 1, 2]):
    - December 1979 + January 1980 + February 1980 = DJF 1980
    - December gets clim_year = calendar_year + 1
    - Years are extended to [years[0]-1, years[1]] when loading
    
This matches the logic in ProcessSource.py exactly.

SPEEDUP SOURCES:
===============
1. Data loading: 1x per source instead of 4x (once per season)
2. Regridding: 1x per source pair instead of 4x (once per season)
3. Regridder building: Cached, only built once per grid pair
4. Memory: Only 'raw' and 'monthly' loaded, seasonal aggregates computed on-demand
"""

# Standard libraries
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
import xesmf as xe
from pathlib import Path
import sys
from importlib import reload

# Custom libraries
PROJECT_ROOT = Path.home() / 'KNMIproject'
sys.path.insert(0, str(PROJECT_ROOT))

from RegionalTrends.Helpers import PlotMaps
reload(PlotMaps)          
from RegionalTrends.Helpers.PlotMaps import plot_map, shared_colorbar, add_shared_cbar_label

from RegionalTrends.Helpers import ProcessNetCDF
reload(ProcessNetCDF)          
from RegionalTrends.Helpers.ProcessNetCDF import subset_space

from RegionalTrends.Outdated import ProcessSource
reload(ProcessSource)
from RegionalTrends.Outdated.ProcessSource import process_source

from RegionalTrends.Helpers import GridBounds
reload(GridBounds)          
from RegionalTrends.Helpers.GridBounds import grid_with_bounds

# Data config custom libraries
import RegionalTrends.Helpers.Config.Constants as Constants
reload(Constants)

import RegionalTrends.Helpers.Config.Plotting as Plotting
reload(Plotting)
from RegionalTrends.Helpers.Config.Plotting import build_corr_cmap, convert_cmap


plt.rcParams['axes.unicode_minus'] = False
dask.config.set(scheduler='threads', num_workers=12)

# Pre-download cartopy features to avoid repeated downloads during plotting
import cartopy.feature as cfeature
_ = cfeature.NaturalEarthFeature('physical', 'coastline', '50m')
_ = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '50m')

#%% User inputs

# Main arguments
var = 'Q_all'
file_freq = 'Monthly'
proc_type = 'Mean'
data_base = ['ERA5', 'RACMO2.4', 'Eobs']
data_compare = None

# Data selection arguments
months_dict = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
years = [1980, 2020] 
lats = [37.7, 63.3]
lons = [-13.3, 22.3]
proj_sel = 'RACMO2.4'
land_only = False
trim_border = None

# Spatial plotting arguments
plot_climatology = False
avg_crange = {'DJF': [-3, 3], 'MAM': [-2, 2], 'JJA': [-1, 1], 'SON': [-2, 2]}
std_mask_ref = data_base
std_dir = 'Lesser'
shared_cbar_avg = False 
shared_cbar_label_avg = False
short_cbar_label_avg = True
cbar_ticks_avg_shared = 0.5
cbar_ticks_avg_sep = [1, 1, 0.5, 1]
cbar_ticks_num_avg = False
save_name_avg = None

# Trend plotting arguments
trend_calc = True
trend_crange = [-0.8, 0.8]
trend_regrid = True
fit_against_gmst = False
shared_cbar_trend = True
shared_cbar_label_trend = True
short_cbar_label_trend = False
cbar_ticks_trend_shared = 0.2
cbar_ticks_trend_sep = [0.5, 1, 1, 1]
cbar_ticks_num_trend = False
save_name_trend = None

# Correlation plotting arguments
corr_calc = True
corr_freq = 'Monthly'
corr_crange = [0.8, 1]
corr_cmap_type = None
shared_cbar_corr = True
shared_cbar_label_corr = True
short_cbar_label_corr = False
cbar_ticks_corr_shared = 0.05
cbar_ticks_corr_sep = 6
cbar_ticks_num_corr = False
save_name_corr = None

# RMSE plotting arguments
rmse_calc = True
rmse_freq = 'Monthly'
rmse_crange = [0, 2]
shared_cbar_rmse = True
shared_cbar_label_rmse = True
short_cbar_label_rmse = False
cbar_ticks_rmse_shared = 0.2
cbar_ticks_rmse_sep = 0.2
cbar_ticks_num_rmse = False
save_name_rmse = None

# Additional plotting arguments
proj_plot = 'RACMO2.4'
plot_lats = [38, 63]
plot_lons = [-13, 22]
switch_sign = False
cut_boundaries = False
cbar_orientation = 'vertical'
swap_rows_cols = False
show_col_titles = True
show_row_titles = True
cmap_type = None
n_disc_colors = 20 
panel_width = 5
panel_height = 5

# Area contour arguments
lats_area = [50.7, 53.6]
lons_area = [3.25, 7.35]
proj_area = 'RACMO2.4'
true_contour = True
grid_contour = False

# Other arguments
rolling_mean_var = False
rolling_mean_years = 3
min_periods = 1
relative_precip = False

#%% Dataset configurations

data_sources = Constants.DATA_SOURCES
station_sources = Constants.STATION_SOURCES
station_coord_cfg = Constants.STATION_COORD_CFG
var_file_cfg = Constants.VAR_FILE_CFG
var_name_cfg = Constants.VAR_NAME_CFG
var_units_cfg = Constants.VAR_UNIT_CFG
var_symbol_cfg = Constants.VAR_SYMBOL_CFG
proj_cfg = Constants.PROJ_CFG
var_colors_cfg = Plotting.VAR_COLORS_CFG

#%% Helper functions for configuration

def get_fit_config(fit_against_gmst):
    """Get fitting configuration based on whether fitting against GMST or time."""
    if fit_against_gmst:
        return {'unit': '°C GMST', 'scaling': 1, 'x_label': r'$\Delta$GMST (°C)'}
    return {'unit': 'decade', 'scaling': 10, 'x_label': 'Year'}

def get_trend_unit(var, var_units_cfg, fit_unit, relative_precip):
    """Get trend unit string based on variable and fit configuration."""
    if var == 'P' and relative_precip:
        return '% / ' + fit_unit
    elif var_units_cfg[var] == '':
        return 'per' + fit_unit
    return var_units_cfg[var] + ' / ' + fit_unit

def get_var_symbol(var, var_symbol_cfg):
    """Get properly formatted variable symbol with math delimiters."""
    raw_sym = var_symbol_cfg[var]
    return raw_sym if '$' in raw_sym else r'$' + raw_sym + r'$'

def get_labels(var, var_name_cfg, var_symbol_cfg, var_units_cfg, trend_unit, 
               short_avg, short_trend, short_corr, short_rmse, has_comparison):
    """Generate all colorbar labels."""
    sym = get_var_symbol(var, var_symbol_cfg)
    full_var = var_name_cfg[var] + ' ' + sym
    if var_name_cfg[var] == var_symbol_cfg[var]:
        full_var = sym
    
    # Average label
    avg_label = (sym if short_avg else full_var) + ' (' + var_units_cfg[var] + ')'
    if has_comparison:
        avg_label = r'$\Delta$' + avg_label if short_avg else 'Difference in ' + avg_label
    
    trend_label = (sym if short_trend else full_var) + ' trend (' + trend_unit + ')'
    corr_label = (sym if short_corr else full_var) + ' correlation'
    rmse_label = (sym if short_rmse else full_var) + ' RMSE (' + var_units_cfg[var] + ')'
    
    return {'avg': avg_label, 'trend': trend_label, 'corr': corr_label, 'rmse': rmse_label}

def setup_colormaps(var, var_colors_cfg, has_comparison, cmap_type, n_disc_colors, corr_cmap_type):
    """Setup all colormaps for the plots."""
    if not var_colors_cfg.get(var):
        var_colors_cfg[var] = var_colors_cfg['Default'].copy()
    elif not var_colors_cfg[var].get('cmap_trend'):
        var_colors_cfg[var]['cmap_trend'] = var_colors_cfg['Default']['cmap_trend']
        var_colors_cfg[var]['extreme_trend'] = var_colors_cfg['Default']['extreme_trend']
    
    avg_cmap_key = 'cmap_trend' if has_comparison else 'cmap_mean'
    avg_ext_key = 'extreme_trend' if has_comparison else 'extreme_mean'
    
    var_colors_cfg[var]['cmap_mean'] = convert_cmap(
        var_colors_cfg[var]['cmap_mean'], cmap_type, n_disc_colors)
    var_colors_cfg[var]['cmap_trend'] = convert_cmap(
        var_colors_cfg[var]['cmap_trend'], cmap_type, n_disc_colors)
    
    rmse_cmap = convert_cmap(cmocean.cm.amp, cmap_type, n_disc_colors)
    corr_meta = build_corr_cmap(corr_cmap_type)
    corr_cmap = convert_cmap(corr_meta['corr_cmap'], cmap_type, n_disc_colors)
    
    return {
        'avg_cmap': var_colors_cfg[var][avg_cmap_key],
        'avg_extreme': var_colors_cfg[var][avg_ext_key],
        'trend_cmap': var_colors_cfg[var]['cmap_trend'],
        'trend_extreme': var_colors_cfg[var]['extreme_trend'],
        'corr_cmap': corr_cmap,
        'corr_extreme': corr_meta['corr_extreme'],
        'rmse_cmap': rmse_cmap,
        'rmse_extreme': [None, "#24050a"]
    }

def ensure_list(param, n):
    """Ensure parameter is a list of length n."""
    return param if isinstance(param, list) else [param] * n

#%% Run configuration

# Set up basic parameters
fit_cfg = get_fit_config(fit_against_gmst)
fit_unit, fit_scaling = fit_cfg['unit'], fit_cfg['scaling']
trend_unit = get_trend_unit(var, var_units_cfg, fit_unit, relative_precip)

n_runs = len(data_base) if isinstance(data_base, list) else 1
has_comparison = data_compare is not None

# Convert to lists
data_base_list = ensure_list(data_base, n_runs)
data_compare_list = ensure_list(data_compare, n_runs)
freq_file_list = ensure_list(file_freq, n_runs)
std_mask_ref_list = ensure_list(std_mask_ref, n_runs)
std_dir_list = ensure_list(std_dir, n_runs)
switch_sign_list = ensure_list(switch_sign, n_runs)
corr_freq_list = ensure_list(corr_freq, n_runs)
rmse_freq_list = ensure_list(rmse_freq, n_runs)

# Setup labels and colormaps
labels = get_labels(var, var_name_cfg, var_symbol_cfg, var_units_cfg, trend_unit,
                    short_cbar_label_avg, short_cbar_label_trend, 
                    short_cbar_label_corr, short_cbar_label_rmse, has_comparison)
cmaps = setup_colormaps(var, var_colors_cfg, has_comparison, cmap_type, n_disc_colors, corr_cmap_type)

# Setup projections
proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())
proj_plot = proj_cfg.get(proj_plot, ccrs.PlateCarree())
proj_area = proj_cfg.get(proj_area, ccrs.PlateCarree())

#%% ============================================================================
#   PHASE 1: LOAD ALL DATA ONCE (with months=None for full year)
#   ============================================================================

print("=" * 60)
print("PHASE 1: Loading all data sources (full year, once per source)")
print("=" * 60)

def get_unique_sources():
    """Get unique (data_source, file_freq) combinations that need loading."""
    sources = set()
    for ii in range(n_runs):
        if data_base_list[ii] is not None:
            sources.add((data_base_list[ii], freq_file_list[ii]))
        if data_compare_list[ii] is not None:
            sources.add((data_compare_list[ii], freq_file_list[ii]))
        if std_mask_ref_list[ii] not in (None, 'Pool', data_base_list[ii], data_compare_list[ii]):
            sources.add((std_mask_ref_list[ii], freq_file_list[ii]))
    return sources

def needs_extended_years(months_dict):
    """Check if any season spans year boundary (needs previous December)."""
    for months in months_dict.values():
        month_start = int(months[0])
        month_end = int(months[-1])
        if month_start > month_end:  # e.g., DJF: [12, 1, 2]
            return True
    return False

def load_source_data(data_source, file_freq):
    """
    Load a data source with ALL months (months=None).
    
    IMPORTANT: We only request 'raw' and 'monthly' from process_source.
    - 'yearly', 'fit', 'avg' computed with months=None would be WRONG for seasonal analysis
    - We'll compute season-specific yearly/fit/avg ourselves after filtering by month
    
    process_source handles:
    - Extended years (years[0]-1) when months span year boundary
    - clim_year assignment (December → year+1 for DJF)
    """
    print(f"  Loading: {data_source} ({file_freq})")
    
    # Extend years if needed for DJF-style seasons
    years_load = list(years)
    if needs_extended_years(months_dict):
        years_load[0] = years[0] - 1
    
    return process_source(
        data_source, var, data_sources, station_sources, file_freq,
        var_file_cfg, proj_cfg, proc_type,
        months=None,  # Load ALL months - we filter by season later
        years=years_load,  # Extended years for DJF
        lats=lats, lons=lons,
        land_only=land_only, trim_border=trim_border,
        rotpole_sel=proj_sel,
        rolling_mean_var=False,  # We'll apply rolling mean ourselves for seasonal data
        fit_against_gmst=False,  # We'll set up fit coordinate ourselves
        rolling_mean_years=1,
        min_periods=1,
        return_items=('raw', 'monthly')  # Only get raw and monthly - we compute seasonal aggregates ourselves
    )

# Load all unique sources once
source_data = {}
for src, freq in get_unique_sources():
    source_data[(src, freq)] = load_source_data(src, freq)

print(f"Loaded {len(source_data)} unique data sources\n")

#%% ============================================================================
#   PHASE 2: BUILD REGRIDDERS AND REGRID FULL TIME SERIES (once per pair)
#   ============================================================================

print("=" * 60)
print("PHASE 2: Building regridders and regridding (once per source pair)")
print("=" * 60)

def get_regrid_method(var):
    """Determine regridding method based on variable."""
    return 'conservative_normed' if var == 'P' else 'bilinear'

def create_regridder(src_grid, trg_grid, method):
    """Create a regridder."""
    return xe.Regridder(src_grid, trg_grid, method, unmapped_to_nan=True)

# Build regridders for unique (base, compare) pairs
regridders = {}
regridded_data = {}

TARGET_CHUNKS = {'latitude': 100, 'longitude': 100}

for ii in range(n_runs):
    base_src = data_base_list[ii]
    comp_src = data_compare_list[ii]
    freq = freq_file_list[ii]
    
    if base_src is None or comp_src is None:
        continue
    
    pair_key = (base_src, comp_src, freq)
    if pair_key in regridders:
        continue  # Already built
    
    print(f"  Building regridder: {comp_src} -> {base_src}")
    
    base_data = source_data[(base_src, freq)]
    comp_data = source_data[(comp_src, freq)]
    
    # For grid definition, compute a temporary avg from monthly
    base_avg_temp = base_data['monthly'].mean('time')
    comp_avg_temp = comp_data['monthly'].mean('time')
    
    method = get_regrid_method(var)
    if var == 'P':
        trg_grid = grid_with_bounds(base_avg_temp, 
                                    rotpole_native=proj_cfg.get(base_src, ccrs.PlateCarree()))
        src_grid = grid_with_bounds(comp_avg_temp, 
                                    rotpole_native=proj_cfg.get(comp_src, ccrs.PlateCarree()))
    else:
        trg_grid = base_avg_temp
        src_grid = comp_avg_temp
    
    regridders[pair_key] = create_regridder(src_grid, trg_grid, method)
    
    # Regrid monthly data ONCE (this is the key optimization)
    regridder = regridders[pair_key]
    print(f"    Regridding monthly time series...")
    regridded_data[(pair_key, 'monthly')] = regridder(comp_data['monthly'], 
                                                       output_chunks=TARGET_CHUNKS).astype('float32')
    
    # Optionally regrid raw if needed for daily correlation/RMSE
    needs_raw = any(corr_freq_list[jj] == 'Daily' or rmse_freq_list[jj] == 'Daily' 
                    for jj in range(n_runs) 
                    if data_base_list[jj] == base_src and data_compare_list[jj] == comp_src)
    
    if needs_raw and comp_data.get('raw') is not None:
        print(f"    Regridding raw (daily) time series...")
        regridded_data[(pair_key, 'raw')] = regridder(comp_data['raw'], 
                                                       output_chunks=TARGET_CHUNKS).astype('float32')

print(f"Built {len(regridders)} regridders\n")

#%% ============================================================================
#   PHASE 3: CLIMATE YEAR SLICING FUNCTIONS
#   ============================================================================

def assign_clim_year(data, months, time_dim='time'):
    """
    Assign climate year coordinate to data based on season months.
    
    Climate year logic (matching ProcessSource.py):
    - If months span year boundary (e.g., DJF: [12, 1, 2]), December belongs to NEXT year's season
    - Example: Dec 1979 + Jan 1980 + Feb 1980 = DJF 1980
    
    NOTE: process_source with months=None already assigns clim_year for the FULL year.
    However, that assignment assumes no year boundary crossing. We need to reassign
    for specific seasons like DJF.
    """
    months_arr = np.asarray(months, dtype=int)
    month_start = int(months_arr[0])
    month_end = int(months_arr[-1])
    spans_year_boundary = month_start > month_end  # e.g., [12, 1, 2]
    
    month_vals = data[time_dim].dt.month
    year_vals = data[time_dim].dt.year
    
    if spans_year_boundary:
        # December gets next year as climate year
        clim_year = xr.where(month_vals >= month_start, year_vals + 1, year_vals)
    else:
        clim_year = year_vals
    
    return data.assign_coords(clim_year=(time_dim, clim_year.values))

def filter_by_season(data, months, years, time_dim='time'):
    """
    Filter data by months AND climate year.
    
    Parameters
    ----------
    data : xr.DataArray
        Data with time dimension
    months : list
        Month numbers [1-12] for the season
    years : list
        [start_year, end_year] for filtering by climate year
    time_dim : str
        Name of time dimension
    """
    if data is None:
        return None
    
    # Handle 2D data (no time dimension)
    if time_dim not in data.dims:
        return data
    
    months_arr = np.asarray(months, dtype=int)
    
    # Filter by months first
    month_vals = data[time_dim].dt.month
    in_season = month_vals.isin(months_arr)
    data_season = data.where(in_season, drop=True)
    
    if len(data_season[time_dim]) == 0:
        return data_season
    
    # Assign climate year for this specific season
    data_season = assign_clim_year(data_season, months, time_dim)
    
    # Filter by climate year range
    y0, y1 = years[0], years[-1]
    return data_season.where((data_season['clim_year'] >= y0) & 
                              (data_season['clim_year'] <= y1), drop=True)

def compute_seasonal_yearly(data_monthly, months, years, proc_type='Mean'):
    """
    Compute yearly means per climate year for a season.
    
    This is the key function that properly computes season-specific yearly aggregates.
    """
    if data_monthly is None:
        return None
    
    filtered = filter_by_season(data_monthly, months, years)
    if filtered is None or len(filtered['time']) == 0:
        return None
    
    # Group by climate year to get seasonal yearly means
    if proc_type == 'Max':
        yearly = filtered.groupby('clim_year').max('time')
    elif proc_type == 'Min':
        yearly = filtered.groupby('clim_year').min('time')
    else:
        yearly = filtered.groupby('clim_year').mean('time')
    
    return yearly.astype('float32')

def compute_seasonal_avg(data_monthly, months, years, proc_type='Mean'):
    """Compute seasonal climatological average."""
    if data_monthly is None:
        return None
    
    yearly = compute_seasonal_yearly(data_monthly, months, years, proc_type)
    if yearly is None:
        return None
    
    return yearly.mean(dim='clim_year').astype('float32').compute()

def compute_seasonal_fit(data_monthly, months, years, proc_type='Mean'):
    """
    Compute fit data (yearly means with year as coordinate) for a season.
    
    Returns data with 'fit_against' dimension containing year values as floats.
    """
    yearly = compute_seasonal_yearly(data_monthly, months, years, proc_type)
    if yearly is None:
        return None
    
    # Apply rolling mean if requested
    if rolling_mean_var:
        yearly = yearly.rolling(clim_year=rolling_mean_years, center=True, 
                                 min_periods=min_periods).mean()
    
    # Set up fit coordinate
    if fit_against_gmst:
        file_GMST = '/nobackup/users/walj/era5/era5_gmst_anom.nc'
        gmst = xr.open_dataset(file_GMST)['__xarray_dataarray_variable__']
        gmst_roll = gmst.rolling(time=rolling_mean_years, center=True, 
                                  min_periods=min_periods).mean()
        # Match years to GMST
        fit_vals = []
        for cy in yearly['clim_year'].values:
            yr_str = str(int(cy))
            try:
                match = gmst_roll.sel(time=yr_str, method='nearest')
                fit_vals.append(float(match.values))
            except:
                fit_vals.append(np.nan)
        fit_coord = np.array(fit_vals)
    else:
        fit_coord = yearly['clim_year'].values.astype(float)
    
    return yearly.rename({'clim_year': 'fit_against'}).assign_coords(
        fit_against=('fit_against', fit_coord)).astype('float32')

#%% ============================================================================
#   PHASE 4: COMPUTE METRICS FOR EACH SEASON (OPTIMIZED)
#   ============================================================================
"""
OPTIMIZATION NOTES:
==================
Key optimizations in this section:
1. Cache seasonal slices - filter once, reuse for all metrics
2. Build lazy computation graphs - delay .compute() until the end
3. Batch compute with dask.compute() - execute multiple graphs in parallel
4. Avoid redundant polyfit calls - compute fit data once, reuse for trend + std_mask
"""

print("=" * 60)
print("PHASE 4: Computing metrics for each season (optimized)")
print("=" * 60)

def compute_correlation_lazy(x, y, time_dim='time'):
    """Compute spatial correlation map - returns LAZY result (no .compute())."""
    x, y = xr.align(x, y, join='inner')
    
    time_chunk = 365 if len(x[time_dim]) > 1000 else -1
    x = x.chunk({time_dim: time_chunk, 'latitude': 100, 'longitude': 100})
    y = y.chunk({time_dim: time_chunk, 'latitude': 100, 'longitude': 100})
    
    valid = np.isfinite(x) & np.isfinite(y)
    xv = x.where(valid)
    yv = y.where(valid)
    
    n = valid.sum(time_dim)
    dx = xv - xv.mean(time_dim)
    dy = yv - yv.mean(time_dim)
    cov = (dx * dy).mean(time_dim)
    sx = xv.std(time_dim)
    sy = yv.std(time_dim)
    
    corr = (cov / (sx * sy)).where((n >= 2) & (sx > 0) & (sy > 0))
    return corr.assign_coords(latitude=x['latitude'], longitude=x['longitude']).astype('float32')

def compute_rmse_lazy(x, y, time_dim='time'):
    """Compute spatial RMSE map - returns LAZY result (no .compute())."""
    x, y = xr.align(x, y, join='inner')
    
    time_chunk = 365 if len(x[time_dim]) > 1000 else -1
    x = x.chunk({time_dim: time_chunk, 'latitude': 100, 'longitude': 100})
    y = y.chunk({time_dim: time_chunk, 'latitude': 100, 'longitude': 100})
    
    valid = np.isfinite(x) & np.isfinite(y)
    err = y.where(valid) - x.where(valid)
    rmse = np.sqrt((err**2).mean(time_dim))
    return rmse.assign_coords(latitude=x['latitude'], longitude=x['longitude']).astype('float32')

def compute_trend_lazy(fit_data, fit_scaling, relative_precip=False, avg_data=None, var='Tg'):
    """Compute trend from fit data - returns LAZY result (no .compute())."""
    if fit_data is None:
        return None
    
    fits = fit_data.polyfit(dim='fit_against', deg=1, skipna=True)
    slope = fits.polyfit_coefficients.sel(degree=1)
    trend = (slope * fit_scaling).astype('float32')
    
    if relative_precip and var == 'P' and avg_data is not None:
        trend = (trend / avg_data) * 100.0
    
    return trend

def compute_std_ref_lazy(base_fit, comp_fit_reg, std_ref_type):
    """Compute standard deviation reference - returns LAZY result."""
    if std_ref_type is None or base_fit is None or comp_fit_reg is None:
        return None
    
    if std_ref_type == 'Pool':
        xb, yc = xr.align(base_fit, comp_fit_reg, join='inner')
        valid = np.isfinite(xb) & np.isfinite(yc)
        xb = xb.where(valid).chunk({'fit_against': -1})
        yc = yc.where(valid).chunk({'fit_against': -1})
        
        fits_xb = xb.polyfit(dim='fit_against', deg=1, skipna=True)
        trend_xb = xr.polyval(xb['fit_against'], fits_xb.polyfit_coefficients)
        resid_xb = xb - trend_xb
        
        fits_yc = yc.polyfit(dim='fit_against', deg=1, skipna=True)
        trend_yc = xr.polyval(yc['fit_against'], fits_yc.polyfit_coefficients)
        resid_yc = yc - trend_yc
        
        sx = resid_xb.std('fit_against')
        sy = resid_yc.std('fit_against')
        std_ref = np.sqrt(0.5 * (sx**2 + sy**2))
        return std_ref.where(valid.sum('fit_against') >= 2)
    else:
        # std_ref_type is 'base' or 'compare'
        ref_fit = base_fit if std_ref_type == 'base' else comp_fit_reg
        _, ref_fit = xr.align(base_fit, ref_fit, join='inner')
        ref_fit = ref_fit.chunk({'fit_against': -1})
        n_ref = np.isfinite(ref_fit).sum('fit_against')
        
        fits_ref = ref_fit.polyfit(dim='fit_against', deg=1, skipna=True)
        trend_ref = xr.polyval(ref_fit['fit_against'], fits_ref.polyfit_coefficients)
        resid_ref = ref_fit - trend_ref
        return resid_ref.std('fit_against').where(n_ref >= 2)

def get_title(base_src, comp_src, data_sources, switch_sign):
    """Generate plot title from source names."""
    base_name = next(key for key in data_sources if key in base_src)
    
    if comp_src is None:
        title = base_name
    else:
        comp_name = next(key for key in data_sources if key in comp_src)
        if switch_sign:
            title = f"{base_name} - {comp_name}"
        else:
            title = f"{comp_name} - {base_name}"
    
    return title.replace('ACMO', '').replace('Eobs', 'E-OBS')

# =============================================================================
# PRE-COMPUTE SEASONAL SLICES (filter once, reuse for all metrics)
# =============================================================================
print("\n  Pre-computing seasonal slices...")

seasonal_slices = {}  # Cache: (source, freq, month_key, data_type) -> filtered data

for month_key, months in months_dict.items():
    for src, freq in source_data.keys():
        data = source_data[(src, freq)]
        
        # Cache filtered monthly data
        monthly_filtered = filter_by_season(data['monthly'], months, years)
        seasonal_slices[(src, freq, month_key, 'monthly')] = monthly_filtered
        
        # Cache yearly aggregates (used for correlation/RMSE with Yearly freq)
        yearly_agg = compute_seasonal_yearly(data['monthly'], months, years, proc_type)
        seasonal_slices[(src, freq, month_key, 'yearly')] = yearly_agg
        
        # Cache fit data (used for trends and std_mask)
        fit_data = compute_seasonal_fit(data['monthly'], months, years, proc_type)
        seasonal_slices[(src, freq, month_key, 'fit')] = fit_data
        
        # Cache raw if available (for daily correlation/RMSE)
        if data.get('raw') is not None:
            raw_filtered = filter_by_season(data['raw'], months, years)
            seasonal_slices[(src, freq, month_key, 'raw')] = raw_filtered

# Also cache seasonal slices for REGRIDDED comparison data
for pair_key in regridded_data.keys():
    base_src, comp_src, freq = pair_key[0]  # Unpack (base, comp, freq)
    data_type = pair_key[1]  # 'monthly' or 'raw'
    
    if data_type == 'monthly':
        comp_monthly_reg = regridded_data[pair_key]
        for month_key, months in months_dict.items():
            # Filtered monthly
            monthly_filtered = filter_by_season(comp_monthly_reg, months, years)
            seasonal_slices[(pair_key, month_key, 'monthly')] = monthly_filtered
            
            # Yearly aggregate
            yearly_agg = compute_seasonal_yearly(comp_monthly_reg, months, years, proc_type)
            seasonal_slices[(pair_key, month_key, 'yearly')] = yearly_agg
            
            # Fit data
            fit_data = compute_seasonal_fit(comp_monthly_reg, months, years, proc_type)
            seasonal_slices[(pair_key, month_key, 'fit')] = fit_data
    
    elif data_type == 'raw':
        comp_raw_reg = regridded_data[pair_key]
        for month_key, months in months_dict.items():
            raw_filtered = filter_by_season(comp_raw_reg, months, years)
            seasonal_slices[(pair_key, month_key, 'raw')] = raw_filtered

print(f"  Cached {len(seasonal_slices)} seasonal slices")

# =============================================================================
# MAIN PROCESSING LOOP - USING CACHED SLICES + BATCH COMPUTE
# =============================================================================
all_results = {}

for month_key, months in months_dict.items():
    print(f"\nProcessing: {month_key}")
    results = []
    
    # Collect all lazy computations for this month, then batch compute
    lazy_computations = []
    computation_map = []  # Track which result each computation belongs to
    
    for ii in range(n_runs):
        base_src = data_base_list[ii]
        comp_src = data_compare_list[ii]
        freq = freq_file_list[ii]
        
        if base_src is None:
            continue
        
        # Initialize result dict with placeholders
        result = {
            'data_avg_plot': None,
            'trend_plot': None,
            'corr_plot': None,
            'rmse_plot': None,
            'mask_std': None,
            'title': get_title(base_src, comp_src, data_sources, switch_sign_list[ii])
        }
        
        # Get cached seasonal slices for base source
        base_yearly = seasonal_slices.get((base_src, freq, month_key, 'yearly'))
        base_fit = seasonal_slices.get((base_src, freq, month_key, 'fit'))
        base_monthly = seasonal_slices.get((base_src, freq, month_key, 'monthly'))
        base_raw = seasonal_slices.get((base_src, freq, month_key, 'raw'))
        
        # Compute base seasonal average (lazy)
        if base_yearly is not None:
            base_avg_season = base_yearly.mean(dim='clim_year').astype('float32')
        else:
            base_avg_season = None
        
        if comp_src is None:
            # Single source mode
            if base_avg_season is not None:
                lazy_computations.append(base_avg_season)
                computation_map.append((ii, 'data_avg_plot'))
            
            if trend_calc and base_fit is not None:
                trend_lazy = compute_trend_lazy(base_fit, fit_scaling, relative_precip, base_avg_season, var)
                if trend_lazy is not None:
                    lazy_computations.append(trend_lazy)
                    computation_map.append((ii, 'trend_plot'))
        
        else:
            # Comparison mode
            pair_key = (base_src, comp_src, freq)
            minus_scaling = -1 if switch_sign_list[ii] else 1
            
            # Get cached regridded comparison data
            comp_yearly = seasonal_slices.get(((pair_key, 'monthly'), month_key, 'yearly'))
            comp_fit = seasonal_slices.get(((pair_key, 'monthly'), month_key, 'fit'))
            comp_monthly = seasonal_slices.get(((pair_key, 'monthly'), month_key, 'monthly'))
            comp_raw = seasonal_slices.get(((pair_key, 'raw'), month_key, 'raw'))
            
            # Compute comparison seasonal average (lazy)
            if comp_yearly is not None:
                comp_avg_season = comp_yearly.mean(dim='clim_year').astype('float32')
            else:
                comp_avg_season = None
            
            if base_avg_season is None or comp_avg_season is None:
                print(f"  Warning: Missing data for {base_src} vs {comp_src}")
                results.append(result)
                continue
            
            # LAZY: Climatology difference
            data_avg_lazy = (minus_scaling * (comp_avg_season - base_avg_season)).astype('float32')
            lazy_computations.append(data_avg_lazy)
            computation_map.append((ii, 'data_avg_plot'))
            
            # LAZY: Trends
            if trend_calc and base_fit is not None:
                trend_base_lazy = compute_trend_lazy(base_fit, fit_scaling, relative_precip, base_avg_season, var)
                
                if trend_regrid:
                    # Get native comparison fit data from cache
                    comp_fit_native = seasonal_slices.get((comp_src, freq, month_key, 'fit'))
                    if relative_precip and var == 'P':
                        comp_yearly_native = seasonal_slices.get((comp_src, freq, month_key, 'yearly'))
                        comp_avg_native = comp_yearly_native.mean(dim='clim_year').astype('float32') if comp_yearly_native is not None else None
                    else:
                        comp_avg_native = None
                    
                    trend_comp_native = compute_trend_lazy(comp_fit_native, fit_scaling, relative_precip, comp_avg_native, var)
                    if trend_comp_native is not None and trend_base_lazy is not None:
                        # Regrid native trend (2D field) - this needs compute
                        trend_comp_lazy = regridders[pair_key](trend_comp_native, output_chunks=TARGET_CHUNKS).astype('float32')
                        trend_diff_lazy = (minus_scaling * (trend_comp_lazy - trend_base_lazy)).astype('float32')
                        lazy_computations.append(trend_diff_lazy)
                        computation_map.append((ii, 'trend_plot'))
                else:
                    # Use regridded fit data
                    if comp_fit is not None and trend_base_lazy is not None:
                        trend_comp_lazy = compute_trend_lazy(comp_fit, fit_scaling, relative_precip, comp_avg_season, var)
                        if trend_comp_lazy is not None:
                            trend_diff_lazy = (minus_scaling * (trend_comp_lazy - trend_base_lazy)).astype('float32')
                            lazy_computations.append(trend_diff_lazy)
                            computation_map.append((ii, 'trend_plot'))
            
            # LAZY: Correlation
            if corr_calc:
                if corr_freq_list[ii] == 'Daily' and base_raw is not None and comp_raw is not None:
                    corr_lazy = compute_correlation_lazy(base_raw, comp_raw, 'time')
                    lazy_computations.append(corr_lazy)
                    computation_map.append((ii, 'corr_plot'))
                elif corr_freq_list[ii] == 'Yearly' and base_yearly is not None and comp_yearly is not None:
                    corr_lazy = compute_correlation_lazy(base_yearly, comp_yearly, 'clim_year')
                    lazy_computations.append(corr_lazy)
                    computation_map.append((ii, 'corr_plot'))
                elif corr_freq_list[ii] == 'Monthly' and base_monthly is not None and comp_monthly is not None:
                    corr_lazy = compute_correlation_lazy(base_monthly, comp_monthly, 'time')
                    lazy_computations.append(corr_lazy)
                    computation_map.append((ii, 'corr_plot'))
            
            # LAZY: RMSE
            if rmse_calc:
                if rmse_freq_list[ii] == 'Daily' and base_raw is not None and comp_raw is not None:
                    rmse_lazy = compute_rmse_lazy(base_raw, comp_raw, 'time')
                    lazy_computations.append(rmse_lazy)
                    computation_map.append((ii, 'rmse_plot'))
                elif rmse_freq_list[ii] == 'Yearly' and base_yearly is not None and comp_yearly is not None:
                    rmse_lazy = compute_rmse_lazy(base_yearly, comp_yearly, 'clim_year')
                    lazy_computations.append(rmse_lazy)
                    computation_map.append((ii, 'rmse_plot'))
                elif rmse_freq_list[ii] == 'Monthly' and base_monthly is not None and comp_monthly is not None:
                    rmse_lazy = compute_rmse_lazy(base_monthly, comp_monthly, 'time')
                    lazy_computations.append(rmse_lazy)
                    computation_map.append((ii, 'rmse_plot'))
            
            # LAZY: Std mask reference
            if std_mask_ref_list[ii] is not None and base_fit is not None and comp_fit is not None:
                if std_mask_ref_list[ii] == 'Pool':
                    ref_type = 'Pool'
                elif std_mask_ref_list[ii] == base_src:
                    ref_type = 'base'
                elif std_mask_ref_list[ii] == comp_src:
                    ref_type = 'compare'
                else:
                    ref_type = None
                
                if ref_type:
                    std_ref_lazy = compute_std_ref_lazy(base_fit, comp_fit, ref_type)
                    if std_ref_lazy is not None:
                        lazy_computations.append(std_ref_lazy)
                        computation_map.append((ii, 'std_ref'))  # Need to process further after compute
        
        results.append(result)
    
    # ==========================================================================
    # BATCH COMPUTE - Execute all lazy computations in parallel
    # ==========================================================================
    if lazy_computations:
        print(f"    Batch computing {len(lazy_computations)} operations...")
        computed_results = dask.compute(*lazy_computations)
        
        # Assign computed results back to result dicts
        for idx, (run_idx, key) in enumerate(computation_map):
            if key == 'std_ref':
                # Process std_ref into mask_std
                std_ref = computed_results[idx]
                data_avg = results[run_idx]['data_avg_plot']
                if data_avg is not None and std_ref is not None:
                    std_dir = std_dir_list[run_idx]
                    if std_dir == 'Greater':
                        results[run_idx]['mask_std'] = (np.abs(data_avg) > std_ref).fillna(False).astype('int8')
                    else:
                        results[run_idx]['mask_std'] = (np.abs(data_avg) < std_ref).fillna(False).astype('int8')
            elif key == 'trend_plot':
                # Assign coordinates from data_avg_plot
                trend_result = computed_results[idx]
                data_avg = results[run_idx].get('data_avg_plot')
                if data_avg is not None and trend_result is not None:
                    results[run_idx][key] = trend_result.assign_coords(
                        latitude=data_avg['latitude'],
                        longitude=data_avg['longitude'])
                else:
                    results[run_idx][key] = trend_result
            else:
                results[run_idx][key] = computed_results[idx]
    
    all_results[month_key] = results

print("\n" + "=" * 60)
print("Processing complete!")
print("=" * 60)

#%% ============================================================================
#   PHASE 5: AREA CONTOUR COMPUTATION
#   ============================================================================

all_area_contours = {}

for month_key in months_dict.keys():
    results = all_results[month_key]
    area_contours = []
    
    for ii in range(n_runs):
        if data_base_list[ii] is None:
            continue
        
        contour_info = {
            'mask_area': None,
            'lat_b_area': None,
            'lon_b_area': None,
            'lats_area_cont': None,
            'lons_area_cont': None,
            'proj_area_cont': None
        }
        
        has_area_bounds = (isinstance(lats_area, (list, tuple)) and len(lats_area) == 2 and
                          isinstance(lons_area, (list, tuple)) and len(lons_area) == 2)
        
        if has_area_bounds and grid_contour:
            contour_area_full = xr.full_like(results[ii]['data_avg_plot'], fill_value=1.0)
            
            if 'rlat' in contour_area_full.dims and 'rlon' in contour_area_full.dims:
                lat2d = contour_area_full['latitude']
                lon2d = contour_area_full['longitude']
                dim_lat, dim_lon = 'rlat', 'rlon'
            else:
                lat1d = contour_area_full['latitude']
                lon1d = contour_area_full['longitude']
                lat2d, lon2d = xr.broadcast(lat1d, lon1d)
                dim_lat, dim_lon = 'latitude', 'longitude'
            
            contour_area = subset_space(
                contour_area_full, lat2d, lon2d, lats_area, lons_area,
                dim_lat, dim_lon, rotpole_sel=proj_area,
                rotpole_native=proj_cfg.get(data_base_list[ii], ccrs.PlateCarree()))
            
            contour_info['mask_area'] = np.isfinite(contour_area.values)
            
            contour_area_bounds = grid_with_bounds(
                contour_area, rotpole_native=proj_cfg.get(data_base_list[ii], ccrs.PlateCarree()))
            contour_info['lon_b_area'] = contour_area_bounds['lon_b']
            contour_info['lat_b_area'] = contour_area_bounds['lat_b']
        
        if has_area_bounds and true_contour:
            contour_info['lats_area_cont'] = lats_area
            contour_info['lons_area_cont'] = lons_area
            contour_info['proj_area_cont'] = proj_area
        
        area_contours.append(contour_info)
    
    all_area_contours[month_key] = area_contours

#%% ============================================================================
#   PLOTTING HELPER FUNCTIONS
#   ============================================================================

def get_cbar_sizing(use_shared_cbar, cbar_orientation, panel_width, panel_height, 
                    fig_width, fig_height, shared_label=False):
    """Calculate colorbar sizing parameters."""
    title_fontsize = max(40, int(panel_height * 7))
    
    cbar_thickness_pct = 0.1 if use_shared_cbar else 0.07
    if cbar_orientation == 'horizontal':
        cbar_thickness = panel_height * cbar_thickness_pct / fig_height
    else:
        cbar_thickness = panel_width * cbar_thickness_pct / fig_width
    
    cbar_pad = 0.028 if cbar_orientation == 'horizontal' else 0.015
    cbar_label_pad = 16 if use_shared_cbar else 6
    
    if use_shared_cbar:
        cbar_tick_size = max(35, int(panel_height * 7))
    else:
        cbar_tick_size = max(25, int(panel_height * 5))
    
    if use_shared_cbar or shared_label:
        cbar_label_size = max(48, int(panel_height * 9))
    else:
        cbar_label_size = max(35, int(panel_height * 7))
    
    return {
        'title_fontsize': title_fontsize,
        'cbar_thickness': cbar_thickness,
        'cbar_pad': cbar_pad,
        'cbar_label_pad': cbar_label_pad,
        'cbar_tick_size': cbar_tick_size,
        'cbar_label_size': cbar_label_size
    }

def plot_spatial_grid(
    all_results, all_area_contours, data_key, crange, label, cmap, extreme_colors,
    shared_cbar, shared_cbar_label, cbar_ticks_shared, cbar_ticks_sep, cbar_ticks_num,
    save_name, months_dict, n_runs, swap_rows_cols, cbar_orientation,
    panel_width, panel_height, proj_plot, plot_lons, plot_lats, cut_boundaries,
    show_col_titles, show_row_titles, std_mask_ref_list=None, data_compare_list=None,
    extendfrac=0.05, length_scale=0.9
):
    """Unified plotting function for spatial grids."""
    # Disable interactive mode during plotting for speed
    was_interactive = plt.isinteractive()
    plt.ioff()
    
    month_keys = list(months_dict.keys())
    n_months = len(months_dict)
    
    crange_is_dict = isinstance(crange, dict)
    use_shared_cbar = (not crange_is_dict) or shared_cbar
    
    if crange_is_dict:
        vmins = [float(v[0]) for v in crange.values()]
        vmaxs = [float(v[1]) for v in crange.values()]
        crange_global = [min(vmins), max(vmaxs)]
    else:
        crange_global = crange
    
    if swap_rows_cols:
        n_rows, n_cols = n_runs, n_months
    else:
        n_rows, n_cols = n_months, n_runs
    
    fig_width = panel_width * n_cols
    fig_height = panel_height * n_rows
    
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_width, fig_height),
        constrained_layout=True, subplot_kw={'projection': proj_plot},
        sharex=True, sharey=True)
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    run_titles = None
    if month_keys:
        run_titles = [r.get('title', '') for r in all_results[month_keys[0]]]
    
    cbar_sizing = get_cbar_sizing(
        use_shared_cbar, cbar_orientation, panel_width, panel_height, 
        fig_width, fig_height, shared_label=shared_cbar_label)
    
    if use_shared_cbar:
        cbar_n_ticks = cbar_ticks_shared
    else:
        cbar_n_ticks = cbar_ticks_sep if not isinstance(cbar_ticks_sep, list) else cbar_ticks_sep[0]
    
    mesh_ref = None
    datasets_all = []
    month_axes_groups = {mk: [] for mk in month_keys}
    month_data_groups = {mk: [] for mk in month_keys}
    month_mesh_groups = {mk: None for mk in month_keys}
    
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            if swap_rows_cols:
                run_idx, month_idx = row_idx, col_idx
            else:
                month_idx, run_idx = row_idx, col_idx
            
            month_key = month_keys[month_idx]
            res = all_results[month_key][run_idx]
            area_contours = all_area_contours[month_key][run_idx]
            
            if res.get(data_key) is None:
                continue
            
            ax = axes[row_idx, col_idx]
            
            if show_col_titles and row_idx == 0:
                title = month_key if swap_rows_cols else res.get('title', None)
            else:
                title = None
            
            if use_shared_cbar:
                crange_this = crange_global
            elif crange_is_dict:
                crange_this = crange[month_key]
            else:
                crange_this = crange
            
            data_plot = res[data_key]
            mesh, _ = plot_map(
                fig, ax, data_plot, data_plot.longitude.values, data_plot.latitude.values,
                crange=crange_this, label=label, cmap=cmap, extreme_colors=extreme_colors,
                show_x_ticks=True, show_y_ticks=True, y_ticks_num=False, y_ticks=5,
                show_y_labels=(col_idx == 0), x_ticks_num=False, x_ticks=10,
                show_x_labels=(row_idx == n_rows - 1), tick_size=16,
                extent=[*plot_lons, *plot_lats], proj=proj_plot, rotated_grid=cut_boundaries,
                title=title, title_size=cbar_sizing['title_fontsize'],
                lats_area=area_contours['lats_area_cont'],
                lons_area=area_contours['lons_area_cont'],
                proj_area=area_contours['proj_area_cont'],
                mask_area=area_contours['mask_area'],
                lat_b_area=area_contours['lat_b_area'],
                lon_b_area=area_contours['lon_b_area'],
                show_plot=False, add_colorbar=False)
            
            if mesh_ref is None:
                mesh_ref = mesh
            
            datasets_all.append(data_plot)
            month_axes_groups[month_key].append(ax)
            month_data_groups[month_key].append(data_plot)
            if month_mesh_groups[month_key] is None:
                month_mesh_groups[month_key] = mesh
            
            # Hatching for std_mask
            if std_mask_ref_list is not None and data_compare_list is not None:
                if std_mask_ref_list[run_idx] is not None and data_compare_list[run_idx] is not None:
                    if res.get('mask_std') is not None:
                        ax.contourf(
                            data_plot.longitude.values, data_plot.latitude.values,
                            res['mask_std'], levels=[0.5, 1.5], colors='none',
                            hatches=['///'], transform=ccrs.PlateCarree(), zorder=50)
    
    # Row titles
    if show_row_titles:
        for row_idx in range(n_rows):
            if swap_rows_cols:
                row_title = run_titles[row_idx] if run_titles else f'Run {row_idx + 1}'
            else:
                row_title = month_keys[row_idx]
            
            axes[row_idx, 0].text(
                -0.25, 0.5, row_title, transform=axes[row_idx, 0].transAxes,
                rotation=90, va='center', ha='center',
                fontsize=cbar_sizing['title_fontsize'], fontweight='bold')
    
    # Colorbars
    if mesh_ref is not None:
        if use_shared_cbar:
            shared_colorbar(
                fig=fig, axes=axes.ravel(), mesh=mesh_ref, datasets=datasets_all,
                crange=crange_global, label=label, orientation=cbar_orientation,
                c_ticks=cbar_n_ticks, c_ticks_num=cbar_ticks_num,
                tick_labelsize=cbar_sizing['cbar_tick_size'],
                labelsize=cbar_sizing['cbar_label_size'],
                pad=cbar_sizing['cbar_pad'], thickness=cbar_sizing['cbar_thickness'],
                label_pad=cbar_sizing['cbar_label_pad'],
                extend_into_axes=True)  # Shared cbar: total size matches axes
        else:
            cbar_axes_list = []
            for kk, month_key in enumerate(month_keys):
                mesh_k = month_mesh_groups[month_key]
                if mesh_k is None:
                    continue
                
                label_k = '' if shared_cbar_label else label
                crange_k = crange[month_key] if crange_is_dict else crange
                
                if isinstance(cbar_ticks_sep, list):
                    cbar_n_ticks_k = cbar_ticks_sep[kk] if kk < len(cbar_ticks_sep) else cbar_ticks_sep[-1]
                else:
                    cbar_n_ticks_k = cbar_ticks_sep
                
                cbar_k = shared_colorbar(
                    fig=fig, axes=month_axes_groups[month_key], mesh=mesh_k,
                    datasets=month_data_groups[month_key], crange=crange_k,
                    label=label_k, orientation=cbar_orientation,
                    c_ticks=cbar_n_ticks_k, c_ticks_num=cbar_ticks_num,
                    tick_labelsize=cbar_sizing['cbar_tick_size'],
                    labelsize=cbar_sizing['cbar_label_size'],
                    pad=cbar_sizing['cbar_pad'], thickness=cbar_sizing['cbar_thickness'],
                    label_pad=cbar_sizing['cbar_label_pad'],
                    extendfrac=extendfrac, length_scale=length_scale)
                cbar_axes_list.append(cbar_k.ax)
            
            if shared_cbar_label and cbar_axes_list:
                add_shared_cbar_label(
                    fig, cbar_axes_list, label, orientation=cbar_orientation,
                    fontsize=cbar_sizing['cbar_label_size'], pad=0.008)
    
    if save_name is not None:
        out = Path.home() / 'KNMIproject' / 'RegionalTrends' / 'Main' / 'figuresproposal' / (save_name + '.pdf')
        plt.savefig(str(out), format='pdf', bbox_inches='tight')
    
    # Restore interactive mode if it was on
    if was_interactive:
        plt.ion()
    
    plt.show()

#%% ============================================================================
#   PLOTTING CALLS
#   ============================================================================

if plot_climatology:
    plot_spatial_grid(
        all_results=all_results, all_area_contours=all_area_contours,
        data_key='data_avg_plot', crange=avg_crange, label=labels['avg'],
        cmap=cmaps['avg_cmap'], extreme_colors=cmaps['avg_extreme'],
        shared_cbar=shared_cbar_avg, shared_cbar_label=shared_cbar_label_avg,
        cbar_ticks_shared=cbar_ticks_avg_shared, cbar_ticks_sep=cbar_ticks_avg_sep,
        cbar_ticks_num=cbar_ticks_num_avg, save_name=save_name_avg,
        months_dict=months_dict, n_runs=n_runs, swap_rows_cols=swap_rows_cols,
        cbar_orientation=cbar_orientation, panel_width=panel_width,
        panel_height=panel_height, proj_plot=proj_plot, plot_lons=plot_lons,
        plot_lats=plot_lats, cut_boundaries=cut_boundaries,
        show_col_titles=show_col_titles, show_row_titles=show_row_titles,
        std_mask_ref_list=std_mask_ref_list, data_compare_list=data_compare_list)

if trend_calc:
    plot_spatial_grid(
        all_results=all_results, all_area_contours=all_area_contours,
        data_key='trend_plot', crange=trend_crange, label=labels['trend'],
        cmap=cmaps['trend_cmap'], extreme_colors=cmaps['trend_extreme'],
        shared_cbar=shared_cbar_trend, shared_cbar_label=shared_cbar_label_trend,
        cbar_ticks_shared=cbar_ticks_trend_shared, cbar_ticks_sep=cbar_ticks_trend_sep,
        cbar_ticks_num=cbar_ticks_num_trend, save_name=save_name_trend,
        months_dict=months_dict, n_runs=n_runs, swap_rows_cols=swap_rows_cols,
        cbar_orientation=cbar_orientation, panel_width=panel_width,
        panel_height=panel_height, proj_plot=proj_plot, plot_lons=plot_lons,
        plot_lats=plot_lats, cut_boundaries=cut_boundaries,
        show_col_titles=show_col_titles, show_row_titles=show_row_titles,
        length_scale=0.9)

if corr_calc:
    plot_spatial_grid(
        all_results=all_results, all_area_contours=all_area_contours,
        data_key='corr_plot', crange=corr_crange, label=labels['corr'],
        cmap=cmaps['corr_cmap'], extreme_colors=cmaps['corr_extreme'],
        shared_cbar=shared_cbar_corr, shared_cbar_label=shared_cbar_label_corr,
        cbar_ticks_shared=cbar_ticks_corr_shared, cbar_ticks_sep=cbar_ticks_corr_sep,
        cbar_ticks_num=cbar_ticks_num_corr, save_name=save_name_corr,
        months_dict=months_dict, n_runs=n_runs, swap_rows_cols=swap_rows_cols,
        cbar_orientation=cbar_orientation, panel_width=panel_width,
        panel_height=panel_height, proj_plot=proj_plot, plot_lons=plot_lons,
        plot_lats=plot_lats, cut_boundaries=cut_boundaries,
        show_col_titles=show_col_titles, show_row_titles=show_row_titles)

if rmse_calc:
    plot_spatial_grid(
        all_results=all_results, all_area_contours=all_area_contours,
        data_key='rmse_plot', crange=rmse_crange, label=labels['rmse'],
        cmap=cmaps['rmse_cmap'], extreme_colors=cmaps['rmse_extreme'],
        shared_cbar=shared_cbar_rmse, shared_cbar_label=shared_cbar_label_rmse,
        cbar_ticks_shared=cbar_ticks_rmse_shared, cbar_ticks_sep=cbar_ticks_rmse_sep,
        cbar_ticks_num=cbar_ticks_num_rmse, save_name=save_name_rmse,
        months_dict=months_dict, n_runs=n_runs, swap_rows_cols=swap_rows_cols,
        cbar_orientation=cbar_orientation, panel_width=panel_width,
        panel_height=panel_height, proj_plot=proj_plot, plot_lons=plot_lons,
        plot_lats=plot_lats, cut_boundaries=cut_boundaries,
        show_col_titles=show_col_titles, show_row_titles=show_row_titles)

#%%
