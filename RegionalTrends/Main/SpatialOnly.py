#%% Imports

# Standard libraries
import numpy as np
import pandas as pd
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
from RegionalTrends.Helpers.ProcessNetCDF import subset_space, is_monthly_time

from RegionalTrends.Helpers import ProcessVar
reload(ProcessVar)
from RegionalTrends.Helpers.ProcessVar import load_var

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

#%% User inputs

# Main arguments
var = 'Rnet'
file_freq = 'Monthly'
proc_type = 'Mean'
# data_base = ['ERA5', 'Eobs', 'ERA5']
# data_compare = ['RACMO2.4', 'RACMO2.4', 'Eobs']
data_base = ['ERA5', 'ERA5L', 'RACMO2.3', 'RACMO2.4']
data_compare = None
save_name_base = None#'AllSeasons19802020'

# Data selection arguments
months_dict = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
years = [1980, 2020] 
lats = [37.7, 63.3]
lons = [-13.3, 22.3]
proj_sel = 'RACMO2.4'
land_only = False
trim_border = None

# Spatial plotting arguments
plot_climatology = True
avg_crange = {'DJF': [-20, 50], 'MAM': [20, 200], 'JJA': [50, 300], 'SON': [0, 80]}
# avg_crange = [10, 50]
std_mask_ref = data_base
std_dir = 'Lesser'
shared_cbar_avg = False
shared_cbar_label_avg = True
short_cbar_label_avg = False
cbar_ticks_avg_shared = 5
cbar_ticks_avg_sep = [20, 40, 50, 20]
cbar_ticks_num_avg = False

# Trend plotting arguments
trend_calc = True
trend_crange = {'DJF': [-2, 2], 'MAM': [-4, 4], 'JJA': [-5, 5], 'SON': [-2, 2]}
# trend_crange = [-1.5,1.5]
trend_regrid = False
fit_against_gmst = False
shared_cbar_trend = False
shared_cbar_label_trend = True
short_cbar_label_trend = False
cbar_ticks_trend_shared = 0.5
cbar_ticks_trend_sep = [1, 2, 2.5, 1]
cbar_ticks_num_trend = False

# Correlation plotting arguments
corr_calc = False
corr_freq = 'Monthly'
corr_crange = [0.8, 1]
corr_cmap_type = None
shared_cbar_corr = True
shared_cbar_label_corr = True
short_cbar_label_corr = False
cbar_ticks_corr_shared = 0.05
cbar_ticks_corr_sep = 6
cbar_ticks_num_corr = False

# RMSE plotting arguments
rmse_calc = False
rmse_freq = 'Monthly'
rmse_crange = [0, 1]
shared_cbar_rmse = True
shared_cbar_label_rmse = True
short_cbar_label_rmse = False
cbar_ticks_rmse_shared = 0.2
cbar_ticks_rmse_sep = 0.2
cbar_ticks_num_rmse = False

# Additional plotting arguments
proj_plot = 'RACMO2.4'
plot_lats = [38, 63]
plot_lons = [-13, 22]
switch_sign = False
cut_boundaries = False
cbar_orientation = 'horizontal'
swap_rows_cols = True
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
true_contour = False
grid_contour = False

# Other arguments
rolling_mean_var = False
rolling_mean_years = 3
min_periods = 1

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
    if fit_against_gmst:
        return {'unit': '°C GMST', 'scaling': 1, 'x_label': r'$\Delta$GMST (°C)'}
    return {'unit': 'decade', 'scaling': 10, 'x_label': 'Year'}

def get_trend_unit(var, var_units_cfg, fit_unit):
    if var_units_cfg[var] == '':
        return 'per' + fit_unit
    return var_units_cfg[var] + ' / ' + fit_unit

def get_var_symbol(var, var_symbol_cfg):
    raw_sym = var_symbol_cfg[var]
    return raw_sym if '$' in raw_sym else r'$' + raw_sym + r'$'

def get_labels(var, var_name_cfg, var_symbol_cfg, var_units_cfg, trend_unit, 
               short_avg, short_trend, short_corr, short_rmse, has_comparison):
    sym = get_var_symbol(var, var_symbol_cfg)
    full_var = var_name_cfg[var] + ' ' + sym
    if var_name_cfg[var] == var_symbol_cfg[var]:
        full_var = sym
    
    avg_label = (sym if short_avg else full_var) + ' (' + var_units_cfg[var] + ')'
    if has_comparison:
        avg_label = r'$\Delta$' + avg_label if short_avg else 'Difference in ' + avg_label
    
    trend_label = (sym if short_trend else full_var) + ' trend (' + trend_unit + ')'
    corr_label = (sym if short_corr else full_var) + ' correlation'
    rmse_label = (sym if short_rmse else full_var) + ' RMSE (' + var_units_cfg[var] + ')'
    
    return {'avg': avg_label, 'trend': trend_label, 'corr': corr_label, 'rmse': rmse_label}

def setup_colormaps(var, var_colors_cfg, has_comparison, cmap_type, n_disc_colors, corr_cmap_type):
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
    return param if isinstance(param, list) else [param]*n

#%% Run configuration

# Set up basic parameters
fit_cfg = get_fit_config(fit_against_gmst)
fit_unit, fit_scaling = fit_cfg['unit'], fit_cfg['scaling']
trend_unit = get_trend_unit(var, var_units_cfg, fit_unit)

n_runs = len(data_base) if isinstance(data_base, list) else 1
has_comparison = data_compare is not None

# Convert to lists
data_base_list = ensure_list(data_base, n_runs)
data_compare_list = ensure_list(data_compare, n_runs)
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
#   PHASE 1: LOAD ALL DATA ONCE
#   ============================================================================

print('='*60)
print('PHASE 1: Loading all data sources')
print('='*60)

def get_unique_sources():
    sources = set()
    for ii in range(n_runs):
        if data_base_list[ii] is not None:
            sources.add(data_base_list[ii])
        if data_compare_list[ii] is not None:
            sources.add(data_compare_list[ii])
        if std_mask_ref_list[ii] not in (None, 'Pool', data_base_list[ii], data_compare_list[ii]):
            sources.add(std_mask_ref_list[ii])
    return sources

def needs_extended_years(months_dict):
    for months in months_dict.values():
        month_start = int(months[0])
        month_end = int(months[-1])
        if month_start > month_end:  # e.g., DJF: [12, 1, 2]
            return True
    return False

def load_source_data(data_source):

    print(f'  Loading: {data_source} ({file_freq})')
    
    # Extend years if needed for DJF-style seasons
    years_load = list(years)
    if needs_extended_years(months_dict):
        years_load[0] = years[0] - 1
    
    # Load the variable
    data_raw = load_var(
        var=var,
        data_source=data_source,
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
    
    # Compute monthly from raw if file_freq is not Monthly
    if file_freq == 'Monthly' or is_monthly_time(data_raw['time']):
        data_monthly = data_raw
    else:
        # Aggregate to monthly based on proc_type
        if proc_type == 'Max':
            data_monthly = data_raw.resample(time='MS').max('time')
        elif proc_type == 'Min':
            data_monthly = data_raw.resample(time='MS').min('time')
        else:
            data_monthly = data_raw.resample(time='MS').mean('time')
    
    return {
        'raw': data_raw,
        'monthly': data_monthly.astype('float32'),
    }

source_data = {}
for src in get_unique_sources():
    source_data[src] = load_source_data(src)

print(f'Loaded {len(source_data)} unique data sources\n')

#%% ============================================================================
#   PHASE 2: BUILD REGRIDDERS AND REGRID FULL TIME SERIES
#   ============================================================================

print('='*60)
print('PHASE 2: Building regridders and regridding')
print('='*60)

def get_regrid_method(var):
    return 'conservative_normed' if var == 'P' else 'bilinear'

def create_regridder(src_grid, trg_grid, method):
    return xe.Regridder(src_grid, trg_grid, method, unmapped_to_nan=True)

regridders = {}
regridded_data = {}

TARGET_CHUNKS = {'latitude': 100, 'longitude': 100}

for ii in range(n_runs):
    base_src = data_base_list[ii]
    comp_src = data_compare_list[ii]
    
    if base_src is None or comp_src is None:
        continue
    
    pair_key = (base_src, comp_src)
    if pair_key in regridders:
        continue  # Already built
    
    print(f'  Building regridder: {comp_src} -> {base_src}')
    
    base_data = source_data[base_src]
    comp_data = source_data[comp_src]
    
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
    
    # Regrid monthly data once
    regridder = regridders[pair_key]
    print(f'    Regridding monthly time series...')
    regridded_data[(pair_key, 'monthly')] = regridder(comp_data['monthly'], 
                                                       output_chunks=TARGET_CHUNKS).astype('float32')
    
    # Optionally regrid raw if needed for daily correlation/RMSE
    needs_raw = any((corr_freq_list[jj] == 'Daily' or rmse_freq_list[jj] == 'Daily')
                    and data_base_list[jj] == base_src and data_compare_list[jj] == comp_src
                    for jj in range(n_runs))
    
    if needs_raw:
        print(f'    Regridding raw (daily) time series...')
        regridded_data[(pair_key, 'raw')] = regridder(comp_data['raw'], 
                                                       output_chunks=TARGET_CHUNKS).astype('float32')

# Build regridders for std_mask_ref sources (when different from base and compare)
for ii in range(n_runs):
    base_src = data_base_list[ii]
    std_ref_src = std_mask_ref_list[ii]
    
    # Skip if no base, no std_ref, or std_ref is Pool/base/compare
    if base_src is None or std_ref_src is None:
        continue
    if std_ref_src in ('Pool', base_src, data_compare_list[ii]):
        continue
    
    pair_key = (base_src, std_ref_src)
    if pair_key in regridders:
        continue  # Already built
    
    print(f'  Building regridder for std_ref: {std_ref_src} -> {base_src}')
    
    base_data = source_data[base_src]
    std_ref_data = source_data[std_ref_src]
    
    base_avg_temp = base_data['monthly'].mean('time')
    std_ref_avg_temp = std_ref_data['monthly'].mean('time')
    
    method = get_regrid_method(var)
    if var == 'P':
        trg_grid = grid_with_bounds(base_avg_temp, 
                                    rotpole_native=proj_cfg.get(base_src, ccrs.PlateCarree()))
        src_grid = grid_with_bounds(std_ref_avg_temp, 
                                    rotpole_native=proj_cfg.get(std_ref_src, ccrs.PlateCarree()))
    else:
        trg_grid = base_avg_temp
        src_grid = std_ref_avg_temp
    
    regridders[pair_key] = create_regridder(src_grid, trg_grid, method)
    
    # Regrid monthly data for std_ref
    regridder = regridders[pair_key]
    print(f'    Regridding std_ref monthly time series...')
    regridded_data[(pair_key, 'monthly')] = regridder(std_ref_data['monthly'], 
                                                       output_chunks=TARGET_CHUNKS).astype('float32')

print(f'Built {len(regridders)} regridders\n')

#%% ============================================================================
#   PHASE 3: CLIMATE YEAR SLICING FUNCTIONS
#   ============================================================================

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

#%% ============================================================================
#   PHASE 4: COMPUTE METRICS FOR EACH SEASON
#   ============================================================================

print('='*60)
print('PHASE 4: Computing metrics for each season')
print('='*60)

def compute_correlation_lazy(x, y, dim='time'):
    x, y = xr.align(x, y, join='inner')
    
    time_chunk = 365 if len(x[dim]) > 1000 else -1
    x = x.chunk({dim: time_chunk, 'latitude': 100, 'longitude': 100})
    y = y.chunk({dim: time_chunk, 'latitude': 100, 'longitude': 100})
    
    valid = np.isfinite(x) & np.isfinite(y)
    xv = x.where(valid)
    yv = y.where(valid)
    
    n = valid.sum(dim)
    dx = xv - xv.mean(dim)
    dy = yv - yv.mean(dim)
    cov = (dx*dy).mean(dim)
    sx = xv.std(dim)
    sy = yv.std(dim)
    
    corr = (cov / (sx*sy)).where((n >= 2) & (sx > 0) & (sy > 0))
    return corr.assign_coords(latitude=x['latitude'], longitude=x['longitude']).astype('float32')

def compute_rmse_lazy(x, y, dim='time'):
    x, y = xr.align(x, y, join='inner')
    
    time_chunk = 365 if len(x[dim]) > 1000 else -1
    x = x.chunk({dim: time_chunk, 'latitude': 100, 'longitude': 100})
    y = y.chunk({dim: time_chunk, 'latitude': 100, 'longitude': 100})
    
    valid = np.isfinite(x) & np.isfinite(y)
    err = y.where(valid) - x.where(valid)
    rmse = np.sqrt((err**2).mean(dim))
    return rmse.assign_coords(latitude=x['latitude'], longitude=x['longitude']).astype('float32')

def compute_trend_lazy(fit_data, fit_scaling):
        
    fits = fit_data.polyfit(dim='fit_against', deg=1, skipna=True)
    slope = fits.polyfit_coefficients.sel(degree=1)
    trend = (slope*fit_scaling).astype('float32')
    
    return trend

def compute_std_ref_lazy(base_fit, comp_fit_reg, std_ref_type):

    if std_ref_type == 'Pool':
        xb, yc = xr.align(base_fit, comp_fit_reg, join='inner')
        valid = np.isfinite(xb) & np.isfinite(yc)
        xb = xb.where(valid).chunk({'fit_against': -1, 'latitude': 100, 'longitude': 100})
        yc = yc.where(valid).chunk({'fit_against': -1, 'latitude': 100, 'longitude': 100})
        
        fits_xb = xb.polyfit(dim='fit_against', deg=1, skipna=True)
        trend_xb = xr.polyval(xb['fit_against'], fits_xb.polyfit_coefficients)
        resid_xb = xb - trend_xb
        
        fits_yc = yc.polyfit(dim='fit_against', deg=1, skipna=True)
        trend_yc = xr.polyval(yc['fit_against'], fits_yc.polyfit_coefficients)
        resid_yc = yc - trend_yc
        
        sx = resid_xb.std('fit_against', ddof=1)
        sy = resid_yc.std('fit_against', ddof=1)
        std_ref = np.sqrt(0.5 * (sx**2 + sy**2))
        return std_ref.where(valid.sum('fit_against') >= 3)
    else:
        # std_ref_type is 'base' or 'compare'
        ref_fit = base_fit if std_ref_type == 'base' else comp_fit_reg
        _, ref_fit = xr.align(base_fit, ref_fit, join='inner')
        ref_fit = ref_fit.chunk({'fit_against': -1, 'latitude': 100, 'longitude': 100})
        n_ref = np.isfinite(ref_fit).sum('fit_against')
        fits_ref = ref_fit.polyfit(dim='fit_against', deg=1, skipna=True)
        trend_ref = xr.polyval(ref_fit['fit_against'], fits_ref.polyfit_coefficients)
        resid_ref = ref_fit - trend_ref
        return resid_ref.std('fit_against', ddof=1).where(n_ref >= 3)

def get_title(base_src, comp_src, data_sources, switch_sign):

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
# PRE-COMPUTE SEASONAL SLICES
# =============================================================================
print('\n  Pre-computing seasonal slices...')

seasonal_slices = {}  # Cache: (source, month_key, data_type)

for month_key, months in months_dict.items():
    for src in source_data.keys():
        data = source_data[src]
        
        # Cache filtered monthly data
        monthly_filtered = filter_by_season(data['monthly'], months, years)
        seasonal_slices[(src, month_key, 'monthly')] = monthly_filtered
        
        # Cache yearly aggregates (used for correlation/RMSE with Yearly freq)
        yearly_agg = compute_seasonal_yearly(data['monthly'], months, years, proc_type)
        seasonal_slices[(src, month_key, 'yearly')] = yearly_agg
        
        # Cache fit data (used for trends and std_mask)
        fit_data = compute_fit_data(yearly_agg, fit_against_gmst, 
                                    rolling_mean_var, rolling_mean_years, min_periods)
        seasonal_slices[(src, month_key, 'fit')] = fit_data
        
        # Cache raw (for daily correlation/RMSE)
        raw_filtered = filter_by_season(data['raw'], months, years)
        seasonal_slices[(src, month_key, 'raw')] = raw_filtered

# Also cache seasonal slices for REGRIDDED comparison data
for pair_key in regridded_data.keys():
    base_src, comp_src = pair_key[0]  # Unpack (base, comp)
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
            fit_data = compute_fit_data(yearly_agg, fit_against_gmst, 
                                        rolling_mean_var, rolling_mean_years, min_periods)
            seasonal_slices[(pair_key, month_key, 'fit')] = fit_data
    
    elif data_type == 'raw':
        comp_raw_reg = regridded_data[pair_key]
        for month_key, months in months_dict.items():
            raw_filtered = filter_by_season(comp_raw_reg, months, years)
            seasonal_slices[(pair_key, month_key, 'raw')] = raw_filtered

print(f'  Cached {len(seasonal_slices)} seasonal slices')

# =============================================================================
# MAIN PROCESSING LOOP - USING CACHED SLICES + BATCH COMPUTE
# =============================================================================
all_results = {}

for month_key, months in months_dict.items():
    print(f'\nProcessing: {month_key}')
    results = []
    
    # Collect all lazy computations for this month, then batch compute
    lazy_computations = []
    computation_map = []  # Track which result each computation belongs to
    
    for ii in range(n_runs):
        base_src = data_base_list[ii]
        comp_src = data_compare_list[ii]
        
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
        base_yearly = seasonal_slices.get((base_src, month_key, 'yearly'))
        base_fit = seasonal_slices.get((base_src, month_key, 'fit'))
        base_monthly = seasonal_slices.get((base_src, month_key, 'monthly'))
        base_raw = seasonal_slices.get((base_src, month_key, 'raw'))
        
        # Compute base seasonal average (lazy)
        base_avg_season = base_yearly.mean(dim='clim_year').astype('float32')
        
        if comp_src is None:
            # Single source mode
            lazy_computations.append(base_avg_season)
            computation_map.append((ii, 'data_avg_plot'))
            
            if trend_calc:
                trend_lazy = compute_trend_lazy(base_fit, fit_scaling)
                lazy_computations.append(trend_lazy)
                computation_map.append((ii, 'trend_plot'))
        
        else:
            # Comparison mode
            pair_key = (base_src, comp_src)
            minus_scaling = -1 if switch_sign_list[ii] else 1
            
            # Get cached regridded comparison data
            comp_yearly = seasonal_slices.get(((pair_key, 'monthly'), month_key, 'yearly'))
            comp_fit = seasonal_slices.get(((pair_key, 'monthly'), month_key, 'fit'))
            comp_monthly = seasonal_slices.get(((pair_key, 'monthly'), month_key, 'monthly'))
            comp_raw = seasonal_slices.get(((pair_key, 'raw'), month_key, 'raw'))
            
            # Compute comparison seasonal average (lazy)
            comp_avg_season = comp_yearly.mean(dim='clim_year').astype('float32')
            
            # LAZY: Climatology difference
            data_avg_lazy = (minus_scaling*(comp_avg_season - base_avg_season)).astype('float32')
            lazy_computations.append(data_avg_lazy)
            computation_map.append((ii, 'data_avg_plot'))
            
            # LAZY: Trends
            if trend_calc:
                trend_base_lazy = compute_trend_lazy(base_fit, fit_scaling)
                
                if trend_regrid:
                    # Get native comparison fit data from cache
                    comp_fit_native = seasonal_slices.get((comp_src, month_key, 'fit'))
                    trend_comp_native = compute_trend_lazy(comp_fit_native, fit_scaling)

                    # Regrid native trend (2D field) - this needs compute
                    trend_comp_lazy = regridders[pair_key](trend_comp_native, output_chunks=TARGET_CHUNKS).astype('float32')
                    trend_diff_lazy = (minus_scaling * (trend_comp_lazy - trend_base_lazy)).astype('float32')
                    lazy_computations.append(trend_diff_lazy)
                    computation_map.append((ii, 'trend_plot'))
                    
                else:
                    # Use regridded fit data
                    trend_comp_lazy = compute_trend_lazy(comp_fit, fit_scaling)
                    trend_diff_lazy = (minus_scaling * (trend_comp_lazy - trend_base_lazy)).astype('float32')
                    lazy_computations.append(trend_diff_lazy)
                    computation_map.append((ii, 'trend_plot'))
            
            # LAZY: Correlation
            if corr_calc:
                if corr_freq_list[ii] == 'Daily':
                    corr_lazy = compute_correlation_lazy(base_raw, comp_raw, 'time')
                    lazy_computations.append(corr_lazy)
                    computation_map.append((ii, 'corr_plot'))
                elif corr_freq_list[ii] == 'Yearly':
                    corr_lazy = compute_correlation_lazy(base_yearly, comp_yearly, 'clim_year')
                    lazy_computations.append(corr_lazy)
                    computation_map.append((ii, 'corr_plot'))
                elif corr_freq_list[ii] == 'Monthly':
                    corr_lazy = compute_correlation_lazy(base_monthly, comp_monthly, 'time')
                    lazy_computations.append(corr_lazy)
                    computation_map.append((ii, 'corr_plot'))
            
            # LAZY: RMSE
            if rmse_calc:
                if rmse_freq_list[ii] == 'Daily':
                    rmse_lazy = compute_rmse_lazy(base_raw, comp_raw, 'time')
                    lazy_computations.append(rmse_lazy)
                    computation_map.append((ii, 'rmse_plot'))
                elif rmse_freq_list[ii] == 'Yearly':
                    rmse_lazy = compute_rmse_lazy(base_yearly, comp_yearly, 'clim_year')
                    lazy_computations.append(rmse_lazy)
                    computation_map.append((ii, 'rmse_plot'))
                elif rmse_freq_list[ii] == 'Monthly':
                    rmse_lazy = compute_rmse_lazy(base_monthly, comp_monthly, 'time')
                    lazy_computations.append(rmse_lazy)
                    computation_map.append((ii, 'rmse_plot'))
            
            # LAZY: Std mask reference
            if std_mask_ref_list[ii] is not None:
                std_ref_src = std_mask_ref_list[ii]
                
                if std_ref_src == 'Pool':
                    # Use pooled std from base and compare
                    std_ref_lazy = compute_std_ref_lazy(base_fit, comp_fit, 'Pool')
                    lazy_computations.append(std_ref_lazy)
                    computation_map.append((ii, 'std_ref'))
                elif std_ref_src == base_src:
                    # Use base source std
                    std_ref_lazy = compute_std_ref_lazy(base_fit, base_fit, 'base')
                    lazy_computations.append(std_ref_lazy)
                    computation_map.append((ii, 'std_ref'))
                elif std_ref_src == comp_src:
                    # Use compare source std (regridded)
                    std_ref_lazy = compute_std_ref_lazy(base_fit, comp_fit, 'compare')
                    lazy_computations.append(std_ref_lazy)
                    computation_map.append((ii, 'std_ref'))
                else:
                    # Third source - get regridded fit data
                    std_ref_pair_key = (base_src, std_ref_src)
                    std_ref_fit = seasonal_slices.get(((std_ref_pair_key, 'monthly'), month_key, 'fit'))
                    std_ref_lazy = compute_std_ref_lazy(base_fit, std_ref_fit, 'compare')
                    lazy_computations.append(std_ref_lazy)
                    computation_map.append((ii, 'std_ref'))
        
        results.append(result)
    
    # ==========================================================================
    # BATCH COMPUTE - Execute all lazy computations in parallel
    # ==========================================================================
    if lazy_computations:
        print(f'    Batch computing {len(lazy_computations)} operations...')
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

print('\n' + '=' * 60)
print('Processing complete!')
print('=' * 60)

#%% ============================================================================
#   AREA CONTOUR COMPUTATION
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

def compute_cbar_pad(fig, axes, cbar_orientation, extra_pad=0.01):

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_width, fig_height = fig.get_size_inches() * fig.dpi
    
    if cbar_orientation == 'horizontal':
        min_y_fig = 1.0
        n_cols = axes.shape[1]
        for col in range(n_cols):
            ax = axes[-1, col]
            for label in ax.get_xticklabels():
                bbox = label.get_window_extent(renderer=renderer)
                y_bottom_fig = bbox.y0 / fig_height
                min_y_fig = min(min_y_fig, y_bottom_fig)
        
        # Get the bottom of the bottom row axes
        bottom_row_y0 = min(axes[-1, col].get_position().y0 for col in range(n_cols))
        
        # Pad is the distance from axes bottom to tick labels bottom, plus extra
        cbar_pad = bottom_row_y0 - min_y_fig + extra_pad
        return max(cbar_pad, 0.02)  # Ensure minimum padding
    
    else:
        return 0.015


def get_cbar_sizing(use_shared_cbar, cbar_orientation, panel_width, panel_height, 
                    fig_width, fig_height, shared_label=False):

    title_fontsize = max(40, int(panel_height * 7))
    
    cbar_thickness_pct = 0.1 if use_shared_cbar else 0.07
    if cbar_orientation == 'horizontal':
        cbar_thickness = panel_height * cbar_thickness_pct / fig_height
    else:
        cbar_thickness = panel_width * cbar_thickness_pct / fig_width
    
    # cbar_pad will be computed dynamically based on tick labels
    cbar_pad = None
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
    
    # Compute colorbar padding based on tick labels
    cbar_pad = compute_cbar_pad(fig, axes, cbar_orientation)
    
    # Colorbars
    if mesh_ref is not None:
        if use_shared_cbar:
            shared_colorbar(
                fig=fig, axes=axes.ravel(), mesh=mesh_ref, datasets=datasets_all,
                crange=crange_global, label=label, orientation=cbar_orientation,
                c_ticks=cbar_n_ticks, c_ticks_num=cbar_ticks_num,
                tick_labelsize=cbar_sizing['cbar_tick_size'],
                labelsize=cbar_sizing['cbar_label_size'],
                pad=cbar_pad, thickness=cbar_sizing['cbar_thickness'],
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
                    pad=cbar_pad, thickness=cbar_sizing['cbar_thickness'],
                    label_pad=cbar_sizing['cbar_label_pad'],
                    extendfrac=extendfrac, length_scale=length_scale)
                cbar_axes_list.append(cbar_k.ax)
            
            if shared_cbar_label and cbar_axes_list:
                add_shared_cbar_label(
                    fig, cbar_axes_list, label, orientation=cbar_orientation,
                    fontsize=cbar_sizing['cbar_label_size'], pad=0.008)
    
    if save_name is not None:
        # Create output directories if they don't exist
        pdf_dir = Path(f'/nobackup/users/walj/Figures/pdf/{var}')
        jpg_dir = Path(f'/nobackup/users/walj/Figures/jpg/{var}')
        pdf_dir.mkdir(parents=True, exist_ok=True)
        jpg_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PDF
        pdf_out = pdf_dir / f'{save_name}.pdf'
        plt.savefig(str(pdf_out), format='pdf', bbox_inches='tight')
        
        # Save JPG
        jpg_out = jpg_dir / f'{save_name}.jpg'
        plt.savefig(str(jpg_out), format='jpg', dpi=300, bbox_inches='tight')
    
    # Restore interactive mode if it was on
    if was_interactive:
        plt.ion()
    
    plt.show()

#%% Plotting Climatology

# # Spatial plotting arguments
# plot_climatology = True
# # avg_crange = {'DJF': [-20, 20], 'MAM': [-30, 30], 'JJA': [-30, 30], 'SON': [-20, 20]}
# avg_crange = [-8,8]
# std_mask_ref = data_base
# std_dir = 'Lesser'
# shared_cbar_avg = True
# shared_cbar_label_avg = True
# short_cbar_label_avg = False
# cbar_ticks_avg_shared = 2
# cbar_ticks_avg_sep = [10, 10, 10, 10]
# cbar_ticks_num_avg = False

# labels = get_labels(var, var_name_cfg, var_symbol_cfg, var_units_cfg, trend_unit,
#                     short_cbar_label_avg, short_cbar_label_trend, 
#                     short_cbar_label_corr, short_cbar_label_rmse, has_comparison)

# avg_crange = {'DJF': [0, 100], 'MAM': [100, 220], 'JJA': [140, 280], 'SON': [20, 140]}

if save_name_base is not None:
    if data_compare is not None:
        save_name_avg = f'{save_name_base}_{var}_bias'
    else:
        save_name_avg = f'{save_name_base}_{var}_climatology'

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


#%% Plotting Trend

# # Trend plotting arguments
# trend_calc = True
# # trend_crange = {'DJF': [-2, 2], 'MAM': [-5, 5], 'JJA': [-5, 5], 'SON': [-2, 2]  }
# trend_crange = [-2,2]
# trend_regrid = False
# fit_against_gmst = False
# shared_cbar_trend = True
# shared_cbar_label_trend = True
# short_cbar_label_trend = False
# cbar_ticks_trend_shared = 0.5
# cbar_ticks_trend_sep = [0.5, 1, 1, 0.6]
# cbar_ticks_num_trend = False

# labels = get_labels(var, var_name_cfg, var_symbol_cfg, var_units_cfg, trend_unit,
#                     short_cbar_label_avg, short_cbar_label_trend, 
#                     short_cbar_label_corr, short_cbar_label_rmse, has_comparison)

# cbar_ticks_trend_shared = 0.2

if save_name_base is not None:
    if data_compare is not None:
        save_name_trend = f'{save_name_base}_{var}_trenddiff'
    else:
        save_name_trend = f'{save_name_base}_{var}_trend'

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


#%% Plotting Correlation

# # Correlation plotting arguments
# corr_calc = True
# corr_freq = 'Monthly'
# corr_crange = [0.5, 1]
# corr_cmap_type = None
# shared_cbar_corr = True
# shared_cbar_label_corr = True
# short_cbar_label_corr = False
# cbar_ticks_corr_shared = 0.1
# cbar_ticks_corr_sep = 6
# cbar_ticks_num_corr = False

# labels = get_labels(var, var_name_cfg, var_symbol_cfg, var_units_cfg, trend_unit,
#                     short_cbar_label_avg, short_cbar_label_trend, 
#                     short_cbar_label_corr, short_cbar_label_rmse, has_comparison)

if save_name_base is not None:
    save_name_corr = f'{save_name_base}_{var}_corr'

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


#%% Plotting RMSE

# # RMSE plotting arguments
# rmse_calc = True
# rmse_freq = 'Monthly'
# rmse_crange = [0, 8]
# shared_cbar_rmse = True
# shared_cbar_label_rmse = True
# short_cbar_label_rmse = False
# cbar_ticks_rmse_shared = 1
# cbar_ticks_rmse_sep = 0.2
# cbar_ticks_num_rmse = False

# labels = get_labels(var, var_name_cfg, var_symbol_cfg, var_units_cfg, trend_unit,
#                     short_cbar_label_avg, short_cbar_label_trend, 
#                     short_cbar_label_corr, short_cbar_label_rmse, has_comparison)

if save_name_base is not None:
    save_name_rmse = f'{save_name_base}_{var}_rmse'

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

