#%% Imports

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

from RegionalTrends.Helpers import ProcessSource
reload(ProcessSource)
from RegionalTrends.Helpers.ProcessSource import process_source

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
var = 'Tg'
file_freq = 'Monthly' #
# data_base = ['ERA5_coarse', 'ERA5_coarse', 'Eobs_fine', 'Eobs_fine'] #
# data_compare = ['RACMO2.3', 'RACMO2.4_KEXT12', 'RACMO2.3', 'RACMO2.4_KEXT12'] #
data_base = ['ERA5_coarse', 'RACMO2.3', 'RACMO2.4_KEXT12', 'Eobs_fine'] #
data_compare = None #

# Data selection arguments
months_dict = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]} #
# months_dict = {'May': [5]} #
years = [1980, 2020] 
lats = [37.7, 63.3]
lons = [-13.3, 22.3]
proj_sel = 'RACMO2.4'
land_only = False
trim_border = None

# Spatial plotting arguments
plot_climatology = False
avg_crange = {'DJF': [-1, 1], 'MAM': [-2, 2], 'JJA': [-2, 2], 'SON': [-2, 2]} #
std_mask_ref = data_base #
std_dir = 'Lesser' #
shared_cbar_avg = False 
shared_cbar_label_avg = False
short_cbar_label_avg = True
cbar_ticks_avg_shared = 11
cbar_ticks_avg_sep = [0.5, 1, 1, 1]
cbar_ticks_num_avg = False
save_name_avg = 'AllClimatologySeasons'

# Trend plotting arguments
trend_calc = True
trend_crange = {'DJF': [-1, 1], 'MAM': [-2, 2], 'JJA': [-2, 2], 'SON': [-2, 2]} #
# trend_crange = {'May': [-0.8, 0.8]}
fit_against_gmst = False
shared_cbar_trend = False
shared_cbar_label_trend = True
short_cbar_label_trend = False
cbar_ticks_trend_shared = 0.2
cbar_ticks_trend_sep = [0.5, 1, 1, 1]
cbar_ticks_num_trend = False
save_name_trend = None

# Correlation plotting arguments
corr_calc = False
corr_freq = 'Monthly' #
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
rmse_calc = False
rmse_freq = 'Monthly' #
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
switch_sign = False #
cut_boundaries = False
cbar_orientation = 'horizontal'
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

if fit_against_gmst:
    fit_unit = '°C GMST'
    fit_scaling = 1
    fit_x_label = r'$\Delta$GMST (°C)'
else:
    fit_unit = 'decade'
    fit_scaling = 10
    fit_x_label = 'Year'

if var == 'P' and relative_precip:
    trend_unit = '% / ' + fit_unit
elif var_units_cfg[var] == '':
    trend_unit = 'per' + fit_unit
else:
    trend_unit = var_units_cfg[var] + ' / ' + fit_unit

# Handle symbols that already contain $ delimiters
raw_sym = var_symbol_cfg[var]
if '$' in raw_sym:
    sym = raw_sym  # Already has math delimiters
else:
    sym = r'$' + raw_sym + r'$'
full_var = var_name_cfg[var] + ' ' + sym
if var_name_cfg[var] == var_symbol_cfg[var]:
    full_var = sym

if short_cbar_label_avg:
    avg_label = sym + ' (' + var_units_cfg[var] + ')'
else:
    avg_label = full_var + ' (' + var_units_cfg[var] + ')'

if short_cbar_label_trend:
    trend_label = sym + ' trend (' + trend_unit + ')'
else:
    trend_label = full_var + ' trend (' + trend_unit + ')'

if short_cbar_label_corr:
    corr_label = sym + ' correlation'
else:
    corr_label = full_var + ' correlation'

if short_cbar_label_rmse:
    rmse_label = sym + ' RMSE (' + var_units_cfg[var] + ')'
else:
    rmse_label = full_var + ' RMSE (' + var_units_cfg[var] + ')'


avg_cmap_key = 'cmap_mean'
avg_ext_key = 'extreme_mean'
if data_compare is not None:
    if short_cbar_label_avg:
        avg_label = r'$\Delta$' + avg_label
    else:
        avg_label = 'Difference in ' + avg_label
    avg_cmap_key = 'cmap_trend'
    avg_ext_key = 'extreme_trend'

if not var_colors_cfg.get(var):
    var_colors_cfg[var] = var_colors_cfg['Default']
elif not var_colors_cfg[var].get('cmap_trend'):
    var_colors_cfg[var]['cmap_trend'] = var_colors_cfg['Default']['cmap_trend']
    var_colors_cfg[var]['extreme_trend'] = var_colors_cfg['Default']['extreme_trend']

var_colors_cfg[var]['cmap_mean'] = convert_cmap(
    var_colors_cfg[var]['cmap_mean'], cmap_type, n_disc_colors
)
var_colors_cfg[var]['cmap_trend'] = convert_cmap(
    var_colors_cfg[var]['cmap_trend'], cmap_type, n_disc_colors
)
rmse_cmap = convert_cmap(
    cmocean.cm.amp, cmap_type, n_disc_colors
)
corr_meta = build_corr_cmap(corr_cmap_type)
corr_cmap = convert_cmap(
    corr_meta['corr_cmap'], cmap_type, n_disc_colors
)

proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())
proj_plot = proj_cfg.get(proj_plot, ccrs.PlateCarree())
proj_area = proj_cfg.get(proj_area, ccrs.PlateCarree())

#%% Ensure lists

def ensure_list(param, n):

    if isinstance(param, list):
        return param
    else:
        return [param]*n

n_runs = len(data_base) if isinstance(data_base, list) else 1
n_months = len(months_dict)
data_base_list = ensure_list(data_base, n_runs)
freq_file_list = ensure_list(file_freq, n_runs)
data_compare_list = ensure_list(data_compare, n_runs)
std_mask_ref_list = ensure_list(std_mask_ref, n_runs)
std_dir_list = ensure_list(std_dir, n_runs)
switch_sign_list = ensure_list(switch_sign, n_runs)
corr_freq_list = ensure_list(corr_freq, n_runs)
rmse_freq_list = ensure_list(rmse_freq, n_runs)

#%% Further processing of base and comparison data

def make_cache_key(data_source, months):
    return (data_source, tuple(np.arange(1, 13)) if months is None else tuple(months))

data_cache = {}
all_results = {}

for month_key, months in months_dict.items():
    
    print(f'Processing months: {month_key}')
    
    results = []

    for ii in range(n_runs):

        if data_base_list[ii] is not None:

            cache_key_base = make_cache_key(data_base_list[ii], months)
            if cache_key_base in data_cache:
                data_base_res = data_cache[cache_key_base]
            else:
                data_base_res = process_source(
                    data_base_list[ii], 
                    var,
                    data_sources,
                    station_sources,
                    freq_file_list[ii],
                    var_file_cfg,
                    proj_cfg,
                    months=months,
                    years=years,
                    lats=lats,
                    lons=lons,
                    land_only=land_only,
                    trim_border=trim_border, 
                    rotpole_sel=proj_sel,
                    rolling_mean_var=rolling_mean_var,
                    fit_against_gmst=fit_against_gmst,
                    rolling_mean_years=rolling_mean_years,
                    min_periods=min_periods
                )
                data_cache[cache_key_base] = data_base_res

            if data_compare_list[ii] is None:

                title = next(key for key in data_sources if key in data_base_list[ii])

                data_avg_plot = data_base_res['avg'].compute()

                if trend_calc:
                    fits_base = data_base_res['fit'].polyfit(dim='fit_against', deg=1, skipna=True)
                    slope_base = fits_base.polyfit_coefficients.sel(degree=1)
                    trend_base = (slope_base*fit_scaling).astype('float32').compute()

                    if relative_precip and var == 'P':
                        trend_plot_base = (trend_base / data_avg_plot)*100.0
                    else:
                        trend_plot_base = trend_base
                    trend_plot = trend_plot_base.assign_coords(
                        latitude=data_avg_plot['latitude'],
                        longitude=data_avg_plot['longitude']
                    )

            elif data_compare_list[ii] is not None:

                cache_key_comp = make_cache_key(data_compare_list[ii], months)
                if cache_key_comp in data_cache:
                    data_comp_res = data_cache[cache_key_comp]
                else:
                    data_comp_res = process_source(
                        data_compare_list[ii], 
                        var,
                        data_sources,
                        station_sources,
                        freq_file_list[ii],
                        var_file_cfg,
                        proj_cfg, 
                        months=months,
                        years=years,
                        lats=lats,
                        lons=lons,
                        land_only=land_only,
                        trim_border=trim_border,
                        rotpole_sel=proj_sel,
                        rolling_mean_var=rolling_mean_var,
                        fit_against_gmst=fit_against_gmst,
                        rolling_mean_years=rolling_mean_years,
                        min_periods=min_periods
                    )
                    data_cache[cache_key_comp] = data_comp_res

                if var == 'P':
                    method = 'conservative_normed'
                    trg_grid = grid_with_bounds(data_base_res['avg'], rotpole_native=proj_cfg.get(data_base_list[ii], ccrs.PlateCarree()))
                    src_grid = grid_with_bounds(data_comp_res['avg'], rotpole_native=proj_cfg.get(data_compare_list[ii], ccrs.PlateCarree()))

                else:
                    method = 'bilinear'
                    trg_grid = data_base_res['avg']
                    src_grid = data_comp_res['avg']

                regridder = xe.Regridder(
                    src_grid,
                    trg_grid,
                    method,
                    unmapped_to_nan=True,
                )

                target_chunks = {'latitude': 100, 'longitude': 100}

                data_avg_comp_reg = regridder(
                    data_comp_res['avg'],
                    output_chunks=target_chunks
                ).astype('float32')

                if switch_sign_list[ii]:
                    minus_scaling = -1
                    title = (
                        next(key for key in data_sources if key in data_base_list[ii]) 
                        + ' - ' 
                        + next(key for key in data_sources if key in data_compare_list[ii]) 
                    )
                else:
                    minus_scaling = 1
                    title = (
                        next(key for key in data_sources if key in data_compare_list[ii]) 
                        + ' - ' 
                        + next(key for key in data_sources if key in data_base_list[ii])
                    )

                title = title.replace('ACMO', '')

                data_avg_plot = minus_scaling*(data_avg_comp_reg - data_base_res['avg']).compute()

                if trend_calc or (std_mask_ref_list[ii] == data_compare_list[ii] or std_mask_ref_list[ii] == 'Pool'):
                    data_fit_comp_reg = regridder(
                        data_comp_res['fit'],
                        output_chunks=target_chunks
                    ).astype('float32').compute()

                if trend_calc:
                    fits_base = data_base_res['fit'].polyfit(dim='fit_against', deg=1, skipna=True)
                    slope_base = fits_base.polyfit_coefficients.sel(degree=1)
                    trend_base = (slope_base*fit_scaling).astype('float32').compute()

                    fits_comp = data_fit_comp_reg.polyfit(dim='fit_against', deg=1, skipna=True)
                    slope_comp = fits_comp.polyfit_coefficients.sel(degree=1)
                    trend_comp = (slope_comp*fit_scaling).astype('float32').compute()

                    if relative_precip and var == 'P':
                        trend_plot_base = (trend_base / data_base_res['avg'])*100.0
                        trend_plot_comp = (trend_comp / data_avg_comp_reg)*100.0
                    else:
                        trend_plot_base = trend_base
                        trend_plot_comp = trend_comp

                    
                    trend_plot = minus_scaling*(trend_plot_comp - trend_plot_base).compute()
                    trend_plot = trend_plot.assign_coords(
                            latitude=data_base_res['avg']['latitude'],
                            longitude=data_base_res['avg']['longitude']
                        ).astype('float32')

                if corr_calc:

                    if corr_freq_list[ii] == 'Daily':
                        data_corr_base = data_base_res['raw']
                        data_corr_comp = data_comp_res['raw']
                    elif corr_freq_list[ii] == 'Monthly':
                        data_corr_base = data_base_res['monthly']
                        data_corr_comp = data_comp_res['monthly']
                    elif corr_freq_list[ii] == 'Yearly':
                        data_corr_base = data_base_res['yearly']
                        data_corr_comp = data_comp_res['yearly']
                    
                    data_corr_comp_reg = regridder(
                        data_corr_comp,
                        output_chunks=target_chunks
                    ).astype('float32')

                    x, y = xr.align(data_corr_base, data_corr_comp_reg, join='inner')

                    x = x.chunk({'time': -1}).compute()
                    y = y.chunk({'time': -1}).compute()

                    valid = np.isfinite(x) & np.isfinite(y)
                    x = x.where(valid)
                    y = y.where(valid)

                    n = valid.sum('time')
                    sx = x.std('time')
                    sy = y.std('time')

                    corr_plot = xr.corr(x, y, dim='time')
                    corr_plot = corr_plot.where((n >= 2) & (sx > 0) & (sy > 0))

                    corr_plot = corr_plot.assign_coords(
                        latitude=data_corr_base['latitude'],
                        longitude=data_corr_base['longitude']
                    ).astype('float32')

                if rmse_calc:

                    if rmse_freq_list[ii] == 'Daily':
                        data_rmse_base = data_base_res['raw']
                        data_rmse_comp = data_comp_res['raw']
                    elif rmse_freq_list[ii] == 'Monthly':
                        data_rmse_base = data_base_res['monthly']
                        data_rmse_comp = data_comp_res['monthly']
                    elif rmse_freq_list[ii] == 'Yearly':
                        data_rmse_base = data_base_res['yearly']
                        data_rmse_comp = data_comp_res['yearly']

                    data_rmse_comp_reg = regridder(
                        data_rmse_comp,
                        output_chunks=target_chunks
                    ).astype('float32')

                    x, y = xr.align(data_rmse_base, data_rmse_comp_reg, join='inner')

                    x = x.chunk({'time': -1}).compute()
                    y = y.chunk({'time': -1}).compute()

                    valid = np.isfinite(x) & np.isfinite(y)
                    x = x.where(valid)
                    y = y.where(valid)

                    err = y - x
                    rmse_plot = np.sqrt((err**2).mean('time'))

                    rmse_plot = rmse_plot.assign_coords(
                        latitude=data_rmse_base['latitude'],
                        longitude=data_rmse_base['longitude']
                    ).astype('float32')

                if std_mask_ref_list[ii] is not None:

                    if std_mask_ref_list[ii] == 'Pool':

                        xb, yc = xr.align(data_base_res['fit'], data_fit_comp_reg, join='inner')

                        valid = np.isfinite(xb) & np.isfinite(yc)
                        xb = xb.where(valid)
                        yc = yc.where(valid)

                        xb = xb.chunk({'fit_against': -1}).compute()
                        yc = yc.chunk({'fit_against': -1}).compute()

                        n = valid.sum('fit_against')

                        fits_xb = xb.polyfit(dim='fit_against', deg=1, skipna=True)
                        trend_xb = xr.polyval(xb['fit_against'], fits_xb.polyfit_coefficients)
                        resid_xb = xb - trend_xb

                        fits_yc = yc.polyfit(dim='fit_against', deg=1, skipna=True)
                        trend_yc = xr.polyval(yc['fit_against'], fits_yc.polyfit_coefficients)
                        resid_yc = yc - trend_yc

                        sx = resid_xb.std('fit_against')
                        sy = resid_yc.std('fit_against')

                        std_ref = np.sqrt(0.5*(sx**2 + sy**2))
                        std_ref = std_ref.where(n >= 2)

                    elif std_mask_ref_list[ii] != 'Pool' and std_mask_ref_list[ii] is not None:
                        
                        if std_mask_ref_list[ii] == data_base_list[ii]:
                            ref_std_reg = data_base_res['fit']
                        elif std_mask_ref_list[ii] == data_compare_list[ii]:
                            ref_std_reg = data_fit_comp_reg 
                        else:
                            cache_key_ref = make_cache_key(std_mask_ref_list[ii], months, var)
                            if cache_key_ref in data_cache:
                                data_ref_res = data_cache[cache_key_ref]
                            else:
                                data_ref_res = process_source(
                                    std_mask_ref_list[ii], 
                                    var,
                                    data_sources,
                                    station_sources,
                                    freq_file_list[ii],
                                    var_file_cfg,
                                    proj_cfg, 
                                    months=months,
                                    years=years,
                                    lats=lats,
                                    lons=lons,
                                    land_only=land_only,
                                    trim_border=trim_border,
                                    rotpole_sel=proj_sel,
                                    rolling_mean_var=rolling_mean_var,
                                    fit_against_gmst=fit_against_gmst,
                                    rolling_mean_years=rolling_mean_years,
                                    min_periods=min_periods
                                )
                                data_cache[cache_key_ref] = data_ref_res

                            if var == 'P':
                                src_grid_ref = grid_with_bounds(
                                    data_ref_res['avg'],
                                    rotpole_native=proj_cfg.get(std_mask_ref_list[ii], ccrs.PlateCarree())
                                )
                            else:
                                src_grid_ref = data_ref_res['avg']

                            regridder_std = xe.Regridder(
                                src_grid_ref,
                                trg_grid,
                                method,
                                unmapped_to_nan=True,
                            )
                            
                            ref_std_reg = regridder_std(
                                data_ref_res['fit'],
                                output_chunks=target_chunks
                            ).astype('float32')

                        _, ref_std_reg = xr.align(data_base_res['fit'], ref_std_reg, join='inner')

                        ref_std_reg = ref_std_reg.chunk({'fit_against': -1}).compute()
                        n_ref = np.isfinite(ref_std_reg).sum('fit_against')

                        fits_ref = ref_std_reg.polyfit(dim='fit_against', deg=1, skipna=True)
                        trend_ref = xr.polyval(ref_std_reg['fit_against'], fits_ref.polyfit_coefficients)
                        resid_ref = ref_std_reg - trend_ref

                        std_ref = resid_ref.std('fit_against')
                        std_ref = std_ref.where(n_ref >= 2)

                    if std_dir_list[ii] == 'Greater':
                        mask_std = (np.abs(data_avg_plot) > std_ref).fillna(False).astype('int8')
                    elif std_dir_list[ii] == 'Lesser':
                        mask_std = (np.abs(data_avg_plot) < std_ref).fillna(False).astype('int8')

            title = title.replace('Eobs', 'E-OBS')

            results.append({
                'data_avg_plot': data_avg_plot,
                'trend_plot': trend_plot if trend_calc else None,
                'corr_plot': corr_plot if (data_compare_list[ii] is not None and corr_calc) else None,
                'rmse_plot': rmse_plot if (data_compare_list[ii] is not None and rmse_calc) else None,
                'mask_std': mask_std if (std_mask_ref_list[ii] is not None and data_compare_list[ii] is not None) else None,
                'title': title
            })
    
    all_results[month_key] = results

#%% Area plotting selection

all_area_contours = {}

for month_key in months_dict.keys():
    
    results = all_results[month_key]
    area_contours = []

    for ii in range(n_runs):

        if data_base_list[ii] is not None: 

            mask_area = None
            lat_b_area = None
            lon_b_area = None

            lats_area_cont = None
            lons_area_cont = None 
            proj_area_cont = None

            if isinstance(lats_area, (list, tuple)) and len(lats_area) == 2 and \
            isinstance(lons_area, (list, tuple)) and len(lons_area) == 2 and grid_contour == True:

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
                    contour_area_full,
                    lat2d,
                    lon2d,
                    lats_area,
                    lons_area,
                    dim_lat,
                    dim_lon,
                    rotpole_sel=proj_area,
                    rotpole_native=proj_cfg.get(data_base_list[ii], ccrs.PlateCarree())
                )

                mask_area = np.isfinite(contour_area.values)

                contour_area_bounds = grid_with_bounds(
                    contour_area,
                    rotpole_native=proj_cfg.get(data_base_list[ii], ccrs.PlateCarree())
                )
                lon_b_area, lat_b_area = contour_area_bounds['lon_b'], contour_area_bounds['lat_b']

            if isinstance(lats_area, (list, tuple)) and len(lats_area) == 2 and \
            isinstance(lons_area, (list, tuple)) and len(lons_area) == 2 and true_contour == True:
                
                lats_area_cont = lats_area
                lons_area_cont = lons_area
                proj_area_cont = proj_area

            area_contours.append({
                'mask_area': mask_area,
                'lat_b_area': lat_b_area,
                'lon_b_area': lon_b_area,
                'lats_area_cont': lats_area_cont,
                'lons_area_cont': lons_area_cont,
                'proj_area_cont': proj_area_cont
            })
    
    all_area_contours[month_key] = area_contours

#%% Colorbar sizing helper function

def get_cbar_sizing(use_shared_cbar, cbar_orientation, panel_width, panel_height, fig_width, fig_height, shared_label=False):
    
    # Title font size - scales with panel size
    title_fontsize = max(40, int(panel_height * 7))
    
    # Colorbar thickness as percentage of panel
    cbar_thickness_pct = 0.1 if use_shared_cbar else 0.07
    if cbar_orientation == 'horizontal':
        cbar_thickness = panel_height * cbar_thickness_pct / fig_height
    else:
        cbar_thickness = panel_width * cbar_thickness_pct / fig_width
    
    # Colorbar padding
    cbar_pad = 0.028 if cbar_orientation == 'horizontal' else 0.015
    
    # Label padding
    cbar_label_pad = 16 if use_shared_cbar else 6
    
    # Tick size - larger for shared colorbar only
    if use_shared_cbar:
        cbar_tick_size = max(35, int(panel_height * 7))
    else:
        cbar_tick_size = max(25, int(panel_height * 5))
    
    # Label size - larger for shared colorbar OR shared label
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

#%% Plot climatology

if plot_climatology:

    month_keys = list(months_dict.keys())

    # avg_crange can be either a dict keyed by month, or a single [vmin, vmax] list
    avg_crange_is_dict = isinstance(avg_crange, dict)
    
    # Determine if we should use a shared colorbar
    # Shared if: avg_crange is a list (not dict), OR shared_cbar is explicitly True
    use_shared_cbar = (not avg_crange_is_dict) or shared_cbar_avg
    
    if avg_crange_is_dict:
        # Build a global range from all month ranges
        vmins = [float(v[0]) for v in avg_crange.values()]
        vmaxs = [float(v[1]) for v in avg_crange.values()]
        avg_crange_global = [min(vmins), max(vmaxs)]
    else:
        avg_crange_global = avg_crange

    # Determine subplot grid shape
    if swap_rows_cols:
        n_rows, n_cols = n_runs, n_months
    else:
        n_rows, n_cols = n_months, n_runs

    fig_width = panel_width * n_cols
    fig_height = panel_height * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        subplot_kw={'projection': proj_plot},
        sharex=True,
        sharey=True
    )
    
    # # No extra padding between figures
    # if swap_rows_cols:
    #     fig.get_layout_engine().set(w_pad=0, wspace=0)
    # else:
    #     fig.get_layout_engine().set(h_pad=0, hspace=0)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Run titles come from the computed results
    run_titles = None
    if month_keys:
        run_titles = [r.get('title', '') for r in all_results[month_keys[0]]]

    # Get colorbar sizing from helper function
    cbar_sizing = get_cbar_sizing(
        use_shared_cbar, cbar_orientation, 
        panel_width, panel_height, fig_width, fig_height,
        shared_label=shared_cbar_label_avg
    )
    title_fontsize = cbar_sizing['title_fontsize']
    cbar_thickness = cbar_sizing['cbar_thickness']
    cbar_pad = cbar_sizing['cbar_pad']
    cbar_label_pad = cbar_sizing['cbar_label_pad']
    cbar_tick_size = cbar_sizing['cbar_tick_size']
    cbar_label_size = cbar_sizing['cbar_label_size']
    
    # Get tick count based on shared vs separate colorbar
    if use_shared_cbar:
        cbar_n_ticks = cbar_ticks_avg_shared
    else:
        cbar_n_ticks = cbar_ticks_avg_sep if not isinstance(cbar_ticks_avg_sep, list) else cbar_ticks_avg_sep[0]

    # Collect for shared colorbar
    mesh_ref = None
    datasets_all = []

    # Collect groups for per-month/run colorbars
    month_axes_groups = {mk: [] for mk in month_keys}
    month_data_groups = {mk: [] for mk in month_keys}
    month_mesh_groups = {mk: None for mk in month_keys}

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):

            if swap_rows_cols:
                run_idx = row_idx
                month_idx = col_idx
            else:
                month_idx = row_idx
                run_idx = col_idx

            month_key = month_keys[month_idx]
            res = all_results[month_key][run_idx]
            area_contours = all_area_contours[month_key][run_idx]

            ax = axes[row_idx, col_idx]

            # Titles
            if show_col_titles and row_idx == 0:
                if swap_rows_cols:
                    title = month_key
                else:
                    title = res.get('title', None)
            else:
                title = None

            # Tick label visibility
            show_y_labels = (col_idx == 0)
            show_x_labels = (row_idx == n_rows - 1)

            # Color range - use global range if shared colorbar
            if use_shared_cbar:
                crange_this = avg_crange_global
            elif avg_crange_is_dict:
                crange_this = avg_crange[month_key]
            else:
                crange_this = avg_crange

            mesh, _ = plot_map(
                fig, ax,
                res['data_avg_plot'],
                res['data_avg_plot'].longitude.values,
                res['data_avg_plot'].latitude.values,
                crange=crange_this,
                label=avg_label,
                cmap=var_colors_cfg[var][avg_cmap_key],
                extreme_colors=var_colors_cfg[var][avg_ext_key],
                show_x_ticks=True,
                show_y_ticks=True,
                y_ticks_num=False,
                y_ticks=5,
                show_y_labels=show_y_labels,
                x_ticks_num=False,
                x_ticks=10,
                show_x_labels=show_x_labels,
                tick_size=16,
                extent=[*plot_lons, *plot_lats],
                proj=proj_plot,
                rotated_grid=cut_boundaries,
                title=title,
                title_size=title_fontsize,
                lats_area=area_contours['lats_area_cont'],
                lons_area=area_contours['lons_area_cont'],
                proj_area=area_contours['proj_area_cont'],
                mask_area=area_contours['mask_area'],
                lat_b_area=area_contours['lat_b_area'],
                lon_b_area=area_contours['lon_b_area'],
                show_plot=False,
                add_colorbar=False
            )

            if mesh_ref is None:
                mesh_ref = mesh

            datasets_all.append(res['data_avg_plot'])

            month_axes_groups[month_key].append(ax)
            month_data_groups[month_key].append(res['data_avg_plot'])
            if month_mesh_groups[month_key] is None:
                month_mesh_groups[month_key] = mesh

            if std_mask_ref_list[run_idx] is not None and data_compare_list[run_idx] is not None:
                ax.contourf(
                    res['data_avg_plot'].longitude.values,
                    res['data_avg_plot'].latitude.values,
                    res['mask_std'],
                    levels=[0.5, 1.5],
                    colors='none',
                    hatches=['///'],
                    transform=ccrs.PlateCarree(),
                    zorder=50
                )

    # Row titles (left side, vertical) - same size as column titles, with proper offset
    if show_row_titles:
        for row_idx in range(n_rows):
            if swap_rows_cols:
                row_title = run_titles[row_idx] if run_titles is not None else f'Run {row_idx + 1}'
            else:
                row_title = month_keys[row_idx]

            ax0 = axes[row_idx, 0]
            # Use larger negative offset to avoid overlap with lat tick labels
            ax0.text(
                -0.25, 0.5,
                row_title,
                transform=ax0.transAxes,
                rotation=90,
                va='center',
                ha='center',
                fontsize=title_fontsize,
                fontweight='bold'
            )

    # Colorbars
    if mesh_ref is not None:
        if use_shared_cbar:
            # Single shared colorbar for all subplots
            cbar = shared_colorbar(
                fig=fig,
                axes=axes.ravel(),
                mesh=mesh_ref,
                datasets=datasets_all,
                crange=avg_crange_global,
                label=avg_label,
                orientation=cbar_orientation,
                c_ticks=cbar_n_ticks,
                c_ticks_num=cbar_ticks_num_avg,
                tick_labelsize=cbar_tick_size,
                labelsize=cbar_label_size,
                pad=cbar_pad,
                thickness=cbar_thickness,
                label_pad=cbar_label_pad
            )
        else:
            # One colorbar per row (per month if not swapped, per run if swapped)
            n_cbars = len(month_keys)
            cbar_axes_avg = []  # Track colorbar axes for shared label
            for kk, month_key in enumerate(month_keys):
                mesh_k = month_mesh_groups[month_key]
                if mesh_k is None:
                    continue

                # Don't put label on individual colorbars if shared label is requested
                if shared_cbar_label_avg:
                    label_k = ''
                else:
                    label_k = avg_label

                # Use the appropriate crange
                if avg_crange_is_dict:
                    crange_k = avg_crange[month_key]
                else:
                    crange_k = avg_crange

                # Get ticks for this row
                if isinstance(cbar_ticks_avg_sep, list):
                    cbar_n_ticks_k = cbar_ticks_avg_sep[kk] if kk < len(cbar_ticks_avg_sep) else cbar_ticks_avg_sep[-1]
                else:
                    cbar_n_ticks_k = cbar_ticks_avg_sep

                cbar_k = shared_colorbar(
                    fig=fig,
                    axes=month_axes_groups[month_key],
                    mesh=mesh_k,
                    datasets=month_data_groups[month_key],
                    crange=crange_k,
                    label=label_k,
                    orientation=cbar_orientation,
                    c_ticks=cbar_n_ticks_k,
                    c_ticks_num=cbar_ticks_num_avg,
                    tick_labelsize=cbar_tick_size,
                    labelsize=cbar_label_size,
                    pad=cbar_pad,
                    thickness=cbar_thickness,
                    label_pad=cbar_label_pad,
                    extendfrac=0.05,
                    length_scale=0.95
                )
                cbar_axes_avg.append(cbar_k.ax)
            
            # Add centered shared label below all colorbars
            if shared_cbar_label_avg and cbar_axes_avg:
                add_shared_cbar_label(
                    fig, cbar_axes_avg, avg_label,
                    orientation=cbar_orientation,
                    fontsize=cbar_label_size,
                    pad=0.008
                )

    if save_name_avg is not None:
        out = Path.home() / 'KNMIproject' / 'RegionalTrends' / 'Main' / 'figuresproposal' / (save_name_avg + '.pdf')
        plt.savefig(str(out), 
                    format='pdf', 
                    bbox_inches='tight')

    plt.show()

#%% Plot linear trends

if trend_calc:

    month_keys = list(months_dict.keys())

    trend_crange_is_dict = isinstance(trend_crange, dict)
    
    use_shared_cbar_trend = (not trend_crange_is_dict) or shared_cbar_trend
    
    if trend_crange_is_dict:
        vmins = [float(v[0]) for v in trend_crange.values()]
        vmaxs = [float(v[1]) for v in trend_crange.values()]
        trend_crange_global = [min(vmins), max(vmaxs)]
    else:
        trend_crange_global = trend_crange

    if swap_rows_cols:
        n_rows, n_cols = n_runs, n_months
    else:
        n_rows, n_cols = n_months, n_runs

    fig_width = panel_width * n_cols
    fig_height = panel_height * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        subplot_kw={'projection': proj_plot},
        sharex=True,
        sharey=True
    )
    
    # # No extra padding between figures
    # if swap_rows_cols:
    #     fig.get_layout_engine().set(w_pad=0, wspace=0)
    # else:
    #     fig.get_layout_engine().set(h_pad=0, hspace=0)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    run_titles = None
    if month_keys:
        run_titles = [r.get('title', '') for r in all_results[month_keys[0]]]

    # Get colorbar sizing from helper function
    cbar_sizing = get_cbar_sizing(
        use_shared_cbar_trend, cbar_orientation, 
        panel_width, panel_height, fig_width, fig_height,
        shared_label=shared_cbar_label_trend
    )
    title_fontsize = cbar_sizing['title_fontsize']
    cbar_thickness = cbar_sizing['cbar_thickness']
    cbar_pad = cbar_sizing['cbar_pad']
    cbar_label_pad = cbar_sizing['cbar_label_pad']
    cbar_tick_size = cbar_sizing['cbar_tick_size']
    cbar_label_size = cbar_sizing['cbar_label_size']
    
    # Get tick count based on shared vs separate colorbar
    if use_shared_cbar_trend:
        cbar_n_ticks = cbar_ticks_trend_shared
    else:
        cbar_n_ticks = cbar_ticks_trend_sep if not isinstance(cbar_ticks_trend_sep, list) else cbar_ticks_trend_sep[0]

    mesh_ref = None
    datasets_all = []

    month_axes_groups = {mk: [] for mk in month_keys}
    month_data_groups = {mk: [] for mk in month_keys}
    month_mesh_groups = {mk: None for mk in month_keys}

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):

            if swap_rows_cols:
                run_idx = row_idx
                month_idx = col_idx
            else:
                month_idx = row_idx
                run_idx = col_idx

            month_key = month_keys[month_idx]
            res = all_results[month_key][run_idx]
            area_contours = all_area_contours[month_key][run_idx]

            ax = axes[row_idx, col_idx]

            if show_col_titles and row_idx == 0:
                if swap_rows_cols:
                    title = month_key
                else:
                    title = res.get('title', None)
            else:
                title = None

            show_y_labels = (col_idx == 0)
            show_x_labels = (row_idx == n_rows - 1)

            if use_shared_cbar_trend:
                crange_this = trend_crange_global
            elif trend_crange_is_dict:
                crange_this = trend_crange[month_key]
            else:
                crange_this = trend_crange

            mesh, _ = plot_map(
                fig, ax,
                res['trend_plot'],
                res['trend_plot'].longitude.values,
                res['trend_plot'].latitude.values,
                crange=crange_this,
                label=trend_label,
                cmap=var_colors_cfg[var]['cmap_trend'],
                extreme_colors=var_colors_cfg[var]['extreme_trend'],
                show_x_ticks=True,
                show_y_ticks=True,
                y_ticks_num=False,
                y_ticks=5,
                show_y_labels=show_y_labels,
                x_ticks_num=False,
                x_ticks=10,
                show_x_labels=show_x_labels,
                tick_size=16,
                extent=[*plot_lons, *plot_lats],
                proj=proj_plot,
                rotated_grid=cut_boundaries,
                title=title,
                title_size=title_fontsize,
                lats_area=area_contours['lats_area_cont'],
                lons_area=area_contours['lons_area_cont'],
                proj_area=area_contours['proj_area_cont'],
                mask_area=area_contours['mask_area'],
                lat_b_area=area_contours['lat_b_area'],
                lon_b_area=area_contours['lon_b_area'],
                show_plot=False,
                add_colorbar=False
            )

            if mesh_ref is None:
                mesh_ref = mesh

            datasets_all.append(res['trend_plot'])

            month_axes_groups[month_key].append(ax)
            month_data_groups[month_key].append(res['trend_plot'])
            if month_mesh_groups[month_key] is None:
                month_mesh_groups[month_key] = mesh

    # Row titles (left side, vertical) - same size as column titles, with proper offset
    if show_row_titles:
        for row_idx in range(n_rows):
            if swap_rows_cols:
                row_title = run_titles[row_idx] if run_titles is not None else f'Run {row_idx + 1}'
            else:
                row_title = month_keys[row_idx]

            ax0 = axes[row_idx, 0]
            # Use larger negative offset to avoid overlap with lat tick labels
            ax0.text(
                -0.25, 0.5,
                row_title,
                transform=ax0.transAxes,
                rotation=90,
                va='center',
                ha='center',
                fontsize=title_fontsize,
                fontweight='bold'
            )

    # Colorbars
    if mesh_ref is not None:
        if use_shared_cbar_trend:
            cbar = shared_colorbar(
                fig=fig,
                axes=axes.ravel(),
                mesh=mesh_ref,
                datasets=datasets_all,
                crange=trend_crange_global,
                label=trend_label,
                orientation=cbar_orientation,
                c_ticks=cbar_n_ticks,
                c_ticks_num=cbar_ticks_num_trend,
                tick_labelsize=cbar_tick_size,
                labelsize=cbar_label_size,
                pad=cbar_pad,
                thickness=cbar_thickness,
                label_pad=cbar_label_pad
            )
        else:
            n_cbars = len(month_keys)
            cbar_axes_trend = []  # Track colorbar axes for shared label
            for kk, month_key in enumerate(month_keys):
                mesh_k = month_mesh_groups[month_key]
                if mesh_k is None:
                    continue

                # Don't put label on individual colorbars if shared label is requested
                if shared_cbar_label_trend:
                    label_k = ''
                else:
                    label_k = trend_label

                if trend_crange_is_dict:
                    crange_k = trend_crange[month_key]
                else:
                    crange_k = trend_crange

                # Get ticks for this row
                if isinstance(cbar_ticks_trend_sep, list):
                    cbar_n_ticks_k = cbar_ticks_trend_sep[kk] if kk < len(cbar_ticks_trend_sep) else cbar_ticks_trend_sep[-1]
                else:
                    cbar_n_ticks_k = cbar_ticks_trend_sep

                cbar_k = shared_colorbar(
                    fig=fig,
                    axes=month_axes_groups[month_key],
                    mesh=mesh_k,
                    datasets=month_data_groups[month_key],
                    crange=crange_k,
                    label=label_k,
                    orientation=cbar_orientation,
                    c_ticks=cbar_n_ticks_k,
                    c_ticks_num=cbar_ticks_num_trend,
                    tick_labelsize=cbar_tick_size,
                    labelsize=cbar_label_size,
                    pad=cbar_pad,
                    thickness=cbar_thickness,
                    label_pad=cbar_label_pad,
                    extendfrac=0.05,
                    length_scale=0.9
                )
                cbar_axes_trend.append(cbar_k.ax)
            
            # Add centered shared label below all colorbars
            if shared_cbar_label_trend and cbar_axes_trend:
                add_shared_cbar_label(
                    fig, cbar_axes_trend, trend_label,
                    orientation=cbar_orientation,
                    fontsize=cbar_label_size,
                    pad=0.008
                )

    if save_name_trend is not None:
        out = Path.home() / 'KNMIproject' / 'RegionalTrends' / 'Main' / 'figuresproposal' / (save_name_trend + '.pdf')
        plt.savefig(str(out), 
                    format='pdf', 
                    bbox_inches='tight')

    plt.show()

#%% Plot correlation map

if corr_calc:

    month_keys = list(months_dict.keys())

    corr_crange_is_dict = isinstance(corr_crange, dict)
    
    use_shared_cbar_corr = (not corr_crange_is_dict) or shared_cbar_corr
    
    if corr_crange_is_dict:
        vmins = [float(v[0]) for v in corr_crange.values()]
        vmaxs = [float(v[1]) for v in corr_crange.values()]
        corr_crange_global = [min(vmins), max(vmaxs)]
    else:
        corr_crange_global = corr_crange

    if swap_rows_cols:
        n_rows, n_cols = n_runs, n_months
    else:
        n_rows, n_cols = n_months, n_runs

    fig_width = panel_width * n_cols
    fig_height = panel_height * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        subplot_kw={'projection': proj_plot},
        sharex=True,
        sharey=True
    )
    
    # # No extra padding between figures
    # if swap_rows_cols:
    #     fig.get_layout_engine().set(w_pad=0, wspace=0)
    # else:
    #     fig.get_layout_engine().set(h_pad=0, hspace=0)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    run_titles = None
    if month_keys:
        run_titles = [r.get('title', '') for r in all_results[month_keys[0]]]

    # Get colorbar sizing from helper function
    cbar_sizing = get_cbar_sizing(
        use_shared_cbar_corr, cbar_orientation, 
        panel_width, panel_height, fig_width, fig_height,
        shared_label=shared_cbar_label_corr
    )
    title_fontsize = cbar_sizing['title_fontsize']
    cbar_thickness = cbar_sizing['cbar_thickness']
    cbar_pad = cbar_sizing['cbar_pad']
    cbar_label_pad = cbar_sizing['cbar_label_pad']
    cbar_tick_size = cbar_sizing['cbar_tick_size']
    cbar_label_size = cbar_sizing['cbar_label_size']
    
    # Get tick count based on shared vs separate colorbar
    if use_shared_cbar_corr:
        cbar_n_ticks = cbar_ticks_corr_shared
    else:
        cbar_n_ticks = cbar_ticks_corr_sep if not isinstance(cbar_ticks_corr_sep, list) else cbar_ticks_corr_sep[0]

    mesh_ref = None
    datasets_all = []

    month_axes_groups = {mk: [] for mk in month_keys}
    month_data_groups = {mk: [] for mk in month_keys}
    month_mesh_groups = {mk: None for mk in month_keys}

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):

            if swap_rows_cols:
                run_idx = row_idx
                month_idx = col_idx
            else:
                month_idx = row_idx
                run_idx = col_idx

            month_key = month_keys[month_idx]
            res = all_results[month_key][run_idx]
            area_contours = all_area_contours[month_key][run_idx]

            if res['corr_plot'] is None:
                continue

            ax = axes[row_idx, col_idx]

            if show_col_titles and row_idx == 0:
                if swap_rows_cols:
                    title = month_key
                else:
                    title = res.get('title', None)
            else:
                title = None

            show_y_labels = (col_idx == 0)
            show_x_labels = (row_idx == n_rows - 1)

            if use_shared_cbar_corr:
                crange_this = corr_crange_global
            elif corr_crange_is_dict:
                crange_this = corr_crange[month_key]
            else:
                crange_this = corr_crange

            mesh, _ = plot_map(
                fig, ax,
                res['corr_plot'],
                res['corr_plot'].longitude.values,
                res['corr_plot'].latitude.values,
                crange=crange_this,
                label=corr_label,
                cmap=corr_cmap,
                extreme_colors=corr_meta['corr_extreme'],
                show_x_ticks=True,
                show_y_ticks=True,
                y_ticks_num=False,
                y_ticks=5,
                show_y_labels=show_y_labels,
                x_ticks_num=False,
                x_ticks=10,
                show_x_labels=show_x_labels,
                tick_size=16,
                extent=[*plot_lons, *plot_lats],
                proj=proj_plot,
                rotated_grid=cut_boundaries,
                title=title,
                title_size=title_fontsize,
                lats_area=area_contours['lats_area_cont'],
                lons_area=area_contours['lons_area_cont'],
                proj_area=area_contours['proj_area_cont'],
                mask_area=area_contours['mask_area'],
                lat_b_area=area_contours['lat_b_area'],
                lon_b_area=area_contours['lon_b_area'],
                show_plot=False,
                add_colorbar=False
            )

            if mesh_ref is None:
                mesh_ref = mesh

            datasets_all.append(res['corr_plot'])

            month_axes_groups[month_key].append(ax)
            month_data_groups[month_key].append(res['corr_plot'])
            if month_mesh_groups[month_key] is None:
                month_mesh_groups[month_key] = mesh

    # Row titles (left side, vertical) - same size as column titles, with proper offset
    if show_row_titles:
        for row_idx in range(n_rows):
            if swap_rows_cols:
                row_title = run_titles[row_idx] if run_titles is not None else f'Run {row_idx + 1}'
            else:
                row_title = month_keys[row_idx]

            ax0 = axes[row_idx, 0]
            # Use larger negative offset to avoid overlap with lat tick labels
            ax0.text(
                -0.25, 0.5,
                row_title,
                transform=ax0.transAxes,
                rotation=90,
                va='center',
                ha='center',
                fontsize=title_fontsize,
                fontweight='bold'
            )

    # Colorbars
    if mesh_ref is not None:
        if use_shared_cbar_corr:
            cbar = shared_colorbar(
                fig=fig,
                axes=axes.ravel(),
                mesh=mesh_ref,
                datasets=datasets_all,
                crange=corr_crange_global,
                label=corr_label,
                orientation=cbar_orientation,
                c_ticks=cbar_n_ticks,
                c_ticks_num=cbar_ticks_num_corr,
                tick_labelsize=cbar_tick_size,
                labelsize=cbar_label_size,
                pad=cbar_pad,
                thickness=cbar_thickness,
                label_pad=cbar_label_pad
            )
        else:
            n_cbars = len(month_keys)
            cbar_axes_corr = []  # Track colorbar axes for shared label
            for kk, month_key in enumerate(month_keys):
                mesh_k = month_mesh_groups[month_key]
                if mesh_k is None:
                    continue

                # Don't put label on individual colorbars if shared label is requested
                if shared_cbar_label_corr:
                    label_k = ''
                else:
                    label_k = corr_label

                if corr_crange_is_dict:
                    crange_k = corr_crange[month_key]
                else:
                    crange_k = corr_crange

                # Get ticks for this row
                if isinstance(cbar_ticks_corr_sep, list):
                    cbar_n_ticks_k = cbar_ticks_corr_sep[kk] if kk < len(cbar_ticks_corr_sep) else cbar_ticks_corr_sep[-1]
                else:
                    cbar_n_ticks_k = cbar_ticks_corr_sep

                cbar_k = shared_colorbar(
                    fig=fig,
                    axes=month_axes_groups[month_key],
                    mesh=mesh_k,
                    datasets=month_data_groups[month_key],
                    crange=crange_k,
                    label=label_k,
                    orientation=cbar_orientation,
                    c_ticks=cbar_n_ticks_k,
                    c_ticks_num=cbar_ticks_num_corr,
                    tick_labelsize=cbar_tick_size,
                    labelsize=cbar_label_size,
                    pad=cbar_pad,
                    thickness=cbar_thickness,
                    label_pad=cbar_label_pad,
                    extendfrac=0.05,
                    length_scale=0.95
                )
                cbar_axes_corr.append(cbar_k.ax)
            
            # Add centered shared label below all colorbars
            if shared_cbar_label_corr and cbar_axes_corr:
                add_shared_cbar_label(
                    fig, cbar_axes_corr, corr_label,
                    orientation=cbar_orientation,
                    fontsize=cbar_label_size,
                    pad=0.008
                )

    if save_name_corr is not None:
        out = Path.home() / 'KNMIproject' / 'RegionalTrends' / 'Main' / 'figuresproposal' / (save_name_corr + '.pdf')
        plt.savefig(str(out), 
                    format='pdf', 
                    bbox_inches='tight')

    plt.show()

#%% Plot RMSE map

if rmse_calc:

    month_keys = list(months_dict.keys())

    rmse_crange_is_dict = isinstance(rmse_crange, dict)
    
    use_shared_cbar_rmse = (not rmse_crange_is_dict) or shared_cbar_rmse
    
    if rmse_crange_is_dict:
        vmins = [float(v[0]) for v in rmse_crange.values()]
        vmaxs = [float(v[1]) for v in rmse_crange.values()]
        rmse_crange_global = [min(vmins), max(vmaxs)]
    else:
        rmse_crange_global = rmse_crange

    if swap_rows_cols:
        n_rows, n_cols = n_runs, n_months
    else:
        n_rows, n_cols = n_months, n_runs

    fig_width = panel_width * n_cols
    fig_height = panel_height * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        subplot_kw={'projection': proj_plot},
        sharex=True,
        sharey=True
    )
    
    # # No extra padding between figures
    # if swap_rows_cols:
    #     fig.get_layout_engine().set(w_pad=0, wspace=0)
    # else:
    #     fig.get_layout_engine().set(h_pad=0, hspace=0)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    run_titles = None
    if month_keys:
        run_titles = [r.get('title', '') for r in all_results[month_keys[0]]]

    # Get colorbar sizing from helper function
    cbar_sizing = get_cbar_sizing(
        use_shared_cbar_rmse, cbar_orientation, 
        panel_width, panel_height, fig_width, fig_height,
        shared_label=shared_cbar_label_rmse
    )
    title_fontsize = cbar_sizing['title_fontsize']
    cbar_thickness = cbar_sizing['cbar_thickness']
    cbar_pad = cbar_sizing['cbar_pad']
    cbar_label_pad = cbar_sizing['cbar_label_pad']
    cbar_tick_size = cbar_sizing['cbar_tick_size']
    cbar_label_size = cbar_sizing['cbar_label_size']

    # Get tick count based on shared vs separate colorbar
    if use_shared_cbar_rmse:
        cbar_n_ticks = cbar_ticks_rmse_shared
    else:
        cbar_n_ticks = cbar_ticks_rmse_sep if not isinstance(cbar_ticks_rmse_sep, list) else cbar_ticks_rmse_sep[0]

    mesh_ref = None
    datasets_all = []

    month_axes_groups = {mk: [] for mk in month_keys}
    month_data_groups = {mk: [] for mk in month_keys}
    month_mesh_groups = {mk: None for mk in month_keys}

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):

            if swap_rows_cols:
                run_idx = row_idx
                month_idx = col_idx
            else:
                month_idx = row_idx
                run_idx = col_idx

            month_key = month_keys[month_idx]
            res = all_results[month_key][run_idx]
            area_contours = all_area_contours[month_key][run_idx]

            if res['rmse_plot'] is None:
                continue

            ax = axes[row_idx, col_idx]

            if show_col_titles and row_idx == 0:
                if swap_rows_cols:
                    title = month_key
                else:
                    title = res.get('title', None)
            else:
                title = None

            show_y_labels = (col_idx == 0)
            show_x_labels = (row_idx == n_rows - 1)

            if use_shared_cbar_rmse:
                crange_this = rmse_crange_global
            elif rmse_crange_is_dict:
                crange_this = rmse_crange[month_key]
            else:
                crange_this = rmse_crange

            mesh, _ = plot_map(
                fig, ax,
                res['rmse_plot'],
                res['rmse_plot'].longitude.values,
                res['rmse_plot'].latitude.values,
                crange=crange_this,
                label=rmse_label,
                cmap=rmse_cmap,
                extreme_colors=[None, "#24050a"],
                show_x_ticks=True,
                show_y_ticks=True,
                y_ticks_num=False,
                y_ticks=5,
                show_y_labels=show_y_labels,
                x_ticks_num=False,
                x_ticks=10,
                show_x_labels=show_x_labels,
                tick_size=16,
                extent=[*plot_lons, *plot_lats],
                proj=proj_plot,
                rotated_grid=cut_boundaries,
                title=title,
                title_size=title_fontsize,
                lats_area=area_contours['lats_area_cont'],
                lons_area=area_contours['lons_area_cont'],
                proj_area=area_contours['proj_area_cont'],
                mask_area=area_contours['mask_area'],
                lat_b_area=area_contours['lat_b_area'],
                lon_b_area=area_contours['lon_b_area'],
                show_plot=False,
                add_colorbar=False
            )

            if mesh_ref is None:
                mesh_ref = mesh

            datasets_all.append(res['rmse_plot'])

            month_axes_groups[month_key].append(ax)
            month_data_groups[month_key].append(res['rmse_plot'])
            if month_mesh_groups[month_key] is None:
                month_mesh_groups[month_key] = mesh

    # Row titles (left side, vertical) - same size as column titles, with proper offset
    if show_row_titles:
        for row_idx in range(n_rows):
            if swap_rows_cols:
                row_title = run_titles[row_idx] if run_titles is not None else f'Run {row_idx + 1}'
            else:
                row_title = month_keys[row_idx]

            ax0 = axes[row_idx, 0]
            # Use larger negative offset to avoid overlap with lat tick labels
            ax0.text(
                -0.25, 0.5,
                row_title,
                transform=ax0.transAxes,
                rotation=90,
                va='center',
                ha='center',
                fontsize=title_fontsize,
                fontweight='bold'
            )

    # Colorbars
    if mesh_ref is not None:
        if use_shared_cbar_rmse:
            cbar = shared_colorbar(
                fig=fig,
                axes=axes.ravel(),
                mesh=mesh_ref,
                datasets=datasets_all,
                crange=rmse_crange_global,
                label=rmse_label,
                orientation=cbar_orientation,
                c_ticks=cbar_n_ticks,
                c_ticks_num=cbar_ticks_num_rmse,
                tick_labelsize=cbar_tick_size,
                labelsize=cbar_label_size,
                pad=cbar_pad,
                thickness=cbar_thickness,
                label_pad=cbar_label_pad
            )
        else:
            n_cbars = len(month_keys)
            cbar_axes_rmse = []  # Track colorbar axes for shared label
            for kk, month_key in enumerate(month_keys):
                mesh_k = month_mesh_groups[month_key]
                if mesh_k is None:
                    continue

                # Don't put label on individual colorbars if shared label is requested
                if shared_cbar_label_rmse:
                    label_k = ''
                else:
                    label_k = rmse_label

                if rmse_crange_is_dict:
                    crange_k = rmse_crange[month_key]
                else:
                    crange_k = rmse_crange

                # Get ticks for this row
                if isinstance(cbar_ticks_rmse_sep, list):
                    cbar_n_ticks_k = cbar_ticks_rmse_sep[kk] if kk < len(cbar_ticks_rmse_sep) else cbar_ticks_rmse_sep[-1]
                else:
                    cbar_n_ticks_k = cbar_ticks_rmse_sep

                cbar_k = shared_colorbar(
                    fig=fig,
                    axes=month_axes_groups[month_key],
                    mesh=mesh_k,
                    datasets=month_data_groups[month_key],
                    crange=crange_k,
                    label=label_k,
                    orientation=cbar_orientation,
                    c_ticks=cbar_n_ticks_k,
                    c_ticks_num=cbar_ticks_num_rmse,
                    tick_labelsize=cbar_tick_size,
                    labelsize=cbar_label_size,
                    pad=cbar_pad,
                    thickness=cbar_thickness,
                    label_pad=cbar_label_pad,
                    extendfrac=0.05,
                    length_scale=0.95
                )
                cbar_axes_rmse.append(cbar_k.ax)
            
            # Add centered shared label below all colorbars
            if shared_cbar_label_rmse and cbar_axes_rmse:
                add_shared_cbar_label(
                    fig, cbar_axes_rmse, rmse_label,
                    orientation=cbar_orientation,
                    fontsize=cbar_label_size,
                    pad=0.008
                )

    if save_name_rmse is not None:
        out = Path.home() / 'KNMIproject' / 'RegionalTrends' / 'Main' / 'figuresproposal' / (save_name_rmse + '.pdf')
        plt.savefig(str(out), 
                    format='pdf', 
                    bbox_inches='tight')

    plt.show()

#%%



# SLIMMERE DATACACHE MAKEN (DUS ALS IK DUS ALL AL HEB, DAN KAN IK NET ZO GOED OOK DIE MAANDEN SELECTEREN.) (is wellicht niet makkelijk)
# SHARED COLORBAR TOEVOEGEN
# KIEZEN VAN ORIENTATIE VAN DE COLORBAR
# KEUZE UIT COLUMNS (DUS MAAND COLUMNS OF RUN COLUMNS)
# TITEL VAN MAANDEN

# Grotere tick labels. (en grotere colorbar labels)