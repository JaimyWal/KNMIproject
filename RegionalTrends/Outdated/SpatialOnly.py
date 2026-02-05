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
from RegionalTrends.Helpers.PlotMaps import plot_map, shared_colorbar

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

#%% User inputs

# Main arguments
n_runs = 3
var = 'Tg' #
file_freq = 'Monthly' #
data_base = ['ERA5_coarse', 'Eobs_fine', 'RACMO2.4_KEXT12'] #
data_compare = None #

# data_base = ['ERA5_coarse', 'RACMO2.3', 'RACMO2.4_KEXT12', 'Eobs_fine'] #
# data_compare = None #

# data_base = ['ERA5_coarse', 'ERA5_coarse', 'Eobs_fine', 'Eobs_fine'] #
# data_compare = ['RACMO2.3', 'RACMO2.4_KEXT12', 'RACMO2.3', 'RACMO2.4_KEXT12'] #

# Data selection arguments
months = None #
years = [1974, 2024] 
lats = [37.7, 63.3]
lons = [-13.3, 22.3]
proj_sel = 'RACMO2.4'
land_only = False
trim_border = None

# Spatial plotting arguments
avg_crange = [0, 20]
plot_climatology = True
proj_plot = 'RACMO2.4'
plot_lats = [38, 63]
plot_lons = [-13, 22]
std_mask_ref = data_base #
std_dir = 'Lesser' #
switch_sign = False #
cut_boundaries = False
cmap_type = None
n_disc_colors = 20 
fixed_aspect = True

# std_mask_ref = ['ERA5_coarse', 'ERA5_coarse', 'Eobs_fine', 'Eobs_fine'] #

# Trend plotting arguments
trend_calc = True
trend_crange = [-1, 1]
fit_against_gmst = False

# Correlation plotting arguments
corr_calc = False
corr_freq = 'Monthly' #
corr_crange = [0, 1] 
corr_cmap_type = None 

# RMSE plotting arguments
rmse_calc = False
rmse_freq = 'Monthly' #
rmse_crange = [0, 2]

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

# lats = [38, 63]
# lons = [-13, 22]
# lats_area = [50.7, 53.6]
# lons_area = [3.25, 7.35]

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
    label_unit = '% / ' + fit_unit
elif var_units_cfg[var] == '':
    label_unit = 'per' + fit_unit
else:
    label_unit = var_units_cfg[var] + ' / ' + fit_unit

avg_label = var_symbol_cfg[var] + ' (' + var_units_cfg[var] + ')'
avg_cmap_key = 'cmap_mean'
avg_ext_key = 'extreme_mean'
if data_compare is not None:
    avg_label = r'$\Delta$' + avg_label
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

def ensure_list(param, n, nested=False):

    if not isinstance(param, list):
        return [param]*n

    if not nested:
        return param

    if any(isinstance(p, list) for p in param):
        return param

    return [param]*n

data_base_list = ensure_list(data_base, n_runs)
freq_file_list = ensure_list(file_freq, n_runs)
data_compare_list = ensure_list(data_compare, n_runs)
months_list = ensure_list(months, n_runs, nested=True)
std_mask_ref_list = ensure_list(std_mask_ref, n_runs)
std_dir_list = ensure_list(std_dir, n_runs)
switch_sign_list = ensure_list(switch_sign, n_runs)
corr_freq_list = ensure_list(corr_freq, n_runs)
rmse_freq_list = ensure_list(rmse_freq, n_runs)

#%% Further processing of base and comparison data

def make_cache_key(data_source, months):
    return (data_source, None if months is None else tuple(months))

data_cache = {}
results = []

for ii in range(n_runs):

    if data_base_list[ii] is not None:

        cache_key_base = make_cache_key(data_base_list[ii], months_list[ii])
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
                months=months_list[ii],
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

            cache_key_comp = make_cache_key(data_compare_list[ii], months_list[ii])
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
                    months=months_list[ii],
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
                        cache_key_ref = make_cache_key(std_mask_ref_list[ii], months_list[ii])
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
                                months=months_list[ii],
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

#%% Area plotting selection

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

#%% Plot climatology

if plot_climatology:

    meshes = []
    data_avg_field = []
    x_tick_bool = [True]*n_runs    
    y_tick_bool = [False]*n_runs
    y_tick_bool[0] = True

    fig, axes = plt.subplots(
        1, n_runs,
        figsize=(18, 5),
        constrained_layout=True,
        subplot_kw={'projection': proj_plot},
        sharex=True,
        sharey=True
    )

    axes = np.atleast_1d(axes).ravel()

    for ii, (ax, res) in enumerate(zip(axes, results)):

        if data_base_list[ii] is not None:

            data_avg_field.append(res['data_avg_plot'])

            title = res['title']

            mesh, _ = plot_map(
                fig, ax,
                res['data_avg_plot'], 
                res['data_avg_plot'].longitude.values, 
                res['data_avg_plot'].latitude.values, 
                crange=avg_crange, 
                label=avg_label, 
                cmap=var_colors_cfg[var][avg_cmap_key], 
                extreme_colors=var_colors_cfg[var][avg_ext_key],
                c_ticks=10,
                show_x_ticks=True,
                show_y_ticks=True,
                y_ticks_num=False,
                y_ticks=5,
                show_y_labels=y_tick_bool[ii],
                x_ticks_num=False,
                x_ticks=10,
                show_x_labels=x_tick_bool[ii],
                tick_size=20,
                extent=[*plot_lons, *plot_lats],
                proj=proj_plot,
                rotated_grid=cut_boundaries,
                title=title,
                lats_area=area_contours[ii]['lats_area_cont'],
                lons_area=area_contours[ii]['lons_area_cont'],
                proj_area=area_contours[ii]['proj_area_cont'],
                mask_area=area_contours[ii]['mask_area'],
                lat_b_area=area_contours[ii]['lat_b_area'],
                lon_b_area=area_contours[ii]['lon_b_area'],
                show_plot=False,
                add_colorbar=False
            )

            meshes.append(mesh)

            if std_mask_ref_list[ii] is not None and data_compare_list[ii] is not None:
                ax.contourf(
                    res['data_avg_plot'].longitude.values, res['data_avg_plot'].latitude.values, 
                    res['mask_std'],
                    levels=[0.5, 1.5],
                    colors='none',
                    hatches=['///'],
                    transform=ccrs.PlateCarree(),
                    zorder=50
                )

    cbar = shared_colorbar(
            fig=fig,
            axes=axes,
            mesh=meshes[0],
            datasets=data_avg_field,
            crange=avg_crange,
            label=avg_label,
            orientation='horizontal',
            c_ticks=10,
            c_ticks_num=True,
            tick_labelsize=26,
            labelsize=32,
            pad=0.1,
            thickness=0.06
        )

    plt.show()

#%% Plot linear trends

if trend_calc:

    meshes = []
    data_trend_field = []
    x_tick_bool = [True]*n_runs    
    y_tick_bool = [False]*n_runs
    y_tick_bool[0] = True

    fig, axes = plt.subplots(
        1, n_runs,
        figsize=(18, 5),
        constrained_layout=True,
        subplot_kw={'projection': proj_plot},
        sharex=True,
        sharey=True
    )

    axes = np.atleast_1d(axes).ravel()

    for ii, (ax, res) in enumerate(zip(axes, results)):

        if data_base_list[ii] is not None:

            data_trend_field.append(res['trend_plot'])
            title = res['title']

            mesh, _ = plot_map(
                fig, ax,
                res['trend_plot'], 
                res['trend_plot'].longitude.values, 
                res['trend_plot'].latitude.values,
                crange=trend_crange, 
                label=var_symbol_cfg[var] + ' trend (' + label_unit + ')', 
                cmap=var_colors_cfg[var]['cmap_trend'], 
                extreme_colors=var_colors_cfg[var]['extreme_trend'],
                c_ticks=10,
                show_x_ticks=True,
                show_y_ticks=True,
                y_ticks_num=False,
                y_ticks=5,
                show_y_labels=y_tick_bool[ii],
                x_ticks_num=False,
                x_ticks=10,
                show_x_labels=x_tick_bool[ii],
                tick_size=20,
                extent=[*plot_lons, *plot_lats],
                proj=proj_plot,
                rotated_grid=cut_boundaries,
                title=title,
                lats_area=area_contours[ii]['lats_area_cont'],
                lons_area=area_contours[ii]['lons_area_cont'],
                proj_area=area_contours[ii]['proj_area_cont'],
                mask_area=area_contours[ii]['mask_area'],
                lat_b_area=area_contours[ii]['lat_b_area'],
                lon_b_area=area_contours[ii]['lon_b_area'],
                show_plot=False,
                add_colorbar=False
            )

            meshes.append(mesh)

    cbar = shared_colorbar(
        fig=fig,
        axes=axes,
        mesh=meshes[0],
        datasets=data_trend_field,
        crange=trend_crange,
        label=var_symbol_cfg[var] + ' trend (' + label_unit + ')',
        orientation='horizontal',
        c_ticks=10,
        c_ticks_num=True,
        tick_labelsize=26,
        labelsize=32,
        pad=0.1,
        thickness=0.06
    )

    plt.show()

#%% Plot correlation map

if corr_calc:

    meshes = []
    data_corr_field = []
    x_tick_bool = [True]*n_runs    
    y_tick_bool = [False]*n_runs
    y_tick_bool[0] = True

    fig, axes = plt.subplots(
        1, n_runs,
        figsize=(18, 5),
        constrained_layout=True,
        subplot_kw={'projection': proj_plot},
        sharex=True,
        sharey=True
    )

    axes = np.atleast_1d(axes).ravel()

    for ii, (ax, res) in enumerate(zip(axes, results)):

        if data_base_list[ii] is not None and data_compare_list[ii] is not None:

            data_corr_field.append(res['corr_plot'])
            title = res['title']

            mesh, _ = plot_map(
                fig, ax,
                res['corr_plot'], 
                res['corr_plot'].longitude.values, 
                res['corr_plot'].latitude.values,
                crange=corr_crange, 
                label=var_symbol_cfg[var] + ' correlation', 
                cmap=corr_cmap, 
                extreme_colors=corr_meta['extreme_colors'],
                c_ticks=10,
                show_x_ticks=True,
                show_y_ticks=True,
                y_ticks_num=False,
                y_ticks=5,
                show_y_labels=y_tick_bool[ii],
                x_ticks_num=False,
                x_ticks=10,
                show_x_labels=x_tick_bool[ii],
                tick_size=20,
                extent=[*plot_lons, *plot_lats],
                proj=proj_plot,
                rotated_grid=cut_boundaries,
                title=title,
                lats_area=area_contours[ii]['lats_area_cont'],
                lons_area=area_contours[ii]['lons_area_cont'],
                proj_area=area_contours[ii]['proj_area_cont'],
                mask_area=area_contours[ii]['mask_area'],
                lat_b_area=area_contours[ii]['lat_b_area'],
                lon_b_area=area_contours[ii]['lon_b_area'],
                show_plot=False,
                add_colorbar=False
            )

            meshes.append(mesh)

    cbar = shared_colorbar(
        fig=fig,
        axes=axes,
        mesh=meshes[0],
        datasets=data_corr_field,
        crange=corr_crange,
        label=var_symbol_cfg[var] + ' correlation',
        orientation='horizontal',
        c_ticks=10,
        c_ticks_num=True,
        tick_labelsize=26,
        labelsize=32,
        pad=0.1,
        thickness=0.06
    )

    plt.show()

#%% Plot RMSE map

if rmse_calc:

    meshes = []
    data_rmse_field = []
    x_tick_bool = [True]*n_runs    
    y_tick_bool = [False]*n_runs
    y_tick_bool[0] = True

    fig, axes = plt.subplots(
        1, n_runs,
        figsize=(18, 5),
        constrained_layout=True,
        subplot_kw={'projection': proj_plot},
        sharex=True,
        sharey=True
    )

    axes = np.atleast_1d(axes).ravel()

    for ii, (ax, res) in enumerate(zip(axes, results)):

        if data_base_list[ii] is not None and data_compare_list[ii] is not None:

            data_rmse_field.append(res['rmse_plot'])
            title = res['title']

            mesh, _ = plot_map(
                fig, ax,
                res['rmse_plot'], 
                res['rmse_plot'].longitude.values, 
                res['rmse_plot'].latitude.values,
                crange=rmse_crange, 
                label=var_symbol_cfg[var] + ' RMSE (' + var_units_cfg[var] + ')', 
                cmap=cmocean.cm.amp, 
                extreme_colors=[None, "#24050a"],
                c_ticks=10,
                show_x_ticks=True,
                show_y_ticks=True,
                y_ticks_num=False,
                y_ticks=5,
                show_y_labels=y_tick_bool[ii],
                x_ticks_num=False,
                x_ticks=10,
                show_x_labels=x_tick_bool[ii],
                tick_size=20,
                extent=[*plot_lons, *plot_lats],
                proj=proj_plot,
                rotated_grid=cut_boundaries,
                title=title,
                lats_area=area_contours[ii]['lats_area_cont'],
                lons_area=area_contours[ii]['lons_area_cont'],
                proj_area=area_contours[ii]['proj_area_cont'],
                mask_area=area_contours[ii]['mask_area'],
                lat_b_area=area_contours[ii]['lat_b_area'],
                lon_b_area=area_contours[ii]['lon_b_area'],
                show_plot=False,
                add_colorbar=False
            )

            meshes.append(mesh)

    cbar = shared_colorbar(
        fig=fig,
        axes=axes,
        mesh=meshes[0],
        datasets=data_rmse_field,
        crange=rmse_crange,
        label=var_symbol_cfg[var] + ' RMSE (' + var_units_cfg[var] + ')',
        orientation='horizontal',
        c_ticks=10,
        c_ticks_num=True,
        tick_labelsize=26,
        labelsize=32,
        pad=0.1,
        thickness=0.06
    )

    plt.show()

#%%
