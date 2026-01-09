#%% Imports

# Standard libraries
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import xesmf as xe
import statsmodels.api as sm
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

from RegionalTrends.Helpers import ProcessSource
reload(ProcessSource)
from RegionalTrends.Helpers.ProcessSource import process_source

from RegionalTrends.Helpers import GridBounds
reload(GridBounds)          
from RegionalTrends.Helpers.GridBounds import grid_with_bounds

from RegionalTrends.Helpers import AreaWeights
reload(AreaWeights)
from RegionalTrends.Helpers.AreaWeights import area_weights, area_weighted_mean

# Data config custom libraries
import RegionalTrends.Helpers.Config.Constants as Constants
reload(Constants)

import RegionalTrends.Helpers.Config.Paths as Paths
reload(Paths)
from RegionalTrends.Helpers.Config.Paths import build_file_cfg, freq_tags

import RegionalTrends.Helpers.Config.Plotting as Plotting
reload(Plotting)
from RegionalTrends.Helpers.Config.Plotting import build_corr_cmap, build_plot_cfg,\
      plot_args, fit_settings


plt.rcParams['axes.unicode_minus'] = False
dask.config.set(scheduler='threads', num_workers=12)

#%% User inputs

# Main arguments
var = 'CloudTotal'
data_base = 'RACMO2.3'
data_compare = 'RACMO2.4'

# Data selection arguments
months = [12, 1, 2]
years = [2016, 2020]
lats = [37.7, 63.3]
lons = [-13.3, 22.3]
proj_sel = 'RACMO2.4'
land_only = False
trim_border = None

# Area selection arguments
data_area = None
stations = None
lats_area = [50.7, 53.6]
lons_area = [3.25, 7.35]
land_only_area = True
proj_area = 'RACMO2.4'

# Spatial plotting arguments
avg_crange = [0, 100]
trend_crange = [-20, 20]
proj_plot = 'RACMO2.4'
plot_lats = [38, 63]
plot_lons = [-13, 22]
true_contour = True
grid_contour = True
switch_sign = False
cut_boundaries = False

# Correlation plotting arguments
corr_calc = True
corr_freq = 'Daily'
corr_crange = [-1, 1]
corr_cmap_neg = True 

# Area plotting arguments
fit_range = None
plots_area = False
crange_area = trend_crange
lats_area_plot = [50.2, 54.1]
lons_area_plot = [2.5, 8.1]
uncertainty_band = False

# Other arguments
relative_precip = False
rolling_mean_var = False
fit_against_gmst = False
rolling_mean_years = 7
min_periods = 1

# lats = [38, 63]
# lons = [-13, 22]
# data_area = ['Observed', 'ERA5_coarse', 'RACMO2.3', 'RACMO2.4']
# stations = ['Bilt', 'Eelde', 'Maastricht', 'Vlissingen', 'Kooy']
# lats_area = [50.7, 53.6]
# lons_area = [3.25, 7.35]

#%% Dataset configurations

data_sources = Constants.DATA_SOURCES
station_sources = Constants.STATION_SOURCES
var_name_cfg = Constants.VAR_NAME_CFG
station_coord_cfg = Constants.STATION_COORD_CFG
proj_cfg = Constants.PROJ_CFG

fit_unit, fit_scaling, fit_x_label = fit_settings(fit_against_gmst)

if corr_calc == True and corr_freq == 'Daily' and data_compare is not None:
    monthly_or_daily = 'Daily'
else:
    monthly_or_daily = 'Monthly'
freq_str, racmo24_sep = freq_tags(monthly_or_daily)

file_cfg = build_file_cfg(freq_str, racmo24_sep)

plot_cfg = build_plot_cfg(
    avg_crange, 
    trend_crange,
    fit_unit,
    fit_range,
    relative_precip=relative_precip
)

cfg_plot = plot_args(plot_cfg, var, data_compare)

proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())
proj_area = proj_cfg.get(proj_area, ccrs.PlateCarree())
proj_plot = proj_cfg.get(proj_plot, ccrs.PlateCarree())

#%% Further processing of base and comparison data

if data_base is not None:

    data_base_res = process_source(
        data_base, 
        var,
        data_sources,
        station_sources,
        var_name_cfg,
        file_cfg,
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

    if data_compare is None:
        fits_base = data_base_res['fit'].polyfit(dim='fit_against', deg=1, skipna=True)
        slope_base = fits_base.polyfit_coefficients.sel(degree=1)
        trend_base = (slope_base*fit_scaling).astype('float32').compute()

        if relative_precip and var == 'P':
            trend_plot_base = (trend_base / data_base_res['avg'])*100.0
        else:
            trend_plot_base = trend_base

        data_avg_plot = data_base_res['avg']
        trend_plot = trend_plot_base

    elif data_compare is not None:

        data_comp_res = process_source(
            data_compare, 
            var,
            data_sources,
            station_sources,
            var_name_cfg,
            file_cfg,
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

        if var == 'P':
            method = 'conservative_normed'
            trg_grid = grid_with_bounds(data_base_res['avg'], rotpole_native=proj_cfg.get(data_base, ccrs.PlateCarree()))
            src_grid = grid_with_bounds(data_comp_res['avg'], rotpole_native=proj_cfg.get(data_compare, ccrs.PlateCarree()))

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

        data_fit_comp_reg = regridder(
            data_comp_res['fit'],
            output_chunks=target_chunks
        ).astype('float32')

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

        if switch_sign:
            minus_scaling = -1
        else:
            minus_scaling = 1

        data_avg_plot = minus_scaling*(data_avg_comp_reg - data_base_res['avg']).compute()
        trend_plot = minus_scaling*(trend_plot_comp - trend_plot_base).compute()

        if corr_calc:

            if corr_freq == 'Daily':
                data_corr_base = data_base_res['raw']
                data_corr_comp = data_comp_res['raw']
            elif corr_freq == 'Monthly':
                data_corr_base = data_base_res['monthly']
                data_corr_comp = data_comp_res['monthly']
            elif corr_freq == 'Yearly':
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

    trend_plot = trend_plot.assign_coords(
                    latitude=data_base_res['avg']['latitude'],
                    longitude=data_base_res['avg']['longitude']
                ).astype('float32')

    lat_plot = data_avg_plot['latitude'].values
    lon_plot = data_avg_plot['longitude'].values

#%% Area plotting selection

if data_base is not None: 

    mask_area = None
    lat_b_area = None
    lon_b_area = None

    lats_area_cont = None
    lons_area_cont = None 
    proj_area_cont = None

    if isinstance(lats_area, (list, tuple)) and len(lats_area) == 2 and \
    isinstance(lons_area, (list, tuple)) and len(lons_area) == 2 and grid_contour == True:

        contour_area_full = xr.full_like(data_avg_plot, fill_value=1.0)

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
            rotpole_native=proj_cfg.get(data_base, ccrs.PlateCarree())
        )

        mask_area = np.isfinite(contour_area.values)

        contour_area_bounds = grid_with_bounds(
            contour_area,
            rotpole_native=proj_cfg.get(data_base, ccrs.PlateCarree())
        )
        lon_b_area, lat_b_area = contour_area_bounds['lon_b'], contour_area_bounds['lat_b']

    if isinstance(lats_area, (list, tuple)) and len(lats_area) == 2 and \
    isinstance(lons_area, (list, tuple)) and len(lons_area) == 2 and true_contour == True:
        
        lats_area_cont = lats_area
        lons_area_cont = lons_area
        proj_area_cont = proj_area

#%% Plot climatology

if data_base is not None:

    fig, ax = plt.subplots(
        1, figsize=(14, 12), 
        constrained_layout=True,
        subplot_kw={'projection': proj_plot}
    )

    plot_map(
        fig, ax,
        data_avg_plot, 
        lon_plot, 
        lat_plot, 
        crange=cfg_plot['crange_mean'], 
        label=cfg_plot['label_mean'], 
        cmap=cfg_plot['cmap_mean'], 
        extreme_colors=cfg_plot['extreme_mean'],
        c_ticks=10,
        show_x_ticks=True,
        show_y_ticks=True,
        y_ticks_num=False,
        y_ticks=5,
        x_ticks_num=False,
        x_ticks=10,
        extent=[*plot_lons, *plot_lats],
        proj=proj_plot,
        rotated_grid=cut_boundaries,
        lats_area=lats_area_cont,
        lons_area=lons_area_cont,
        proj_area=proj_area_cont,
        mask_area=mask_area,
        lat_b_area=lat_b_area,
        lon_b_area=lon_b_area
    )

#%% Plot linear trends

if data_base is not None:
    fig, ax = plt.subplots(
        1, figsize=(14, 12), 
        constrained_layout=True,
        subplot_kw={'projection': proj_plot}
    )

    plot_map(
        fig, ax,
        trend_plot, 
        lon_plot, 
        lat_plot, 
        crange=cfg_plot['crange_trend'], 
        label=cfg_plot['label_trend'], 
        cmap=cfg_plot['cmap_trend'], 
        extreme_colors=cfg_plot['extreme_trend'],
        c_ticks=10,
        show_x_ticks=True,
        show_y_ticks=True,
        y_ticks_num=False,
        y_ticks=5,
        x_ticks_num=False,
        x_ticks=10,
        extent=[*plot_lons, *plot_lats],
        proj=proj_plot,
        rotated_grid=cut_boundaries,
        lats_area=lats_area_cont,
        lons_area=lons_area_cont,
        proj_area=proj_area_cont,
        mask_area=mask_area,
        lat_b_area=lat_b_area,
        lon_b_area=lon_b_area
    )

#%% Plot correlation map

if data_base is not None and data_compare is not None and corr_calc:

    corr_meta = build_corr_cmap(corr_cmap_neg)
    corr_cmap = corr_meta['corr_cmap']
    corr_extreme = corr_meta['corr_extreme']

    fig, ax = plt.subplots(
        1, figsize=(14, 12), 
        constrained_layout=True,
        subplot_kw={'projection': proj_plot}
    )

    plot_map(
        fig, ax,
        corr_plot, 
        lon_plot, 
        lat_plot, 
        crange=corr_crange, 
        label='Correlation', 
        cmap=corr_cmap,
        extreme_colors=corr_extreme,
        c_ticks=10,
        show_x_ticks=True,
        show_y_ticks=True,
        y_ticks_num=False,
        y_ticks=5,
        x_ticks_num=False,
        x_ticks=10,
        extent=[*plot_lons, *plot_lats],
        proj=proj_plot,
        rotated_grid=cut_boundaries
    )

#%% Loading data for chosen area

def combine_lists(a, b):
    if a is None and b is None:
        return None
    return (a or []) + (b or [])

data_area_all = combine_lists(data_area, stations)

if data_area_all is not None:

    data_area_avg_raw = {}
    data_area_fit_raw = {}
    data_area_avg = {}
    data_area_monthly = {}
    data_area_fit = {}

    for src in data_area_all:

        is_station = src in station_sources

        if not is_station:
            if isinstance(lats_area, str) or isinstance(lons_area, str):
                station_name = lats_area if isinstance(lats_area, str) else lons_area
                lat_sel = station_coord_cfg[station_name]['latitude']
                lon_sel = station_coord_cfg[station_name]['longitude']
            else:
                lat_sel = lats_area
                lon_sel = lons_area
        else:
            lat_sel = None
            lon_sel = None

        data_res = process_source(
            src,
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
            land_only=land_only_area,
            trim_border=trim_border,
            rotpole_sel=proj_area,
            rolling_mean_var=rolling_mean_var,
            fit_against_gmst=fit_against_gmst,
            rolling_mean_years=rolling_mean_years,
            min_periods=min_periods
        )

        data_area_avg_raw[src] = data_res['avg']
        data_area_fit_raw[src] = data_res['fit']

        weights = area_weights(data_res['avg'], rotpole_native=proj_cfg.get(src, ccrs.PlateCarree()))
        data_area_avg[src] = area_weighted_mean(data_res['avg'], weights=weights)
        monthly_raw = area_weighted_mean(data_res['monthly'], weights=weights)
        yearly_raw = area_weighted_mean(data_res['fit'], weights=weights)

        if var == 'P' and relative_precip:
            data_area_monthly[src] = 100*monthly_raw / data_area_avg[src]
            data_area_fit[src] = 100*yearly_raw / data_area_avg[src]
        else:
            data_area_monthly[src] = monthly_raw
            data_area_fit[src] = yearly_raw

    station_keys = [k for k in data_area_avg if k in station_sources]

    if station_keys:
        data_area_avg['Stations'] = xr.concat(
            [data_area_avg[k] for k in station_keys],
            dim='station'
        ).mean(dim='station')

        data_area_monthly['Stations'] = xr.concat(
            [data_area_monthly[k] for k in station_keys],
            dim='station'
        ).mean(dim='station')

        data_area_fit['Stations'] = xr.concat(
            [data_area_fit[k] for k in station_keys],
            dim='station'
        ).mean(dim='station')

#%% Fit statistics for area

data_area_sources = (
    ['Stations'] + (data_area or [])
    if stations is not None
    else data_area
)

if data_area_sources is not None:

    trend_stats = {}

    for src in data_area_sources:

        x_arr = data_area_fit[src]['fit_against'].values
        y_arr = data_area_fit[src].values

        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_clean = x_arr[mask]
        y_clean = y_arr[mask]

        X = sm.add_constant(x_clean)

        model = sm.OLS(y_clean, X).fit()

        slope = model.params[1]
        intercept = model.params[0]

        slope_std = model.bse[1]
        slope_trend = slope*fit_scaling
        slope_trend_std = slope_std*fit_scaling

        trend_stats[src] = {
            'model': model,
            'x_clean': x_clean,
            'y_clean': y_clean,
            'slope': slope,
            'intercept': intercept,
            'slope_trend': slope_trend,
            'slope_trend_std': slope_trend_std,
        }

#%% Temporal plotting for area

if data_area_sources is not None:

    colors = ['#000000', '#DB2525', '#0168DE', '#00A236', "#CA721B", '#7B2CBF']

    fig, ax = plt.subplots(1, figsize=(16, 6)) #12, 8

    for ii, src in enumerate(data_area_sources):

        stats = trend_stats[src]

        model = stats['model']
        x_clean = stats['x_clean']
        y_clean = stats['y_clean']

        slope_trend = stats['slope_trend']
        slope_trend_std = stats['slope_trend_std']

        color = colors[ii]

        if src == 'Stations':
            base_name = 'Stations'
        else:
            base_name = next(key for key in data_sources if key in src)

        if base_name == 'Eobs':
            base_name = 'E-OBS'

        label = (
            f'{base_name} (trend: {slope_trend:.2f} ± {slope_trend_std:.2f} '
            f'{cfg_plot["trend_unit"]})'
        )

        order = np.argsort(x_clean)
        x_sorted = x_clean[order]
        y_sorted = y_clean[order]
        
        X_sorted = sm.add_constant(x_sorted)
        pred = model.get_prediction(X_sorted)
        frame = pred.summary_frame(alpha=0.05)

        y_trend = frame['mean'].values
        y_lo = frame['mean_ci_lower'].values
        y_hi = frame['mean_ci_upper'].values

        ax.plot(
                x_sorted,
                y_sorted,
                c=color,
                linewidth=2.5,
                zorder=10,
                ms=10,
                marker='o',
                linestyle='--',
        )

        ax.plot(
            x_sorted,
            y_trend,
            c=color,
            linewidth=3,
            alpha=1,
            label=label,
            zorder=15
        )

        if uncertainty_band:
            ax.fill_between(
                x_sorted,
                y_lo,
                y_hi,
                color=color,
                alpha=0.15,
                linewidth=0,
            )

    ax.grid()
    ax.set_xlabel(fit_x_label, fontsize=28)
    ax.set_ylabel(cfg_plot['label_plot'], fontsize=28)
    ax.tick_params(axis='both', labelsize=20, length=6)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if cfg_plot['ylim_fit'] is not None:
        ax.set_ylim(*cfg_plot['ylim_fit'])
    
    leg = ax.legend(fontsize=18, handlelength=1.5, handletextpad=0.4, loc='best')
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    leg.set_zorder(20)

#%% Spatial plotting for area

if isinstance(lats_area, (list, tuple)) and len(lats_area) == 2 and \
    isinstance(lons_area, (list, tuple)) and len(lons_area) == 2 and plots_area == True:

    meshes = []
    trend_fields = [] 

    n_panels = len(data_area)

    if n_panels == 4:
        nrows, ncols = 2, 2
        figsize = (14, 12)
        x_tick_bool = [False, False, True, True]
        y_tick_bool = [True, False, True, False]
    else:
        nrows, ncols = 1, n_panels
        figsize = (18, 5)
        x_tick_bool = [True]*n_panels    
        y_tick_bool = [False]*n_panels
        y_tick_bool[0] = True

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        constrained_layout=True,
        subplot_kw={'projection': proj_plot},
        sharex=True,
        sharey=True
    )

    axes = np.atleast_1d(axes).ravel()

    for ii, (ax, src) in enumerate(zip(axes, data_area)):

        data_fit_area = data_area_fit_raw[src]

        fits_area = data_fit_area.polyfit(dim='fit_against', deg=1, skipna=True)
        slope_area = fits_area.polyfit_coefficients.sel(degree=1)
        trend_area = (slope_area*fit_scaling).astype('float32').compute()

        if relative_precip and var == 'P':
            trend_plot_area = (trend_area / data_area_avg_raw[src])*100.0
        else:
            trend_plot_area = trend_area

        trend_plot_area = trend_plot_area.assign_coords(
            latitude=data_area_avg_raw[src]['latitude'],
            longitude=data_area_avg_raw[src]['longitude']
        )

        trend_fields.append(trend_plot_area)

        title = next(key for key in data_sources if key in src)
        if title == 'Eobs':
            title = 'E-OBS'

        mesh, _ = plot_map(
            fig, ax,
            trend_plot_area,
            trend_plot_area['longitude'],
            trend_plot_area['latitude'],
            crange=plot_cfg[var]['crange_trend'], 
            cmap=plot_cfg[var]['cmap_trend'],
            extreme_colors=plot_cfg[var]['extreme_trend'],
            show_plot=False,
            x_ticks=1,
            y_ticks=1,
            x_ticks_num=False,
            y_ticks_num=False,
            show_x_labels=x_tick_bool[ii],
            show_y_labels=y_tick_bool[ii],
            tick_size=24,
            extent=[*lons_area_plot, *lats_area_plot],
            lats_area=lats_area,
            lons_area=lons_area,
            proj=proj_plot,
            proj_area=proj_area,
            add_colorbar=False,
            title=title
        )

        meshes.append(mesh)

    cbar = shared_colorbar(
        fig=fig,
        axes=axes,
        mesh=meshes[0],
        datasets=trend_fields,
        crange=plot_cfg[var]['crange_trend'],
        label=plot_cfg[var]['label_trend'],
        orientation='horizontal',
        c_ticks=10,
        c_ticks_num=True,
        tick_labelsize=28,
        labelsize=34,
        pad=0.11,
        thickness=0.06
    )

    plt.show()

#%%

# Combined plots for only one variable and one colorbar....
# Maybe average over several station_coord_cfg stations for netcdf (similar to normal stations)

# Ipv corr ook misschien iets van SDEV?

    
