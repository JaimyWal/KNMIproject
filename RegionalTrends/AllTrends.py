#%% Imports

# Standard libraries
import numpy as np
import xarray as xr
import pandas as pd
import colormaps as cmaps
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cmocean
import xesmf as xe
import statsmodels.api as sm
import os
from dask.distributed import Client, get_client
from importlib import reload

# Custom functions
import PlotMaps
reload(PlotMaps)          
from PlotMaps import plot_map 

import ProcessNetCDF
reload(ProcessNetCDF)          
from ProcessNetCDF import preprocess_netcdf_monthly, subset_space

import ProcessStation
reload(ProcessStation)          
from ProcessStation import preprocess_station_monthly

#%% User inputs

# Main arguments
var = 'Tg'
data_base = 'RACMO2.4'
data_compare = None

# Data selection arguments
months = [12, 1, 2]
years = [2016, 2020]
lats = [38, 63]
lons = [-13, 22]
trim_border = None
proj_sel = 'RACMO2.4'
# Hier misschien ook argument voor land only? (of bij plotten?)

# Area selection arguments
data_area = None
lats_area = [50.7, 53.6]
lons_area = [3.25, 7.35]
proj_area = 'RACMO2.4'
land_only = True

# Plotting arguments
avg_crange = [-15, 15]
trend_crange = [-4, 4]
fit_range = None
proj_plot = 'RACMO2.4'
plot_lats = [38, 63]
plot_lons = [-13, 22]
true_contour = True
grid_contour = True
switch_sign = False
cut_boundaries = False

# Other arguments
relative_precip = False
rolling_mean_var = False
fit_against_gmst = False
rolling_mean_years = 5
min_periods = 1

# lats = [38, 63]
# lons = [-13, 22]
# data_area = ['Observed', 'ERA5_coarse', 'RACMO2.3', 'RACMO2.4']
# lats_area = [50.7, 53.6]
# lons_area = [3.25, 7.35]

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
station_sources = ['Bilt', 'Cabauw']

if fit_against_gmst:
    fit_unit = '°C GMST'
    fit_scaling = 1
    fit_x_label = 'ΔGMST (°C)'
else:
    fit_unit = 'decade'
    fit_scaling = 10
    fit_x_label = 'Year'

if relative_precip:
    precip_trend_label = 'Relative trend (% / ' + fit_unit + ')'
    precip_ylabel = 'Precipitation (% of climatology)'
    precip_trend_unit = '% / ' + fit_unit
else:
    precip_trend_label = 'Trend (mm / ' + fit_unit + ')'
    precip_ylabel = 'Precipitation (mm)'
    precip_trend_unit = 'mm / ' + fit_unit

plot_cfg = {
    'Tg': {
        'label_mean': 'Temperature (°C)',
        'label_trend': 'Trend (°C / ' + fit_unit + ')',
        'cmap_mean': 'Spectral_r',
        'cmap_trend': cmaps.temp_19lev,
        'crange_mean': avg_crange,
        'crange_trend': trend_crange,
        'extreme_mean': ("#0a0a86", "#700c0c"),
        'extreme_trend': ("#000020", "#350000"),
        'ylabel_fit': 'Temperature (°C)',
        'trend_unit': '°C / ' + fit_unit,
        'ylim_fit': fit_range,
    },
    'P': {
        'label_mean': 'Precipitation (mm)',
        'label_trend': precip_trend_label,
        'cmap_mean': cmocean.cm.rain,
        'cmap_trend': plt.get_cmap('BrBG', 20),
        'crange_mean': avg_crange,
        'crange_trend': trend_crange,
        'extreme_mean': (None, "#040812"),
        'extreme_trend': ("#271500", "#001f1f"),
        'ylabel_fit': precip_ylabel,
        'trend_unit': precip_trend_unit,
        'ylim_fit': fit_range,
    },
}

cfg_plot = plot_cfg[var]
if data_compare is not None:
    cfg_plot['cmap_mean'] = cfg_plot['cmap_trend']
    cfg_plot['extreme_mean'] = cfg_plot['extreme_trend']
    cfg_plot['label_mean'] = 'Difference in ' + cfg_plot['label_mean']
    cfg_plot['label_trend'] = 'Difference in ' + cfg_plot['label_trend']

var_name_cfg = {
    'Eobs': {
        'Tg': 'tg',
        'P': 'rr',
    },
    'ERA5': {
        'Tg': 't2m',
        'P': 'tp',
    },
    'RACMO2.3': {
        'Tg': 't2m',
        'P': 'precip',
    },
    'RACMO2.4': {
        'Tg': 'tas',
        'P': 'pr',
    },
    'Station': {
        'Tg': 'TG',
        'P': 'RH',
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
}

file_cfg = {
    'Eobs_fine': {
        'Tg': 'tg_ens_mean_0.1deg_reg_v31.0e.nc',
        'P': 'rr_ens_mean_0.1deg_reg_v31.0e.nc',
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
    },

    'RACMO2.3': {
        'Tg': 't2m/*.nc',
        'P': 'precip/*.nc',
    },
    'RACMO2.4': {
        'Tg': 'tas_*.nc',
        'P': 'pr_*.nc',
    },
    'Station': {
        'Bilt': 'KNMI_Bilt.txt',
        'Cabauw': 'KNMI_Cabauw.txt',
    },
}

base_dir_cfg = {
    'Eobs': '/nobackup/users/walj/eobs',
    'ERA5': '/nobackup/users/walj/era5',
    'RACMO2.3': '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Monthly_data',
    'RACMO2.4': '/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly',
    'Station': '/nobackup/users/walj/knmi',
}

proj_cfg = {
    'RACMO2.3': rotpole23,
    'RACMO2.4': rotpole24,
}

# Assign projections
proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())
proj_area = proj_cfg.get(proj_area, ccrs.PlateCarree())
proj_plot = proj_cfg.get(proj_plot, ccrs.PlateCarree())

#%% Some functions

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
                   trim_border=None,
                   rotpole_sel=ccrs.PlateCarree(),
                   rolling_mean_var=False,
                   fit_against_gmst=False,
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
        data = preprocess_netcdf_monthly(
            source=cfg['file_key'],
            file_path=input_file_data,
            var_name=cfg['variable'],
            months=months_local,
            years=years_load,
            lats=lats,
            lons=lons,
            trim_border=trim_local,
            rotpole_sel=rotpole_sel,
            rotpole_native=cfg['proj']
        ).squeeze()

    elif cfg['datatype'] == 'station':
        data = preprocess_station_monthly(
            file_path=input_file_data,
            var_name=cfg['variable'],
            months=months,
            years=years_load,
        ).squeeze()

    month = data['time'].dt.month
    year = data['time'].dt.year

    month_start = months_local[0]
    month_end = months_local[-1]

    if month_start <= month_end:
        clim_year = year
    else:
        clim_year = xr.where(month >= month_start, year + 1, year)

    if month_start <= month_end:
        clim_year = year
    else:
        clim_year = xr.where(month >= month_start, year + 1, year)

    data = data.assign_coords(clim_year=clim_year)
    data_year = data.groupby('clim_year').mean('time')

    if years_req is not None:
        data_year = data_year.sel(clim_year=slice(years_req[0], years_req[-1]))

    data_avg = data_year.mean(dim='clim_year').astype('float32')

    time_coord = pd.to_datetime(data_year['clim_year'].astype(str))
    data_year_time = data_year.assign_coords(time=('clim_year', time_coord)).swap_dims({'clim_year': 'time'})

    if rolling_mean_var:
        data_year_time = data_year_time.rolling(
            time=rolling_mean_years,
            center=True,
            min_periods=min_periods
        ).mean()

    if fit_against_gmst:
        file_GMST = '/nobackup/users/walj/era5/era5_gmst_anom.nc'
        data_GMST = xr.open_dataset(file_GMST)

        gmst_roll = data_GMST.rolling(
            time=rolling_mean_years,
            center=True,
            min_periods=min_periods
        ).mean()

        gmst_full = gmst_roll['GMST']

        gmst_sel = gmst_full.sel(time=data_year_time['time'])
        fit_coord = gmst_sel.astype(float)

    else:
        fit_coord = data_year_time['clim_year'].astype(float)

    data_fit = (
        data_year_time
        .rename({'time': 'fit_against'})
        .assign_coords(fit_against=('fit_against', fit_coord.values))
    ).astype('float32')

    return data, data_avg, data_fit


def bounds_from_centers(coord):

    coord = np.asarray(coord)
    n = coord.size
    steps = coord[1:] - coord[:-1]

    bounds = np.empty(n + 1, dtype=coord.dtype)
    bounds[0] = coord[0] - 0.5*steps[0]
    bounds[1:n] = coord[:-1] + 0.5*steps
    bounds[-1] = coord[-1] + 0.5*steps[-1]

    return bounds


def rotated_bounds(ds_rot, rotpole_crs):

    rlat_1d = ds_rot['rlat'].values
    rlon_1d = ds_rot['rlon'].values

    rlat_b_1d = bounds_from_centers(rlat_1d)
    rlon_b_1d = bounds_from_centers(rlon_1d)

    rlon_b_2d, rlat_b_2d = np.meshgrid(rlon_b_1d, rlat_b_1d)

    plate = ccrs.PlateCarree()

    pts = plate.transform_points(rotpole_crs,
                                 rlon_b_2d,  
                                 rlat_b_2d) 

    lon_b = pts[..., 0]
    lat_b = pts[..., 1]

    return lat_b, lon_b


def racmo_bounds_grid(ds_racmo_grid, rotpole_native):

    lat_rac = ds_racmo_grid['latitude'].values
    lon_rac = ds_racmo_grid['longitude'].values

    lat_b_full, lon_b_full = rotated_bounds(ds_racmo_grid, rotpole_native)

    grid = xr.Dataset(
        {
            'lon':   (('rlat', 'rlon'), lon_rac),
            'lat':   (('rlat', 'rlon'), lat_rac),
            'lon_b': (('rlat_b', 'rlon_b'), lon_b_full),
            'lat_b': (('rlat_b', 'rlon_b'), lat_b_full),
        }
    )

    return grid

#%% Process data

if data_base is not None:

    data_base_ds, data_avg_base, data_fit_base = process_source(
        data_base, 
        var,
        months=months,
        years=years,
        lats=lats,
        lons=lons,
        trim_border=trim_border, 
        rotpole_sel=proj_sel,
        rolling_mean_var=rolling_mean_var,
        fit_against_gmst=fit_against_gmst,
        rolling_mean_years=rolling_mean_years,
        min_periods=min_periods
    )

    if data_compare is None:
        fits_base = data_fit_base.polyfit(dim='fit_against', deg=1, skipna=True)
        slope_base = fits_base.polyfit_coefficients.sel(degree=1)
        trend_base = (slope_base*fit_scaling).astype('float32').compute()

        if relative_precip and var == 'P':
            trend_plot_base = (trend_base / data_avg_base)*100.0
        else:
            trend_plot_base = trend_base

        data_avg_plot = data_avg_base
        trend_plot = trend_plot_base

    elif data_compare is not None:

        data_comp_ds, data_avg_comp, data_fit_comp = process_source(
            data_compare, 
            var, 
            months=months,
            years=years,
            lats=lats,
            lons=lons,
            trim_border=trim_border,
            rotpole_sel=proj_sel,
            rolling_mean_var=rolling_mean_var,
            fit_against_gmst=fit_against_gmst,
            rolling_mean_years=rolling_mean_years,
            min_periods=min_periods
        )

        trg_grid = data_avg_base
        src_grid = data_avg_comp

        if var == 'P':
            method = 'conservative_normed'

            if data_base == 'RACMO2.3':
                trg_grid = racmo_bounds_grid(data_avg_base, rotpole23)
            elif data_base == 'RACMO2.4':
                trg_grid = racmo_bounds_grid(data_avg_base, rotpole24)
            
            if data_compare == 'RACMO2.3':
                src_grid = racmo_bounds_grid(data_avg_comp, rotpole23)
            elif data_compare == 'RACMO2.4':
                src_grid = racmo_bounds_grid(data_avg_comp, rotpole24)

        elif var == 'Tg':
            method = 'bilinear'

        regridder = xe.Regridder(
            src_grid,
            trg_grid,
            method,
            unmapped_to_nan=True,
        )

        target_chunks = {'latitude': 100, 'longitude': 100}

        data_avg_comp_reg = regridder(
            data_avg_comp,
            output_chunks=target_chunks
        ).astype('float32')

        data_fit_comp_reg = regridder(
            data_fit_comp,
            output_chunks=target_chunks
        ).astype('float32')

        fits_base = data_fit_base.polyfit(dim='fit_against', deg=1, skipna=True)
        slope_base = fits_base.polyfit_coefficients.sel(degree=1)
        trend_base = (slope_base*fit_scaling).astype('float32').compute()

        fits_comp = data_fit_comp_reg.polyfit(dim='fit_against', deg=1, skipna=True)
        slope_comp = fits_comp.polyfit_coefficients.sel(degree=1)
        trend_comp = (slope_comp*fit_scaling).astype('float32').compute()

        if relative_precip and var == 'P':
            trend_plot_base = (trend_base / data_avg_base)*100.0
            trend_plot_comp = (trend_comp / data_avg_comp_reg)*100.0
        else:
            trend_plot_base = trend_base
            trend_plot_comp = trend_comp

        if switch_sign:
            minus_scaling = -1
        else:
            minus_scaling = 1

        data_avg_plot = minus_scaling*(data_avg_comp_reg - data_avg_base).compute()
        trend_plot = minus_scaling*(trend_plot_comp - trend_plot_base).compute()

    trend_plot = trend_plot.assign_coords(
                latitude=data_avg_base['latitude'],
                longitude=data_avg_base['longitude']
                )

    lat_plot = data_avg_plot['latitude'].values
    lon_plot = data_avg_plot['longitude'].values

#%% Area plotting selection

if data_base is not None: 

    mask_area = None
    lat_b_area = None
    lon_b_area = None

    lats_area_plot = None
    lons_area_plot = None 
    proj_area_plot = None

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

        if 'rlat' in contour_area.dims and 'rlon' in contour_area.dims:
            rotpole_native = rotpole23 if data_base == 'RACMO2.3' else rotpole24
            lat_b_area, lon_b_area = rotated_bounds(contour_area, rotpole_native)
        else:
            lat1d_area = contour_area['latitude'].values
            lon1d_area = contour_area['longitude'].values
            lat_b_1d = bounds_from_centers(lat1d_area)
            lon_b_1d = bounds_from_centers(lon1d_area)
            lon_b_area, lat_b_area = np.meshgrid(lon_b_1d, lat_b_1d)

    if isinstance(lats_area, (list, tuple)) and len(lats_area) == 2 and \
    isinstance(lons_area, (list, tuple)) and len(lons_area) == 2 and true_contour == True:
        
        lats_area_plot = lats_area
        lons_area_plot = lons_area
        proj_area_plot = proj_area

#%% Calculate mean and plot

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
        lats_area=lats_area_plot,
        lons_area=lons_area_plot,
        proj_area=proj_area_plot,
        mask_area=mask_area,
        lat_b_area=lat_b_area,
        lon_b_area=lon_b_area
    )

#%% Linear trends and plot

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
        lats_area=lats_area_plot,
        lons_area=lons_area_plot,
        proj_area=proj_area_plot,
        mask_area=mask_area,
        lat_b_area=lat_b_area,
        lon_b_area=lon_b_area
    )

#%% Loading data for chosen area

if lats_area is not None and lons_area is not None and data_area is not None:

    if 'Observed' in data_area:

        if isinstance(lats_area, str) or isinstance(lons_area, str):
            replacement = lats_area if isinstance(lats_area, str) else lons_area
        else:
            replacement = 'Eobs_fine'

        data_area = [
            replacement if x == 'Observed' else x
            for x in data_area
        ]

    data_area_avg = {}
    data_area_monthly = {}
    data_area_yearly = {}

    for src in data_area:

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

        data_ds, data_avg, data_fit = process_source(
            src,
            var,
            months=months,
            years=years,
            lats=lat_sel,
            lons=lon_sel,
            trim_border=trim_border,
            rotpole_sel=proj_area,
            rolling_mean_var=rolling_mean_var,
            fit_against_gmst=fit_against_gmst,
            rolling_mean_years=rolling_mean_years,
            min_periods=min_periods
        )

        spatial_dims = [
            d for d in data_avg.dims
            if d in ('rlat', 'rlon', 'latitude', 'longitude')
        ]

        if spatial_dims:
            # build 2D lat
            if 'rlat' in data_avg.dims and 'rlon' in data_avg.dims:
                lat2d = data_avg['latitude']
            else:
                lat1d = data_avg['latitude']
                lon1d = data_avg['longitude']
                lat2d, lon2d = xr.broadcast(lat1d, lon1d)

            weights = np.cos(np.deg2rad(lat2d))

            w_da = xr.DataArray(
                weights,
                coords=lat2d.coords,
                dims=lat2d.dims,
            )

            mask_spatial = data_avg.notnull()
            w_masked = w_da.where(mask_spatial)
            w_sum = w_masked.sum(dim=spatial_dims)

            data_area_avg[src] = (data_avg*w_masked).sum(dim=spatial_dims) / w_sum
            monthly_raw = (data_ds*w_masked).sum(dim=spatial_dims) / w_sum
            yearly_raw = (data_fit*w_masked).sum(dim=spatial_dims) / w_sum

        else:
            data_area_avg[src] = data_avg
            monthly_raw = data_ds
            yearly_raw = data_fit

        if var == 'P' and relative_precip:
            data_area_monthly[src] = 100*monthly_raw / data_area_avg[src]
            data_area_yearly[src] = 100*yearly_raw  / data_area_avg[src]
        else:
            data_area_monthly[src] = monthly_raw
            data_area_yearly[src] = yearly_raw

#%% Fit stastitics for area

if lats_area is not None and lons_area is not None and data_area is not None:

    trend_stats = {}

    for src in data_area:

        x_arr = data_area_yearly[src]['fit_against'].values
        y_arr = data_area_yearly[src].values

        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_clean = x_arr[mask]
        y_clean = y_arr[mask]

        # Add intercept column
        X = sm.add_constant(x_clean)

        # Fit Ordinary Least Squares
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

if lats_area is not None and lons_area is not None and data_area is not None:

    colors = ['#000000', '#DB2525', '#0168DE', '#00A236']

    fig, ax = plt.subplots(1, figsize=(12, 8))

    for ii, src in enumerate(data_area):

        stats = trend_stats[src]

        model = stats['model']
        x_clean = stats['x_clean']
        y_clean = stats['y_clean']

        slope_trend = stats['slope_trend']
        slope_trend_std = stats['slope_trend_std']

        color = colors[ii]

        if 'Bilt' in src or 'Cabauw' in src:
            base_name = 'Station'
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
            label=label
        )

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
    ax.set_ylabel(cfg_plot['ylabel_fit'], fontsize=28)
    ax.tick_params(axis='both', labelsize=20, length=6)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if cfg_plot['ylim_fit'] is not None:
        ax.set_ylim(*cfg_plot['ylim_fit'])
    
    leg = ax.legend(fontsize=18, handlelength=1.5, handletextpad=0.4, loc='best')
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    leg.set_zorder(20)


#%%



# Mask sea values voor area!
# Plotjes maken op het laatst van hoe de geselecteerde data eruit ziet


# Gedaan: 
# Lijnen toevoegen over gebied waarover ik average
# Kijk of lijn 'blokkerig' kan zodat het langs de gridcells heen gaat!
# For contour, first make array full of ones -> then subset_space
# Wanneer ik gemiddelde neem over een gebied, moet ik wel area weighted doen!
# Misschien optie voor alleen temporal of alleen spatial?
# Kijk naar nieuwe versie van subset_space
# Optie voor exacte contour of ongeveer contour! (voor ongeveer contour, gewoon simpel de 4 hoeken nemen...)




# Ambitieus of niet slim om te doen:
# # Is het wel mogelijk om over bepaalde gebieden te masken? (Bijvoorbeeld Utrecht / Nederland) (niet doen!)
# Marker toevoegen wanneer point coordinate???



    
