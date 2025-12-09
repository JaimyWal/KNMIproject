#%% Imports

# Standard libraries
import numpy as np
import xarray as xr
import pandas as pd
import colormaps as cmaps
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean
import xesmf as xe
import os
from dask.distributed import Client, get_client
from importlib import reload

# Custom functions
import PlotMaps
reload(PlotMaps)          
from PlotMaps import plot_map 

import ProcessEobs
reload(ProcessEobs)          
from ProcessEobs import preprocess_eobs_monthly 

import ProcessERA5
reload(ProcessERA5)          
from ProcessERA5 import preprocess_era5 

import ProcessRACMO
reload(ProcessRACMO)          
from ProcessRACMO import preprocess_racmo_monthly 

#%% User inputs

var = 'P'
data_base = 'ERA5_coarse'
data_compare = 'RACMO'

months = [12, 1, 2]
years = [1987, 2020]
lats = [20, 75]
lons = [-40, 56]

avg_crange = [-1, 1]
trend_crange = [-0.3, 0.3]
use_rotpole = True
cut_boundaries = False
plot_lats = [35, 70]
plot_lons = [-10, 35]
switch_sign = False

relative_precip = False
rolling_mean_var = False
fit_against_gmst = False
rolling_mean_years = 5
min_periods = 1

#%% Dataset configurations

try:
    client = get_client()
except ValueError:
    client = Client(n_workers=1, threads_per_worker=8, processes=False)

sources = ['Eobs', 'ERA5', 'RACMO']

if fit_against_gmst:
    fit_unit = '°C GMST'
    fit_scaling = 1
else:
    fit_unit = 'decade'
    fit_scaling = 10

if relative_precip:
    precip_trend_label = 'Relative trend (% / ' + fit_unit + ')'
else:
    precip_trend_label = 'Trend (mm / ' + fit_unit + ')'

if data_compare is not None:
    label_prefix = 'Difference in '
if data_compare is None:
    label_prefix = ''

plot_cfg = {
    'Tg': {
        'label_mean': label_prefix + r'Temperature (°C)',
        'label_trend': label_prefix + r'Trend (°C / ' + fit_unit + ')',
        'cmap_mean': 'Spectral_r',
        'cmap_trend': cmaps.temp_19lev,
        'crange_mean': avg_crange,
        'crange_trend': trend_crange,
        'extreme_mean': ("#0a0a86", "#700c0c"),
        'extreme_trend': ("#000020", "#350000"),
    },
    'P': {
        'label_mean': label_prefix + 'Precipitation (mm)',
        'label_trend': label_prefix + precip_trend_label,
        'cmap_mean': cmocean.cm.rain,
        'cmap_trend': plt.get_cmap('BrBG', 20),
        'crange_mean': avg_crange,
        'crange_trend': trend_crange,
        'extreme_mean': (None, "#040812"),
        'extreme_trend': ("#271500", "#001f1f"),
    },
}

cfg_plot = plot_cfg[var]
if data_compare is not None:
    cfg_plot['cmap_mean'] = cfg_plot['cmap_trend']
    cfg_plot['extreme_mean'] = cfg_plot['extreme_trend']

var_name_cfg = {
    'Eobs': {
        'Tg': 'tg',
        'P': 'rr',
    },
    'ERA5': {
        'Tg': 't2m',
        'P': 'tp',
    },
    'RACMO': {
        'Tg': 't2m',
        'P': 'precip',
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

    'RACMO': {
        'Tg': 't2m',
        'P': 'precip',
    },
}

base_dir_cfg = {
    'Eobs': '/nobackup/users/walj/eobs',
    'ERA5': '/nobackup/users/walj/era5',
    'RACMO': '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Monthly_data',
}

#%% Some functions

def make_cfg(data_source, var):
    file_key = next(src for src in sources if src in data_source)
    cfg = {
        **plot_cfg[var],
        'variable': var_name_cfg[file_key][var],
        'file': file_cfg[data_source][var],
        'base_dir': base_dir_cfg[file_key],
        'file_key': file_key,
    }
    return cfg


def process_source(data_source):
    cfg = make_cfg(data_source, var)
    file_key = cfg['file_key']

    if months is None:
        months_local = np.arange(1, 13)
    else:
        months_local = np.asarray(months, dtype=int)

    years_req = list(years)
    years_load = list(years_req)

    month_start = int(months_local[0])
    month_end = int(months_local[-1])

    if month_start > month_end:
        years_load[0] = years_req[0] - 1

    input_file_data = os.path.join(cfg['base_dir'], cfg['file'])

    if file_key == 'Eobs':
        data = preprocess_eobs_monthly(
            file_path=input_file_data,
            var_name=cfg['variable'],
            months=months_local,
            years=years_load,
            lats=lats,
            lons=lons,
            chunks_time=180,
            chunks_lat=200,
            chunks_lon=200,
        )

    elif file_key == 'ERA5':
        data = preprocess_era5(
            file_path=input_file_data,
            var_name=cfg['variable'],
            months=months_local,
            years=years_load,
            lats=lats,
            lons=lons,
            chunks_time=180,
            chunks_lat=200,
            chunks_lon=200,
        )

    elif file_key == 'RACMO':
        data = preprocess_racmo_monthly(
            dir_path=input_file_data,
            var_name=cfg['variable'],
            months=months_local,
            years=years_load,
            lats=lats,
            lons=lons,
            chunks_time=180,
            chunks_lat=200,
            chunks_lon=200,
            already_monthly=True,
        ).squeeze()

    lat_vals = data['latitude'].values
    lon_vals = data['longitude'].values

    month = data['time'].dt.month
    year = data['time'].dt.year

    if month_start <= month_end:
        clim_year = year
    else:
        clim_year = xr.where(month >= month_start, year + 1, year)

    data = data.assign_coords(clim_year=clim_year)
    data_year = data.groupby('clim_year').mean('time')
    data_year = data_year.sel(clim_year=slice(years_req[0], years_req[-1]))

    data_avg = data_year.mean(dim='clim_year').astype('float32')

    time_coord = pd.to_datetime(data_year['clim_year'].values.astype(int).astype(str))
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
        fit_coord = data_year_time['time'].dt.year.astype(float)

    data_fit = (
        data_year_time
        .rename({'time': 'fit_against'})
        .assign_coords(fit_against=('fit_against', fit_coord.values))
    ).astype('float32')

    data_dict = {
        'cfg': cfg,
        'data_avg': data_avg,
        'data_fit': data_fit,
        'lat': lat_vals,
        'lon': lon_vals,
    }

    return data_dict


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


def racmo_bounds_grid(ds_racmo_grid, lats, lons, rotpole_native):
    lat_full = ds_racmo_grid['lat']
    lon_full = ds_racmo_grid['lon']

    lat_min, lat_max = sorted(lats)
    lon_min, lon_max = sorted(lons)

    mask = (
        (lat_full >= lat_min) & (lat_full <= lat_max) &
        (lon_full >= lon_min) & (lon_full <= lon_max)
    )

    ii, jj = np.where(mask)

    i0, i1 = ii.min(), ii.max()
    j0, j1 = jj.min(), jj.max()

    lat_b_full, lon_b_full = rotated_bounds(ds_racmo_grid, rotpole_native)

    lat_rac = lat_full.isel(rlat=slice(i0, i1 + 1), rlon=slice(j0, j1 + 1)).values
    lon_rac = lon_full.isel(rlat=slice(i0, i1 + 1), rlon=slice(j0, j1 + 1)).values
    lat_b_rac = lat_b_full[i0:i1 + 2, j0:j1 + 2]
    lon_b_rac = lon_b_full[i0:i1 + 2, j0:j1 + 2]

    grid = xr.Dataset(
        {
            'lon':   (('rlat', 'rlon'), lon_rac),
            'lat':   (('rlat', 'rlon'), lat_rac),
            'lon_b': (('rlat_b', 'rlon_b'), lon_b_rac),
            'lat_b': (('rlat_b', 'rlon_b'), lat_b_rac),
        }
    )

    return grid

#%% Obtain rotated grid

# Rotated projection:
rotpole_dir = '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/precip'
rotpole_file = 'precip.KNMI-2000.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc'
data_rotpole = xr.open_dataset(os.path.join(rotpole_dir, rotpole_file))
ds_racmo_grid = data_rotpole.isel(time=0)

rp = data_rotpole['rotated_pole']

pole_lat = rp.grid_north_pole_latitude
pole_lon = rp.grid_north_pole_longitude
central_rlon = 18.0

rotpole_native = ccrs.RotatedPole(
    pole_latitude=pole_lat,
    pole_longitude=pole_lon,
)

rotpole_plot = ccrs.RotatedPole(
    pole_latitude=pole_lat,
    pole_longitude=pole_lon,
    central_rotated_longitude=central_rlon,
)

if use_rotpole:
    proj = rotpole_plot
else:
    proj = ccrs.PlateCarree()

#%% Process data

base_res = process_source(data_base)
cfg_base = base_res['cfg']
data_avg_base = base_res['data_avg']
data_fit_base = base_res['data_fit']

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

    comp_res = process_source(data_compare)
    data_avg_comp = comp_res['data_avg']
    data_fit_comp = comp_res['data_fit']

    trg_grid = data_avg_base
    src_grid = data_avg_comp

    is_racmo_base = (cfg_base['file_key'] == 'RACMO')
    is_racmo_comp = (comp_res['cfg']['file_key'] == 'RACMO')

    if var == 'P' and is_racmo_comp and not is_racmo_base:
        src_grid = racmo_bounds_grid(ds_racmo_grid, lats, lons, rotpole_native)

    elif var == 'P' and is_racmo_base and not is_racmo_comp:
        trg_grid = racmo_bounds_grid(ds_racmo_grid, lats, lons, rotpole_native)

    if var == 'P':
        method = 'conservative_normed'
    else:
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

lat_plot = data_avg_plot['latitude'].values
lon_plot = data_avg_plot['longitude'].values

#%% Calculate mean and plot

fig, ax = plt.subplots(
        1, figsize=(14, 12), 
        constrained_layout=True,
        subplot_kw={'projection': proj}
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
    proj=proj,
    rotated_grid=cut_boundaries
)

#%% Linear trends and plot

fig, ax = plt.subplots(
        1, figsize=(14, 12), 
        constrained_layout=True,
        subplot_kw={'projection': proj}
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
    proj=proj,
    rotated_grid=cut_boundaries
)


#%%


# Andere denominator bij relative trends?
# Zonneuren ook plotten!!

# Misschien ook nog kijken of era5 boundaries niet nodig zijn ofzo.

    # if var == 'P' and (is_racmo_base or is_racmo_comp):
        
    #     if is_racmo_base: 
    #         data_carree = data_avg_comp 
    #     elif is_racmo_comp: 
    #         data_carree = data_avg_base 

    #     lon_carree = data_carree['longitude'].values 
    #     lat_carree = data_carree['latitude'].values 
        
    #     # grid spacing 
    #     d_lon = float(np.abs(lon_carree[1] - lon_carree[0])) 
    #     d_lat = float(np.abs(lat_carree[1] - lat_carree[0])) 
        
    #     # bounds from centers 
    #     lon0_b = float(lon_carree[0] - d_lon / 2) 
    #     lon1_b = float(lon_carree[-1] + d_lon / 2) 
    #     lat0_b = float(lat_carree[0] - d_lat / 2) 
    #     lat1_b = float(lat_carree[-1] + d_lat / 2) 
        
    #     # build rectilinear ERA5 grid for xESMF 
    #     grid_carree = xe.util.grid_2d(lon0_b, lon1_b, d_lon, lat0_b, lat1_b, d_lat)

    #     if is_racmo_base: 
    #         src_grid = grid_carree 
    #     elif is_racmo_comp: 
    #         trg_grid = grid_carree


    # if var == 'P' and is_racmo_comp and not is_racmo_base:
    #     if 'y' in data_avg_comp_reg.dims:
    #         data_avg_comp_reg = data_avg_comp_reg.rename({'y': 'latitude', 'x': 'longitude'})
    #         data_fit_comp_reg = data_fit_comp_reg.rename({'y': 'latitude', 'x': 'longitude'})

    #     data_avg_comp_reg = data_avg_comp_reg.assign_coords(
    #         latitude=data_avg_base['latitude'],
    #         longitude=data_avg_base['longitude'],
    #     )
    #     data_fit_comp_reg = data_fit_comp_reg.assign_coords(
    #         latitude=data_fit_base['latitude'],
    #         longitude=data_fit_base['longitude'],
    #     )

    # elif var == 'P' and is_racmo_base and not is_racmo_comp:
    #     if 'rlat' in data_avg_comp_reg.dims:
    #         data_avg_comp_reg = data_avg_comp_reg.rename({'rlat': 'latitude', 'rlon': 'longitude'})
    #         data_fit_comp_reg = data_fit_comp_reg.rename({'rlat': 'latitude', 'rlon': 'longitude'})

    #     data_avg_comp_reg = data_avg_comp_reg.assign_coords(
    #         latitude=data_avg_base['latitude'],
    #         longitude=data_avg_base['longitude'],
    #     )
    #     data_fit_comp_reg = data_fit_comp_reg.assign_coords(
    #         latitude=data_fit_base['latitude'],
    #         longitude=data_fit_base['longitude'],
    #     )
