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
from PlotMaps import plot_map, shared_colorbar
import ProcessNetCDF
reload(ProcessNetCDF)          
from ProcessNetCDF import preprocess_netcdf_monthly

#%% User inputs

# Main arguments
var = 'P' #
data_base = ['ERA5_coarse', 'RACMO2.4', 'Eobs_fine'] #
data_compare = None
n_runs = 3

# Data selection arguments
months = [6, 7, 8] #
years = [2016, 2024]
lats = [38, 63] 
lons = [-13, 22]
proj_sel = 'RACMO2.4'
land_only = False
trim_border = None

# Spatial plotting arguments
avg_crange = [0, 5]
trend_crange = [-2, 2]
proj_plot = 'RACMO2.4'
plot_lats = [38, 63] 
plot_lons = [-13, 22]
switch_sign = [False, True, False] #
cut_boundaries = False

# Other arguments
relative_precip = False
rolling_mean_var = False
fit_against_gmst = False
rolling_mean_years = 5
min_periods = 1

# lats = [38, 63]
# lons = [-13, 22]

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

if fit_against_gmst:
    fit_unit = '°C GMST'
    fit_scaling = 1
else:
    fit_unit = 'decade'
    fit_scaling = 10

if relative_precip:
    precip_trend_label = 'Relative trend (% / ' + fit_unit + ')'
    precip_trend_unit = '% / ' + fit_unit
else:
    precip_trend_label = 'Trend (mm / ' + fit_unit + ')'
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
    },
}

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
    }
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
}

proj_cfg = {
    'RACMO2.3': rotpole23,
    'RACMO2.4': rotpole24,
}

# Assign projections
proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())
proj_plot = proj_cfg.get(proj_plot, ccrs.PlateCarree())

#%% Some functions

def make_cfg(data_source, var):

    file_key = next(src for src in data_sources if src in data_source)
    cfg = {
        'variable': var_name_cfg[file_key][var],
        'file': file_cfg[data_source][var],
        'base_dir': base_dir_cfg[file_key],
        'file_key': file_key,
        'proj': proj_cfg.get(file_key, ccrs.PlateCarree()),
    }

    return cfg

def process_source(data_source,
                   var,
                   months=None,
                   years=None,
                   lats=None,
                   lons=None,
                   land_only=False,
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

    data = preprocess_netcdf_monthly(
        source=cfg['file_key'],
        file_path=input_file_data,
        var_name=cfg['variable'],
        months=months_local,
        years=years_load,
        lats=lats,
        lons=lons,
        land_only=land_only,
        trim_border=trim_local,
        rotpole_sel=rotpole_sel,
        rotpole_native=cfg['proj'],
        chunks_time=60,
        chunks_lat=50,
        chunks_lon=50
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

#%% Ensure lists for looped parameters

def ensure_list(param, n, nested=False):

    if not isinstance(param, list):
        return [param]*n

    if not nested:
        return param

    if any(isinstance(p, list) for p in param):
        return param

    return [param]*n

var_list = ensure_list(var, n_runs)
data_base_list = ensure_list(data_base, n_runs)
data_compare_list = ensure_list(data_compare, n_runs)
months_list = ensure_list(months, n_runs, nested=True)
switch_sign_list = ensure_list(switch_sign, n_runs)

#%% Process data

results = []

for ii in range(n_runs):

    if data_base_list[ii] is not None:

        cfg_plot = plot_cfg[var_list[ii]].copy()

        data_base_ds, data_avg_base, data_fit_base = process_source(
            data_base_list[ii], 
            var_list[ii],
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

        if data_compare_list[ii] is None:

            title = next(key for key in data_sources if key in data_base_list[ii])

            fits_base = data_fit_base.polyfit(dim='fit_against', deg=1, skipna=True)
            slope_base = fits_base.polyfit_coefficients.sel(degree=1)
            trend_base = (slope_base*fit_scaling).astype('float32').compute()

            if relative_precip and var_list[ii] == 'P':
                trend_plot_base = (trend_base / data_avg_base)*100.0
            else:
                trend_plot_base = trend_base

            data_avg_plot = data_avg_base
            trend_plot = trend_plot_base

        elif data_compare_list[ii] is not None:

            cfg_plot['cmap_mean'] = cfg_plot['cmap_trend']
            cfg_plot['extreme_mean'] = cfg_plot['extreme_trend']
            cfg_plot['label_mean'] = 'Difference in ' + cfg_plot['label_mean']
            cfg_plot['label_trend'] = 'Difference in ' + cfg_plot['label_trend']

            data_comp_ds, data_avg_comp, data_fit_comp = process_source(
                data_compare_list[ii], 
                var_list[ii], 
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

            trg_grid = data_avg_base
            src_grid = data_avg_comp

            if var_list[ii] == 'P':
                method = 'conservative_normed'

                if data_base_list[ii] == 'RACMO2.3':
                    trg_grid = racmo_bounds_grid(data_avg_base, rotpole23)
                elif data_base_list[ii] == 'RACMO2.4':
                    trg_grid = racmo_bounds_grid(data_avg_base, rotpole24)
                
                if data_compare_list[ii] == 'RACMO2.3':
                    src_grid = racmo_bounds_grid(data_avg_comp, rotpole23)
                elif data_compare_list[ii] == 'RACMO2.4':
                    src_grid = racmo_bounds_grid(data_avg_comp, rotpole24)

            elif var_list[ii] == 'Tg':
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

            if relative_precip and var_list[ii] == 'P':
                trend_plot_base = (trend_base / data_avg_base)*100.0
                trend_plot_comp = (trend_comp / data_avg_comp_reg)*100.0
            else:
                trend_plot_base = trend_base
                trend_plot_comp = trend_comp

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

            data_avg_plot = minus_scaling*(data_avg_comp_reg - data_avg_base).compute()
            trend_plot = minus_scaling*(trend_plot_comp - trend_plot_base).compute()

        trend_plot = trend_plot.assign_coords(
                        latitude=data_avg_base['latitude'],
                        longitude=data_avg_base['longitude']
                    )

        lat_plot = data_avg_plot['latitude'].values
        lon_plot = data_avg_plot['longitude'].values

        if 'Eobs' in title:
            title = title.replace('Eobs', 'E-OBS')
        cfg_plot['title'] = title

        results.append({
            'data_avg_plot': data_avg_plot,
            'trend_plot': trend_plot,
            'latitude': lat_plot,
            'longitude': lon_plot,
            'cfg_plot': cfg_plot
        })

#%% Plot climatology

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

        title = res['cfg_plot']['title']
        if 'Eobs' in title:
            title = title.replace('Eobs', 'E-OBS')

        mesh, _ = plot_map(
            fig, ax,
            res['data_avg_plot'], 
            res['longitude'], 
            res['latitude'], 
            crange=res['cfg_plot']['crange_mean'], 
            label=res['cfg_plot']['label_mean'], 
            cmap=res['cfg_plot']['cmap_mean'], 
            extreme_colors=res['cfg_plot']['extreme_mean'],
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
            show_plot=False,
            add_colorbar=False
        )

        meshes.append(mesh)

cbar = shared_colorbar(
        fig=fig,
        axes=axes,
        mesh=meshes[0],
        datasets=data_avg_field,
        crange=cfg_plot['crange_mean'],
        label=cfg_plot['label_mean'],
        orientation='horizontal',
        c_ticks=10,
        c_ticks_num=True,
        tick_labelsize=26,
        labelsize=32,
        pad=0.1,
        thickness=0.06
    )

plt.show()

#%% Linear trends and plot


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

        title = res['cfg_plot']['title']
        if 'Eobs' in title:
            title = title.replace('Eobs', 'E-OBS')

        mesh, _ = plot_map(
            fig, ax,
            res['trend_plot'], 
            res['longitude'], 
            res['latitude'], 
            crange=res['cfg_plot']['crange_trend'], 
            label=res['cfg_plot']['label_trend'], 
            cmap=res['cfg_plot']['cmap_trend'], 
            extreme_colors=res['cfg_plot']['extreme_trend'],
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
            show_plot=False,
            add_colorbar=False
        )

        meshes.append(mesh)

cbar = shared_colorbar(
        fig=fig,
        axes=axes,
        mesh=meshes[0],
        datasets=data_trend_field,
        crange=cfg_plot['crange_trend'],
        label=cfg_plot['label_trend'],
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



# Plotjes maken op het laatst van hoe de geselecteerde data eruit ziet
# Correlation plots?
# Correlation per maand? En if so, andere climate years weghalen.


# Gedaan: 
# Lijnen toevoegen over gebied waarover ik average
# Kijk of lijn 'blokkerig' kan zodat het langs de gridcells heen gaat!
# For contour, first make array full of ones -> then subset_space
# Wanneer ik gemiddelde neem over een gebied, moet ik wel area weighted doen!
# Misschien optie voor alleen temporal of alleen spatial?
# Kijk naar nieuwe versie van subset_space
# Optie voor exacte contour of ongeveer contour! (voor ongeveer contour, gewoon simpel de 4 hoeken nemen...)
# Mask sea values voor area!





# Ambitieus of niet slim om te doen:
# # Is het wel mogelijk om over bepaalde gebieden te masken? (Bijvoorbeeld Utrecht / Nederland) (niet doen!)
# Marker toevoegen wanneer point coordinate???


# Vraagjes:
# Over ECWMF account praten.
# Over correlatie per maand praten.
# Over sunshine duration praten.
    
