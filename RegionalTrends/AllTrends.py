#%% Imports

# Standard libraries
import numpy as np
import xarray as xr
import pandas as pd
import dask
import colormaps as cmaps
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as clr
import cartopy.crs as ccrs
import cmocean
import xesmf as xe
import statsmodels.api as sm
import os
from importlib import reload

# Custom libraries
import PlotMaps
reload(PlotMaps)          
from PlotMaps import plot_map, shared_colorbar

import ProcessNetCDF
reload(ProcessNetCDF)          
from ProcessNetCDF import preprocess_netcdf, subset_space

import ProcessStation
reload(ProcessStation)          
from ProcessStation import preprocess_station

import GridBounds
reload(GridBounds)          
from GridBounds import grid_with_bounds

import AreaWeights
reload(AreaWeights)
from AreaWeights import area_weights, area_weighted_mean

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

proj_cfg = {
    'RACMO2.3': rotpole23,
    'RACMO2.4': rotpole24,
}

# Assign projections
proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())
proj_area = proj_cfg.get(proj_area, ccrs.PlateCarree())
proj_plot = proj_cfg.get(proj_plot, ccrs.PlateCarree())

#%% Dataset configurations (misschien in een aparte script doen?)

dask.config.set(scheduler='threads', num_workers=12)

data_sources = ['Eobs', 'ERA5', 'RACMO2.3', 'RACMO2.4']
station_sources = ['Bilt', 'Cabauw', 'Eelde', 'Maastricht', 'Vlissingen', 'Kooy']

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

if corr_freq == 'Daily' and corr_calc == True and data_compare is not None:
    freq_str = 'Daily'
    racmo24_sep = '.'
else:
    freq_str = 'Monthly'
    racmo24_sep = '_'

sun_colors = [
    '#2b0a3d',
    '#5c1a1b',
    '#8b2f1c',
    '#c45a1a',
    '#e39b2d',
    '#f4e27a'
]
cmap_sun = LinearSegmentedColormap.from_list('sunshine', sun_colors, N=256)

if corr_cmap_neg:
    colors = [
        "#570088", "#3700b3", "#1d00d7", "#0300f6", "#0231be",
        "#056775", "#079d2c", "#35c13b", "#80d883",
        "#ffffff", "#ffffff", "#ffffff",
        "#fff400", "#ffe400", "#ffc900", "#ffad00",
        "#ff8200", "#ff5500", "#ff2800", "#a30e03", "#6b0902"
    ]
    corr_cmap = clr.ListedColormap(colors)
    corr_extreme = (None, None)

else:
    corr_colors = [
        '#ffffff',
        '#fff7bc',
        '#fee391',
        '#fec44f',
        '#fe9929',
        '#ec7014',
        '#cc4c02',
        '#993404',
        '#662506',
        '#3d1f0f',
    ]
    corr_cmap = clr.ListedColormap(corr_colors)
    corr_extreme = ("#999898", None)

cmap_trend_signed = ListedColormap(cmaps.cmp_b2r(np.linspace(0, 1, 20)))

cmap_sw_mean = cmocean.cm.solar
extreme_sw_mean = (None, '#fff3b2')
extreme_sw_trend = ('#1B1C70', '#7e060c')

cmap_lw_mean = cmocean.cm.balance
extreme_lw_mean = ('#0a0a86', '#700c0c')
extreme_lw_trend = ('#000020', '#350000')

cmap_turb_mean = cmocean.cm.balance
extreme_turb_mean = ('#0a0a86', '#700c0c')
extreme_turb_trend = ('#000020', '#350000')

cmap_cloud_mean = cmocean.cm.gray
extreme_cloud_mean = ('#000000', '#ffffff')
extreme_cloud_trend = ('#1B1C70', '#7e060c')

cmap_wp_mean = cmocean.cm.matter
extreme_wp_mean = ('#0a0a86', '#fff3b2')
extreme_wp_trend = ('#1B1C70', '#7e060c')

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
    'Sq': {
        'label_mean': 'Sund. (hours/day)',
        'label_trend': 'Trend (hours/day / ' + fit_unit + ')',
        'cmap_mean': cmap_sun,
        'cmap_trend': ListedColormap(cmaps.cmp_b2r(np.linspace(0, 1, 20))),
        'crange_mean': avg_crange,
        'crange_trend': trend_crange,
        'extreme_mean': (None, "#fff3b2"),
        'extreme_trend': ("#1B1C70", "#7e060c"),
        'ylabel_fit': 'Sund. (hours/day)',
        'trend_unit': 'hours/day / ' + fit_unit,
        'ylim_fit': fit_range,
    }
}

def add_plot_cfg(
    label_mean,
    label_trend,
    cmap_mean,
    cmap_trend,
    extreme_mean,
    extreme_trend,
    ylabel_fit,
    trend_unit,
    ylim_fit,
    crange_mean=avg_crange,
    crange_trend=trend_crange,
):
    return {
        'label_mean': label_mean,
        'label_trend': label_trend,
        'cmap_mean': cmap_mean,
        'cmap_trend': cmap_trend,
        'crange_mean': crange_mean,
        'crange_trend': crange_trend,
        'extreme_mean': extreme_mean,
        'extreme_trend': extreme_trend,
        'ylabel_fit': ylabel_fit,
        'trend_unit': trend_unit,
        'ylim_fit': ylim_fit,
    }

sw_vars = {
    'SWin': r'SW_{in} (W/m$^2$)',
    'SWnet': r'SW_{net} (W/m$^2$)',
    'SWincs': r'SW_{in,cs} (W/m$^2$)',
    'SWnetcs': r'SW_{net,cs} (W/m$^2$)',
}
for v, lab in sw_vars.items():
    plot_cfg[v] = add_plot_cfg(
        label_mean=lab,
        label_trend=r'Trend (W/m$^2$ / ' + fit_unit + ')',
        cmap_mean=cmap_sw_mean,
        cmap_trend=cmap_trend_signed,
        extreme_mean=extreme_sw_mean,
        extreme_trend=extreme_sw_trend,
        ylabel_fit=lab,
        trend_unit=r'W/m$^2$ / ' + fit_unit,
        ylim_fit=fit_range,
    )

lw_vars = {
    'LWin': r'LW_{in} (W/m$^2$)',
    'LWincs': r'LW_{in,cs} (W/m$^2$)',
    'LWnet': r'LW_{net} (W/m$^2$)',
    'LWnetcs': r'LW_{net,cs} (W/m$^2$)',
}
for v, lab in lw_vars.items():
    plot_cfg[v] = add_plot_cfg(
        label_mean=lab,
        label_trend=r'Trend (W/m$^2$ / ' + fit_unit + ')',
        cmap_mean=cmap_lw_mean,
        cmap_trend=cmap_trend_signed,
        extreme_mean=extreme_lw_mean,
        extreme_trend=extreme_lw_trend,
        ylabel_fit=lab,
        trend_unit=r'W/m$^2$ / ' + fit_unit,
        ylim_fit=fit_range,
    )

turb_vars = {
    'SHF': r'SHF (W/m$^2$)',
    'LHF': r'LHF (W/m$^2$)',
}
for v, lab in turb_vars.items():
    plot_cfg[v] = add_plot_cfg(
        label_mean=lab,
        label_trend=r'Trend (W/m$^2$ / ' + fit_unit + ')',
        cmap_mean=cmap_turb_mean,
        cmap_trend=cmap_trend_signed,
        extreme_mean=extreme_turb_mean,
        extreme_trend=extreme_turb_trend,
        ylabel_fit=lab,
        trend_unit=r'W/m$^2$ / ' + fit_unit,
        ylim_fit=fit_range,
    )

cloud_vars = {
    'CloudLow': 'Low cloud (%)',
    'CloudMid': 'Mid cloud (%)',
    'CloudHigh': 'High cloud (%)',
    'CloudTotal': 'Total cloud (%)',
}
for v, lab in cloud_vars.items():
    plot_cfg[v] = add_plot_cfg(
        label_mean=lab,
        label_trend='Trend (% / ' + fit_unit + ')',
        cmap_mean=cmap_cloud_mean,
        cmap_trend=cmap_trend_signed,
        extreme_mean=extreme_cloud_mean,
        extreme_trend=extreme_cloud_trend,
        ylabel_fit=lab,
        trend_unit='% / ' + fit_unit,
        ylim_fit=fit_range,
    )

wp_vars = {
    'LWP': 'LWP (g/m$^2$)',
    'IWP': 'IWP (g/m$^2$)',
}
for v, lab in wp_vars.items():
    plot_cfg[v] = add_plot_cfg(
        label_mean=lab,
        label_trend=r'Trend (g/m$^2$ / ' + fit_unit + ')',
        cmap_mean=cmap_wp_mean,
        cmap_trend=cmap_trend_signed,
        extreme_mean=extreme_wp_mean,
        extreme_trend=extreme_wp_trend,
        ylabel_fit=lab,
        trend_unit=r'g/m$^2$ / ' + fit_unit,
        ylim_fit=fit_range,
    )

cfg_plot = plot_cfg[var].copy()
if data_compare is not None:
    cfg_plot['cmap_mean'] = cfg_plot['cmap_trend']
    cfg_plot['extreme_mean'] = cfg_plot['extreme_trend']
    cfg_plot['label_mean'] = 'Difference in ' + cfg_plot['label_mean']
    cfg_plot['label_trend'] = 'Difference in ' + cfg_plot['label_trend']

var_name_cfg = {
    'Eobs': {
        'Tg': 'tg',
        'P': 'rr',
        'SWin': 'qq',
    },
    'ERA5': {
        'Tg': 't2m',
        'P': 'tp',
        'SWin': 'ssrd',
    },
    'RACMO2.3': {
        'Tg': 't2m',
        'P': 'precip',
        'Sq': 'sund',
        'SWin': 'swsd',
        'SWnet': 'swsn',
        'SWincs': 'swsdcs',
        'SWnetcs': 'swsncs',
        'LWin': 'lwsd',
        'LWnet': 'lwsn',
        'LWincs': 'lwsdcs',
        'LWnetcs': 'lwsncs',
        'SHF': 'senf',
        'LHF': 'latf',
        'CloudLow': 'aclcovL',
        'CloudMid': 'aclcovM',
        'CloudHigh': 'aclcovH',
        'CloudTotal': 'aclcov',
        'LWP': 'qli',
        'IWP': 'qii',
    },
    'RACMO2.4': {
        'Tg': 'tas',
        'P': 'pr',
        'Sq': 'sund',
        'SWin': 'rsds',
        'SWnet': 'ssr',
        'SWincs': 'rsdscs',
        'SWnetcs': 'ssrc',
        'LWin': 'rlds',
        'LWnet': 'str',
        'LWincs': 'rldscs',
        'LWnetcs': 'strc',
        'SHF': 'hfss',
        'LHF': 'hfls',
        'CloudLow': 'cll',
        'CloudMid': 'clm',
        'CloudHigh': 'clh',
        'CloudTotal': 'clt',
        'LWP': 'clwvi',
        'IWP': 'clivi',
    },
    'Station': {
        'Tg': 'TG',
        'P': 'RH',
        'Sq': 'SQ',
        'SWin': 'Q'
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
    'Eelde': {
        'latitude': 53.12385846866912,
        'longitude': 6.584799434350561
    },
    'Maastricht': {
        'latitude': 50.90548320406765,
        'longitude': 5.761839846736004
    },
    'Vlissingen': {
        'latitude': 51.441328455552586,
        'longitude': 3.5958610840956884
    },
    'Kooy': {
        'latitude': 52.924172463538795,
        'longitude': 4.779336630180403
    },
}

base_dir_cfg = {
    'Eobs': '/nobackup/users/walj/eobs',
    'ERA5': '/nobackup/users/walj/era5',
    'RACMO2.3': f'/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/{freq_str}_data',
    'RACMO2.3_ERIK': '/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data',
    'RACMO2.4': '/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning',
    'Station': '/nobackup/users/walj/knmi',
}

file_cfg = {
    'Eobs_fine': {
        'Tg': os.path.join(base_dir_cfg['Eobs'], 'tg_ens_mean_0.1deg_reg_v31.0e.nc'),
        'P': os.path.join(base_dir_cfg['Eobs'], 'rr_ens_mean_0.1deg_reg_v31.0e.nc'),
        'SWin': os.path.join(base_dir_cfg['Eobs'], 'qq_ens_mean_0.1deg_reg_v31.0e.nc'),
    },
    'Eobs_coarse': {
        'Tg': os.path.join(base_dir_cfg['Eobs'], 'tg_ens_mean_0.25deg_reg_v31.0e.nc'),
        'P': os.path.join(base_dir_cfg['Eobs'], 'rr_ens_mean_0.25deg_reg_v31.0e.nc'),
    },

    'ERA5_fine': {
        'Tg': os.path.join(base_dir_cfg['ERA5'], 'era5_fine.nc'),
        'P': os.path.join(base_dir_cfg['ERA5'], 'era5_fine.nc'),
    },
    'ERA5_coarse': {
        'Tg': os.path.join(base_dir_cfg['ERA5'], 'era5_coarse_full_t2m.nc'),
        'P': os.path.join(base_dir_cfg['ERA5'], 'era5_coarse_full_tp.nc'),
        'SWin': os.path.join(base_dir_cfg['ERA5'], 'era5_rsds.nc'),
    },

    'RACMO2.3': {
        'Tg': os.path.join(base_dir_cfg['RACMO2.3'], 't2m/*.nc'),
        'P': os.path.join(base_dir_cfg['RACMO2.3'], 'precip/*.nc'),
        'Sq': os.path.join(base_dir_cfg['RACMO2.3'], 'sund/*.nc'),
        'SWin': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'swsd/*.nc'),
        'SWnet': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'swsn/*.nc'),
        'SWincs': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'swsdcs/*.nc'),
        'SWnetcs': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'swsncs/*.nc'),
        'LWin': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'lwsd/*.nc'),
        'LWnet': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'lwsn/*.nc'),
        'LWincs': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'lwsdcs/*.nc'),
        'LWnetcs': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'lwsncs/*.nc'),
        'SHF': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'senf/*.nc'),
        'LHF': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'latf/*.nc'),
        'CloudLow': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'aclcovL/*.nc'),
        'CloudMid': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'aclcovM/*.nc'),
        'CloudHigh': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'aclcovH/*.nc'),
        'CloudTotal': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'aclcov/*.nc'),
        'LWP': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'qli/*.nc'),
        'IWP': os.path.join(base_dir_cfg['RACMO2.3_ERIK'], 'qii/*.nc'),
    },
    'RACMO2.4': {
        'Tg': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/tas{racmo24_sep}*.nc'),
        'P': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/pr{racmo24_sep}*.nc'),
        'Sq': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/sund{racmo24_sep}*.nc'),
        'SWin': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/rsds{racmo24_sep}*.nc'),
        'SWnet': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/ssr{racmo24_sep}*.nc'),
        'SWincs': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/rsdscs{racmo24_sep}*.nc'),
        'SWnetcs': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/ssrc{racmo24_sep}*.nc'),
        'LWin': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/rlds{racmo24_sep}*.nc'),
        'LWnet': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/str{racmo24_sep}*.nc'),
        'LWincs': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/rldscs{racmo24_sep}*.nc'),
        'LWnetcs': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/strc{racmo24_sep}*.nc'),
        'SHF': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/hfss{racmo24_sep}*.nc'),
        'LHF': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/hfls{racmo24_sep}*.nc'),
        'CloudLow': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/cll{racmo24_sep}*.nc'),
        'CloudMid': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/clm{racmo24_sep}*.nc'),
        'CloudHigh': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/clh{racmo24_sep}*.nc'),
        'CloudTotal': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/clt{racmo24_sep}*.nc'),
        'LWP': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/clwvi{racmo24_sep}*.nc'),
        'IWP': os.path.join(base_dir_cfg['RACMO2.4'], f'{freq_str}/clivi{racmo24_sep}*.nc'),
    },

    'Station': {
        'Bilt': os.path.join(base_dir_cfg['Station'], 'KNMI_Bilt.txt'),
        'Cabauw': os.path.join(base_dir_cfg['Station'], 'KNMI_Cabauw.txt'),
        'Eelde': os.path.join(base_dir_cfg['Station'], 'KNMI_Eelde.txt'),
        'Maastricht': os.path.join(base_dir_cfg['Station'], 'KNMI_Maastricht.txt'),
        'Vlissingen': os.path.join(base_dir_cfg['Station'], 'KNMI_Vlissingen.txt'),
        'Kooy': os.path.join(base_dir_cfg['Station'], 'KNMI_Kooy.txt')
    },
}

#%% Loading and processing data

def make_cfg(data_source, var):

    if any(src in data_source for src in data_sources):
        file_key = next(src for src in data_sources if src in data_source)
        cfg = {
            'variable': var_name_cfg[file_key][var],
            'file': file_cfg[data_source][var],
            'file_key': file_key,
            'datatype': 'netcdf',
            'proj': proj_cfg.get(file_key, ccrs.PlateCarree()),
        }
        return cfg
    
    elif data_source in station_sources:
        cfg = {
            'variable': var_name_cfg['Station'][var],
            'file': file_cfg['Station'][data_source],
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

    trim_local = trim_border
    if data_source == 'RACMO2.4' and trim_border is None:
        trim_local = 8

    if cfg['datatype'] == 'netcdf':
        data = preprocess_netcdf(
            source=cfg['file_key'],
            file_path=cfg['file'],
            var_name=cfg['variable'],
            months=months_local,
            years=years_load,
            lats=lats,
            lons=lons,
            land_only=land_only,
            trim_border=trim_local,
            rotpole_sel=rotpole_sel,
            rotpole_native=cfg['proj']
        ).squeeze()

    elif cfg['datatype'] == 'station':
        data = preprocess_station(
            file_path=cfg['file'],
            var_name=cfg['variable'],
            months=months,
            years=years_load,
        ).squeeze()

    month_d = data['time'].dt.month
    year_d = data['time'].dt.year

    month_start = months_local[0]
    month_end = months_local[-1]

    if month_start <= month_end:
        clim_year_d = year_d
    else:
        clim_year_d = xr.where(month_d >= month_start, year_d + 1, year_d)

    data = data.assign_coords(clim_year=clim_year_d)

    data_monthly = data.resample(time='MS').mean('time')
    existing = pd.DatetimeIndex(data['time'].values).to_period('M').unique().to_timestamp()
    data_monthly = data_monthly.sel(time=existing)

    month_m = data_monthly['time'].dt.month
    year_m = data_monthly['time'].dt.year

    if month_start <= month_end:
        clim_year_m = year_m
    else:
        clim_year_m = xr.where(month_m >= month_start, year_m + 1, year_m)

    data_monthly = data_monthly.assign_coords(clim_year=clim_year_m)

    data_year = data_monthly.groupby('clim_year').mean('time')

    if years_req is not None:
        y0, y1 = years_req[0], years_req[-1]

        data = data.where((data['clim_year'] >= y0) & (data['clim_year'] <= y1), drop=True)
        data_monthly = data_monthly.where(
            (data_monthly['clim_year'] >= y0) & (data_monthly['clim_year'] <= y1),
            drop=True
        )

        data_year = data_year.sel(clim_year=slice(y0, y1))

    data_avg = data_year.mean(dim='clim_year').astype('float32')

    time_coord = pd.to_datetime(data_year['clim_year'].astype(str))
    data_year_time = data_year.assign_coords(time=('clim_year', time_coord)).swap_dims({'clim_year': 'time'})
    data_yearly = data_year_time.copy()

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

    return data, data_monthly, data_yearly, data_fit, data_avg

#%% Further processing of base and comparison data

if data_base is not None:

    data_raw_base, data_mm_base, data_yy_base, data_fit_base, data_avg_base = process_source(
        data_base, 
        var,
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

        data_raw_comp, data_mm_comp, data_yy_comp, data_fit_comp, data_avg_comp = process_source(
            data_compare, 
            var, 
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
            trg_grid = grid_with_bounds(data_avg_base, rotpole_native=proj_cfg.get(data_base, ccrs.PlateCarree()))
            src_grid = grid_with_bounds(data_avg_comp, rotpole_native=proj_cfg.get(data_compare, ccrs.PlateCarree()))

        else:
            method = 'bilinear'
            trg_grid = data_avg_base
            src_grid = data_avg_comp

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

        if corr_calc:

            if corr_freq == 'Daily':
                data_corr_base = data_raw_base
                data_corr_comp = data_raw_comp
            elif corr_freq == 'Monthly':
                data_corr_base = data_mm_base
                data_corr_comp = data_mm_comp
            elif corr_freq == 'Yearly':
                data_corr_base = data_yy_base
                data_corr_comp = data_yy_comp
            
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
                    latitude=data_avg_base['latitude'],
                    longitude=data_avg_base['longitude']
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

        data_raw, data_mm, data_yy, data_fit, data_avg = process_source(
            src,
            var,
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

        data_area_avg_raw[src] = data_avg
        data_area_fit_raw[src] = data_fit

        weights = area_weights(data_avg, rotpole_native=proj_cfg.get(src, ccrs.PlateCarree()))
        data_area_avg[src] = area_weighted_mean(data_avg, weights=weights)
        monthly_raw = area_weighted_mean(data_mm, weights=weights)
        yearly_raw = area_weighted_mean(data_fit, weights=weights)

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
    ax.set_ylabel(cfg_plot['ylabel_fit'], fontsize=28)
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

# Opletten met dagelijkse station waardes voor NaNs!!

# Gedaan: 
# Lijnen toevoegen over gebied waarover ik average
# Kijk of lijn 'blokkerig' kan zodat het langs de gridcells heen gaat!
# For contour, first make array full of ones -> then subset_space
# Wanneer ik gemiddelde neem over een gebied, moet ik wel area weighted doen!
# Misschien optie voor alleen temporal of alleen spatial?
# Kijk naar nieuwe versie van subset_space
# Optie voor exacte contour of ongeveer contour! (voor ongeveer contour, gewoon simpel de 4 hoeken nemen...)
# Mask sea values voor area!
# Plotjes maken op het laatst van hoe de geselecteerde data eruit ziet
# Sunshine duration van stations!!!!
# Voor zoomed in plots the difference weghalen!!!
# Nieuwe presentatie maken van main results
# Main results opsommen
# Correlation plots toegevoegd
# Variabelen tegen elkaar plotten!!! Dus van ene dataset tegenover de andere






# Ambitieus of niet slim om te doen:
# # Is het wel mogelijk om over bepaalde gebieden te masken? (Bijvoorbeeld Utrecht / Nederland) (niet doen!)
# Marker toevoegen wanneer point coordinate???
# Internal variability toevoegen!?



    
