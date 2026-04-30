#%% Imports

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

import geopandas as gpd
import cartopy.io.shapereader as shpreader
import regionmask

import cartopy.crs as ccrs
import dask
from pathlib import Path
import sys
from importlib import reload

# Custom libraries
PROJECT_ROOT = Path.home() / 'KNMIproject'
sys.path.insert(0, str(PROJECT_ROOT))

from RegionalTrends.Helpers import ProcessVar
reload(ProcessVar)
from RegionalTrends.Helpers.ProcessVar import load_var

from RegionalTrends.Helpers import AreaWeights
reload(AreaWeights)
from RegionalTrends.Helpers.AreaWeights import area_weights, area_weighted_mean

import RegionalTrends.Helpers.Config.Constants as Constants
reload(Constants)

plt.rcParams['axes.unicode_minus'] = False
dask.config.set(scheduler='threads', num_workers=12)

#%% Data loading inputs

tends_to_load = ['dyntot', 'phystot', 'numtot', 'tendtot', 'swnet', 'lwnet', 'radtot']
t2m_datasets = ['RACMO2.4A', 'ERA5', 'Stations']
file_freq = 'Daily'
years = [1960, 2024]

lats = [50.7, 53.6]
lons = [3.25, 7.35]
proj_sel = 'RACMO2.4'
land_only = True
trim_border = None
shapefile_NL = True

#%% Fixed configuration

data_sources = Constants.DATA_SOURCES
station_sources = Constants.STATION_SOURCES
station_coord_cfg = Constants.STATION_COORD_CFG
proj_cfg = Constants.PROJ_CFG

suffix = {'Daily': 'dailyA', 'Monthly': 'monthlyA', 'Seasonal': 'seasonalA'}[file_freq]
years_load = list(range(years[0], years[1] + 1))
proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())
station_dir = Path('/nobackup/users/walj/knmi') / file_freq
station_names = ['Bilt', 'Eelde', 'Kooy', 'Maastricht', 'Vlissingen']
l1_source = 'RACMO2.4A'

var_file_cfg = {key: value.copy() for key, value in Constants.VAR_FILE_CFG.items()}
for key in ('ERA5', 'RACMO2.4', 'Station'):
    var_file_cfg[key]['temp'] = 'temp'
    var_file_cfg[key]['tendtot'] = 'tendtot'
    var_file_cfg[key]['temp_tendtot'] = 'tendtot'
var_file_cfg['RACMO2.4']['tendtotl1'] = 'tendtot'

build_file_cfg_base = ProcessVar.build_file_cfg


def build_file_cfg_A(freq_str):
    freq_suffix = {'Daily': 'dailyA', 'Monthly': 'monthlyA', 'Seasonal': 'seasonalA'}[freq_str]
    cfg = build_file_cfg_base(freq_str)
    cfg['ERA5']['temp'] = f'/nobackup/users/walj/era5/{freq_str}/temp_{freq_suffix}_EU_ERA5_*.nc'
    cfg['ERA5']['temp_tendtot'] = cfg['ERA5']['temp']
    cfg['RACMO2.4A']['temp'] = f'/nobackup_1/users/walj/racmo24/{freq_str}/temp_{freq_suffix}_KEXT12_RACMO24p1v7_FINAL_enbud_fix_*.nc'
    cfg['RACMO2.4A']['temp_tendtot'] = cfg['RACMO2.4A']['temp']
    for var in Constants.ENBUD_VARS:
        cfg['RACMO2.4A'][var] = f'/nobackup_1/users/walj/racmo24/{freq_str}/{var}_{freq_suffix}_KEXT12_RACMO24p1v7_FINAL_enbud_fix_*.nc'
    return cfg


ProcessVar.build_file_cfg = build_file_cfg_A


def normalize_loaded(da, var_name):
    da = da.astype('float32')
    if var_name in ['temp', 'templ1'] and da.attrs.get('units') == 'K':
        da = (da - np.float32(273.15)).astype('float32')
        da.attrs['units'] = 'degC'
    elif var_name in ['tendtot', 'temp_tendtot', 'tendtotl1'] and da.attrs.get('units') == 'K':
        da.attrs['units'] = 'degC'
    return da


def load_series(var_name, data_source):
    da = load_var(
        var=var_name,
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

    out = normalize_loaded(da, var_name)
    return out


def load_station_mean(var_name):
    arrays = []
    for station in station_names:
        file_path = next(station_dir.glob(f'temp_{suffix}_KNMI_{station}_*.nc'), None)
        with xr.open_dataset(file_path) as ds:
            arrays.append(ds[var_name].where(ds['time'].dt.year.isin(years_load), drop=True).load())
    return xr.concat(arrays, dim='station', join='inner').mean('station', skipna=True).astype('float32')


def load_t2m_dataset(dataset):
    if dataset == 'Stations':
        return {name: load_station_mean(name) for name in ('temp', 'tendtot')}
    return {
        'temp': load_series('temp', dataset),
        'tendtot': load_series('temp_tendtot', dataset),
    }


def x_axis_format(da):
    x = month_day_axis(da['month_day'].values)
    ax = plt.gca()
    month_ticks = month_tick_positions()
    ax.set_xticks(month_ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    return x

#%% ============================================================================
#   PHASE 1: LOAD ALL DATA ONCE
#   ============================================================================

print('=' * 60)
print('PHASE 1: Loading all data')
print('=' * 60)

tend_data = {var: load_series(var, 'RACMO2.4A') for var in tends_to_load}
t2m_data = {dataset: load_t2m_dataset(dataset) for dataset in t2m_datasets}
t2m = {dataset: values['temp'] for dataset, values in t2m_data.items()}
t2m_tendtot = {dataset: values['tendtot'] for dataset, values in t2m_data.items()}
templ1 = load_series('templ1', l1_source)

#%% ============================================================================
#   PHASE 2: DOMAIN AVERAGING OF RAW DATA
#   ============================================================================

print('='*60)
print('PHASE 2: Domain averaging of raw data')
print('='*60)

def area_average(da, data_source):
    sample = da.squeeze()
    for dim in ['time', 'year']:
        if dim in sample.dims:
            sample = sample.isel({dim: 0})

    weights = area_weights(
        sample,
        rotpole_native=proj_cfg.get(data_source, ccrs.PlateCarree()),
    )

    if shapefile_NL and weights is not None and {'latitude', 'longitude'}.issubset(sample.coords):
        countries = gpd.read_file(shpreader.natural_earth('10m', 'cultural', 'admin_0_countries'))
        region = countries[countries['ADMIN'] == 'Netherlands'].to_crs('EPSG:4326')
        mask = regionmask.mask_geopandas(region, sample['longitude'], sample['latitude']).notnull()
        weights = weights.where(mask)
        # area_average.mask = mask

    out = area_weighted_mean(
        da,
        rotpole_native=proj_cfg.get(data_source, ccrs.PlateCarree()),
        weights=weights,
    ).compute().astype('float32')
    return out

for var in tend_data:
    print(f'  Area-weighting tendency: {var}')
    tend_data[var] = area_average(tend_data[var], 'RACMO2.4A')

for dataset in t2m:
    print(f'  Area-weighting {dataset}: temp')
    t2m[dataset] = area_average(t2m[dataset], dataset)

for dataset in t2m_tendtot:
    print(f'  Area-weighting {dataset}: tendtot')
    t2m_tendtot[dataset] = area_average(t2m_tendtot[dataset], dataset)

print(f'  Area-weighting {l1_source}: templ1')
templ1 = area_average(templ1, l1_source)

#%% Phase 3 inputs

year_start_month = 12
rolling_window = 91
rolling_method = 'Triangular'
gaussian_sigma = rolling_window / 4

#%% ============================================================================
#   PHASE 3: NO-LEAP, SMOOTHING, GROUPING
#   ============================================================================

print('='*60)
print('PHASE 3: No-leap, smoothing, grouping')
print('='*60)

periods = {
    'Pref':  [1964, 1985],
    'PrefL': [1964, 1974],
    'PrefR': [1975, 1985],

    'Ptrend':  [1986, 2023],
    'PtrendL': [1986, 2004],
    'PtrendR': [2005, 2023],
}


def rolling_weights(window):
    if rolling_method == 'Gaussian':
        offsets = np.arange(-(window // 2), window // 2 + 1, dtype='float32')
        weights = np.exp(-0.5 * (offsets / max(gaussian_sigma, 1.0))**2).astype('float32')
    elif rolling_method == 'Triangular':
        offsets = np.arange(-(window // 2), window // 2 + 1, dtype='float32')
        weights = (window // 2 + 1 - np.abs(offsets)).astype('float32')
    return xr.DataArray(weights / weights.sum(), dims=('window',))


def smooth_time(da):

    window = min(int(rolling_window), da.sizes['time'])
    if window % 2 == 0:
        window = max(1, window - 1)

    if rolling_method == 'Boxcar':
        out = da.rolling(time=window, center=True, min_periods=window).mean().astype('float32')
        return out

    weights = rolling_weights(window)
    out = (
        da.rolling(time=window, center=True, min_periods=window)
        .construct('window')
        .dot(weights)
        .astype('float32')
    )
    return out


def to_noleap_temp(da):
    mask = ~((da['time'].dt.month == 2) & (da['time'].dt.day == 29))
    return da.where(mask, drop=True).astype('float32')


def to_noleap_tendency(da):
    feb29 = (da['time'].dt.month == 2) & (da['time'].dt.day == 29)
    time_nl = xr.where(feb29, da['time'] - np.timedelta64(1, 'D'), da['time'])
    out = (
        da.assign_coords(time_nl=('time', time_nl.data))
        .groupby('time_nl')
        .sum('time', skipna=True)
        .rename({'time_nl': 'time'})
        .sortby('time')
        .astype('float32')
    )
    return out


def analysis_year(da):
    year = da['time'].dt.year
    if year_start_month == 1:
        return year
    return year + (da['time'].dt.month >= year_start_month)


def month_day_axis(month_day_values):
    x = pd.to_datetime(['2000-' + month_day for month_day in month_day_values])
    if year_start_month == 1:
        return pd.DatetimeIndex(x)
    plot_year = np.where(x.month >= year_start_month, 2000, 2001)
    plot_dates = pd.to_datetime({
        'year': plot_year.astype(int),
        'month': x.month.astype(int),
        'day': x.day.astype(int),
    })
    return pd.DatetimeIndex(plot_dates)


def reorder_month_day(da):
    if 'month_day' not in da.dims or year_start_month == 1:
        return da
    x = month_day_axis(da['month_day'].values)
    return da.isel(month_day=np.argsort(x.values))


def month_tick_positions():
    return pd.date_range(pd.Timestamp(2000, year_start_month, 1), periods=12, freq='MS')


def select_period(da, period):
    years_period = range(period[0], period[1] + 1)
    return da.where(analysis_year(da).isin(years_period), drop=True).astype('float32')


def average_period(da, period):
    da_period = select_period(da, period)

    month_day = xr.DataArray(
        da_period['time'].dt.strftime('%m-%d').data,
        dims=('time',),
        coords={'time': da_period['time']},
        name='month_day',
    )
    out = (
        da_period.assign_coords(month_day=month_day)
        .groupby('month_day')
        .mean('time', skipna=True)
        .astype('float32')
    )
    return reorder_month_day(out)


noleap_tend = {}
for var, da in tend_data.items():
    print(f'  No-leap tendency: {var}')
    noleap_tend[var] = to_noleap_tendency(da)

noleap_t2m = {}
for dataset, da in t2m.items():
    print(f'  No-leap {dataset}: temp')
    noleap_t2m[dataset] = to_noleap_temp(da)

noleap_t2m_tendtot = {}
for dataset, da in t2m_tendtot.items():
    print(f'  No-leap {dataset}: tendtot')
    noleap_t2m_tendtot[dataset] = to_noleap_tendency(da)

noleap_templ1 = to_noleap_temp(templ1)

smooth_tend = {}
for var, da in noleap_tend.items():
    print(f'  Smoothing tendency: {var}')
    smooth_tend[var] = smooth_time(da)

smooth_t2m = {}
for dataset, da in noleap_t2m.items():
    print(f'  Smoothing {dataset}: temp')
    smooth_t2m[dataset] = smooth_time(da)

smooth_t2m_tendtot = {}
for dataset, da in noleap_t2m_tendtot.items():
    print(f'  Smoothing {dataset}: tendtot')
    smooth_t2m_tendtot[dataset] = smooth_time(da)

smooth_templ1 = smooth_time(noleap_templ1)

clim_tend = {}
for var, da in smooth_tend.items():
    print(f'  Grouping tendency: {var}')
    clim_tend[var] = {
        period_key: average_period(da, period)
        for period_key, period in periods.items()
    }

clim_t2m = {}
for dataset, da in smooth_t2m.items():
    print(f'  Grouping {dataset}: temp')
    clim_t2m[dataset] = {
        period_key: average_period(da, period)
        for period_key, period in periods.items()
    }

clim_t2m_tendtot = {}
for dataset, da in smooth_t2m_tendtot.items():
    print(f'  Grouping {dataset}: tendtot')
    clim_t2m_tendtot[dataset] = {
        period_key: average_period(da, period)
        for period_key, period in periods.items()
    }

clim_templ1 = {
    period_key: average_period(smooth_templ1, period)
    for period_key, period in periods.items()
}

#%% Obtaining annual offset

def weights_kappa(period1, period2):
    y0_1, y1_1 = period1
    y0_2, y1_2 = period2

    n1 = y1_1 - y0_1 + 1
    n2 = y1_2 - y0_2 + 1

    years = np.arange(y0_1, y1_2, dtype=int)
    weights = np.empty(years.size, dtype='float32')

    mask_p1 = years <= y1_1
    mask_p2 = years >= y0_2

    weights[mask_p1] = (years[mask_p1] - y0_1 + 1) / n1
    weights[mask_p2] = (y1_2 - years[mask_p2]) / n2

    return xr.DataArray(weights, dims='year', coords={'year': years}, name='kappa')


def weights_alpha(month_day):
    n = len(month_day)
    values = (np.arange(n, dtype='float32') + 1.0) / n
    return xr.DataArray(values, dims='month_day', coords={'month_day': month_day}, name='alpha')


def weights_beta(month_day):
    n = len(month_day)
    values = (n - 1 - np.arange(n, dtype='float32')) / n
    return xr.DataArray(values, dims='month_day', coords={'month_day': month_day}, name='beta')


def to_year_monthday(da):
    year = xr.DataArray(analysis_year(da).data, dims='time', coords={'time': da['time']})
    month_day = xr.DataArray(da['time'].dt.strftime('%m-%d').data, dims='time', coords={'time': da['time']})

    out = (
        da.assign_coords(year=year, month_day=month_day)
          .set_index(time=['year', 'month_day'])
          .unstack('time')
          .transpose('year', 'month_day')
          .astype('float32')
    )
    return reorder_month_day(out)


def C_ann_from_A(A_y_d):
    alpha = weights_alpha(A_y_d['month_day'])
    beta = weights_beta(A_y_d['month_day'])

    A_start = A_y_d.isel(year=slice(0, -1))
    A_end = A_y_d.isel(year=slice(1, None)).assign_coords(year=A_start['year'])

    return (alpha * A_start + beta * A_end).sum('month_day').astype('float32')


def split_kappa(kappa, period_left, period_right):
    y0_l, y1_l = period_left
    y0_r, y1_r = period_right

    kappa_left = kappa.where((kappa['year'] >= y0_l) & (kappa['year'] <= y1_l), 0).astype('float32')
    kappa_right = kappa.where((kappa['year'] >= y0_r) & (kappa['year'] <= y1_r - 1), 0).astype('float32')

    return kappa_left, kappa_right


def weighted_Ahat(A_y_d, kappa_side):
    weight_sum = kappa_side.sum('year').astype('float32')

    Ahat_start = (
        (A_y_d.sel(year=kappa_side['year']) * kappa_side).sum('year') / weight_sum
    ).astype('float32')

    A_end_source = (
        A_y_d.sel(year=(kappa_side['year'].values + 1))
             .assign_coords(year=kappa_side['year'])
    )
    Ahat_end = (
        (A_end_source * kappa_side).sum('year') / weight_sum
    ).astype('float32')

    return Ahat_start, Ahat_end, weight_sum


def new_cumsum(da):
    return da.shift(month_day=1, fill_value=0).cumsum('month_day').astype('float32')


def aggregate_calendar(da, split='day'):
    if split == 'day':
        return da.astype('float32')

    month = xr.DataArray(
        pd.Index(da['month_day'].values.astype(str)).str.slice(0, 2).astype(int),
        dims='month_day',
        coords={'month_day': da['month_day']},
        name='month',
    )

    if split == 'month':
        return da.assign_coords(month=month).groupby('month').sum('month_day').astype('float32')

    if split == 'season':
        season_labels = np.empty(month.size, dtype='<U3')
        season_labels[np.isin(month, [12, 1, 2])] = 'DJF'
        season_labels[np.isin(month, [3, 4, 5])] = 'MAM'
        season_labels[np.isin(month, [6, 7, 8])] = 'JJA'
        season_labels[np.isin(month, [9, 10, 11])] = 'SON'

        season = xr.DataArray(
            season_labels,
            dims='month_day',
            coords={'month_day': da['month_day']},
            name='season',
        )
        out = da.assign_coords(season=season).groupby('season').sum('month_day').astype('float32')
        order = [s for s in ['DJF', 'MAM', 'JJA', 'SON'] if s in out['season'].values]
        return out.sel(season=order)


def determine_mu(A_time, period_left, period_right, split='day'):
    A_y_d = to_year_monthday(A_time)
    C_ann = C_ann_from_A(A_y_d)

    kappa = weights_kappa(period_left, period_right)
    kappa_left, kappa_right = split_kappa(kappa, period_left, period_right)

    mu_total = (kappa * C_ann.sel(year=kappa['year'])).sum('year').astype('float32')

    Ahat_start_left, Ahat_end_left, weight_left = weighted_Ahat(A_y_d, kappa_left)
    Ahat_start_right, Ahat_end_right, weight_right = weighted_Ahat(A_y_d, kappa_right)

    alpha = weights_alpha(A_y_d['month_day'])
    beta = weights_beta(A_y_d['month_day'])

    B_left = (alpha * Ahat_start_left + beta * Ahat_end_left).astype('float32')
    B_right = (alpha * Ahat_start_right + beta * Ahat_end_right).astype('float32')

    curve_left = aggregate_calendar(weight_left * B_left, split=split)
    curve_right = aggregate_calendar(weight_right * B_right, split=split)
    curve_total = (curve_left + curve_right).astype('float32')

    return {
        'mu_total': mu_total,
        'mu_left': curve_left.sum().astype('float32'),
        'mu_right': curve_right.sum().astype('float32'),
        'curve_left': curve_left,
        'curve_right': curve_right,
        'curve_total': curve_total,
    }


def determine_shape(A_left_clim, A_right_clim):
    deltaA = (A_right_clim - A_left_clim).astype('float32')
    Theta = (new_cumsum(deltaA) - new_cumsum(deltaA).mean('month_day')).astype('float32')

    return {
        'deltaA': deltaA,
        'Theta': Theta,
    }


def period_label(period_key):
    return f'{periods[period_key][0]}-{periods[period_key][1]}'


#%% Plotting climatological tendencies

fig, ax = plt.subplots(figsize=(12, 6))
x_axis = x_axis_format(clim_tend['tendtot']['Ptrend'])
ax2 = ax.twinx()
ax2.grid(False)

ax.plot(x_axis, 10*clim_tend['tendtot']['PtrendL'], color='black', label='Total x10')
ax.plot(x_axis, 10*clim_tend['tendtot']['PtrendR'], color='black', linestyle='dashed')

ax.plot(x_axis, clim_tend['dyntot']['PtrendL'], color='red', label='Dynamical')
ax.plot(x_axis, clim_tend['dyntot']['PtrendR'], color='red', linestyle='dashed')

ax.plot(x_axis, clim_tend['phystot']['PtrendL'], color='blue', label='Diabatic')
ax.plot(x_axis, clim_tend['phystot']['PtrendR'], color='blue', linestyle='dashed')

ax.plot(x_axis, clim_tend['numtot']['PtrendL'], color='green', label='Numerical')
ax.plot(x_axis, clim_tend['numtot']['PtrendR'], color='green', linestyle='dashed')

ax2.plot(x_axis, clim_templ1['PtrendL'], color='purple', label='Temperature')
ax2.plot(x_axis, clim_templ1['PtrendR'], color='purple', linestyle='dashed')

ax.set_xlabel('Month')
ax.set_ylabel(r'Tendency contribution [$^\circ$C day$^{-1}$]')
ax2.set_ylabel(r'Temperature [$^\circ$C]')

ax.axhline(0, color='gray', linewidth=0.8, alpha=0.7)
ax.grid(alpha=0.25)

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

legend1 = ax.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc='upper left',
    ncol=2,
    frameon=True,
)

period_handles = [
    Line2D([0], [0], color='gray', linestyle='solid', label=period_label('PtrendL')),
    Line2D([0], [0], color='gray', linestyle='dashed', label=period_label('PtrendR')),
]

legend2 = ax.legend(
    handles=period_handles,
    loc='upper right',
    frameon=True,
)

ax.add_artist(legend1)

fig.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(12, 6))
x_axis = x_axis_format(clim_tend['tendtot']['Ptrend'])
ax2 = ax.twinx()
ax2.grid(False)

delta_tendtot = clim_tend['tendtot']['PtrendR'] - clim_tend['tendtot']['PtrendL']
delta_dyntot = clim_tend['dyntot']['PtrendR'] - clim_tend['dyntot']['PtrendL']
delta_phystot = clim_tend['phystot']['PtrendR'] - clim_tend['phystot']['PtrendL']
delta_numtot = clim_tend['numtot']['PtrendR'] - clim_tend['numtot']['PtrendL']
delta_temp = clim_templ1['PtrendR'] - clim_templ1['PtrendL']

ax.plot(x_axis, 10*delta_tendtot, color='black', label='Total x10')
ax.plot(x_axis, delta_dyntot, color='red', label='Dynamical')
ax.plot(x_axis, delta_phystot, color='blue', label='Diabatic')
ax.plot(x_axis, delta_numtot, color='green', label='Numerical')

ax2.plot(x_axis, delta_temp, color='purple', label='Temperature')
ax2.plot(
    x_axis,
    new_cumsum(delta_tendtot),
    color='purple',
    linestyle='dashed',
    label='Reconstructed temperature',
)

ax.set_xlabel('Month')
ax.set_ylabel(r'Change in tendency contribution [$^\circ$C day$^{-1}$]')
ax2.set_ylabel(r'Change in temperature [$^\circ$C]')

ax.axhline(0, color='gray', linewidth=0.8, alpha=0.7)
ax.grid(alpha=0.25)

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc='upper left',
    ncol=2,
    frameon=True,
)

fig.tight_layout()
plt.show()


#%% Radiative tendencies for the two periods

fig, ax = plt.subplots(figsize=(12, 6))
x_axis = x_axis_format(clim_tend['radtot']['Ptrend'])

ax.plot(x_axis, clim_tend['radtot']['PtrendL'], color='black', label='Radiative total')
ax.plot(x_axis, clim_tend['radtot']['PtrendR'], color='black', linestyle='dashed')

ax.plot(x_axis, clim_tend['swnet']['PtrendL'], color='red', label='SW net')
ax.plot(x_axis, clim_tend['swnet']['PtrendR'], color='red', linestyle='dashed')

ax.plot(x_axis, clim_tend['lwnet']['PtrendL'], color='blue', label='LW net')
ax.plot(x_axis, clim_tend['lwnet']['PtrendR'], color='blue', linestyle='dashed')

ax.set_xlabel('Month')
ax.set_ylabel(r'Tendency contribution [$^\circ$C day$^{-1}$]')
ax.axhline(0, color='gray', linewidth=0.8, alpha=0.7)
ax.grid(alpha=0.25)

handles_main, labels_main = ax.get_legend_handles_labels()
legend1 = ax.legend(handles_main, labels_main, loc='upper left', ncol=2, frameon=True)

period_handles = [
    Line2D([0], [0], color='gray', linestyle='solid', label=period_label('PtrendL')),
    Line2D([0], [0], color='gray', linestyle='dashed', label=period_label('PtrendR')),
]
legend2 = ax.legend(handles=period_handles, loc='upper right', frameon=True)

ax.add_artist(legend1)

fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
x_axis = x_axis_format(clim_tend['radtot']['Ptrend'])

delta_radtot = clim_tend['radtot']['PtrendR'] - clim_tend['radtot']['PtrendL']
delta_swnet  = clim_tend['swnet']['PtrendR']  - clim_tend['swnet']['PtrendL']
delta_lwnet  = clim_tend['lwnet']['PtrendR']  - clim_tend['lwnet']['PtrendL']

ax.plot(x_axis, delta_radtot, color='black', label='Radiative total')
ax.plot(x_axis, delta_swnet, color='red', label='SW net')
ax.plot(x_axis, delta_lwnet, color='blue', label='LW net')

ax.set_xlabel('Month')
ax.set_ylabel(r'Change in tendency contribution [$^\circ$C day$^{-1}$]')
ax.axhline(0, color='gray', linewidth=0.8, alpha=0.7)
ax.grid(alpha=0.25)
ax.legend(ncol=2, frameon=True)

fig.tight_layout()
plt.show()

#%% Offsets

offset_periods = {
    'Pref': ('PrefL', 'PrefR'),
    'Ptrend': ('PtrendL', 'PtrendR'),
}

offset_tend = {
    period_key: {
        var: determine_mu(
            A_time=smooth_tend[var],
            period_left=periods[left_key],
            period_right=periods[right_key],
            split='day',
        )
        for var in tends_to_load
    }
    for period_key, (left_key, right_key) in offset_periods.items()
}

offset_L = {
    period_key: weights_kappa(periods[left_key], periods[right_key]).sum('year').astype('float32')
    for period_key, (left_key, right_key) in offset_periods.items()
}


def offset_curve(period_key, var):
    return offset_tend[period_key][var]['curve_total'] / offset_L[period_key]


def offset_mu(period_key, var='tendtot'):
    return offset_tend[period_key][var]['mu_total'] / offset_L[period_key]


def offset_mu_per_decade(period_key, var='tendtot'):
    return 10 * offset_mu(period_key, var)


offset_period_handles = [
    Line2D([0], [0], color='gray', linestyle='solid', label=f"Ptrend ({period_label('Ptrend')})"),
    Line2D([0], [0], color='gray', linestyle='dashed', label=f"Pref ({period_label('Pref')})"),
]


fig, ax = plt.subplots(figsize=(12, 6))
x_axis = x_axis_format(offset_tend['Ptrend']['tendtot']['curve_total'])
ax2 = ax.twinx()
ax2.grid(False)

ax.plot(x_axis, 10*offset_curve('Ptrend', 'tendtot'), color='black', label='Total x10')
ax.plot(x_axis, 10*offset_curve('Pref', 'tendtot'), color='black', linestyle='dashed')

ax.plot(x_axis, offset_curve('Ptrend', 'dyntot'), color='red', label='Dynamical')
ax.plot(x_axis, offset_curve('Pref', 'dyntot'), color='red', linestyle='dashed')

ax.plot(x_axis, offset_curve('Ptrend', 'phystot'), color='blue', label='Diabatic')
ax.plot(x_axis, offset_curve('Pref', 'phystot'), color='blue', linestyle='dashed')

ax.plot(x_axis, offset_curve('Ptrend', 'numtot'), color='green', label='Numerical')
ax.plot(x_axis, offset_curve('Pref', 'numtot'), color='green', linestyle='dashed')

ax2.axhline(offset_mu_per_decade('Ptrend').item(), color='purple', label='Total annual offset trend')
ax2.axhline(offset_mu_per_decade('Pref').item(), color='purple', linestyle='dashed')

ax.set_xlabel('Month')
ax.set_ylabel(r'Normalized annual-offset contribution [$^\circ$C day$^{-1}$]')
ax2.set_ylabel(r'Annual offset trend [$^\circ$C decade$^{-1}$]')

ax.axhline(0, color='gray', linewidth=0.8, alpha=0.7)
ax.grid(alpha=0.25)

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

legend1 = ax.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc='upper left',
    ncol=2,
    frameon=True,
)

legend2 = ax.legend(
    handles=offset_period_handles,
    loc='upper right',
    frameon=True,
)

ax.add_artist(legend1)

fig.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(12, 6))
x_axis = x_axis_format(offset_tend['Ptrend']['tendtot']['curve_total'])
ax2 = ax.twinx()
ax2.grid(False)

delta_offset_tendtot = offset_curve('Ptrend', 'tendtot') - offset_curve('Pref', 'tendtot')
delta_offset_dyntot = offset_curve('Ptrend', 'dyntot') - offset_curve('Pref', 'dyntot')
delta_offset_phystot = offset_curve('Ptrend', 'phystot') - offset_curve('Pref', 'phystot')
delta_offset_numtot = offset_curve('Ptrend', 'numtot') - offset_curve('Pref', 'numtot')

ax.plot(x_axis, 10*delta_offset_tendtot, color='black', label='Total x10')
ax.plot(x_axis, delta_offset_dyntot, color='red', label='Dynamical')
ax.plot(x_axis, delta_offset_phystot, color='blue', label='Diabatic')
ax.plot(x_axis, delta_offset_numtot, color='green', label='Numerical')

ax2.axhline(
    (offset_mu_per_decade('Ptrend') - offset_mu_per_decade('Pref')).item(),
    color='purple',
    label='Total annual offset trend difference',
)

ax.set_xlabel('Month')
ax.set_ylabel(r'Normalized annual-offset contribution difference [$^\circ$C day$^{-1}$]')
ax2.set_ylabel(r'Annual offset trend difference [$^\circ$C decade$^{-1}$]')

ax.axhline(0, color='gray', linewidth=0.8, alpha=0.7)
ax.grid(alpha=0.25)

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc='upper left',
    ncol=2,
    frameon=True,
)

fig.tight_layout()
plt.show()


#%% Radiative annual offsets

fig, ax = plt.subplots(figsize=(12, 6))
x_axis = x_axis_format(offset_tend['Ptrend']['radtot']['curve_total'])

ax.plot(x_axis, offset_curve('Ptrend', 'radtot'), color='black', label='Radiative total')
ax.plot(x_axis, offset_curve('Pref', 'radtot'), color='black', linestyle='dashed')

ax.plot(x_axis, offset_curve('Ptrend', 'swnet'), color='red', label='SW net')
ax.plot(x_axis, offset_curve('Pref', 'swnet'), color='red', linestyle='dashed')

ax.plot(x_axis, offset_curve('Ptrend', 'lwnet'), color='blue', label='LW net')
ax.plot(x_axis, offset_curve('Pref', 'lwnet'), color='blue', linestyle='dashed')

ax.set_xlabel('Month')
ax.set_ylabel(r'Normalized annual-offset contribution [$^\circ$C day$^{-1}$]')
ax.axhline(0, color='gray', linewidth=0.8, alpha=0.7)
ax.grid(alpha=0.25)

handles_main, labels_main = ax.get_legend_handles_labels()
legend1 = ax.legend(handles_main, labels_main, loc='upper left', ncol=2, frameon=True)

legend2 = ax.legend(handles=offset_period_handles, loc='upper right', frameon=True)

ax.add_artist(legend1)

fig.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(12, 6))
x_axis = x_axis_format(offset_tend['Ptrend']['radtot']['curve_total'])

delta_offset_radtot = offset_curve('Ptrend', 'radtot') - offset_curve('Pref', 'radtot')
delta_offset_swnet = offset_curve('Ptrend', 'swnet') - offset_curve('Pref', 'swnet')
delta_offset_lwnet = offset_curve('Ptrend', 'lwnet') - offset_curve('Pref', 'lwnet')

ax.plot(x_axis, delta_offset_radtot, color='black', label='Radiative total')
ax.plot(x_axis, delta_offset_swnet, color='red', label='SW net')
ax.plot(x_axis, delta_offset_lwnet, color='blue', label='LW net')

ax.set_xlabel('Month')
ax.set_ylabel(r'Normalized annual-offset contribution difference [$^\circ$C day$^{-1}$]')
ax.axhline(0, color='gray', linewidth=0.8, alpha=0.7)
ax.grid(alpha=0.25)
ax.legend(ncol=2, frameon=True)

fig.tight_layout()
plt.show()




























































































#%% IGNORE


# mu1_2 = determine_mu(
#     A_time=smooth_tend['tendtot'],
#     period_left=periods['P1'],
#     period_right=periods['P2'],
#     split='day',   # or 'month' or 'season'
# )

# mu2_3 = determine_mu(
#     A_time=smooth_tend['tendtot'],
#     period_left=periods['P2'],
#     period_right=periods['P3'],
#     split='day',   # or 'month' or 'season'
# )

# shape1_2 = determine_shape(
#     A_left_clim=clim_tend['tendtot']['P1'],
#     A_right_clim=clim_tend['tendtot']['P2'],
# )

# shape2_3 = determine_shape(
#     A_left_clim=clim_tend['tendtot']['P2'],
#     A_right_clim=clim_tend['tendtot']['P3'],
# )

# shape1_3 = determine_shape(
#     A_left_clim=clim_tend['tendtot']['P1'],
#     A_right_clim=clim_tend['tendtot']['P3'],
# )

# # deltaT_recon = mu['mu_total'] + shape['Theta']
# # deltaT_true = clim_templ1['P2'] - clim_templ1['P1']

# #%%

# fig, ax = plt.subplots(figsize=(12, 6))

# x_axis = x_axis_format(clim_templ1['P1'])

# ax.plot(x_axis, (mu2_3['curve_total'] - mu1_2['curve_total']), color='black', linewidth=2, label='true')

# ax.plot(x_axis, shape1_3['Theta'], color='black', linewidth=2, label='true')






# #%% Annual offset part 2

# A_y_d_total = to_year_monthday(smooth_tend['tendtot'])
# C_ann_total = C_ann_from_A(A_y_d_total)

# # Exact kappa weights and exact side split
# kappa = weights_kappa(periods['P1'], periods['P2'])
# kappa_left, kappa_right = split_kappa(kappa, periods['P1'], periods['P2'])

# mu_ann_total = (kappa*C_ann_total.sel(year=kappa['year'])).sum('year').astype('float32')

# # Same side split, but now written through weighted climatological bridge objects
# Ahat_start_left, Ahat_end_left, weight_left = weighted_Ahat(A_y_d_total, kappa_left)
# Ahat_start_right, Ahat_end_right, weight_right = weighted_Ahat(A_y_d_total, kappa_right)


# # Calendar-year bridge curves
# alpha = weights_alpha(Ahat_start_left['month_day'])
# beta = weights_beta(Ahat_end_left['month_day'])

# B_left = (alpha * Ahat_start_left + beta * Ahat_end_left).astype('float32')
# B_right = (alpha * Ahat_start_right + beta * Ahat_end_right).astype('float32')

# # Daily contribution densities whose yearly sums equal the side contributions
# D_left = (weight_left * B_left).astype('float32')
# D_right = (weight_right * B_right).astype('float32')
# D_total = (D_left + D_right).astype('float32')

# # Grouped-climatology shape term and full exact reconstruction
# deltaA = (clim_tend['tendtot']['P2'] - clim_tend['tendtot']['P1']).astype('float32')
# Q = new_cumsum(deltaA)
# Theta = (Q - Q.mean('month_day')).astype('float32')

# deltaT_recon = (mu_ann_total + Theta).astype('float32')
# deltaT_true = (clim_templ1['P2'] - clim_templ1['P1']).astype('float32')


# # Function for return dictionary of trend mu split up in left and right but also per 

# #%% Plot 1. Exact daily reconstruction of P3 - P2

# def x_axis_format(da):
#     return month_day_axis(da['month_day'].values)

# fig, ax = plt.subplots(figsize=(12, 6))

# ax.plot(x_axis_format(deltaT_true), deltaT_true, color='black', linewidth=2, label='true')
# ax.plot(x_axis_format(deltaT_recon), deltaT_recon, color='red', linestyle='--', linewidth=2, label='exact recon')

# month_ticks = month_tick_positions()
# ax.set_xticks(month_ticks)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
# ax.legend()


# #%% Plot 2. Exact annual-offset contribution densities over the calendar year

# fig, ax = plt.subplots(figsize=(12, 6))

# ax.plot(x_axis_format(D_left), D_left, color='red', linewidth=2, label=f'{'P1'} side')
# ax.plot(x_axis_format(D_right), D_right, color='blue', linewidth=2, label=f'{'P2'} side')
# ax.plot(x_axis_format(D_total), D_total, color='black', linewidth=2, label='total')

# month_ticks = month_tick_positions()
# ax.set_xticks(month_ticks)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
# ax.legend()


# #%% Plot 3. Weighted bridge climatologies themselves

# fig, ax = plt.subplots(figsize=(12, 6))

# ax.plot(x_axis_format(Ahat_start_left), Ahat_start_left, color='red', linewidth=2, label=f'{'P1'} start')
# ax.plot(x_axis_format(Ahat_end_left), Ahat_end_left, color='red', linestyle='--', linewidth=2, label=f'{'P2'} end')

# ax.plot(x_axis_format(Ahat_start_right), Ahat_start_right, color='blue', linewidth=2, label=f'{'P1'} start')
# ax.plot(x_axis_format(Ahat_end_right), Ahat_end_right, color='blue', linestyle='--', linewidth=2, label=f'{'P2'} end')

# month_ticks = month_tick_positions()
# ax.set_xticks(month_ticks)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
# ax.legend()

# #%% Plotting of climatological A

# fig, ax = plt.subplots(figsize=(12, 6))

# x_axis = x_axis_format(clim_templ1['P1'])
# ax.plot(x_axis, clim_templ1['P1'], color='red')
# ax.plot(x_axis, clim_templ1['P2'], color='blue')
# ax.plot(x_axis, clim_templ1['P3'], color='green')


# fig, ax = plt.subplots(figsize=(12, 6))

# x_axis = x_axis_format(clim_templ1['P1'])
# ax.plot(x_axis, clim_templ1['P2'] - clim_templ1['P1'], color='red')
# ax.plot(x_axis, clim_templ1['P3'] - clim_templ1['P2'], color='blue')
# ax.plot(x_axis, clim_templ1['P3'] - clim_templ1['P1'], color='green')





























# #%% ============================================================================
# #   PHASE 4: PLOTTING
# #   ============================================================================

# print('='*60)
# print('PHASE 4: Plotting')
# print('='*60)



# # fig, ax = plt.subplots(figsize=(12, 6))

# # ax.plot(x_axis_format(clim_temp['RACMO2.4A']['P1']), clim_temp['RACMO2.4A']['P1'], label='RACMO2.4A P1', color='red')
# # ax.plot(x_axis_format(clim_temp['RACMO2.4A']['P2']), clim_temp['RACMO2.4A']['P2'], label='RACMO2.4A P2', color='blue')
# # ax.plot(x_axis_format(clim_temp['RACMO2.4A']['P3']), clim_temp['RACMO2.4A']['P3'], label='RACMO2.4A P3', color='green')

# # month_ticks = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
# # ax.set_xticks(month_ticks)
# # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


# # fig, ax = plt.subplots(figsize=(12, 6))

# # ax.plot(x_axis_format(clim_temp['RACMO2.4A']['P1']), clim_temp['RACMO2.4A']['P2'] - clim_temp['RACMO2.4A']['P1'], label='RACMO2.4A P1', color='red')
# # ax.plot(x_axis_format(clim_temp['RACMO2.4A']['P1']), clim_temp['RACMO2.4A']['P3'] - clim_temp['RACMO2.4A']['P2'], label='RACMO2.4A P1', color='blue')
# # ax.plot(x_axis_format(clim_temp['RACMO2.4A']['P1']), clim_temp['RACMO2.4A']['P3'] - clim_temp['RACMO2.4A']['P1'], label='RACMO2.4A P1', color='green')

# # month_ticks = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
# # ax.set_xticks(month_ticks)
# # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))



# # fig, ax = plt.subplots(figsize=(12, 6))

# # ax.plot(x_axis_format(clim_tendtot['RACMO2.4A']['P1']), clim_tendtot['RACMO2.4A']['P1'], label='RACMO2.4A P1', color='red')
# # ax.plot(x_axis_format(clim_tendtot['RACMO2.4A']['P2']), clim_tendtot['RACMO2.4A']['P2'], label='RACMO2.4A P2', color='blue')
# # ax.plot(x_axis_format(clim_tendtot['RACMO2.4A']['P3']), clim_tendtot['RACMO2.4A']['P3'], label='RACMO2.4A P3', color='green')

# # month_ticks = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
# # ax.set_xticks(month_ticks)
# # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


# # fig, ax = plt.subplots(figsize=(12, 6))

# # ax.plot(x_axis_format(clim_tendtot['RACMO2.4A']['P1']), clim_tendtot['RACMO2.4A']['P2'] - clim_tendtot['RACMO2.4A']['P1'], label='RACMO2.4A P1', color='red')
# # ax.plot(x_axis_format(clim_tendtot['RACMO2.4A']['P1']), clim_tendtot['RACMO2.4A']['P3'] - clim_tendtot['RACMO2.4A']['P2'], label='RACMO2.4A P1', color='blue')
# # ax.plot(x_axis_format(clim_tendtot['RACMO2.4A']['P1']), clim_tendtot['RACMO2.4A']['P3'] - clim_tendtot['RACMO2.4A']['P1'], label='RACMO2.4A P1', color='green')
# # month_ticks = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
# # ax.set_xticks(month_ticks)
# # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


# #%%

# def new_cumsum(da):
#     da_new = da.shift(month_day=1, fill_value=0)
#     return da_new.cumsum(dim='month_day')

# plot_test = new_cumsum(clim_tend['tendtot']['P2']) + clim_templ1['P2'].isel(month_day=0)

# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(x_axis_format(clim_templ1['P1']), clim_templ1['P2'], color='red')
# ax.plot(x_axis_format(clim_templ1['P1']), plot_test, color='blue', linestyle='--')

# month_ticks = month_tick_positions()

# ax.set_xticks(month_ticks)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))



# #%%

# def deoy(da):
#     da_cumsum = new_cumsum(da)
#     return da_cumsum.isel(month_day=-1) + da.isel(month_day=-1) 

# recon_temp1 = new_cumsum(clim_tend['tendtot']['P2'])
# recon_temp2 = new_cumsum(clim_tend['tendtot']['P3'])

# plot_test = recon_temp2 - recon_temp1 + 10*deoy(clim_tend['tendtot']['P2']) + 10*deoy(clim_tend['tendtot']['P3'])
# # plot_test = recon_temp2 - recon_temp1 + clim_t2m['RACMO2.4A']['P3'].isel(month_day=0) - clim_t2m['RACMO2.4A']['P2'].isel(month_day=0)


# fig, ax = plt.subplots(figsize=(12, 6))

# ax.plot(x_axis_format(clim_templ1['P1']), clim_templ1['P3'] - clim_templ1['P2'], color='red')
# ax.plot(x_axis_format(clim_tend['tendtot']['P1']), recon_temp2 - recon_temp1, color='blue')
# ax.plot(x_axis_format(clim_tend['tendtot']['P1']), plot_test, color='green')

# month_ticks = month_tick_positions()
# ax.set_xticks(month_ticks)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


# #%%

# def int_constant0(da):
#     raw_int = new_cumsum(da)
#     move_int = raw_int.mean(dim='month_day')
#     return raw_int - move_int


# recon_temp1 = int_constant0(clim_tend['tendtot']['P1'])
# recon_temp2 = int_constant0(clim_tend['tendtot']['P2'])

# plot_test = recon_temp2 - recon_temp1 + 0*(recon_temp2 - recon_temp1).sum(dim='month_day')
# # plot_test = recon_temp2 - recon_temp1 + clim_t2m['RACMO2.4A']['P3'].isel(month_day=0) - clim_t2m['RACMO2.4A']['P2'].isel(month_day=0)


# fig, ax = plt.subplots(figsize=(12, 6))

# ax.plot(x_axis_format(clim_templ1['P1']), clim_templ1['P2'] - clim_templ1['P1'], color='red')
# ax.plot(x_axis_format(clim_tend['tendtot']['P1']), recon_temp2 - recon_temp1, color='blue')
# ax.plot(x_axis_format(clim_tend['tendtot']['P1']), plot_test, color='green')

# month_ticks = month_tick_positions()
# ax.set_xticks(month_ticks)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


#%%









#%%


# (noleap_templ1.sel(time=slice('2001-01-01', '2021-01-01')).values - noleap_templ1.sel(time=slice('1961-01-01', '1981-01-01')).values).mean()

# Hoe ziet een gemiddelde dag eruit in een bepaald seizoen in een bepaalde periode?

# Do shapefile in this file itself.

# Periodes per decade...



# Shapefile en ding meer als notebook maken
# daarna figuren met kalenderdag links en stacked barplots per seizoen rechts
