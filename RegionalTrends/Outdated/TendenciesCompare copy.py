#%% Imports

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

#%% User inputs

# tends_to_load = ['dyntot', 'phystot', 'numtot', 'tendtot']
tends_to_load = ['tendtot']
t2m_datasets = ['RACMO2.4A']
file_freq = 'Daily'
periods = {'P1': [1961, 1980], 'P2': [1981, 2000], 'P3': [2001, 2020]}

rolling_window = 91
rolling_method = 'Triangular'
gaussian_sigma = rolling_window / 4
tend_contrib_source = 'smooth_tend'
# options: 'raw_tend', 'smooth_tend', 'clim_bound', 'loaded'
loaded_tend_contrib = None
loaded_t2m_tendtot_contrib = None

lats = [50.7, 53.6]
lons = [3.25, 7.35]
proj_sel = 'RACMO2.4'
land_only = True
trim_border = None

#%% Fixed configuration

data_sources = Constants.DATA_SOURCES
station_sources = Constants.STATION_SOURCES
station_coord_cfg = Constants.STATION_COORD_CFG
proj_cfg = Constants.PROJ_CFG

suffix = {'Daily': 'dailyA', 'Monthly': 'monthlyA', 'Seasonal': 'seasonalA'}[file_freq]
years = sorted({
    year
    for period in periods.values()
    for year in range(period[0], period[1] + 1)
})
years_load = sorted({
    year
    for period in periods.values()
    for year in range(period[0] - 1, period[1] + 2)
})
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
    for var in tends_to_load:
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
    weights = area_weights(
        da.isel(time=0).squeeze() if 'time' in da.dims else da.squeeze(),
        rotpole_native=proj_cfg.get(data_source, ccrs.PlateCarree()),
    )
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

#%% ============================================================================
#   PHASE 3: NO-LEAP, SMOOTHING, GROUPING
#   ============================================================================

print('='*60)
print('PHASE 3: No-leap, smoothing, grouping')
print('='*60)

# rolling_method = 'Triangular'
# rolling_window = 91

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


def select_period(da, period):
    years_period = range(period[0], period[1] + 1)
    return da.where(da['time'].dt.year.isin(years_period), drop=True).astype('float32')


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
        .sortby('month_day')
        .astype('float32')
    )
    return out


def period_bounds(period):
    bounds = {
        'early': [period[0], period[1] - 1],
        'late': [period[0] + 1, period[1]],
    }
    return bounds


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

clim_tend_bound = {}
for var, da in smooth_tend.items():
    print(f'  Grouping tendency bounds: {var}')
    clim_tend_bound[var] = {
        period_key: {
            bound_key: average_period(da, bound_period)
            for bound_key, bound_period in period_bounds(period).items()
        }
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

clim_t2m_tendtot_bound = {}
for dataset, da in smooth_t2m_tendtot.items():
    print(f'  Grouping {dataset} tendency bounds')
    clim_t2m_tendtot_bound[dataset] = {
        period_key: {
            bound_key: average_period(da, bound_period)
            for bound_key, bound_period in period_bounds(period).items()
        }
        for period_key, period in periods.items()
    }

clim_templ1 = {
    period_key: average_period(smooth_templ1, period)
    for period_key, period in periods.items()
}


#%% Yearly contributions


def assign_clim_time(da, year):
    if 'time' in da.dims:
        return da
    time = pd.to_datetime([f'{year}-{month_day}' for month_day in da['month_day'].values])
    return da.rename(month_day='time').assign_coords(time=time)


def tendency_contributions(da, split='Seasonal', weighted=False, rescale=False, start_month=None, da_next=None, base_year=2001):

    if da_next is not None:
        da = xr.concat(
            [
                assign_clim_time(da, base_year),
                assign_clim_time(da_next, base_year + 1),
            ],
            dim='time',
        )
    else:
        da = assign_clim_time(da, base_year)

    time_index = pd.DatetimeIndex(da['time'].values)
    month_index = time_index.month.astype(int)

    if split == 'Daily':
        split_keys = sorted(np.unique(time_index.strftime('%m-%d')))
        split_index = np.asarray(time_index.strftime('%m-%d'))
        start_month = 1 if start_month is None else start_month
    elif split == 'Monthly':
        split_keys = list(range(1, 13))
        split_index = month_index
        start_month = 1 if start_month is None else start_month
    else:
        split_keys = ['DJF', 'MAM', 'JJA', 'SON']
        split_index = np.array([
            'DJF' if month in [12, 1, 2] else
            'MAM' if month in [3, 4, 5] else
            'JJA' if month in [6, 7, 8] else
            'SON'
            for month in month_index
        ])
        start_month = 12 if start_month is None else start_month

    if split == 'Seasonal' and 'season_year' in da.coords:
        year_index = np.asarray(da['season_year'].values, dtype=int)
    elif start_month == 1:
        year_index = time_index.year.astype(int)
    else:
        year_index = (time_index.year + (month_index >= start_month)).astype(int)
# Hier gebleven
    available_years = np.unique(year_index)
    target_years = np.array([year for year in available_years if year + 1 in available_years], dtype=int) if weighted else available_years

    def select_year(target_year):
        year_mask = xr.DataArray(
            year_index == target_year,
            dims='time',
            coords={'time': da['time']},
        )
        da_year = da.where(year_mask, drop=True)
        split_year = split_index[year_index == target_year]
        return da_year, split_year

    def make_weights(da_year, reverse=False):
        time_year = pd.DatetimeIndex(da_year['time'].values)
        n_time = da_year.sizes['time']
        values = np.arange(n_time - 1, -1, -1) if reverse else np.arange(1, n_time + 1)
        denom = n_time
        if rescale:
            month_step = np.diff(time_year.year*12 + time_year.month)
            if time_year.size > 1 and np.all(month_step == 1):
                lengths = np.asarray(time_year.days_in_month, dtype='float32')
            elif time_year.size > 1 and np.all(month_step == 3):
                lengths = np.asarray(
                    (time_year - pd.DateOffset(months=1)).days_in_month
                    + time_year.days_in_month
                    + (time_year + pd.DateOffset(months=1)).days_in_month,
                    dtype='float32',
                )
            values = np.cumsum(lengths[::-1])[::-1] - lengths if reverse else np.cumsum(lengths)
            denom = lengths.sum()
        return xr.DataArray(
            values.astype('float32') / denom,
            dims='time',
            coords={'time': da_year['time']},
        )

    def split_sum(da_year, split_year, key, weights=None):
        time_sel = np.where(split_year == key)[0]
        out = da_year.isel(time=time_sel)
        if weights is not None:
            w = weights.isel(time=time_sel)
            out = xr.where(w == 0, 0, out*w)
        return out.sum('time', skipna=False)

    out = {}
    for key in split_keys:
        year_values = []
        for year in target_years:
            da_y, split_y = select_year(year)

            if weighted:
                da_y1, split_y1 = select_year(year + 1)
                value = (
                    split_sum(da_y, split_y, key, make_weights(da_y))
                    + split_sum(da_y1, split_y1, key, make_weights(da_y1, reverse=True))
                )
            else:
                value = split_sum(da_y, split_y, key)

            year_values.append(value)

        out[key] = (
            xr.concat(year_values, dim='year')
            .assign_coords(year=target_years)
            .astype('float32')
        )

    return out


def average_daily_contributions(contrib, period):
    years_sel = range(period[0], period[1])
    out = []
    for key in sorted(contrib):
        da = contrib[key]
        if 'year' in da.dims:
            da = da.where(da['year'].isin(years_sel), drop=True).mean('year', skipna=True)
        out.append(da.expand_dims(month_day=[key]))
    return xr.concat(out, dim='month_day').sortby('month_day').astype('float32')


def average_loaded_daily_contrib(da, period):
    if 'time' in da.dims:
        return average_period(to_noleap_tendency(da), [period[0], period[1]-1])
    if 'year' in da.dims and 'month_day' in da.dims:
        years_sel = range(period[0], period[1]) if period[1] > period[0] else range(period[0], period[1] + 1)
        return da.where(da['year'].isin(years_sel), drop=True).mean('year', skipna=True).astype('float32')
    if 'month_day' in da.dims:
        return da.astype('float32')
    raise ValueError("Loaded contribution data must have 'time', or ('year' and 'month_day'), or 'month_day'")


loaded_tend_contrib = {} if loaded_tend_contrib is None else loaded_tend_contrib
loaded_t2m_tendtot_contrib = {} if loaded_t2m_tendtot_contrib is None else loaded_t2m_tendtot_contrib


def compute_daily_contrib_clim(source_dict):
    out = {}
    for key, da in source_dict.items():
        daily_contrib = tendency_contributions(da, split='Daily', weighted=True)
        out[key] = {
            period_key: average_daily_contributions(daily_contrib, period)
            for period_key, period in periods.items()
        }
    return out


def compute_clim_bound_contrib(bound_dict):
    return {
        key: {
            period_key: average_daily_contributions(
                tendency_contributions(
                    period_bounds_data['early'],
                    split='Daily',
                    weighted=True,
                    da_next=period_bounds_data['late'],
                    base_year=period[0],
                ),
                period,
            )
            for period_key, period in periods.items()
            for period_bounds_data in [bound_dict[key][period_key]]
        }
        for key in bound_dict
    }


if tend_contrib_source == 'raw_tend':
    clim_tend_contrib = compute_daily_contrib_clim(noleap_tend)
    clim_t2m_tendtot_contrib = compute_daily_contrib_clim(noleap_t2m_tendtot)
elif tend_contrib_source == 'smooth_tend':
    clim_tend_contrib = compute_daily_contrib_clim(smooth_tend)
    clim_t2m_tendtot_contrib = compute_daily_contrib_clim(smooth_t2m_tendtot)
elif tend_contrib_source == 'clim_bound':
    clim_tend_contrib = compute_clim_bound_contrib(clim_tend_bound)
    clim_t2m_tendtot_contrib = compute_clim_bound_contrib(clim_t2m_tendtot_bound)
elif tend_contrib_source == 'loaded':
    if not loaded_tend_contrib or not loaded_t2m_tendtot_contrib:
        raise ValueError("tend_contrib_source='loaded' requires loaded_tend_contrib and loaded_t2m_tendtot_contrib")
    clim_tend_contrib = {
        key: {
            period_key: average_loaded_daily_contrib(da, period)
            for period_key, period in periods.items()
        }
        for key, da in loaded_tend_contrib.items()
    }
    clim_t2m_tendtot_contrib = {
        key: {
            period_key: average_loaded_daily_contrib(da, period)
            for period_key, period in periods.items()
        }
        for key, da in loaded_t2m_tendtot_contrib.items()
    }
else:
    raise ValueError("tend_contrib_source must be one of 'raw_tend', 'smooth_tend', 'clim_bound', or 'loaded'")






#%% Plotting of climatological A

def x_axis_format(da):
    x = pd.to_datetime('2000-' + da['month_day'].values)
    ax = plt.gca()
    month_ticks = pd.date_range(x.min().replace(day=1), x.max().replace(day=1), freq='MS')
    ax.set_xticks(month_ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    return x

fig, ax = plt.subplots(figsize=(12, 6))

x_axis = x_axis_format(clim_templ1['P1'])
ax.plot(x_axis, clim_templ1['P1'], color='red')
ax.plot(x_axis, clim_templ1['P2'], color='blue')
ax.plot(x_axis, clim_templ1['P3'], color='green')


fig, ax = plt.subplots(figsize=(12, 6))

x_axis = x_axis_format(clim_templ1['P1'])
ax.plot(x_axis, clim_templ1['P2'] - clim_templ1['P1'], color='red')
ax.plot(x_axis, clim_templ1['P3'] - clim_templ1['P2'], color='blue')
ax.plot(x_axis, clim_templ1['P3'] - clim_templ1['P1'], color='green')



#%% ============================================================================
#   PHASE 4: PLOTTING
#   ============================================================================

print('='*60)
print('PHASE 4: Plotting')
print('='*60)



# fig, ax = plt.subplots(figsize=(12, 6))

# ax.plot(x_axis_format(clim_temp['RACMO2.4A']['P1']), clim_temp['RACMO2.4A']['P1'], label='RACMO2.4A P1', color='red')
# ax.plot(x_axis_format(clim_temp['RACMO2.4A']['P2']), clim_temp['RACMO2.4A']['P2'], label='RACMO2.4A P2', color='blue')
# ax.plot(x_axis_format(clim_temp['RACMO2.4A']['P3']), clim_temp['RACMO2.4A']['P3'], label='RACMO2.4A P3', color='green')

# month_ticks = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
# ax.set_xticks(month_ticks)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


# fig, ax = plt.subplots(figsize=(12, 6))

# ax.plot(x_axis_format(clim_temp['RACMO2.4A']['P1']), clim_temp['RACMO2.4A']['P2'] - clim_temp['RACMO2.4A']['P1'], label='RACMO2.4A P1', color='red')
# ax.plot(x_axis_format(clim_temp['RACMO2.4A']['P1']), clim_temp['RACMO2.4A']['P3'] - clim_temp['RACMO2.4A']['P2'], label='RACMO2.4A P1', color='blue')
# ax.plot(x_axis_format(clim_temp['RACMO2.4A']['P1']), clim_temp['RACMO2.4A']['P3'] - clim_temp['RACMO2.4A']['P1'], label='RACMO2.4A P1', color='green')

# month_ticks = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
# ax.set_xticks(month_ticks)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))



# fig, ax = plt.subplots(figsize=(12, 6))

# ax.plot(x_axis_format(clim_tendtot['RACMO2.4A']['P1']), clim_tendtot['RACMO2.4A']['P1'], label='RACMO2.4A P1', color='red')
# ax.plot(x_axis_format(clim_tendtot['RACMO2.4A']['P2']), clim_tendtot['RACMO2.4A']['P2'], label='RACMO2.4A P2', color='blue')
# ax.plot(x_axis_format(clim_tendtot['RACMO2.4A']['P3']), clim_tendtot['RACMO2.4A']['P3'], label='RACMO2.4A P3', color='green')

# month_ticks = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
# ax.set_xticks(month_ticks)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


# fig, ax = plt.subplots(figsize=(12, 6))

# ax.plot(x_axis_format(clim_tendtot['RACMO2.4A']['P1']), clim_tendtot['RACMO2.4A']['P2'] - clim_tendtot['RACMO2.4A']['P1'], label='RACMO2.4A P1', color='red')
# ax.plot(x_axis_format(clim_tendtot['RACMO2.4A']['P1']), clim_tendtot['RACMO2.4A']['P3'] - clim_tendtot['RACMO2.4A']['P2'], label='RACMO2.4A P1', color='blue')
# ax.plot(x_axis_format(clim_tendtot['RACMO2.4A']['P1']), clim_tendtot['RACMO2.4A']['P3'] - clim_tendtot['RACMO2.4A']['P1'], label='RACMO2.4A P1', color='green')
# month_ticks = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
# ax.set_xticks(month_ticks)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


#%%

def new_cumsum(da):
    da_new = da.shift(month_day=1, fill_value=0)
    return da_new.cumsum(dim='month_day')

plot_test = new_cumsum(clim_tend['tendtot']['P2']) + clim_templ1['P2'].isel(month_day=0)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_axis_format(clim_templ1['P1']), clim_templ1['P2'], color='red')
ax.plot(x_axis_format(clim_templ1['P1']), plot_test, color='blue', linestyle='--')

month_ticks = pd.date_range('2000-01-01', '2000-12-01', freq='MS')

ax.set_xticks(month_ticks)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))



#%%

def deoy(da):
    da_cumsum = new_cumsum(da)
    return da_cumsum.isel(month_day=-1) + da.isel(month_day=-1) 

recon_temp1 = new_cumsum(clim_tend['tendtot']['P2'])
recon_temp2 = new_cumsum(clim_tend['tendtot']['P3'])

plot_test = recon_temp2 - recon_temp1 + 10*deoy(clim_tend['tendtot']['P2']) + 10*deoy(clim_tend['tendtot']['P3'])
# plot_test = recon_temp2 - recon_temp1 + clim_t2m['RACMO2.4A']['P3'].isel(month_day=0) - clim_t2m['RACMO2.4A']['P2'].isel(month_day=0)


fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(x_axis_format(clim_templ1['P1']), clim_templ1['P3'] - clim_templ1['P2'], color='red')
ax.plot(x_axis_format(clim_tend['tendtot']['P1']), recon_temp2 - recon_temp1, color='blue')
ax.plot(x_axis_format(clim_tend['tendtot']['P1']), plot_test, color='green')

month_ticks = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
ax.set_xticks(month_ticks)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


#%%

recon_temp1 = new_cumsum(clim_tend['tendtot']['P1'])
recon_temp2 = new_cumsum(clim_tend['tendtot']['P2'])

plot_test = recon_temp2 - recon_temp1 + 10*(recon_temp2 - recon_temp1).cumsum(dim='month_day').mean(dim='month_day')/365
# plot_test = recon_temp2 - recon_temp1 + clim_t2m['RACMO2.4A']['P3'].isel(month_day=0) - clim_t2m['RACMO2.4A']['P2'].isel(month_day=0)


fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(x_axis_format(clim_templ1['P1']), clim_templ1['P2'] - clim_templ1['P1'], color='red')
ax.plot(x_axis_format(clim_tend['tendtot']['P1']), recon_temp2 - recon_temp1, color='blue')
ax.plot(x_axis_format(clim_tend['tendtot']['P1']), plot_test, color='green')

month_ticks = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
ax.set_xticks(month_ticks)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))



#%%

# Count how many leap years in period and make sure 29th of february has weight leap years / years ?
# Is tendency at 31 dec the same as 1 jan - 31 dec? Nee... sluit niet. OPLETTEN!!!


# Niet deoy doen maar gebruik ook de basistrend, tel die er ook bij op!


# (noleap_templ1.sel(time=slice('2001-01-01', '2021-01-01')).values - noleap_templ1.sel(time=slice('1961-01-01', '1981-01-01')).values).mean()


# Check if leap years are in contribution files, if so, merge with 28th of feb