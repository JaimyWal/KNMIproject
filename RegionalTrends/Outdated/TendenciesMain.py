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

from RegionalTrends.Helpers import ComputeTendencies
reload(ComputeTendencies)
from RegionalTrends.Helpers.ComputeTendencies import construct_tendency, full_interval_time

import RegionalTrends.Helpers.Config.Constants as Constants
reload(Constants)

plt.rcParams['axes.unicode_minus'] = False
dask.config.set(scheduler='threads', num_workers=12)

#%% User inputs

# Main arguments
var_tend_main = {
    'Total Tendency': 'tendtot',
    'Total Tendency2': 'tendtot', 
    # 'Total Radiative Tendency': 'radtot',
    # 'Shortwave Radiative Tendency': 'swnet',
    # 'Longwave Radiative Tendency': 'lwnet',
    # 'Total Dynamical Tendency': 'dyntot',
}
var_tend_sums = {
    'Total Tendency': ['dyntot', 'radtot', 'senstot', 'phasetot', 'frictot', 'numtot'],
    'Total Tendency2': ['dyntot', 'diabatic', 'numtot'], 
    # 'Total Radiative Tendency': ['swnet', 'lwnet'],
    # 'Shortwave Radiative Tendency': ['swtop', 'swbot'],
    # 'Longwave Radiative Tendency': ['lwtop', 'lwbot'],
    # 'Total Dynamical Tendency': ['horadv', 'vertadv', 'adicomp', 'orography'],
}
var_tend_close_pr = ['dyntot', 'radtot', 'senstot', 'phasetot', 'frictot', 'numtot']
var_tend_close_tot = 'tendtot'
var_temp = 'templ1'
file_freq = 'Seasonal'
proc_freq = 'Seasonal'
tend_trans = False
save_name_base = None

# Data selection arguments
years = np.arange(1961, 2023 + 1)
months_dict = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
# months_dict = {'Jan': [1], 'Feb': [2], 'Mar': [3], 'Apr': [4], 'May': [5], 'Jun': [6], 'Jul': [7], 'Aug': [8], 'Sep': [9], 'Oct': [10], 'Nov': [11], 'Dec': [12]}
# months_dict = None
lats = [50.7, 53.6]
lons = [3.25, 7.35]
proj_sel = 'RACMO2.4'
land_only = True
trim_border = None

# Trend arguments
trend_box = [-9, 9]
trend_method = 'MeanDiff'
trend_bandwidth = 5.0
trend_ref = [1970, 1970]  # None or [year0, year1]

# Raw group time series plot arguments
groupload_plotb = True
groupload_layout = (2, 2)
groupload_panel_width = 6
groupload_panel_height = 4
groupload_y_range = None
groupload_shared_labels = True
groupload_mirror_y_axes = True

# Yearly contribution plot arguments
cont_plotb = True
cont_layout = (2, 2)
cont_panel_width = 6
cont_panel_height = 4
cont_y_range = None
cont_shared_labels = True
cont_mirror_y_axes = True

# Trend contribution plot arguments
conttrend_plotb = True
conttrend_layout = (2, 2)
conttrend_panel_width = 6
conttrend_panel_height = 4
conttrend_y_range = None
conttrend_shared_labels = True
conttrend_mirror_y_axes = True

# Integrated trend contribution plot arguments
conttrendint_plotb = True
conttrendint_layout = (2, 2)
conttrendint_panel_width = 6
conttrendint_panel_height = 4
conttrendint_y_range = None
conttrendint_shared_labels = True
conttrendint_mirror_y_axes = True

# Closure time series of plot arguments
closeload_plotb = True
closeload_layout = (2, 2)
closeload_panel_width = 6
closeload_panel_height = 4
closeload_y_range = None
closeload_shared_labels = True
closeload_mirror_y_axes = True

# Yearly closure plot arguments
closeyear_plotb = True
closeyear_panel_width = 6
closeyear_panel_height = 4
closeyear_y_range = None
closeyear_shared_labels = True

# Trend closure plot arguments
closetrend_plotb = True
closetrend_panel_width = 6
closetrend_panel_height = 4
closetrend_y_range = None
closetrend_shared_labels = True

#%% Dataset configurations

data_sources = Constants.DATA_SOURCES
station_sources = Constants.STATION_SOURCES
station_coord_cfg = Constants.STATION_COORD_CFG
enbud_vars = Constants.ENBUD_VARS
tendency_vars = Constants.TENDENCY_VARS
var_file_cfg = Constants.VAR_FILE_CFG
var_name_cfg = Constants.VAR_NAME_CFG
var_units_cfg = Constants.VAR_UNIT_CFG
var_symbol_cfg = Constants.VAR_SYMBOL_CFG
proj_cfg = Constants.PROJ_CFG

proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())

#%% ============================================================================
#   PHASE 1: LOAD ALL DATA ONCE
#   ============================================================================

print('='*60)
print('PHASE 1: Loading all data')
print('='*60)

var_tend_main_vars = list(var_tend_main.values())
var_tend_sum_vars = [var for vars_group in var_tend_sums.values() for var in vars_group]
vars_all = set(var_tend_main_vars + var_tend_sum_vars + var_tend_close_pr + [var_tend_close_tot] +
               ([var_temp] if var_temp is not None else []))
vars_tendencies = set(var_tend_main_vars + var_tend_sum_vars + var_tend_close_pr + [var_tend_close_tot])

if file_freq in ['Monthly', 'Seasonal']:
    build_file_cfg_base = ProcessVar.build_file_cfg

    def build_file_cfg_select(freq_str):
        file_cfg = build_file_cfg_base(freq_str)

        if freq_str in ['Monthly', 'Seasonal']:
            suffix = {
                'Monthly': 'monthlyA' if tend_trans else 'monthlycont',
                'Seasonal': 'seasonalA' if tend_trans else 'seasonalcont',
            }[freq_str]

            for var in tendency_vars:
                if var in file_cfg['RACMO2.4A']:
                    freq_dir = Path(file_cfg['RACMO2.4A'][var]).parent
                    file_cfg['RACMO2.4A'][var] = str(freq_dir / f'{var}_{suffix}_*.nc')

        return file_cfg

    ProcessVar.build_file_cfg = build_file_cfg_select


def months_span_year_boundary(months):
    if months is None or len(months) == 0:
        return False
    months_arr = np.asarray(months, dtype=int)
    return int(months_arr[0]) > int(months_arr[-1])

def needs_extended_years(months_dict=None):
    if months_dict:
        for months in months_dict.values():
            if months_span_year_boundary(months):
                return True
    return False


# Extend years if needed for DJF-style seasons
years_load = list(years)
years_load.append(years_load[-1] + 1) 
if (needs_extended_years(months_dict=months_dict) or proc_freq == 'Seasonal') and file_freq != 'Seasonal':
    extra_years = {y - 1 for y in years} - set(years)
    years_load = sorted(set(years_load) | extra_years)

# Load all var data
var_data = {}
weights = None

for var in vars_all:
    print(f'  Loading: {var}')
    
    # Load raw data
    data_raw = load_var(
        var=var,
        data_source='RACMO2.4A',
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
    
    if weights is None:
        weights = area_weights(data_raw.isel(time=0), 
                                rotpole_native=proj_cfg.get('RACMO2.4A', ccrs.PlateCarree()))
    
    var_data[var] = data_raw.astype('float32')

print(f'Loaded {len(var_data)} datasets\n')

#%% ============================================================================
#   PHASE 2: DOMAIN AVERAGING OF RAW DATA
#   ============================================================================

print('='*60)
print('PHASE 2: Domain averaging of raw data')
print('='*60)

var_weighted = {}

for var in vars_all:
    print(f'  Area-weighting: {var}')
    var_weighted[var] = area_weighted_mean(var_data[var], weights=weights).compute().astype('float32')
    del var_data[var]

var_data = None

print(f'Area-weighted {len(var_weighted)} datasets\n')

#%% ============================================================================
#   PHASE 3: PROCESSING OF LOADED TENDENCIES AND TEMPERATURE
#   ============================================================================

print('='*60)
print('PHASE 3: Processing of loaded tendencies and temperature')
print('='*60)

var_tendencies = {}

for var in vars_tendencies:
    print(f'  Computing tendency: {var}')

    if file_freq == proc_freq:
        var_tendencies[var] = var_weighted[var]
    else:
        var_tendencies[var] = construct_tendency(
            var_weighted[var],
            interval=proc_freq        
        )

var_temperature = None
var_temp_close = None
if var_temp is not None:
    temp_da = var_weighted[var_temp]

    if file_freq == proc_freq:
        var_temperature = temp_da.astype('float32')
    elif proc_freq == 'Raw':
        var_temperature = temp_da.astype('float32')
    elif proc_freq == 'Daily':
        var_temperature = temp_da.resample(time='1D').mean(skipna=False).reindex(
            time=full_interval_time(temp_da['time'], interval='Daily')
        ).astype('float32')
    elif proc_freq == 'Monthly':
        var_temperature = temp_da.resample(time='MS').mean(skipna=False).reindex(
            time=full_interval_time(temp_da['time'], interval='Monthly')
        ).astype('float32')
    elif proc_freq == 'Seasonal':
        if file_freq == 'Monthly':
            monthly_temperature = temp_da
        else:
            monthly_temperature = temp_da.resample(time='MS').mean(skipna=False).reindex(
                time=full_interval_time(temp_da['time'], interval='Monthly')
            ).astype('float32')
        years_temp = np.append(years, years[-1] + 1)
        month_weights = monthly_temperature['time'].dt.days_in_month.astype('float32')
        weighted_sum = (monthly_temperature*month_weights).resample(time='QS-DEC').sum()
        weight_sum = month_weights.where(monthly_temperature.notnull()).resample(time='QS-DEC').sum()
        var_temperature = weighted_sum / weight_sum

        seasonal_time = pd.DatetimeIndex(var_temperature['time'].values) + pd.DateOffset(months=1)
        var_temperature = var_temperature.assign_coords(
            time=('time', seasonal_time),
            season_year=('time', seasonal_time.year.astype(int)),
        )
        var_temperature = var_temperature.where(
            var_temperature['season_year'].isin(years_temp),
            drop=True
        ).astype('float32')
    
    if tend_trans:
        var_temp_close = var_temperature.shift(time=-1) - var_temperature

#%% Yearly tendencies

temp_year = None
delta_temp_year = None

if var_temperature is not None and proc_freq in ['Monthly', 'Seasonal']:

    temp_time = pd.DatetimeIndex(var_temperature['time'].values)

    if proc_freq == 'Monthly':
        year_labels = temp_time.year.astype(int)
        year_weights = temp_time.days_in_month.astype('float32')

    else:
        season_period = temp_time.to_period('Q-NOV')
        year_labels = season_period.qyear.astype(int)
        year_weights = (
            (season_period.end_time.normalize() - season_period.start_time) / np.timedelta64(1, 'D') + 1
        ).astype('float32')

    year_coord = xr.DataArray(
        year_labels,
        dims='time',
        coords={'time': var_temperature['time']},
        name='year',
    )
    year_weights = xr.DataArray(
        year_weights,
        dims='time',
        coords={'time': var_temperature['time']},
    )

    temp_year = (
        (var_temperature*year_weights).groupby(year_coord).sum(skipna=False)
        / year_weights.where(var_temperature.notnull()).groupby(year_coord).sum(skipna=False)
    ).astype('float32')

    available_years = np.asarray(temp_year['year'].values, dtype=int)
    delta_years = np.array([y for y in years if y in available_years and y + 1 in available_years], dtype=int)

    delta_temp_year = (temp_year.shift(year=-1) - temp_year).sel(year=delta_years).astype('float32')


var_tend_cont = {}

def tendency_contributions(da, split='Seasonal', weighted=False, rescale=False, start_month=None):

    time_index = pd.DatetimeIndex(da['time'].values)
    month_index = time_index.month.astype(int)

    if split == 'Monthly':
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


for var in var_tendencies:
    var_tend_cont[var] = tendency_contributions(
        var_tendencies[var],
        split=proc_freq,
        weighted=tend_trans,
        rescale=tend_trans,
    )

#%% Final processing

def select_years_da(da, years_sel):
    if da is None:
        return None
    if 'year' in da.coords:
        return da.where(da['year'].isin(years_sel), drop=True)
    if 'season_year' in da.coords:
        return da.where(da['season_year'].isin(years_sel), drop=True)
    if 'time' in da.coords:
        return da.where(da['time'].dt.year.isin(years_sel), drop=True)
    return da


extra_years = np.array(
    sorted(set(years) | ({y + 1 for y in years} - set(years))),
    dtype=int,
)

var_temp_close = select_years_da(var_temp_close, years)
var_temperature = select_years_da(var_temperature, extra_years)
temp_year = select_years_da(temp_year, extra_years)
delta_temp_year = select_years_da(delta_temp_year, years)

var_tendencies = {
    var: select_years_da(da, years)
    for var, da in var_tendencies.items()
}
var_tend_cont = {
    var: {
        key: select_years_da(da, years)
        for key, da in cont.items()
    }
    for var, cont in var_tend_cont.items()
}


#%% Checking closures

def sum_dataarrays(arrays):
    arrays = [da for da in arrays if da is not None]
    if not arrays:
        return None
    out = arrays[0].copy()
    for da in arrays[1:]:
        out = out + da
    return out

def sum_tend_cont(cont):
    return sum_dataarrays(list(cont.values()))

var_temp_close_tot = None
var_temp_close_pr = None
if tend_trans:
    var_temp_close_tot = var_tendencies.get(var_tend_close_tot)
    var_temp_close_pr = sum_dataarrays(
        [var_tendencies.get(var) for var in var_tend_close_pr]
    )

delta_temp_year_tot = None
delta_temp_year_pr = None
if var_tend_close_tot in var_tend_cont:
    delta_temp_year_tot = sum_tend_cont(var_tend_cont[var_tend_close_tot])
delta_temp_year_pr = sum_dataarrays(
    [sum_tend_cont(var_tend_cont[var]) for var in var_tend_close_pr]
)

#%% ============================================================================
#   PHASE 4: LOCAL TRENDING AND ATTRIBUTION
#   ============================================================================

print('='*60)
print('PHASE 4: LOCAL TRENDING AND ATTRIBUTION')
print('='*60)

def build_trend_weights(method='OLS', trend_box=None, bandwidth=None):
        
    k = np.arange(trend_box[0], trend_box[1] + 1, dtype=float)

    if method == 'LocalLinearTricubeSlope':
        h = np.max(np.abs(k))
        if h <= 0:
            raise ValueError('trend_box must span at least one year on each side.')

        u = np.abs(k) / h
        w = np.where(u < 1.0, (1.0 - u**3)**3, 0.0)

        X = np.column_stack([
            np.ones_like(k),
            k,
        ])

        XtWX = (X.T * w) @ X
        XtW = X.T * w
        lam = (np.linalg.pinv(XtWX) @ XtW)[1, :]

        # enforce exact zero response to constants
        lam = lam - lam.sum() / lam.size

        return lam.astype('float32')

    if method == 'LocalQuadGaussSlope':
        w = np.exp(-0.5 * (k / bandwidth)**2)
        X = np.column_stack([np.ones_like(k), k, k**2])
        lam = (np.linalg.pinv((X.T * w) @ X) @ (X.T * w))[1, :]
        lam = lam - lam.sum() / lam.size
        return lam.astype('float32')

    if method == 'OLS':
        kc = k - k.mean()
        return (kc / np.sum(kc**2)).astype('float32')

    if method == 'MeanDiff':
        n = len(k)
        return (np.sign(np.arange(n) - (n - 1) / 2) / (n // 2)).astype('float32')

    raise ValueError(f'Unknown trend method: {method}')


def local_trend(da, trend_weights, trend_box):

    years = np.asarray(da['year'].values, dtype=int)

    trend_years = years[
        (years + trend_box[0] >= years[0])
        & (years + trend_box[1] <= years[-1])
    ]
    weight_da = xr.DataArray(
        np.asarray(trend_weights, dtype='float32'),
        dims='window',
    )
    trend = (
        da
        .rolling(year=len(trend_weights), min_periods=len(trend_weights))
        .construct('window')
        * weight_da
    ).sum('window', skipna=False)
    trend = trend.assign_coords(year=years - trend_box[1])

    return trend.sel(year=trend_years).assign_coords(
        time=('year', pd.to_datetime(dict(year=trend_years, month=1, day=1))),
    ).astype('float32')

def trend_reference_constant(da, trend_ref):
    if da is None or trend_ref is None:
        return None
    ref = da.sel(year=slice(*trend_ref)).mean('year', skipna=False)
    if np.asarray(ref.values).size == 1:
        return float(np.asarray(ref.values))
    return ref

def trend_reference(da, trend_ref):
    if da is None or trend_ref is None:
        return da
    return (da - da.sel(year=slice(*trend_ref)).mean('year', skipna=False)).astype('float32')


trend_weights = build_trend_weights(
    method=trend_method,
    trend_box=trend_box,
    bandwidth=trend_bandwidth if trend_method == 'LocalQuadGaussSlope' else None,
)

temp_trend = None
if temp_year is not None:
    temp_trend = local_trend(
        temp_year,
        trend_weights,
        trend_box,
    )
temp_trend_ref_const = trend_reference_constant(temp_trend, trend_ref)
if temp_trend_ref_const is not None:
    print(f'  baseline constant for temp_trend over {trend_ref}: {temp_trend_ref_const}')
temp_trend = trend_reference(temp_trend, trend_ref)

temp_trend_close = None
if temp_trend is not None:
    temp_trend_close = temp_trend.shift(year=-1) - temp_trend

delta_temp_year_trend = None
if delta_temp_year is not None:
    delta_temp_year_trend = local_trend(
        delta_temp_year,
        trend_weights,
        trend_box,
    )
if temp_trend_close is not None and delta_temp_year_trend is not None:
    temp_trend_close = temp_trend_close.sel(year=delta_temp_year_trend['year'])

var_tend_cont_trend = {
    var: {
        key: local_trend(da, trend_weights, trend_box)
        for key, da in cont.items()
    }
    for var, cont in var_tend_cont.items()
}

#%% Integration of tendency trends

def integrate_trend_acceleration(da, trend_ref=None, anchor_value=0.0, label=None):
    if da is None:
        return None

    years = np.asarray(da['year'].values, dtype=int)
    first = xr.full_like(da.isel(year=0), anchor_value).expand_dims(year=[years[0]])
    rest = da.cumsum('year').assign_coords(year=('year', years + 1))
    years_new = np.concatenate(([years[0]], years + 1))
    out = xr.concat([first, rest], dim='year').assign_coords(
        time=('year', pd.to_datetime(dict(year=years_new, month=1, day=1))),
    )
    ref_const = trend_reference_constant(out, trend_ref)
    if label is not None and ref_const is not None:
        print(f'  baseline constant for {label} over {trend_ref}: {ref_const}')
    return trend_reference(out, trend_ref)

delta_temp_year_trend_int = None
delta_temp_year_trend_int = integrate_trend_acceleration(
    delta_temp_year_trend,
    trend_ref=trend_ref,
    label='delta_temp_year_trend_int',
)

var_tend_cont_trend_int = {
    var: {
        key: integrate_trend_acceleration(da, trend_ref=trend_ref)
        for key, da in cont.items()
    }
    for var, cont in var_tend_cont_trend.items()
}

#%% Plotting

PANEL_KEYS = list(months_dict.keys()) if months_dict is not None else ['All']
PANEL_MONTHS = months_dict if months_dict is not None else {'All': None}
COLORS = ['#000000', '#DB2525', '#0168DE', '#00A236', '#CA721B', '#7B2CBF', '#E91E8C', '#808080']

def format_panel_title(panel_key):
    if not tend_trans or months_dict is None or panel_key == 'All':
        return panel_key

    next_key = PANEL_KEYS[(PANEL_KEYS.index(panel_key) + 1) % len(PANEL_KEYS)]
    return f'{panel_key} $\\rightarrow$ {next_key}'

def plot_time_values(data):
    if data is None:
        return None
    if 'time' in data.coords:
        return data['time'].values
    if 'year' in data.coords:
        years_plot = np.asarray(data['year'].values, dtype=int)
        return pd.to_datetime(dict(year=years_plot, month=1, day=1))
    return np.arange(data.size)

def normalize_axes(axes, n_rows, n_cols):
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    if n_rows == 1:
        return axes.reshape(1, -1)
    if n_cols == 1:
        return axes.reshape(-1, 1)
    return axes

def select_panel_time_data(da, months):
    if da is None or months is None:
        return da
    return da.where(da['time'].dt.month.isin(months), drop=True)

def select_panel_cont_data(cont, months):
    if cont is None:
        return None
    if months is None:
        return sum_dataarrays(list(cont.values()))

    season_months = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
    keys = []
    for key in cont:
        if isinstance(key, str) and any(month in months for month in season_months[key]):
            keys.append(key)
        if isinstance(key, (int, np.integer)) and key in months:
            keys.append(key)
    return sum_dataarrays([cont[key] for key in keys])

def build_group_panel_data(group_name, tendency_data):
    main_var = var_tend_main[group_name]
    return {
        'main': tendency_data.get(main_var),
        'components': {
            var: tendency_data.get(var)
            for var in var_tend_sums.get(group_name, [])
        },
    }

def build_closure_panel_data(tendency_data, temp_close):
    summed_processes = sum_dataarrays([tendency_data.get(var) for var in var_tend_close_pr])
    true_tendency = tendency_data.get(var_tend_close_tot)
    return {
        'summed_processes': summed_processes,
        'true_tendency': true_tendency,
        'temp_close': temp_close,
        'diff_true': None if temp_close is None or true_tendency is None else temp_close - true_tendency,
        'diff_sum': None if temp_close is None or summed_processes is None else temp_close - summed_processes,
    }

def group_plot_series(group_name, panel_data):
    series = [(group_name, panel_data['main'], COLORS[0], '-')]
    for ii, var in enumerate(var_tend_sums.get(group_name, [])):
        series.append((var_name_cfg.get(var, var), panel_data['components'].get(var), COLORS[(ii + 1) % len(COLORS)], '-'))
    return series

def closure_plot_series(panel_data):
    return [
        ('Summed processes', panel_data['summed_processes'], 'tab:blue', '-'),
        (var_name_cfg.get(var_tend_close_tot, var_tend_close_tot), panel_data['true_tendency'], 'tab:orange', '-'),
        (r'$\Delta T$', panel_data['temp_close'], 'tab:green', '-'),
    ]

def closure_difference_plot_series(panel_data):
    return [
        (r'$\Delta T$ - tendtot', panel_data['diff_true'], 'tab:red', '-'),
        (r'$\Delta T$ - summed processes', panel_data['diff_sum'], 'tab:purple', '-'),
    ]

def plot_series_list(ax, panel_series):
    for label, data, color, linestyle in panel_series:
        if data is None or data.size == 0:
            continue
        ax.plot(
            plot_time_values(data),
            data.values,
            c=color,
            lw=2.5,
            marker='o',
            ms=4,
            ls=linestyle,
            label=label,
        )

def format_time_axis(ax, title, title_fs, tick_fs, zero_line=False):
    if zero_line:
        ax.axhline(0.0, color='k', lw=1.0, alpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=title_fs, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_fs)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

def style_legend(ax, legend_fs):
    leg = ax.legend(fontsize=legend_fs, handlelength=1.5, loc='best')
    leg.set_zorder(20)
    for line in leg.get_lines():
        line.set_linewidth(3.0)

def add_shared_axis_labels(fig, axes, x_label, y_label, fontsize, pad=0.01):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    fig_width, fig_height = fig.get_size_inches() * fig.dpi
    n_cols = axes.shape[1]

    min_x_fig = 1.0
    for row in range(axes.shape[0]):
        ax = axes[row, 0]
        for label in ax.get_yticklabels():
            bbox = label.get_window_extent(renderer=renderer)
            min_x_fig = min(min_x_fig, bbox.x0 / fig_width)

    min_y_fig = 1.0
    for col in range(n_cols):
        ax = axes[-1, col]
        for label in ax.get_xticklabels():
            bbox = label.get_window_extent(renderer=renderer)
            min_y_fig = min(min_y_fig, bbox.y0 / fig_height)

    x_center = 0.5 * (axes[-1, 0].get_position().x0 + axes[-1, -1].get_position().x1)
    y_bottom = min_y_fig - pad - 0.01
    fig.text(x_center, y_bottom, x_label, ha='center', va='top', fontsize=fontsize*1.2)

    y_center = 0.5 * (axes[0, 0].get_position().y1 + axes[-1, 0].get_position().y0)
    x_left = min_x_fig - pad
    fig.text(x_left, y_center, y_label, ha='right', va='center', rotation=90, fontsize=fontsize)

def save_figure(save_var, suffix):
    if save_name_base is None:
        return

    save_name = f'{save_name_base}_{save_var}_{suffix}'
    pdf_dir = Path(f'/nobackup/users/walj/Figures_tendencies/pdf/{save_var}')
    jpg_dir = Path(f'/nobackup/users/walj/Figures_tendencies/jpg/{save_var}')
    pdf_dir.mkdir(parents=True, exist_ok=True)
    jpg_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(pdf_dir / f'{save_name}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(str(jpg_dir / f'{save_name}.jpg'), format='jpg', dpi=300, bbox_inches='tight')

def plot_panel_grid(panel_series_dict, layout, panel_width, panel_height, y_range, shared_labels,
                    mirror_y_axes, y_label, save_var, suffix, zero_line=False):
    n_rows, n_cols = layout
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        sharex=True,
        sharey=False,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.95, wspace=0.06 if mirror_y_axes else 0.16, hspace=0.18)
    axes = normalize_axes(axes, n_rows, n_cols)

    title_fs = max(18, int(panel_height * 5))
    tick_fs = max(12, int(panel_height * 3))
    legend_fs = max(10, int(panel_height * 2.5))

    for idx, panel_key in enumerate(PANEL_KEYS):
        ax = axes[idx // n_cols, idx % n_cols]
        plot_series_list(ax, panel_series_dict[panel_key])
        format_time_axis(ax, format_panel_title(panel_key), title_fs, tick_fs, zero_line=zero_line)

        if mirror_y_axes and (idx % n_cols == n_cols - 1) and n_cols > 1:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        panel_y_range = y_range.get(panel_key) if isinstance(y_range, dict) else y_range
        if panel_y_range is not None:
            ax.set_ylim(*panel_y_range)

        if ax.lines:
            style_legend(ax, legend_fs)

    for idx in range(len(PANEL_KEYS), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    if shared_labels:
        add_shared_axis_labels(fig, axes, 'Time', y_label, fontsize=max(20, int(panel_height * 4)))

    save_figure(save_var, suffix)
    plt.show()

def plot_single_panel(panel_series, panel_width, panel_height, y_range, shared_labels, y_label, save_var, suffix, title,
                      zero_line=False):
    fig, ax = plt.subplots(1, 1, figsize=(panel_width * 1.4, panel_height * 1.1))

    title_fs = max(18, int(panel_height * 5))
    tick_fs = max(12, int(panel_height * 3))
    legend_fs = max(10, int(panel_height * 2.5))

    plot_series_list(ax, panel_series)
    format_time_axis(ax, title, title_fs, tick_fs, zero_line=zero_line)

    if y_range is not None:
        ax.set_ylim(*y_range)

    if shared_labels:
        ax.set_xlabel('Time', fontsize=max(18, int(panel_height * 4)))
        ax.set_ylabel(y_label, fontsize=max(18, int(panel_height * 4)))

    if ax.lines:
        style_legend(ax, legend_fs)

    save_figure(save_var, suffix)
    plt.show()

raw_panel_tendencies = {
    panel_key: {
        var: select_panel_time_data(da, months)
        for var, da in var_tendencies.items()
    }
    for panel_key, months in PANEL_MONTHS.items()
}
raw_panel_temp_close = {
    panel_key: select_panel_time_data(var_temp_close, months)
    for panel_key, months in PANEL_MONTHS.items()
}
cont_panel_tendencies = {
    panel_key: {
        var: select_panel_cont_data(cont, months)
        for var, cont in var_tend_cont.items()
    }
    for panel_key, months in PANEL_MONTHS.items()
}
conttrend_panel_tendencies = {
    panel_key: {
        var: select_panel_cont_data(cont, months)
        for var, cont in var_tend_cont_trend.items()
    }
    for panel_key, months in PANEL_MONTHS.items()
}
conttrendint_panel_tendencies = {
    panel_key: {
        var: select_panel_cont_data(cont, months)
        for var, cont in var_tend_cont_trend_int.items()
    }
    for panel_key, months in PANEL_MONTHS.items()
}
close_label = 'Closure'
close_diff_label = 'Closure Difference'
trend_label = 'Trend'
trend_diff_label = 'Trend Difference'

if tend_trans and groupload_plotb:
    for group_name in var_tend_main:
        group_label = group_name
        plot_panel_grid(
            {
                panel_key: group_plot_series(group_name, build_group_panel_data(group_name, raw_panel_tendencies[panel_key]))
                for panel_key in PANEL_KEYS
            },
            layout=groupload_layout,
            panel_width=groupload_panel_width,
            panel_height=groupload_panel_height,
            y_range=groupload_y_range,
            shared_labels=groupload_shared_labels,
            mirror_y_axes=groupload_mirror_y_axes,
            y_label=group_label,
            save_var=var_tend_main[group_name],
            suffix='raw_grouped',
        )

if cont_plotb:
    for group_name in var_tend_main:
        group_label = group_name
        plot_panel_grid(
            {
                panel_key: group_plot_series(group_name, build_group_panel_data(group_name, cont_panel_tendencies[panel_key]))
                for panel_key in PANEL_KEYS
            },
            layout=cont_layout,
            panel_width=cont_panel_width,
            panel_height=cont_panel_height,
            y_range=cont_y_range,
            shared_labels=cont_shared_labels,
            mirror_y_axes=cont_mirror_y_axes,
            y_label=group_label,
            save_var=var_tend_main[group_name],
            suffix='yearly_contributions',
        )

if conttrend_plotb:
    for group_name in var_tend_main:
        group_label = group_name
        plot_panel_grid(
            {
                panel_key: group_plot_series(group_name, build_group_panel_data(group_name, conttrend_panel_tendencies[panel_key]))
                for panel_key in PANEL_KEYS
            },
            layout=conttrend_layout,
            panel_width=conttrend_panel_width,
            panel_height=conttrend_panel_height,
            y_range=conttrend_y_range,
            shared_labels=conttrend_shared_labels,
            mirror_y_axes=conttrend_mirror_y_axes,
            y_label=group_label,
            save_var=var_tend_main[group_name],
            suffix='yearly_contribution_trends',
        )

if conttrendint_plotb:
    for group_name in var_tend_main:
        group_label = group_name
        plot_panel_grid(
            {
                panel_key: group_plot_series(group_name, build_group_panel_data(group_name, conttrendint_panel_tendencies[panel_key]))
                for panel_key in PANEL_KEYS
            },
            layout=conttrendint_layout,
            panel_width=conttrendint_panel_width,
            panel_height=conttrendint_panel_height,
            y_range=conttrendint_y_range,
            shared_labels=conttrendint_shared_labels,
            mirror_y_axes=conttrendint_mirror_y_axes,
            y_label=group_label,
            save_var=var_tend_main[group_name],
            suffix='yearly_contribution_trends_int',
        )

if var_temp_close is not None and closeload_plotb:
    plot_panel_grid(
        {
            panel_key: closure_plot_series(build_closure_panel_data(raw_panel_tendencies[panel_key], raw_panel_temp_close[panel_key]))
            for panel_key in PANEL_KEYS
        },
        layout=closeload_layout,
        panel_width=closeload_panel_width,
        panel_height=closeload_panel_height,
        y_range=closeload_y_range,
        shared_labels=closeload_shared_labels,
        mirror_y_axes=closeload_mirror_y_axes,
        y_label=close_label,
        save_var=var_tend_close_tot,
        suffix='closure_raw',
    )

if tend_trans and var_temp_close is not None and closeload_plotb:
    plot_panel_grid(
        {
            panel_key: closure_difference_plot_series(build_closure_panel_data(raw_panel_tendencies[panel_key], raw_panel_temp_close[panel_key]))
            for panel_key in PANEL_KEYS
        },
        layout=closeload_layout,
        panel_width=closeload_panel_width,
        panel_height=closeload_panel_height,
        y_range=closeload_y_range,
        shared_labels=closeload_shared_labels,
        mirror_y_axes=closeload_mirror_y_axes,
        y_label=close_diff_label,
        save_var=var_tend_close_tot,
        suffix='closure_raw_diff',
        zero_line=True,
    )

if closeyear_plotb and delta_temp_year is not None:
    plot_single_panel(
        [
            ('Summed processes', delta_temp_year_pr, 'tab:blue', '-'),
            (var_name_cfg.get(var_tend_close_tot, var_tend_close_tot), delta_temp_year_tot, 'tab:orange', '-'),
            (r'$\Delta T$', delta_temp_year, 'tab:green', '-'),
        ],
        panel_width=closeyear_panel_width,
        panel_height=closeyear_panel_height,
        y_range=closeyear_y_range,
        shared_labels=closeyear_shared_labels,
        y_label=close_label,
        save_var=var_tend_close_tot,
        suffix='closure_yearly',
        title='Yearly closure',
    )

if closeyear_plotb and delta_temp_year is not None:
    plot_single_panel(
        [
            (r'$\Delta T$ - tendtot', None if delta_temp_year_tot is None else delta_temp_year - delta_temp_year_tot, 'tab:red', '-'),
            (r'$\Delta T$ - summed processes', None if delta_temp_year_pr is None else delta_temp_year - delta_temp_year_pr, 'tab:purple', '-'),
        ],
        panel_width=closeyear_panel_width,
        panel_height=closeyear_panel_height,
        y_range=closeyear_y_range,
        shared_labels=closeyear_shared_labels,
        y_label=close_diff_label,
        save_var=var_tend_close_tot,
        suffix='closure_yearly_diff',
        title='Yearly closure difference',
        zero_line=True,
    )

if closetrend_plotb and temp_trend_close is not None:
    trend_sum = sum_dataarrays(
        [sum_tend_cont(var_tend_cont_trend[var]) for var in var_tend_close_pr if var in var_tend_cont_trend]
    )
    plot_single_panel(
        [
            ('Summed processes', trend_sum, 'tab:blue', '-'),
            (r'$\Omega$', delta_temp_year_trend, 'tab:orange', '-'),
            (r'$\Delta T_{trend}$', temp_trend_close, 'tab:green', '-'),
        ],
        panel_width=closetrend_panel_width,
        panel_height=closetrend_panel_height,
        y_range=closetrend_y_range,
        shared_labels=closetrend_shared_labels,
        y_label=trend_label,
        save_var=var_tend_close_tot,
        suffix='trend_accel_closure',
        title='Trend acceleration closure',
    )

if closetrend_plotb and temp_trend_close is not None:
    trend_sum = sum_dataarrays(
        [sum_tend_cont(var_tend_cont_trend[var]) for var in var_tend_close_pr if var in var_tend_cont_trend]
    )
    plot_single_panel(
        [
            (r'$\Delta T_{trend} - \Omega$', None if delta_temp_year_trend is None else temp_trend_close - delta_temp_year_trend, 'tab:red', '-'),
            (r'$\Delta T_{trend} - \Psi$', None if trend_sum is None else temp_trend_close - trend_sum, 'tab:purple', '-'),
        ],
        panel_width=closetrend_panel_width,
        panel_height=closetrend_panel_height,
        y_range=closetrend_y_range,
        shared_labels=closetrend_shared_labels,
        y_label=trend_diff_label,
        save_var=var_tend_close_tot,
        suffix='trend_accel_closure_diff',
        title='Trend acceleration closure difference',
        zero_line=True,
    )

if closetrend_plotb and temp_trend is not None:
    trend_sum = sum_dataarrays(
        [sum_tend_cont(var_tend_cont_trend_int[var]) for var in var_tend_close_pr if var in var_tend_cont_trend_int]
    )
    plot_single_panel(
        [
            ('Integrated summed processes', trend_sum, 'tab:blue', '-'),
            (r'$\int \Omega \, dy$', delta_temp_year_trend_int, 'tab:orange', '-'),
            ('Temperature trend', temp_trend, 'tab:green', '-'),
        ],
        panel_width=closetrend_panel_width,
        panel_height=closetrend_panel_height,
        y_range=closetrend_y_range,
        shared_labels=closetrend_shared_labels,
        y_label=trend_label,
        save_var=var_tend_close_tot,
        suffix='trend_closure',
        title='Trend closure',
    )

if closetrend_plotb and temp_trend is not None:
    trend_sum = sum_dataarrays(
        [sum_tend_cont(var_tend_cont_trend_int[var]) for var in var_tend_close_pr if var in var_tend_cont_trend_int]
    )
    plot_single_panel(
        [
            (r'$T_{trend} - \int \Omega \, dy$', None if delta_temp_year_trend_int is None else temp_trend - delta_temp_year_trend_int, 'tab:red', '-'),
            (r'$T_{trend} - \int \Psi \, dy$', None if trend_sum is None else temp_trend - trend_sum, 'tab:purple', '-'),
        ],
        panel_width=closetrend_panel_width,
        panel_height=closetrend_panel_height,
        y_range=closetrend_y_range,
        shared_labels=closetrend_shared_labels,
        y_label=trend_diff_label,
        save_var=var_tend_close_tot,
        suffix='trend_closure_diff',
        title='Trend closure difference',
        zero_line=True,
    )



#%%


# OTHER PLOTS:
# Bar plot of end - begin value of something
# Spatial plot of end - begin values
# Trend plot in time
# Option for end interval and begin interval


# Andere temperaturen er ook in krijgen
# Toch iets met seasonal transitions dan trouwens.
# Integrate again!
# Opsplitsen van between and within interval (alleen wanneer suffix is yearlyB) (twee nieuwe plots)

# Ook verkrijgen van ERA5 en station (station is wel wat meer moeite)
# Ook T2m van racmo24 zelf erbij pakken. En van oude racmo24
# Ook baseline trend printen

# Shapefiles!!
