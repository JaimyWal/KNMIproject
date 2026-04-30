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

# Main arguments
var_tend_main = {
    'Total Tendency': 'tendtot',
    'Total Radiative Tendency': 'radtot',
    'Shortwave Radiative Tendency': 'swnet',
    'Longwave Radiative Tendency': 'lwnet',
    'Total Dynamical Tendency': 'dyntot',
}
var_tend_sums = {
    'Total Tendency': ['dyntot', 'radtot', 'senstot', 'phasetot', 'frictot', 'numtot'],
    'Total Radiative Tendency': ['swnet', 'lwnet'],
    'Shortwave Radiative Tendency': ['swtopdn', 'swtopup', 'swbotdn', 'swbotup'],
    'Longwave Radiative Tendency': ['lwtopdn', 'lwtopup', 'lwbotdn', 'lwbotup'],
    'Total Dynamical Tendency': ['horadv', 'vertadv', 'adicomp', 'orography'],
}
var_tend_close_pr = ['dyntot', 'radtot', 'senstot', 'phasetot', 'frictot', 'numtot']
var_tend_close_true = 'tendtot'
var_temp = 'templ1'

# Daily/monthly/seasonal files are supported and much faster than raw.
# Use raw only when you explicitly want the native highest-frequency input.
file_freq = 'Monthly'
compare_file_freqs = ['Raw', 'Daily', 'Monthly', 'Seasonal']
run_frequency_comparison = False
primary_plot_metric = 'CenteredAnomalyFingerprint'
plot_tau_diagnostic = False
reference_center_year = None

save_name_base = None

# Data selection arguments
years = np.arange(1961, 2023 + 1)
months_dict = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
lats = [50.7, 53.6]
lons = [3.25, 7.35]
proj_sel = 'RACMO2.4'
land_only = True
trim_border = None

# Group time series plot arguments
group_layout = (2, 2)
group_panel_width = 6
group_panel_height = 4
group_y_range = None
group_shared_labels = True
group_mirror_y_axes = True
group_x_year = True

# Closure time series plot arguments
close_layout = (2, 2)
close_panel_width = 6
close_panel_height = 4
close_y_range = None
close_shared_labels = True
close_mirror_y_axes = True
close_x_year = True

# Frequency comparison plot arguments
comp_layout = (2, 2)
comp_panel_width = 6
comp_panel_height = 4
comp_y_range = None
comp_shared_labels = True
comp_mirror_y_axes = True
comp_x_year = True

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

# Setup projections
proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())

RUNNING_WINDOW = 11
RUNNING_HALF = RUNNING_WINDOW // 2
COLORS = ['#000000', '#DB2525', '#0168DE', '#00A236', '#CA721B', '#7B2CBF', '#E91E8C', '#808080']
PANEL_KEYS = list(months_dict.keys()) if months_dict is not None else ['All']
SEASON_FILE_MONTH = {
    (12, 1, 2): 1,
    (3, 4, 5): 4,
    (6, 7, 8): 7,
    (9, 10, 11): 10,
}
FREQ_COLORS = {
    'Raw': '#000000',
    'Daily': '#DB2525',
    'Monthly': '#0168DE',
    'Seasonal': '#00A236',
}
SUPPORTED_FILE_FREQS = {'Raw', 'Daily', 'Monthly', 'Seasonal'}
SUPPORTED_PLOT_METRICS = {'Tau', 'CenteredAnomaly', 'CenteredAnomalyFingerprint'}

#%% ============================================================================
#   HELPER FUNCTIONS
#   ============================================================================

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


def assign_clim_year_from_time(time_coord, months):
    months_arr = np.asarray(months, dtype=int)
    time_index = pd.DatetimeIndex(time_coord.values)
    year_vals = np.asarray(time_index.year, dtype=int)
    month_vals = np.asarray(time_index.month, dtype=int)

    if months_span_year_boundary(months_arr):
        return np.where(month_vals >= int(months_arr[0]), year_vals + 1, year_vals)

    return year_vals


def seasonal_anchor_month(months):
    month_key = tuple(int(month) for month in np.asarray(months, dtype=int))
    return SEASON_FILE_MONTH.get(month_key)


def prepare_years_load(target_years, months_dict=None, source_freq='Raw'):
    target_years = np.asarray(target_years, dtype=int)
    years_core = np.arange(target_years.min() - RUNNING_HALF,
                           target_years.max() + RUNNING_HALF + 2)
    years_load = list(years_core)

    if source_freq != 'Seasonal' and needs_extended_years(months_dict=months_dict):
        extra_years = {int(year) - 1 for year in years_core}
        years_load = sorted(set(years_load) | extra_years)

    return years_load


def build_target_time(year_vals):
    year_vals = np.asarray(year_vals, dtype=int)
    return pd.to_datetime(dict(year=year_vals, month=1, day=1))


def to_year_da(values, tau_years, name=None):
    tau_years = np.asarray(tau_years, dtype=int)
    data = xr.DataArray(
        np.asarray(values, dtype='float64'),
        dims=['time'],
        coords={'time': build_target_time(tau_years)},
        name=name,
    )
    return data.assign_coords(season_year=('time', tau_years))


def build_seasonal_weight_matrix(temp, months, source_freq='Raw'):
    if 'time' not in temp.dims:
        raise ValueError('Temperature input must contain a time dimension')

    temp = temp.sortby('time').astype('float64')
    time_index = pd.DatetimeIndex(temp['time'].values)
    temp_values = np.asarray(temp.values, dtype='float64')
    valid_temp = np.isfinite(temp_values)
    n_time = temp_values.size

    months_arr = np.asarray(months, dtype=int)
    month_vals = np.asarray(time_index.month, dtype=int)
    clim_year = assign_clim_year_from_time(temp['time'], months_arr)

    if source_freq == 'Seasonal':
        anchor_month = seasonal_anchor_month(months_arr)
        if anchor_month is None:
            raise ValueError('Seasonal source frequency only supports DJF/MAM/JJA/SON targets')
        in_target = month_vals == anchor_month
        sample_weights = np.ones(n_time, dtype='float64')
    elif source_freq == 'Monthly':
        in_target = np.isin(month_vals, months_arr)
        sample_weights = np.asarray(time_index.days_in_month, dtype='float64')
    else:
        in_target = np.isin(month_vals, months_arr)
        sample_weights = np.ones(n_time, dtype='float64')

    active = in_target & valid_temp
    season_years_all = np.unique(clim_year[active]).astype(int)
    season_years_all.sort()

    if season_years_all.size == 0:
        return np.empty((0, n_time), dtype='float64'), np.empty(0, dtype=int), np.empty(0, dtype='float64')

    rows = []
    valid_years = []
    for season_year in season_years_all:
        idx = np.where(active & (clim_year == season_year))[0]
        if idx.size == 0:
            continue

        weights = sample_weights[idx].astype('float64')
        total_weight = float(weights.sum())
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            continue

        row = np.zeros(n_time, dtype='float64')
        row[idx] = weights / total_weight
        rows.append(row)
        valid_years.append(int(season_year))

    if not rows:
        return np.empty((0, n_time), dtype='float64'), np.empty(0, dtype=int), np.empty(0, dtype='float64')

    weight_matrix = np.vstack(rows).astype('float64')
    season_years = np.asarray(valid_years, dtype=int)
    seasonal_means = weight_matrix @ np.nan_to_num(temp_values, nan=0.0)

    return weight_matrix, season_years, seasonal_means


def build_interval_influence_matrix(weight_matrix):
    if weight_matrix.size == 0:
        n_time = 0 if weight_matrix.ndim < 2 else max(weight_matrix.shape[1] - 1, 0)
        return np.empty((0, n_time), dtype='float64')

    reverse_cumsum = np.cumsum(weight_matrix[:, ::-1], axis=1)[:, ::-1]
    return reverse_cumsum[:, 1:].astype('float64')


def build_centered_running_weight_matrix(weight_matrix, season_years):
    season_years = np.asarray(season_years, dtype=int)

    if weight_matrix.shape[0] < RUNNING_WINDOW:
        return (
            np.empty((0, weight_matrix.shape[1]), dtype='float64'),
            np.empty(0, dtype=int),
        )

    beta_rows = []
    beta_years = []

    for center_idx in range(RUNNING_HALF, weight_matrix.shape[0] - RUNNING_HALF):
        window_years = season_years[center_idx - RUNNING_HALF:center_idx + RUNNING_HALF + 1]
        expected_years = np.arange(window_years[0], window_years[0] + RUNNING_WINDOW)
        if not np.array_equal(window_years, expected_years):
            continue

        beta_rows.append(weight_matrix[center_idx - RUNNING_HALF:center_idx + RUNNING_HALF + 1].mean(axis=0))
        beta_years.append(int(season_years[center_idx]))

    if not beta_rows:
        return (
            np.empty((0, weight_matrix.shape[1]), dtype='float64'),
            np.empty(0, dtype=int),
        )

    return np.vstack(beta_rows).astype('float64'), np.asarray(beta_years, dtype=int)


def build_tau_weight_matrix(weight_matrix, season_years):
    beta_matrix, beta_years = build_centered_running_weight_matrix(weight_matrix, season_years)

    if beta_matrix.shape[0] < 2:
        return (
            np.empty((0, weight_matrix.shape[1]), dtype='float64'),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
        )

    c_rows = []
    tau_years = []
    low_years = []
    high_years = []

    for idx in range(beta_matrix.shape[0] - 1):
        if beta_years[idx + 1] - beta_years[idx] != 1:
            continue

        tau_year = int(beta_years[idx])
        c_rows.append(beta_matrix[idx + 1] - beta_matrix[idx])
        tau_years.append(tau_year)
        low_years.append(tau_year - RUNNING_HALF)
        high_years.append(tau_year + RUNNING_HALF + 1)

    if not c_rows:
        return (
            np.empty((0, weight_matrix.shape[1]), dtype='float64'),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
        )

    return (
        np.vstack(c_rows).astype('float64'),
        np.asarray(tau_years, dtype=int),
        np.asarray(low_years, dtype=int),
        np.asarray(high_years, dtype=int),
    )


def filter_tau_to_target_years(da, target_years):
    if da is None:
        return None
    year_vals = da['season_year'].values.astype(int)
    mask = xr.DataArray(
        np.isin(year_vals, np.asarray(target_years, dtype=int)),
        dims=['time'],
        coords={'time': da['time']},
    )
    return da.where(mask, drop=True)


def filter_centered_to_target_years(da, target_years):
    if da is None:
        return None
    year_vals = da['season_year'].values.astype(int)
    mask = xr.DataArray(
        np.isin(year_vals, np.asarray(target_years, dtype=int)),
        dims=['time'],
        coords={'time': da['time']},
    )
    return da.where(mask, drop=True)


def sum_tendency_terms(data_dict, vars_to_sum):
    total = 0
    for var in vars_to_sum:
        total = total + data_dict[var]
    return total


def build_group_panel_data(group_name, tendency_data):
    main_var = var_tend_main[group_name]
    group_vars = var_tend_sums.get(group_name, [])

    return {
        'main': tendency_data.get(main_var),
        'components': {var: tendency_data.get(var) for var in group_vars},
    }


def build_closure_panel_data(tendency_data, temp_close):
    summed_processes = sum_tendency_terms(tendency_data, var_tend_close_pr)
    true_tendency = tendency_data.get(var_tend_close_true)

    diff_true = None if temp_close is None or true_tendency is None else temp_close - true_tendency
    diff_sum = None if temp_close is None or summed_processes is None else temp_close - summed_processes

    return {
        'summed_processes': summed_processes,
        'true_tendency': true_tendency,
        'temp_close': temp_close,
        'diff_true': diff_true,
        'diff_sum': diff_sum,
    }


def compute_increment_closure_stats(temp, total_increment):
    temp = temp.sortby('time').astype('float64')
    interval_time = temp['time'].values[:-1]
    total_interval = total_increment.sortby('time').reindex(time=interval_time).astype('float64')

    temp_diff = temp.diff('time').assign_coords(time=interval_time)
    residual = temp_diff - total_interval
    residual_values = np.asarray(residual.values, dtype='float64')

    return {
        'mean_abs': float(np.nanmean(np.abs(residual_values))),
        'max_abs': float(np.nanmax(np.abs(residual_values))),
    }


def compute_season_tau(temp, process_data, months, source_freq='Raw', target_years=None):
    temp = temp.sortby('time').astype('float64')
    weight_matrix, season_years, _ = build_seasonal_weight_matrix(
        temp,
        months,
        source_freq=source_freq,
    )
    return compute_tau_from_weight_matrix(
        temp,
        process_data,
        weight_matrix,
        season_years,
        target_years=target_years,
    )


def compute_tau_from_weight_matrix(temp, process_data, weight_matrix, season_years, target_years=None):
    tau_weight_matrix, tau_years, low_years, high_years = build_tau_weight_matrix(weight_matrix, season_years)
    influence_matrix = build_interval_influence_matrix(tau_weight_matrix)

    temp_values = np.asarray(temp.values, dtype='float64')
    tau_temp_values = tau_weight_matrix @ np.nan_to_num(temp_values, nan=0.0)
    tau_temp = to_year_da(tau_temp_values, tau_years, name='tau_temp')
    tau_temp = tau_temp.assign_coords(window_low=('time', low_years), window_high=('time', high_years))

    interval_time = temp['time'].values[:-1]
    season_process_values = {}
    for var, proc in process_data.items():
        proc_interval = proc.sortby('time').reindex(time=interval_time).astype('float64')
        proc_values = np.nan_to_num(np.asarray(proc_interval.values, dtype='float64'), nan=0.0)
        season_process_values[var] = influence_matrix @ proc_values

    tau_process = {}
    for var, season_vals in season_process_values.items():
        tau_da = to_year_da(season_vals, tau_years, name=var)
        tau_da = tau_da.assign_coords(window_low=('time', low_years), window_high=('time', high_years))
        tau_process[var] = tau_da

    if target_years is not None:
        tau_temp = filter_tau_to_target_years(tau_temp, target_years)
        tau_process = {var: filter_tau_to_target_years(data, target_years) for var, data in tau_process.items()}

    return tau_temp, tau_process


def compute_season_centered_anomaly(temp, process_data, months, source_freq='Raw',
                                    target_years=None, reference_year=None):
    temp = temp.sortby('time').astype('float64')
    weight_matrix, season_years, _ = build_seasonal_weight_matrix(
        temp,
        months,
        source_freq=source_freq,
    )
    return compute_centered_anomaly_from_weight_matrix(
        temp,
        process_data,
        weight_matrix,
        season_years,
        target_years=target_years,
        reference_year=reference_year,
    )


def compute_centered_anomaly_from_weight_matrix(temp, process_data, weight_matrix, season_years,
                                                target_years=None, reference_year=None):
    beta_matrix, beta_years = build_centered_running_weight_matrix(weight_matrix, season_years)
    if beta_matrix.shape[0] == 0:
        return None, {}

    if target_years is not None:
        keep_mask = np.isin(beta_years, np.asarray(target_years, dtype=int))
        beta_matrix = beta_matrix[keep_mask]
        beta_years = beta_years[keep_mask]

    if beta_matrix.shape[0] == 0:
        return None, {}

    if reference_year is not None and reference_year in beta_years:
        ref_year = int(reference_year)
    else:
        ref_year = int(beta_years[0])

    ref_idx = int(np.where(beta_years == ref_year)[0][0])
    anomaly_weight_matrix = beta_matrix - beta_matrix[ref_idx]
    influence_matrix = build_interval_influence_matrix(anomaly_weight_matrix)

    temp_values = np.asarray(temp.values, dtype='float64')
    centered_temp_values = anomaly_weight_matrix @ np.nan_to_num(temp_values, nan=0.0)
    centered_temp_anom = to_year_da(centered_temp_values, beta_years, name='ctr11_temp')
    centered_temp_anom = centered_temp_anom.assign_coords(
        window_low=('time', beta_years - RUNNING_HALF),
        window_high=('time', beta_years + RUNNING_HALF),
    )

    interval_time = temp['time'].values[:-1]
    centered_process_anom = {}
    for var, proc in process_data.items():
        proc_interval = proc.sortby('time').reindex(time=interval_time).astype('float64')
        proc_values = np.nan_to_num(np.asarray(proc_interval.values, dtype='float64'), nan=0.0)
        centered_vals = influence_matrix @ proc_values
        centered_da = to_year_da(centered_vals, beta_years, name=var)
        centered_da = centered_da.assign_coords(
            window_low=('time', beta_years - RUNNING_HALF),
            window_high=('time', beta_years + RUNNING_HALF),
        )
        centered_process_anom[var] = centered_da

    centered_temp_anom.attrs['reference_year'] = ref_year
    for data in centered_process_anom.values():
        data.attrs['reference_year'] = ref_year

    return centered_temp_anom, centered_process_anom


def compute_season_outputs(temp, process_data, months, source_freq='Raw', target_years=None,
                           include_tau=False, include_centered_anomaly=False, reference_year=None):
    temp = temp.sortby('time').astype('float64')
    weight_matrix, season_years, _ = build_seasonal_weight_matrix(
        temp,
        months,
        source_freq=source_freq,
    )

    tau_temp = None
    tau_process = {}
    if include_tau:
        tau_temp, tau_process = compute_tau_from_weight_matrix(
            temp,
            process_data,
            weight_matrix,
            season_years,
            target_years=target_years,
        )

    centered_temp = None
    centered_process = {}
    if include_centered_anomaly:
        centered_temp, centered_process = compute_centered_anomaly_from_weight_matrix(
            temp,
            process_data,
            weight_matrix,
            season_years,
            target_years=target_years,
            reference_year=reference_year,
        )

    return tau_temp, tau_process, centered_temp, centered_process


def compute_cross_season_mean(data_by_season):
    valid_items = [(season, data) for season, data in data_by_season.items() if data is not None]
    if not valid_items:
        return None, {}

    aligned_arrays = xr.align(*[data.astype('float64') for _, data in valid_items], join='inner')
    if not aligned_arrays:
        return None, {}

    season_labels = [season for season, _ in valid_items]
    stacked = xr.concat(aligned_arrays, dim=pd.Index(season_labels, name='season'))
    season_mean = stacked.mean('season')

    aligned_dict = {
        season: aligned
        for (season, _), aligned in zip(valid_items, aligned_arrays)
    }
    return season_mean, aligned_dict


def compute_seasonal_fingerprint(temp_by_season, process_by_season):
    season_mean_temp, aligned_temp = compute_cross_season_mean(temp_by_season)
    if season_mean_temp is None:
        return {}, {}, None

    fingerprint_temp = {}
    for season, data in aligned_temp.items():
        fingerprint = (data - season_mean_temp).astype('float64')
        fingerprint.name = data.name
        fingerprint.attrs.update(data.attrs)
        fingerprint.attrs['fingerprint_reference'] = 'cross-season mean'
        fingerprint_temp[season] = fingerprint

    fingerprint_process = {season: {} for season in aligned_temp}
    process_vars = sorted({
        var
        for season_data in process_by_season.values()
        for var in season_data
    })

    for var in process_vars:
        var_by_season = {
            season: process_by_season.get(season, {}).get(var)
            for season in aligned_temp
        }
        season_mean_var, aligned_var = compute_cross_season_mean(var_by_season)
        if season_mean_var is None:
            continue

        for season, data in aligned_var.items():
            fingerprint = (data - season_mean_var).astype('float64')
            fingerprint.name = data.name
            fingerprint.attrs.update(data.attrs)
            fingerprint.attrs['fingerprint_reference'] = 'cross-season mean'
            fingerprint_process[season][var] = fingerprint

    return fingerprint_temp, fingerprint_process, season_mean_temp


def run_tau_workflow(source_freq='Raw', vars_to_load=None):
    if source_freq not in SUPPORTED_FILE_FREQS:
        raise ValueError(f"Unsupported file_freq '{source_freq}'. Choose from {sorted(SUPPORTED_FILE_FREQS)}")
    if primary_plot_metric not in SUPPORTED_PLOT_METRICS:
        raise ValueError(
            f"Unsupported primary_plot_metric '{primary_plot_metric}'. "
            f"Choose from {sorted(SUPPORTED_PLOT_METRICS)}"
        )

    print('='*60)
    print(f'RUNNING TAU ATTRIBUTION FOR {source_freq}')
    print('='*60)

    vars_to_load = set(vars_to_load) if vars_to_load is not None else set()
    if var_temp is not None:
        vars_to_load.add(var_temp)

    years_load = prepare_years_load(years, months_dict=months_dict, source_freq=source_freq)

    print('='*60)
    print('PHASE 1: Loading all data')
    print('='*60)

    var_data = {}
    weights = None

    for var in vars_to_load:
        print(f'  Loading: {var}')
        data_raw = load_var(
            var=var,
            data_source='RACMO2.4A',
            data_sources=data_sources,
            station_sources=station_sources,
            file_freq=source_freq,
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
            weights = area_weights(
                data_raw.isel(time=0),
                rotpole_native=proj_cfg.get('RACMO2.4A', ccrs.PlateCarree()),
            )

        var_data[var] = data_raw.astype('float32')

    print(f'Loaded {len(var_data)} datasets\n')

    print('='*60)
    print('PHASE 2: Domain averaging of raw data')
    print('='*60)

    var_weighted = {}
    for var in vars_to_load:
        print(f'  Area-weighting: {var}')
        var_weighted[var] = area_weighted_mean(var_data[var], weights=weights).compute().astype('float64')
        del var_data[var]

    var_data = None

    print(f'Area-weighted {len(var_weighted)} datasets\n')

    print('='*60)
    print('PHASE 3: Increment closure check')
    print('='*60)

    closure_stats = None
    if var_temp is not None and var_tend_close_true in var_weighted:
        closure_stats = compute_increment_closure_stats(var_weighted[var_temp], var_weighted[var_tend_close_true])
        print(f"  mean(|ΔT - {var_tend_close_true}|) = {closure_stats['mean_abs']:.3e}")
        print(f"  max(|ΔT - {var_tend_close_true}|) = {closure_stats['max_abs']:.3e}\n")

    print('='*60)
    print('PHASE 4: Computing seasonal tau attribution')
    print('='*60)

    tau_temp = {}
    tau_process = {}
    centered_anom_temp = {}
    centered_anom_process = {}
    process_data = {var: data for var, data in var_weighted.items() if var != var_temp}
    include_tau = (primary_plot_metric == 'Tau') or plot_tau_diagnostic
    include_centered_anomaly = primary_plot_metric in {'CenteredAnomaly', 'CenteredAnomalyFingerprint'}

    for month_key, months in months_dict.items():
        print(f'  Season: {month_key}')
        temp_da, proc_dict, ctr_temp_da, ctr_proc_dict = compute_season_outputs(
            var_weighted[var_temp],
            process_data,
            months,
            source_freq=source_freq,
            target_years=years,
            include_tau=include_tau,
            include_centered_anomaly=include_centered_anomaly,
            reference_year=reference_center_year,
        )
        if include_tau:
            tau_temp[month_key] = temp_da
            tau_process[month_key] = proc_dict

        if include_centered_anomaly:
            centered_anom_temp[month_key] = ctr_temp_da
            centered_anom_process[month_key] = ctr_proc_dict

    return {
        'source_freq': source_freq,
        'weighted_data': var_weighted,
        'tau_temp': tau_temp,
        'tau_process': tau_process,
        'centered_anom_temp': centered_anom_temp,
        'centered_anom_process': centered_anom_process,
        'increment_closure': closure_stats,
    }


#%% ============================================================================
#   PLOTTING HELPERS
#   ============================================================================

def format_panel_title(month_key):
    return month_key


def plot_time_values(data, use_year_axis=False):
    if data is None or not use_year_axis:
        return None if data is None else data['time'].values

    year_vals = data['time'].dt.year.values
    return pd.to_datetime(dict(year=year_vals, month=1, day=1))


def group_plot_series(group_name, panel_data):
    group_vars = var_tend_sums.get(group_name, [])
    group_colors = COLORS[1:]

    panel_series = [(group_name, panel_data['main'], COLORS[0], '-')]
    for ii, var in enumerate(group_vars):
        label = var_name_cfg.get(var, var)
        panel_series.append((label, panel_data['components'].get(var), group_colors[ii % len(group_colors)], '-'))

    return panel_series


def closure_actual_plot_series(panel_data, temp_label=r'$\tau_{y,s}^{temp}$',
                               summed_label='Summed processes',
                               true_label=None):
    if true_label is None:
        true_label = var_name_cfg.get(var_tend_close_true, var_tend_close_true)

    return [
        (summed_label, panel_data['summed_processes'], 'tab:blue', '-'),
        (true_label, panel_data['true_tendency'], 'tab:orange', '-'),
        (temp_label, panel_data['temp_close'], 'tab:green', '-'),
    ]


def closure_difference_plot_series(panel_data, temp_label=r'$\tau_{y,s}^{temp}$'):
    return [
        (f'{temp_label} - tendtot', panel_data['diff_true'], 'tab:red', '-'),
        (f'{temp_label} - summed processes', panel_data['diff_sum'], 'tab:purple', '-'),
    ]


def comparison_plot_series(panel_data):
    panel_series = []
    for freq in compare_file_freqs:
        data = panel_data.get(freq)
        if data is None:
            continue
        panel_series.append((freq, data, FREQ_COLORS.get(freq, 'k'), '-'))
    return panel_series


def plot_series_list(ax, panel_series, use_year_axis=False):
    for label, data, color, linestyle in panel_series:
        if data is None:
            continue
        ax.plot(
            plot_time_values(data, use_year_axis=use_year_axis),
            data.values,
            c=color,
            lw=2.5,
            ms=4,
            marker='o',
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


def normalize_axes(axes, n_rows, n_cols):
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    if n_rows == 1:
        return axes.reshape(1, -1)
    if n_cols == 1:
        return axes.reshape(-1, 1)
    return axes


def hide_unused_axes(axes, n_used, n_rows, n_cols):
    for idx in range(n_used, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)


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
            x_left_fig = bbox.x0 / fig_width
            min_x_fig = min(min_x_fig, x_left_fig)

    min_y_fig = 1.0
    for col in range(n_cols):
        ax = axes[-1, col]
        for label in ax.get_xticklabels():
            bbox = label.get_window_extent(renderer=renderer)
            y_bottom_fig = bbox.y0 / fig_height
            min_y_fig = min(min_y_fig, y_bottom_fig)

    x_center = 0.5 * (axes[-1, 0].get_position().x0 + axes[-1, -1].get_position().x1)
    y_bottom = min_y_fig - pad - 0.01
    fig.text(x_center, y_bottom, x_label, ha='center', va='top', fontsize=fontsize * 1.2)

    y_center = 0.5 * (axes[0, 0].get_position().y1 + axes[-1, 0].get_position().y0)
    x_left = min_x_fig - pad
    fig.text(x_left, y_center, y_label, ha='right', va='center', rotation=90, fontsize=fontsize)


def save_figure(save_name_base, var, suffix):
    if save_name_base is None:
        return

    save_name = f'{save_name_base}_{var}_{suffix}'
    pdf_dir = Path(f'/nobackup/users/walj/Figures_tendencies/pdf/{var}')
    jpg_dir = Path(f'/nobackup/users/walj/Figures_tendencies/jpg/{var}')
    pdf_dir.mkdir(parents=True, exist_ok=True)
    jpg_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(str(pdf_dir / f'{save_name}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(str(jpg_dir / f'{save_name}.jpg'), format='jpg', dpi=300, bbox_inches='tight')


def plot_panel_grid(panel_series_dict, layout, panel_width, panel_height, y_range, shared_labels,
                    mirror_y_axes, use_year_axis, y_label, save_var, suffix, zero_line=False):
    n_panels = len(PANEL_KEYS)
    n_rows, n_cols = layout

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        sharex=True, sharey=False
    )
    wspace = 0.06 if mirror_y_axes else 0.16
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.95, wspace=wspace, hspace=0.18)
    axes = normalize_axes(axes, n_rows, n_cols)

    title_fs = max(18, int(panel_height * 5))
    tick_fs = max(12, int(panel_height * 3))
    legend_fs = max(10, int(panel_height * 2.5))

    for idx, month_key in enumerate(PANEL_KEYS):
        ax = axes[idx // n_cols, idx % n_cols]
        plot_series_list(ax, panel_series_dict[month_key], use_year_axis=use_year_axis)
        format_time_axis(ax, format_panel_title(month_key), title_fs, tick_fs, zero_line=zero_line)

        is_right_col = (idx % n_cols == n_cols - 1) and n_cols > 1
        if mirror_y_axes and is_right_col:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        panel_y_range = y_range.get(month_key) if isinstance(y_range, dict) else y_range
        if panel_y_range is not None:
            ax.set_ylim(*panel_y_range)

        style_legend(ax, legend_fs)

    hide_unused_axes(axes, n_panels, n_rows, n_cols)

    if shared_labels:
        add_shared_axis_labels(
            fig,
            axes,
            'Time',
            y_label,
            fontsize=max(28, int(panel_height * 7))
        )

    save_figure(save_name_base, save_var, suffix)
    plt.show()


#%% ============================================================================
#   MAIN WORKFLOW
#   ============================================================================

def main():
    var_tend_main_vars = list(var_tend_main.values())
    var_tend_sum_vars = [var for vars_group in var_tend_sums.values() for var in vars_group]
    vars_all = set(var_tend_main_vars + var_tend_sum_vars + var_tend_close_pr + [var_tend_close_true] +
                   ([var_temp] if var_temp is not None else []))

    results_main = run_tau_workflow(source_freq=file_freq, vars_to_load=vars_all)

    plot_group_data = None
    plot_closure_data = None
    plot_group_label = None
    plot_closure_label = None
    plot_closure_diff_label = None
    plot_group_suffix = None
    plot_closure_suffix = None
    plot_true_label = None
    plot_temp_label = None

    if primary_plot_metric == 'CenteredAnomalyFingerprint':
        fingerprint_temp, fingerprint_process, _ = compute_seasonal_fingerprint(
            results_main['centered_anom_temp'],
            results_main['centered_anom_process'],
        )
        plot_group_data = {
            group_name: {
                month_key: build_group_panel_data(group_name, fingerprint_process[month_key])
                for month_key in PANEL_KEYS
            }
            for group_name in var_tend_main
        }
        plot_closure_data = {
            month_key: build_closure_panel_data(
                fingerprint_process[month_key],
                fingerprint_temp[month_key],
            )
            for month_key in PANEL_KEYS
        }
        plot_group_label = r'$\delta A_{y,s}^{\mathrm{ctr11}}$ (K)'
        plot_closure_label = r'$\delta A_{y,s}^{\mathrm{ctr11}}$ (K)'
        plot_closure_diff_label = r'Fingerprint residual (K)'
        plot_group_suffix = 'ctr11_fingerprint_group'
        plot_closure_suffix = 'ctr11_fingerprint'
        plot_true_label = r'$\delta A_{y,s}^{\mathrm{ctr11,tendtot}}$'
        plot_temp_label = r'$\delta A_{y,s}^{\mathrm{ctr11,temp}}$'
    elif primary_plot_metric == 'CenteredAnomaly':
        plot_group_data = {
            group_name: {
                month_key: build_group_panel_data(group_name, results_main['centered_anom_process'][month_key])
                for month_key in PANEL_KEYS
            }
            for group_name in var_tend_main
        }
        plot_closure_data = {
            month_key: build_closure_panel_data(
                results_main['centered_anom_process'][month_key],
                results_main['centered_anom_temp'][month_key],
            )
            for month_key in PANEL_KEYS
        }
        plot_group_label = r'$A_{y,s}^{\mathrm{ctr11}}$ (K)'
        plot_closure_label = r'$A_{y,s}^{\mathrm{ctr11}}$ (K)'
        plot_closure_diff_label = r'Centered anomaly residual (K)'
        plot_group_suffix = 'ctr11_anomaly_group'
        plot_closure_suffix = 'ctr11_anomaly'
        plot_true_label = r'$A_{y,s}^{\mathrm{ctr11,tendtot}}$'
        plot_temp_label = r'$A_{y,s}^{\mathrm{ctr11,temp}}$'
    else:
        plot_group_data = {
            group_name: {
                month_key: build_group_panel_data(group_name, results_main['tau_process'][month_key])
                for month_key in PANEL_KEYS
            }
            for group_name in var_tend_main
        }
        plot_closure_data = {
            month_key: build_closure_panel_data(
                results_main['tau_process'][month_key],
                results_main['tau_temp'][month_key],
            )
            for month_key in PANEL_KEYS
        }
        close_unit = var_units_cfg.get(var_tend_close_true, 'K')
        plot_group_label = rf'$\tau_{{y,s}}$ ({close_unit})'
        plot_closure_label = rf'$\tau_{{y,s}}$ ({close_unit})'
        plot_closure_diff_label = rf'Closure residual ({close_unit})'
        plot_group_suffix = 'tau_group'
        plot_closure_suffix = 'tau_closure'
        plot_true_label = None
        plot_temp_label = r'$\tau_{y,s}^{temp}$'

    for group_name in var_tend_main:
        main_var = var_tend_main[group_name]
        group_panel_series = {
            month_key: group_plot_series(group_name, plot_group_data[group_name][month_key])
            for month_key in PANEL_KEYS
        }
        plot_panel_grid(
            group_panel_series,
            layout=group_layout,
            panel_width=group_panel_width,
            panel_height=group_panel_height,
            y_range=group_y_range,
            shared_labels=group_shared_labels,
            mirror_y_axes=group_mirror_y_axes,
            use_year_axis=group_x_year,
            y_label=plot_group_label,
            save_var=main_var,
            suffix=plot_group_suffix,
            zero_line=True,
        )

    closure_actual_series = {
        month_key: closure_actual_plot_series(
            plot_closure_data[month_key],
            temp_label=plot_temp_label,
            summed_label='Summed processes',
            true_label=plot_true_label,
        )
        for month_key in PANEL_KEYS
    }
    closure_diff_series = {
        month_key: closure_difference_plot_series(
            plot_closure_data[month_key],
            temp_label=plot_temp_label,
        )
        for month_key in PANEL_KEYS
    }

    plot_panel_grid(
        closure_actual_series,
        layout=close_layout,
        panel_width=close_panel_width,
        panel_height=close_panel_height,
        y_range=close_y_range,
        shared_labels=close_shared_labels,
        mirror_y_axes=close_mirror_y_axes,
        use_year_axis=close_x_year,
        y_label=plot_closure_label,
        save_var=var_tend_close_true,
        suffix=f'{plot_closure_suffix}_actual',
        zero_line=True,
    )
    plot_panel_grid(
        closure_diff_series,
        layout=close_layout,
        panel_width=close_panel_width,
        panel_height=close_panel_height,
        y_range=close_y_range,
        shared_labels=close_shared_labels,
        mirror_y_axes=close_mirror_y_axes,
        use_year_axis=close_x_year,
        y_label=plot_closure_diff_label,
        save_var=var_tend_close_true,
        suffix=f'{plot_closure_suffix}_diff',
        zero_line=True,
    )

    if plot_tau_diagnostic:
        tau_group_panel_data = {
            group_name: {
                month_key: build_group_panel_data(group_name, results_main['tau_process'][month_key])
                for month_key in PANEL_KEYS
            }
            for group_name in var_tend_main
        }
        tau_closure_panel_data = {
            month_key: build_closure_panel_data(
                results_main['tau_process'][month_key],
                results_main['tau_temp'][month_key],
            )
            for month_key in PANEL_KEYS
        }

        for group_name in var_tend_main:
            main_var = var_tend_main[group_name]
            tau_group_series = {
                month_key: group_plot_series(group_name, tau_group_panel_data[group_name][month_key])
                for month_key in PANEL_KEYS
            }
            plot_panel_grid(
                tau_group_series,
                layout=group_layout,
                panel_width=group_panel_width,
                panel_height=group_panel_height,
                y_range=group_y_range,
                shared_labels=group_shared_labels,
                mirror_y_axes=group_mirror_y_axes,
                use_year_axis=group_x_year,
                y_label=r'$\tau_{y,s}$ (K)',
                save_var=main_var,
                suffix='tau_group',
                zero_line=True,
            )

        tau_closure_actual_series = {
            month_key: closure_actual_plot_series(tau_closure_panel_data[month_key])
            for month_key in PANEL_KEYS
        }
        tau_closure_diff_series = {
            month_key: closure_difference_plot_series(tau_closure_panel_data[month_key])
            for month_key in PANEL_KEYS
        }

        plot_panel_grid(
            tau_closure_actual_series,
            layout=close_layout,
            panel_width=close_panel_width,
            panel_height=close_panel_height,
            y_range=close_y_range,
            shared_labels=close_shared_labels,
            mirror_y_axes=close_mirror_y_axes,
            use_year_axis=close_x_year,
            y_label=r'$\tau_{y,s}$ (K)',
            save_var=var_tend_close_true,
            suffix='tau_closure_actual',
            zero_line=True,
        )
        plot_panel_grid(
            tau_closure_diff_series,
            layout=close_layout,
            panel_width=close_panel_width,
            panel_height=close_panel_height,
            y_range=close_y_range,
            shared_labels=close_shared_labels,
            mirror_y_axes=close_mirror_y_axes,
            use_year_axis=close_x_year,
            y_label='Tau residual (K)',
            save_var=var_tend_close_true,
            suffix='tau_closure_diff',
            zero_line=True,
        )

    if run_frequency_comparison:
        vars_compare = {var_temp, var_tend_close_true}
        results_by_freq = {file_freq: results_main}

        for compare_freq in compare_file_freqs:
            if compare_freq == file_freq:
                continue
            results_by_freq[compare_freq] = run_tau_workflow(source_freq=compare_freq, vars_to_load=vars_compare)

        print('='*60)
        print('FREQUENCY COMPARISON AGAINST PRIMARY CHOICE')
        print('='*60)

        for compare_freq in compare_file_freqs:
            closure_stats = results_by_freq[compare_freq]['increment_closure']
            if closure_stats is None:
                continue
            print(f'{compare_freq:>8s}: mean_abs={closure_stats["mean_abs"]:.3e}, max_abs={closure_stats["max_abs"]:.3e}')

        if primary_plot_metric == 'CenteredAnomalyFingerprint':
            temp_comp_label = r'$\delta A_{y,s}^{\mathrm{ctr11,temp}}$ (K)'
            proc_comp_label = rf'$\delta A_{{y,s}}^{{\mathrm{{ctr11,{var_tend_close_true}}}}}$ (K)'
            temp_comp_suffix = 'ctr11_fingerprint_frequency_comparison_temp'
            proc_comp_suffix = 'ctr11_fingerprint_frequency_comparison_tendtot'

            temp_comp_series = {}
            tendtot_comp_series = {}
            for month_key in PANEL_KEYS:
                temp_panel_data = {}
                proc_panel_data = {}
                for freq in compare_file_freqs:
                    if freq not in results_by_freq:
                        continue
                    temp_fp, proc_fp, _ = compute_seasonal_fingerprint(
                        results_by_freq[freq]['centered_anom_temp'],
                        results_by_freq[freq]['centered_anom_process'],
                    )
                    temp_panel_data[freq] = temp_fp.get(month_key)
                    proc_panel_data[freq] = proc_fp.get(month_key, {}).get(var_tend_close_true)
                temp_comp_series[month_key] = comparison_plot_series(temp_panel_data)
                tendtot_comp_series[month_key] = comparison_plot_series(proc_panel_data)
        elif primary_plot_metric == 'CenteredAnomaly':
            temp_comp_label = r'$A_{y,s}^{\mathrm{ctr11,temp}}$ (K)'
            proc_comp_label = rf'$A_{{y,s}}^{{\mathrm{{ctr11,{var_tend_close_true}}}}}$ (K)'
            temp_comp_suffix = 'ctr11_anomaly_frequency_comparison_temp'
            proc_comp_suffix = 'ctr11_anomaly_frequency_comparison_tendtot'
            temp_comp_series = {
                month_key: comparison_plot_series({
                    freq: results_by_freq[freq]['centered_anom_temp'][month_key]
                    for freq in compare_file_freqs
                    if freq in results_by_freq
                })
                for month_key in PANEL_KEYS
            }
            tendtot_comp_series = {
                month_key: comparison_plot_series({
                    freq: results_by_freq[freq]['centered_anom_process'][month_key].get(var_tend_close_true)
                    for freq in compare_file_freqs
                    if freq in results_by_freq
                })
                for month_key in PANEL_KEYS
            }
        else:
            temp_comp_label = r'$\tau_{y,s}^{temp}$ (K)'
            proc_comp_label = rf'$\tau_{{y,s}}^{{{var_tend_close_true}}}$ (K)'
            temp_comp_suffix = 'tau_frequency_comparison_temp'
            proc_comp_suffix = 'tau_frequency_comparison_tendtot'
            temp_comp_series = {
                month_key: comparison_plot_series({
                    freq: results_by_freq[freq]['tau_temp'][month_key]
                    for freq in compare_file_freqs
                    if freq in results_by_freq
                })
                for month_key in PANEL_KEYS
            }
            tendtot_comp_series = {
                month_key: comparison_plot_series({
                    freq: results_by_freq[freq]['tau_process'][month_key].get(var_tend_close_true)
                    for freq in compare_file_freqs
                    if freq in results_by_freq
                })
                for month_key in PANEL_KEYS
            }

        plot_panel_grid(
            temp_comp_series,
            layout=comp_layout,
            panel_width=comp_panel_width,
            panel_height=comp_panel_height,
            y_range=comp_y_range,
            shared_labels=comp_shared_labels,
            mirror_y_axes=comp_mirror_y_axes,
            use_year_axis=comp_x_year,
            y_label=temp_comp_label,
            save_var='tau_temp',
            suffix=temp_comp_suffix,
            zero_line=True,
        )
        plot_panel_grid(
            tendtot_comp_series,
            layout=comp_layout,
            panel_width=comp_panel_width,
            panel_height=comp_panel_height,
            y_range=comp_y_range,
            shared_labels=comp_shared_labels,
            mirror_y_axes=comp_mirror_y_axes,
            use_year_axis=comp_x_year,
            y_label=proc_comp_label,
            save_var=var_tend_close_true,
            suffix=proc_comp_suffix,
            zero_line=True,
        )


if __name__ == '__main__':
    main()
