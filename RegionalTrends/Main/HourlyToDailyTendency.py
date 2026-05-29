"""Convert 3-hourly RACMO ENBUD output to daily-mean tendency products.

The daily tendency for process p is the contribution to the change in daily
mean temperature:

    Tbar(d + 1) - Tbar(d) = sum_p D_p(d)

For eight 3-hourly intervals per day and increments A_p(d, k),

    D_p(d) = sum_k A_p(d, k) + W_p(d + 1) - W_p(d)
    W_p(d) = sum_k ((7 - k) / 8) * A_p(d, k)

This is the daily part of the archived ``TendencyToFreq.py`` conversion,
written explicitly here so it does not depend on ``ComputeTendencies.py``.
"""

import glob
import os
from pathlib import Path
import sys

import dask
import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path.home() / 'KNMIproject'
sys.path.insert(0, str(PROJECT_ROOT))

import RegionalTrends.Helpers.Config.Constants as Constants


dask.config.set(scheduler='single-threaded')


INPUT_DIR = Path('/nobackup/users/walj/racmo24A_rerun/Threehourly')
OUTPUT_DIR = Path('/nobackup_1/users/walj/racmo24A_rerun/Daily')

N_INTERVALS_PER_DAY = 8
CLOSURE_ATOL = 1.0e-4
EXPECTED_DAILY_START = pd.Timestamp('1978-01-01')
EXPECTED_DAILY_END = pd.Timestamp('2024-12-31')

# Keep the same DailyA base-variable selection as TendencyToFreq.py, adjusted
# for the rerun transfer. This rerun contains advtot, while adiadj and horadv
# are not raw files in the Threehourly directory. The notebook derives adiadj
# from orography + adicomp, and numtot from numbnd + numdif.
DERIVED_OR_UNAVAILABLE_TENDENCY_VARS = {'adiadj', 'horadv'}
RERUN_EXTRA_TENDENCY_VARS = ['advtot']
TENDENCY_VARS = [
    var for var in Constants.TENDENCY_VARS
    if var not in DERIVED_OR_UNAVAILABLE_TENDENCY_VARS
] + RERUN_EXTRA_TENDENCY_VARS
AVERAGE_VARS = ['templ1', 'templ1s', 'templ1spr', 'mlmid', 'mltop']
RAW_VARS = TENDENCY_VARS + AVERAGE_VARS

# Process the principal closure terms first. numtot is reconstructed downstream
# as numbnd + numdif, so its raw source terms are prioritized here.
PRIORITY_VARS = [
    'templ1',
    'tendtot',
    'dyntot',
    'phystot',
    'numbnd',
    'numdif',
    'udtdx',
    'vdtdy',
    'vertadv',
    'orography',
    'adicomp',
    'advtot',
]
ALL_VARS = PRIORITY_VARS + [var for var in RAW_VARS if var not in PRIORITY_VARS]


def raw_files(var):
    files = sorted(glob.glob(str(INPUT_DIR / f'{var}.*.nc')))
    if not files:
        raise FileNotFoundError(f'No raw files found for {var}')
    return files


def open_raw_dataset(files):
    ds = xr.open_mfdataset(
        files,
        combine='nested',
        concat_dim='time',
        chunks='auto',
        data_vars='minimal',
        coords='minimal',
        compat='override',
        parallel=False,
    )
    return ds.sortby('time')


def full_daily_time(time_coord):
    time_index = pd.DatetimeIndex(time_coord.values)
    return pd.date_range(
        f'{int(time_index.year.min())}-01-01',
        f'{int(time_index.year.max())}-12-31',
        freq='D',
    )


def daily_counting_weights(time_coord):
    """Return 7-k after checking for complete 3-hourly days."""

    time_index = pd.DatetimeIndex(time_coord.values)
    if time_index.empty:
        raise ValueError('Cannot construct DailyA output from an empty time axis')
    if not time_index.is_monotonic_increasing or time_index.has_duplicates:
        raise ValueError('Threehourly input times must be unique and increasing')

    time_steps = np.diff(time_index.asi8)
    expected_step = pd.Timedelta(hours=3).value
    if time_steps.size and not np.all(time_steps == expected_step):
        raise ValueError('Threehourly input contains a missing or non-3-hourly step')

    day_counts = pd.Series(1, index=time_index.normalize()).groupby(level=0).sum()
    incomplete = day_counts[day_counts != N_INTERVALS_PER_DAY]
    if not incomplete.empty:
        examples = ', '.join(
            f'{timestamp:%Y-%m-%d}: {int(count)}'
            for timestamp, count in incomplete.iloc[:5].items()
        )
        raise ValueError(
            'DailyA conversion requires eight 3-hourly samples per day; '
            f'incomplete days include {examples}'
        )

    slot = np.tile(np.arange(N_INTERVALS_PER_DAY), len(day_counts))
    count_weight = N_INTERVALS_PER_DAY - 1 - slot
    return xr.DataArray(
        count_weight.astype('float32'),
        dims=['time'],
        coords={'time': time_coord},
    )


def construct_daily_tendency(raw_da):
    """Construct D(d), the process contribution to Tbar(d+1)-Tbar(d)."""

    count_weight = daily_counting_weights(raw_da['time'])

    # The artificial missing final increment has zero contribution to W(d)
    # because its k=7 counting weight is zero.
    weighted_valid = xr.where(count_weight > 0, raw_da.notnull(), True)
    weighted_valid = weighted_valid.resample(time='1D').all('time')
    weighted_sum = (
        (raw_da.fillna(0) * count_weight)
        .resample(time='1D')
        .sum('time', skipna=False)
    )
    weighted_offset = (weighted_sum / np.float32(N_INTERVALS_PER_DAY)).where(weighted_valid)

    daily_sum = raw_da.resample(time='1D').sum('time', skipna=False)
    daily_da = daily_sum + weighted_offset.shift(time=-1) - weighted_offset

    daily_da = daily_da.reindex(time=full_daily_time(raw_da['time']))
    next_time = daily_da['time'].shift(time=-1).values
    daily_da = daily_da.assign_coords(interval_next=('time', next_time))
    return daily_da.astype('float32')


def construct_daily_mean(raw_da):
    daily_counting_weights(raw_da['time'])
    return (
        raw_da.resample(time='1D')
        .mean('time', skipna=False)
        .reindex(time=full_daily_time(raw_da['time']))
        .astype('float32')
    )


def build_output_dataset(template_ds, out_da, var):
    ds_out = out_da.astype('float32').to_dataset(name=var)
    ds_out.attrs = dict(template_ds.attrs)
    ds_out[var].attrs = dict(template_ds[var].attrs)
    return ds_out


def output_filename(template_ds, var, out_time):
    domain = template_ds.attrs['Domain']
    experiment = template_ds.attrs['Experiment']
    time_index = pd.DatetimeIndex(out_time.values)
    start = time_index[0].strftime('%Y%m%d')
    end = time_index[-1].strftime('%Y%m%d')
    return OUTPUT_DIR / f'{var}_dailyA_{domain}_{experiment}_{start}_{end}.nc'


def existing_complete_output(var):
    suffix = (
        f'_{EXPECTED_DAILY_START:%Y%m%d}_'
        f'{EXPECTED_DAILY_END:%Y%m%d}.nc'
    )
    matches = sorted(OUTPUT_DIR.glob(f'{var}_dailyA_*{suffix}'))
    if not matches:
        return None
    if len(matches) > 1:
        names = ', '.join(path.name for path in matches)
        raise ValueError(f'Multiple existing DailyA outputs found for {var}: {names}')

    out_path = matches[0]
    try:
        with xr.open_dataset(out_path) as ds:
            if var not in ds:
                return None
            time_index = pd.DatetimeIndex(ds['time'].values)
            if time_index.empty:
                return None
            if time_index[0] != EXPECTED_DAILY_START:
                return None
            if time_index[-1] != EXPECTED_DAILY_END:
                return None
    except Exception:
        return None

    return out_path


def write_dataset(out_path, ds_out):
    tmp_path = out_path.with_name(f'{out_path.name}.tmp.{os.getpid()}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        ds_out.to_netcdf(tmp_path, unlimited_dims=['time'])
        tmp_path.replace(out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def process_variable(var):
    print(f'\nProcessing {var}', flush=True)

    out_path = existing_complete_output(var)
    if out_path is not None:
        print(f'  Reusing existing daily output {out_path}', flush=True)
        return out_path

    files = raw_files(var)
    template_ds = xr.open_dataset(files[0])
    raw_ds = open_raw_dataset(files)

    try:
        raw_da = raw_ds[var].astype('float32')

        if var in AVERAGE_VARS:
            daily_da = construct_daily_mean(raw_da)
        else:
            # Raw ENBUD values are aligned as in TendencyToFreq.py and
            # ProcessNetCDF.py before constructing D(d).
            daily_da = construct_daily_tendency(raw_da.shift(time=-1))

        out_path = output_filename(template_ds, var, daily_da['time'])
        print(f'  Writing daily output to {out_path}', flush=True)
        ds_out = build_output_dataset(template_ds, daily_da, var)
        try:
            write_dataset(out_path, ds_out)
        finally:
            ds_out.close()
    finally:
        raw_ds.close()
        template_ds.close()

    return out_path


def check_temperature_closure(output_paths):
    """Check tendtot DailyA against the daily change in templ1."""

    print('\nChecking DailyA closure: templ1(d+1) - templ1(d) = tendtot(d)', flush=True)
    with xr.open_dataset(output_paths['templ1'], chunks='auto') as temp_ds:
        with xr.open_dataset(output_paths['tendtot'], chunks='auto') as tendency_ds:
            delta_temp = temp_ds['templ1'].shift(time=-1) - temp_ds['templ1']
            residual = delta_temp - tendency_ds['tendtot']
            max_abs_residual = float(np.abs(residual).max(skipna=True).compute())

    if not np.isfinite(max_abs_residual):
        raise ValueError('DailyA closure could not be evaluated: no finite residuals')
    if max_abs_residual > CLOSURE_ATOL:
        raise ValueError(
            'DailyA closure failed: maximum absolute residual is '
            f'{max_abs_residual:.6g} K (tolerance {CLOSURE_ATOL:.6g} K)'
        )

    print(
        '  DailyA closure passed: maximum absolute residual is '
        f'{max_abs_residual:.6g} K',
        flush=True,
    )


def main():
    output_paths = {}

    print('Generating daily ENBUD products only', flush=True)
    print(f'Input directory: {INPUT_DIR}', flush=True)
    print(f'Output directory: {OUTPUT_DIR}', flush=True)

    for var in ALL_VARS:
        output_paths[var] = process_variable(var)

    check_temperature_closure(output_paths)
    print('\nAll DailyA ENBUD products generated and closure checked.', flush=True)


if __name__ == '__main__':
    main()
