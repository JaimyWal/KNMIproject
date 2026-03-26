import glob
from pathlib import Path
import shutil
import sys

import dask
import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path.home() / 'KNMIproject'
sys.path.insert(0, str(PROJECT_ROOT))

from RegionalTrends.Helpers.ComputeTendencies import construct_tendency, full_interval_time
import RegionalTrends.Helpers.Config.Constants as Constants


dask.config.set(scheduler='single-threaded')


INPUT_DIR = Path('/nobackup/users/walj/racmo24_aer/threehourly')
OUTPUT_DIRS = {
    'Daily': Path('/nobackup/users/walj/racmo24_aer/daily'),
    'Monthly': Path('/nobackup/users/walj/racmo24_aer/monthly'),
    'Seasonal': Path('/nobackup/users/walj/racmo24_aer/seasonal'),
}
TEMP_ROOT = Path('/nobackup/users/walj/racmo24_aer/.tendency_to_freq_tmp')

TENDENCY_VARS = list(Constants.TENDENCY_VARS)
AVERAGE_VARS = ['templ1', 'templ1s', 'templ1spr', 'mlmid', 'mltop']
ALL_VARS = TENDENCY_VARS + AVERAGE_VARS

SEASONS = (
    ((12, 1, 2), 1),
    ((3, 4, 5), 4),
    ((6, 7, 8), 7),
    ((9, 10, 11), 10),
)


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


def seasonal_average(monthly_da):
    full_time = pd.DatetimeIndex(full_interval_time(monthly_da['time'], interval='Seasonal'))

    seasonal_parts = []

    for months, time_month in SEASONS:
        month_arr = np.asarray(months, dtype=int)
        output_time = full_time[full_time.month == time_month]
        output_years = output_time.year.values
        in_season = monthly_da['time'].dt.month.isin(month_arr)
        season_da = monthly_da.where(in_season, drop=True)

        month_vals = season_da['time'].dt.month
        year_vals = season_da['time'].dt.year

        if month_arr[0] > month_arr[-1]:
            season_year = xr.where(month_vals >= month_arr[0], year_vals + 1, year_vals)
        else:
            season_year = year_vals

        season_da = season_da.assign_coords(season_year=('time', season_year.values))
        season_da = season_da.where(season_da['season_year'].isin(output_years), drop=True)

        year_months = pd.DataFrame({
            'season_year': season_da['season_year'].values,
            'month': season_da['time'].dt.month.values,
        })
        complete_years = year_months.groupby('season_year')['month'].nunique()
        complete_years = complete_years[complete_years == len(month_arr)].index.values

        month_weights = season_da['time'].dt.days_in_month.astype('float32')
        month_weights = month_weights.assign_coords(season_year=season_da['season_year'])

        weighted_sum = (season_da * month_weights).groupby('season_year').sum('time', skipna=False)
        weight_sum = month_weights.groupby('season_year').sum('time')
        season_yearly = (weighted_sum / weight_sum).reindex(season_year=output_years).astype('float32')
        complete_mask = xr.DataArray(
            np.isin(output_years, complete_years),
            dims=['season_year'],
            coords={'season_year': output_years},
        )
        season_yearly = season_yearly.where(complete_mask)

        season_yearly = season_yearly.rename({'season_year': 'time'}).assign_coords(
            time=('time', output_time),
            season_year=('time', output_years),
        )
        seasonal_parts.append(season_yearly)

    return xr.concat(seasonal_parts, dim='time').sortby('time').astype('float32')


def build_output_dataset(template_ds, out_da, var):
    ds_out = out_da.astype('float32').to_dataset(name=var)
    ds_out.attrs = dict(template_ds.attrs)
    ds_out[var].attrs = dict(template_ds[var].attrs)
    return ds_out


def output_filename(template_ds, var, frequency, out_time):
    domain = template_ds.attrs['Domain']
    experiment = template_ds.attrs['Experiment']
    freq_token = {'Daily': 'dailyA', 'Monthly': 'monthlyA', 'Seasonal': 'seasonalA'}[frequency]
    stamp_fmt = '%Y%m%d' if frequency == 'Daily' else '%Y%m'

    time_index = pd.DatetimeIndex(out_time.values)
    start = time_index[0].strftime(stamp_fmt)
    end = time_index[-1].strftime(stamp_fmt)

    return OUTPUT_DIRS[frequency] / f'{var}_{freq_token}_{domain}_{experiment}_{start}_{end}.nc'


def write_dataset(out_path, ds_out):
    tmp_path = Path(f'{out_path}.tmp')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()
    if out_path.exists():
        out_path.unlink()

    ds_out.to_netcdf(tmp_path, unlimited_dims=['time'])
    tmp_path.replace(out_path)


def process_variable(var):
    print(f'\nProcessing {var}', flush=True)
    files = raw_files(var)

    template_ds = xr.open_dataset(files[0])
    raw_ds = open_raw_dataset(files)

    try:
        raw_da = raw_ds[var].astype('float32')

        if var in TENDENCY_VARS:
            # Match the raw tendency handling in ProcessNetCDF: shift 3-hourly
            # tendency values one step backward and keep the final timestamp as NaN.
            raw_da = raw_da.shift(time=-1)
            daily_da = construct_tendency(raw_da, interval='Daily', relation='previous_interval').astype('float32')
            monthly_da = construct_tendency(raw_da, interval='Monthly', relation='previous_interval').astype('float32')
            seasonal_da = construct_tendency(raw_da, interval='Seasonal', relation='previous_interval').astype('float32')
        else:
            daily_da = raw_da.resample(time='1D').mean(skipna=False).reindex(
                time=full_interval_time(raw_da['time'], interval='Daily')
            ).astype('float32')
            monthly_da = raw_da.resample(time='MS').mean(skipna=False).reindex(
                time=full_interval_time(raw_da['time'], interval='Monthly')
            ).astype('float32')
            seasonal_da = seasonal_average(monthly_da)

        for frequency, out_da in [
            ('Daily', daily_da),
            ('Monthly', monthly_da),
            ('Seasonal', seasonal_da),
        ]:
            out_path = output_filename(template_ds, var, frequency, out_da['time'])
            print(f'  Writing {frequency.lower()} output to {out_path}', flush=True)
            ds_out = build_output_dataset(template_ds, out_da, var)
            try:
                write_dataset(out_path, ds_out)
            finally:
                ds_out.close()
    finally:
        raw_ds.close()
        template_ds.close()


def main():
    for out_dir in OUTPUT_DIRS.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT)

    print('Generating daily, monthly, and seasonal ENBUD products', flush=True)
    print(f'Input directory: {INPUT_DIR}', flush=True)

    for var in ALL_VARS:
        process_variable(var)

    print('\nAll ENBUD frequency products generated.', flush=True)


if __name__ == '__main__':
    main()
