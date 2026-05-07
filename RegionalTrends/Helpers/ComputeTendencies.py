import numpy as np
import pandas as pd
import xarray as xr


def contiguous_groups(keys):

    keys = np.asarray(keys)
    if keys.size == 0:
        return []

    idx = np.where(keys[1:] != keys[:-1])[0] + 1
    starts = np.concatenate(([0], idx))
    stops = np.concatenate((idx, [keys.size]))

    return [(keys[s], int(s), int(e)) for s, e in zip(starts, stops)]


def interval_setup(tendency, interval='Monthly'):

    time_index = pd.DatetimeIndex(tendency['time'].values)
    month_vals = time_index.month.values
    year_vals = time_index.year.values
    keep_idx = np.arange(time_index.size)

    if interval == 'Raw':
        keys = keep_idx
        years = year_vals

    elif interval == 'Daily':
        keys = np.asarray(time_index.to_period('D'))
        years = year_vals

    elif interval == 'Monthly':
        keys = np.asarray(time_index.to_period('M'))
        years = year_vals

    elif interval == 'Yearly':
        keys = year_vals
        years = year_vals

    elif interval == 'Seasonal':
        season_id = np.select(
            [
                np.isin(month_vals, [12, 1, 2]),
                np.isin(month_vals, [3, 4, 5]),
                np.isin(month_vals, [6, 7, 8]),
                np.isin(month_vals, [9, 10, 11]),
            ],
            [0, 1, 2, 3],
            default=-1,
        ).astype(int)

        keep_idx = np.where(season_id >= 0)[0]
        years = year_vals[keep_idx].copy()
        years = np.where(month_vals[keep_idx] == 12, years + 1, years)
        keys = years*10 + season_id[keep_idx]

    da_use = tendency.isel(time=keep_idx)
    groups = contiguous_groups(keys)

    group_start = np.array([np.datetime64(da_use['time'].values[i0]) for _, i0, _ in groups])
    group_year = np.array([int(years[i0]) for _, i0, _ in groups])

    return da_use, groups, group_start, group_year


def within_group_D(da_slice, interval):
    time_index = pd.DatetimeIndex(da_slice['time'].values)
    lengths = np.ones(time_index.size, dtype='float32')

    if interval in {'Seasonal', 'Yearly'}:
        month_step = np.diff(time_index.year*12 + time_index.month)
        if time_index.size > 1 and (time_index.day == 1).all() and np.all(month_step == 1):
            lengths = np.asarray(time_index.days_in_month, dtype='float32')
        elif time_index.size > 1 and (time_index.day == 1).all() and np.all(month_step == 3):
            lengths = np.asarray(
                (
                (time_index - pd.DateOffset(months=1)).days_in_month
                + time_index.days_in_month
                + (time_index + pd.DateOffset(months=1)).days_in_month
                ),
                dtype='float32'
            )

    omega = np.cumsum(lengths[::-1])[::-1] - lengths
    w = xr.DataArray(omega, dims=['time'], coords={'time': da_slice['time']})
    valid = xr.where(w > 0, da_slice.notnull(), True).all('time')
    weighted_sum = (da_slice.fillna(0)*w).sum('time')
    return (weighted_sum / np.sum(lengths)).where(valid)


def full_interval_time(time_coord, interval='Monthly'):

    time_index = pd.DatetimeIndex(time_coord.values)
    if time_index.size == 0:
        return pd.DatetimeIndex([])

    start_year = int(time_index.year.min())
    end_year = int(time_index.year.max())

    if interval == 'Daily':
        return pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='D')
    if interval == 'Monthly':
        return pd.date_range(f'{start_year}-01-01', f'{end_year}-12-01', freq='MS')
    if interval == 'Yearly':
        return pd.date_range(f'{start_year}-01-01', f'{end_year}-01-01', freq='YS')
    if interval == 'Seasonal':
        years = np.arange(start_year, end_year + 1)

        months = [1, 4, 7, 10]
        return pd.DatetimeIndex(
            [pd.Timestamp(year=year, month=month, day=1) for year in years for month in months]
        )


def construct_tendency(tendency,
                       interval='Monthly'):

    if 'time' not in tendency.dims and 'interval' in tendency.dims:
        da_full = tendency.rename({'interval': 'time'})
    else:
        da_full = tendency

    da_use, groups, group_start, group_year = interval_setup(
        da_full,
        interval=interval
    )
    d_vals = []
    for _, i0, i1 in groups:
        da_slice = da_use.isel(time=slice(i0, i1))
        d_vals.append(within_group_D(da_slice, interval=interval))

    A_vals = []
    prev_start = np.asarray(group_start[:-1])
    curr_start = np.asarray(group_start[1:])

    for (_, i_prev, _), (_, i_curr, _), d_prev, d_curr in zip(
        groups[:-1], groups[1:], d_vals[:-1], d_vals[1:]
    ):
        B_val = da_use.isel(time=slice(i_prev, i_curr)).sum('time', skipna=False)

        A_val = B_val + d_curr - d_prev

        A_vals.append(A_val)

    A_out = xr.concat(A_vals, dim=xr.IndexVariable('interval', prev_start))
    A_out = A_out.assign_coords(interval_next=('interval', curr_start))

    if interval == 'Seasonal':
        A_out = A_out.assign_coords(season_year=('interval', group_year[:-1]))

    if interval in {'Daily', 'Monthly', 'Seasonal', 'Yearly'}:
        if interval == 'Seasonal':
            month = np.select(
                [np.isin(pd.DatetimeIndex(A_out['interval'].values).month, [12, 1, 2]),
                 np.isin(pd.DatetimeIndex(A_out['interval'].values).month, [3, 4, 5]),
                 np.isin(pd.DatetimeIndex(A_out['interval'].values).month, [6, 7, 8]),
                 np.isin(pd.DatetimeIndex(A_out['interval'].values).month, [9, 10, 11])],
                [1, 4, 7, 10], -1
            )
            shifted_time = pd.to_datetime(dict(year=A_out['season_year'].values, month=month, day=1))
            A_out = A_out.assign_coords(interval=np.asarray(shifted_time))
        full_time = full_interval_time(da_full['time'], interval)
        if 'interval' in A_out.dims:
            A_out = A_out.rename({'interval': 'time'})
        A_out = A_out.reindex(time=full_time)
        if interval == 'Seasonal':
            A_out = A_out.assign_coords(season_year=('time', pd.DatetimeIndex(full_time).year.values))

    return A_out
