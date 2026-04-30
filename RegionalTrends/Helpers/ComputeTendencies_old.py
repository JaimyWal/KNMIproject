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

    if interval == 'Raw':
        keep_idx = np.arange(time_index.size)
        keys = keep_idx
        years = year_vals
        same_key = list(zip(time_index.month.values, time_index.day.values, time_index.hour.values))

    elif interval == 'Daily':
        keep_idx = np.arange(time_index.size)
        keys = np.asarray(time_index.to_period('D'))
        years = year_vals
        same_key = list(zip(time_index.month.values, time_index.day.values))

    elif interval == 'Monthly':
        keep_idx = np.arange(time_index.size)
        keys = np.asarray(time_index.to_period('M'))
        years = year_vals
        same_key = month_vals

    elif interval == 'Yearly':
        keep_idx = np.arange(time_index.size)
        keys = year_vals
        years = year_vals
        same_key = np.zeros_like(year_vals)

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
        season_id = season_id[keep_idx]
        keys = years*10 + season_id
        same_key = season_id

    da_use = tendency.isel(time=keep_idx)
    groups = contiguous_groups(keys)

    group_start = np.array([np.datetime64(da_use['time'].values[i0]) for _, i0, _ in groups])
    group_year = np.array([int(years[i0]) for _, i0, _ in groups])
    group_same_key = [same_key[i0] for _, i0, _ in groups]

    return da_use, groups, group_start, group_year, group_same_key


def interval_pairs(group_year, group_same_key, relation='Adjacent'):

    n = len(group_year)

    if relation == 'Adjacent':
        return [(ii - 1, ii) for ii in range(1, n)]

    if relation == 'Yearly':
        lookup = {(int(group_year[ii]), group_same_key[ii]): ii for ii in range(n)}
        pairs = []
        for ii in range(n):
            prev = lookup.get((int(group_year[ii]) - 1, group_same_key[ii]))
            if prev is not None:
                pairs.append((prev, ii))
        return pairs


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
                       interval='Monthly',
                       relation='Adjacent',
                       return_intermediates=False):

    if 'time' not in tendency.dims and 'interval' in tendency.dims:
        da_full = tendency.rename({'interval': 'time'})
    else:
        da_full = tendency

    da_use, groups, group_start, group_year, group_same_key = interval_setup(
        da_full,
        interval=interval
    )
    pairs = interval_pairs(group_year, group_same_key, relation=relation)

    d_vals = []
    for _, i0, i1 in groups:
        da_slice = da_use.isel(time=slice(i0, i1))
        d_vals.append(within_group_D(da_slice, interval=interval))

    D_out = xr.concat(d_vals, dim=xr.IndexVariable('interval', group_start))

    A_vals = []
    B_vals = []
    prev_start = []
    curr_start = []
    prev_year = []
    full_time = pd.DatetimeIndex(da_full['time'].values)

    for prev, curr in pairs:
        i_prev = groups[prev][1]
        i_curr = groups[curr][1]
        B_val = da_use.isel(time=slice(i_prev, i_curr)).sum('time', skipna=False)

        A_val = B_val + d_vals[curr] - d_vals[prev]

        B_vals.append(B_val)
        A_vals.append(A_val)
        prev_start.append(group_start[prev])
        curr_start.append(group_start[curr])
        prev_year.append(group_year[prev])

    A_out = xr.concat(A_vals, dim=xr.IndexVariable('interval', np.asarray(prev_start)))
    B_out = xr.concat(B_vals, dim=xr.IndexVariable('interval', np.asarray(prev_start)))
    A_out = A_out.assign_coords(interval_next=('interval', np.asarray(curr_start)))
    B_out = B_out.assign_coords(interval_next=('interval', np.asarray(curr_start)))

    if interval == 'Seasonal':
        D_out = D_out.assign_coords(season_year=('interval', np.asarray(group_year)))
        A_out = A_out.assign_coords(season_year=('interval', np.asarray(prev_year)))
        B_out = B_out.assign_coords(season_year=('interval', np.asarray(prev_year)))

    if return_intermediates:
        return {'A': A_out, 'B': B_out, 'D': D_out}

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
