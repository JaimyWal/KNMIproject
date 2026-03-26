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


def interval_setup(tendency, interval='Monthly', season_months=None):

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

    elif interval == 'Seasonal':
        if season_months is None:
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
        else:
            season_months = np.asarray(season_months)
            keep_mask = np.isin(month_vals, season_months)
            keep_idx = np.where(keep_mask)[0]

            years = year_vals[keep_idx].copy()
            if season_months[-1] < season_months[0]:
                years = np.where(month_vals[keep_idx] >= season_months[0], years + 1, years)

            year_months = pd.DataFrame({'season_year': years, 'month': month_vals[keep_idx]})
            months_per_year = year_months.groupby('season_year')['month'].nunique()

            if len(months_per_year) > 0 and months_per_year.max() > 1:
                full_years = months_per_year[months_per_year == len(season_months)].index.values
                full_mask = np.isin(years, full_years)
                keep_idx = keep_idx[full_mask]
                years = years[full_mask]
            elif len(months_per_year) > 0:
                full_mask = month_vals[keep_idx] == season_months[0]
                keep_idx = keep_idx[full_mask]
                years = years[full_mask]

            keys = years
            same_key = [tuple(season_months)]*keep_idx.size

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


def within_group_D(da_slice, interval, season_months):

    n = da_slice.sizes['time']

    if interval == 'Seasonal' and season_months is not None:
        t_slice = pd.DatetimeIndex(da_slice['time'].values)
        if n > 1 and (t_slice.day == 1).all():
            month_w = np.asarray(t_slice.days_in_month)
            omega = np.cumsum(month_w[::-1])[::-1] - month_w
            w = xr.DataArray(omega, dims=['time'], coords={'time': da_slice['time']})
            return (da_slice*w).sum('time', skipna=False) / month_w.sum()

    w = xr.DataArray(np.arange(n - 1, -1, -1), dims=['time'], coords={'time': da_slice['time']})
    return (da_slice*w).sum('time', skipna=False) / n


def full_interval_time(time_coord, interval='Monthly', season_months=None):

    time_index = pd.DatetimeIndex(time_coord.values)
    if time_index.size == 0:
        return pd.DatetimeIndex([])

    start_year = int(time_index.year.min())
    end_year = int(time_index.year.max())

    if interval == 'Daily':
        return pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='D')
    if interval == 'Monthly':
        return pd.date_range(f'{start_year}-01-01', f'{end_year}-12-01', freq='MS')
    if interval == 'Seasonal':
        years = np.arange(start_year, end_year + 1)

        if season_months is None:
            months = [1, 4, 7, 10]
            return pd.DatetimeIndex(
                [pd.Timestamp(year=year, month=month, day=1) for year in years for month in months]
            )

        season_months = np.asarray(season_months)
        month = int(season_months[(len(season_months) - 1)//2])
        return pd.DatetimeIndex(
            [pd.Timestamp(year=year, month=month, day=1) for year in years]
        )


def construct_tendency(tendency,
                       interval='Monthly',
                       relation='Adjacent',
                       season_months=None,
                       return_intermediates=False):

    if 'time' not in tendency.dims and 'interval' in tendency.dims:
        da_full = tendency.rename({'interval': 'time'})
    else:
        da_full = tendency

    da_use, groups, group_start, group_year, group_same_key = interval_setup(
        da_full,
        interval=interval,
        season_months=season_months,
    )
    pairs = interval_pairs(group_year, group_same_key, relation=relation)

    d_vals = []
    for _, i0, i1 in groups:
        da_slice = da_use.isel(time=slice(i0, i1))
        d_vals.append(within_group_D(da_slice, interval=interval, season_months=season_months))

    D_out = xr.concat(d_vals, dim=xr.IndexVariable('interval', group_start))

    A_vals = []
    B_vals = []
    prev_start = []
    curr_start = []
    prev_year = []
    full_time = pd.DatetimeIndex(da_full['time'].values)

    for prev, curr in pairs:
        if interval == 'Seasonal' and season_months is not None:
            i_prev = full_time.get_loc(pd.Timestamp(group_start[prev]))
            i_curr = full_time.get_loc(pd.Timestamp(group_start[curr]))
            B_val = da_full.isel(time=slice(i_prev, i_curr)).sum('time', skipna=False)
        else:
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

    if interval in {'Daily', 'Monthly', 'Seasonal'}:
        if interval == 'Seasonal':
            month = np.select(
                [np.isin(pd.DatetimeIndex(A_out['interval'].values).month, [12, 1, 2]),
                 np.isin(pd.DatetimeIndex(A_out['interval'].values).month, [3, 4, 5]),
                 np.isin(pd.DatetimeIndex(A_out['interval'].values).month, [6, 7, 8]),
                 np.isin(pd.DatetimeIndex(A_out['interval'].values).month, [9, 10, 11])],
                [1, 4, 7, 10], default=int(np.asarray(season_months)[(len(season_months) - 1)//2]) if season_months is not None else -1
            )
            shifted_time = pd.to_datetime(dict(year=A_out['season_year'].values, month=month, day=1))
            A_out = A_out.assign_coords(interval=np.asarray(shifted_time))
        full_time = full_interval_time(da_full['time'], interval, season_months=season_months)
        if 'interval' in A_out.dims:
            A_out = A_out.rename({'interval': 'time'})
        A_out = A_out.reindex(time=full_time)
        if interval == 'Seasonal':
            A_out = A_out.assign_coords(season_year=('time', pd.DatetimeIndex(full_time).year.values))

    return A_out
