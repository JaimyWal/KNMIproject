#!/usr/bin/env python3

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd


DAILY_DIRS = [
    Path('/nobackup_1/users/walj/racmo24/Daily'),
    Path('/nobackup/users/walj/racmo24/Daily'),
    Path('/nobackup/users/walj/era5/Daily'),
    Path('/nobackup/users/walj/knmi/Daily'),
]

PATTERN = '*_dailycont_*.nc'
BACKUP_NAME = 'dailycont_year_month_day_backup_20260428'
CALENDAR = 'proleptic_gregorian'
COMP_LEVEL = 4
TIME_CHUNK = 31


def log(message: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def copy_attrs(src, dst, skip=()) -> None:
    for name in src.ncattrs():
        if name not in skip:
            setattr(dst, name, getattr(src, name))


def strings(values) -> np.ndarray:
    return np.asarray(values).astype(str)


def variable_name(path: Path) -> str:
    return path.name.split('_dailycont_', 1)[0]


def dailycont_variable_name(src, path: Path) -> str:
    guessed = variable_name(path)
    if guessed in src.variables:
        return guessed

    candidates = [
        name for name, var in src.variables.items()
        if {'year', 'month_day'}.issubset(var.dimensions)
    ]
    if len(candidates) == 1:
        return candidates[0]

    raise KeyError(
        f"{path.name}: expected variable '{guessed}', "
        f"but found dailycont candidates {candidates}"
    )


def create_var(dst, name: str, src_var, dims: tuple[str, ...]):
    fill_value = getattr(src_var, '_FillValue', None)
    kwargs = {}
    if fill_value is not None:
        kwargs['fill_value'] = fill_value
    if np.issubdtype(src_var.dtype, np.floating):
        kwargs.update(zlib=True, complevel=COMP_LEVEL, shuffle=True)
        if dims and dims[0] == 'time':
            kwargs['chunksizes'] = (TIME_CHUNK,) + tuple(dst.dimensions[dim].size for dim in dims[1:])
    out = dst.createVariable(name, src_var.datatype, dims, **kwargs)
    copy_attrs(src_var, out, skip={'_FillValue'})
    return out


def time_plan(years: np.ndarray, month_days: np.ndarray):
    month_day_index = {day: idx for idx, day in enumerate(strings(month_days))}
    feb28 = month_day_index['02-28']
    feb29 = month_day_index['02-29']

    dates_all = []
    plan = []

    for year_idx, year_value in enumerate(years.astype(int)):
        year = int(year_value)
        dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
        year_plan = []

        for date in dates:
            month_day = date.strftime('%m-%d')
            if month_day == '02-28' and not is_leap(year):
                year_plan.append((feb28, feb29))
            else:
                year_plan.append((month_day_index[month_day],))

        dates_all.extend(dates)
        plan.append((year_idx, year, year_plan))

    return pd.DatetimeIndex(dates_all), plan


def write_converted(src_path: Path, out_path: Path) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
    if tmp_path.exists():
        log(f"  remove partial {tmp_path.name}")
        tmp_path.unlink()

    with nc.Dataset(src_path) as src:
        if 'time' in src.dimensions:
            log(f"  backup already has time, skip {src_path.name}")
            return
        if not {'year', 'month_day'}.issubset(src.dimensions):
            raise ValueError(f"{src_path} has neither time nor year/month_day dailycont dimensions")

        name = dailycont_variable_name(src, src_path)
        years = np.asarray(src.variables['year'][:], dtype=int)
        month_days = strings(src.variables['month_day'][:])
        time_index, plan = time_plan(years, month_days)
        time_units = f'days since {int(years[0]):04d}-01-01 00:00:00'
        time_values = (
            time_index.values.astype('datetime64[D]')
            - np.datetime64(f'{int(years[0]):04d}-01-01')
        ).astype('timedelta64[D]').astype('int64')

        src_data = src.variables[name]
        space_dims = tuple(dim for dim in src_data.dimensions if dim not in ('year', 'month_day'))

        with nc.Dataset(tmp_path, 'w', format='NETCDF4') as dst:
            for dim_name, dim in src.dimensions.items():
                if dim_name not in ('year', 'month_day'):
                    dst.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)
            dst.createDimension('time', None)

            copy_attrs(src, dst)
            old_comment = str(getattr(src, 'comment', '')).strip()
            leap_note = (
                'Converted from year/month_day dailycont layout to a real proleptic_gregorian daily time axis. '
                'For non-leap target years, the stored 02-29 contribution was added to 02-28 so annual sums are preserved; '
                'for leap target years, 02-29 remains its own time step.'
            )
            dst.CreationDate = datetime.now(timezone.utc).strftime('%a %b %d %H:%M:%S %Y')
            dst.comment = f'{old_comment} | {leap_note}' if old_comment else leap_note
            dst.dailycont_original_layout = 'year, month_day, ...'
            dst.dailycont_original_file = str(src_path)
            dst.dailycont_leap_day_handling = (
                'Non-leap target-year 02-29 bins are merged into target-year 02-28. '
                'Leap-year 02-29 bins are kept on Feb 29.'
            )

            time_var = dst.createVariable('time', 'i8', ('time',))
            time_var[:] = time_values
            time_var.units = time_units
            time_var.calendar = CALENDAR
            time_var.long_name = 'target calendar date'
            time_var.comment = 'Daily target dates for exact yearly contribution bins.'

            for var_name, src_var in src.variables.items():
                if var_name in (name, 'year', 'month_day', 'time'):
                    continue
                if any(dim in ('year', 'month_day') for dim in src_var.dimensions):
                    continue
                out_var = create_var(dst, var_name, src_var, src_var.dimensions)
                out_var[:] = src_var[:]

            out_data = create_var(dst, name, src_data, ('time',) + space_dims)
            out_data.cell_methods = 'time: exact yearly contribution from raw 3-hourly increments'
            coords = [
                coord for coord in ('lat', 'lon', 'latitude', 'longitude', 'rlat', 'rlon', 'height')
                if coord in dst.variables
            ]
            if coords:
                out_data.coordinates = ' '.join(coords)
            old_definition = str(getattr(src_data, 'contribution_definition', '')).strip()
            leap_definition = 'Leap-day conversion for standard time axis: non-leap target-year 02-29 contributions are merged into 02-28.'
            out_data.contribution_definition = f'{old_definition} {leap_definition}'.strip()

            start = 0
            max_error = 0.0
            for year_idx, year, year_plan in plan:
                source = np.asarray(src_data[year_idx, ...], dtype='float64')
                converted = np.empty((len(year_plan),) + source.shape[1:], dtype='float64')

                for time_idx, source_indices in enumerate(year_plan):
                    if len(source_indices) == 1:
                        converted[time_idx, ...] = source[source_indices[0], ...]
                    else:
                        converted[time_idx, ...] = source[source_indices[0], ...] + source[source_indices[1], ...]

                source_sum = np.nansum(source, axis=0)
                converted_sum = np.nansum(converted, axis=0)
                max_error = max(max_error, float(np.nanmax(np.abs(source_sum - converted_sum))))

                stop = start + len(year_plan)
                out_data[start:stop, ...] = converted
                start = stop

            dst.dailycont_max_abs_annual_sum_conversion_error = max_error

    tmp_path.replace(out_path)
    log(f"  wrote {out_path} ({len(time_index)} time steps)")


def dims(path: Path) -> set[str]:
    with nc.Dataset(path) as ds:
        return set(ds.dimensions)


def convert_target(daily_dir: Path, filename: str) -> None:
    out_path = daily_dir / filename
    backup_dir = daily_dir / BACKUP_NAME
    backup_path = backup_dir / filename

    if out_path.exists():
        current_dims = dims(out_path)
        if 'time' in current_dims:
            log(f"  already converted: {out_path}")
            return
        if {'year', 'month_day'}.issubset(current_dims):
            backup_dir.mkdir(exist_ok=True)
            if backup_path.exists():
                raise FileExistsError(f"Backup already exists while top-level is still unconverted: {backup_path}")
            log(f"  move original to {backup_path}")
            shutil.move(str(out_path), str(backup_path))
        else:
            raise ValueError(f"Unexpected dimensions in {out_path}: {sorted(current_dims)}")

    if not backup_path.exists():
        raise FileNotFoundError(f"No source file for conversion: {out_path} or {backup_path}")

    write_converted(backup_path, out_path)


def dailycont_names(daily_dir: Path) -> list[str]:
    backup_dir = daily_dir / BACKUP_NAME
    names = {path.name for path in daily_dir.glob(PATTERN) if path.parent == daily_dir}
    if backup_dir.exists():
        names.update(path.name for path in backup_dir.glob(PATTERN))
    return sorted(names)


def main() -> None:
    for daily_dir in DAILY_DIRS:
        if not daily_dir.exists():
            continue

        names = dailycont_names(daily_dir)
        log(f"{daily_dir}: {len(names)} dailycont files")
        for index, filename in enumerate(names, start=1):
            log(f"[{index}/{len(names)}] {filename}")
            convert_target(daily_dir, filename)

    log('All requested dailycont conversions complete.')


if __name__ == '__main__':
    main()
