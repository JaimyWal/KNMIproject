#!/usr/bin/env python3

import argparse
import glob
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr


INPUT_DIR = Path("/nobackup_1/users/walj/racmo24/Threehourly")
DAILY_DIR = Path("/nobackup_1/users/walj/racmo24/Daily")
MONTHLY_DIR = Path("/nobackup_1/users/walj/racmo24/Monthly")
SEASONAL_DIR = Path("/nobackup_1/users/walj/racmo24/Seasonal")
ERA5_THREEHOURLY_DIR = Path("/nobackup/users/walj/era5/Threehourly")
ERA5_DAILY_DIR = Path("/nobackup/users/walj/era5/Daily")
KNMI_THREEHOURLY_DIR = Path("/nobackup/users/walj/knmi/Threehourly")
KNMI_DAILY_DIR = Path("/nobackup/users/walj/knmi/Daily")
RACMO_V5_THREEHOURLY_DIR = Path("/nobackup/users/walj/racmo24/Threehourly")
RACMO_V5_DAILY_DIR = Path("/nobackup/users/walj/racmo24/Daily")
TENDENCY_VARS = (
    "adiadj adicomp confric conphase consens contot dyntot horadv lcbotdn lctopdn "
    "lcbotup lctopup lcnet lscld lwbotdn lwbotup lwnet lwtopdn lwtopup numbnd numdif "
    "orography phystot radtot scbotdn scbotup scnet sctopdn sctopup swbotdn swbotup "
    "swnet swtopdn swtopup tendtot tendtotpr udtdx vdffric vdfphase vdfsens vdftot "
    "vdtdy vertadv"
).split()
SEASONS = [("DJF", (12, 1, 2), 1), ("MAM", (3, 4, 5), 4), ("JJA", (6, 7, 8), 7), ("SON", (9, 10, 11), 10)]
PROCESS_GROUPS = {
    "dyntot": ["dyntot"],
    "radtot": ["radtot"],
    "senstot": ["consens", "vdfsens"],
    "phasetot": ["conphase", "vdfphase", "lscld"],
    "frictot": ["confric", "vdffric"],
    "numtot": ["numbnd", "numdif"],
}
SPATIAL_DIMS = ("height", "rlat", "rlon")
DAY_KEYS = tuple(pd.date_range("2000-01-01", "2000-12-31", freq="D").strftime("%m-%d"))
DAY_INDEX = {key: idx for idx, key in enumerate(DAY_KEYS)}
TEMP_GRID_SPECS = (
    {
        "name": "ERA5",
        "input_dir": ERA5_THREEHOURLY_DIR,
        "daily_dir": ERA5_DAILY_DIR,
        "glob": "temp.ERA5-*.EU.3H.nc",
        "pattern": "temp_dailycont_EU_ERA5_{start}_{end}.nc",
    },
    {
        "name": "RACMO2.4p1_v5",
        "input_dir": RACMO_V5_THREEHOURLY_DIR,
        "daily_dir": RACMO_V5_DAILY_DIR,
        "glob": "temp.KNMI-*.KEXT12.RACMO2.4p1_v5_trends_bugfixes.3H.nc",
        "pattern": "temp_dailycont_KEXT12_RACMO2.4p1_v5_trends_bugfixes_{start}_{end}.nc",
    },
)
SYNC_EVERY = 8
DEC_STEPS = 31 * 8
YEAR_RE = re.compile(r"-(\d{4})\.")
YEAR_META = {}
TOL = 1e-10


def parse_args():
    p = argparse.ArgumentParser(description="Build exact monthly and seasonal yearly tendency contributions from raw 3-hourly increments.")
    p.add_argument("--variables", nargs="+", default=TENDENCY_VARS)
    p.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    p.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    p.add_argument("--monthly-dir", type=Path, default=MONTHLY_DIR)
    p.add_argument("--seasonal-dir", type=Path, default=SEASONAL_DIR)
    p.add_argument("--time-chunk", type=int, default=124)
    p.add_argument("--shift-minus-one", action="store_true")
    p.add_argument("--check-total-var", default=None)
    p.add_argument("--check-temp-var", default=None)
    p.add_argument("--daily-only", action="store_true")
    p.add_argument("--skip-temp-dailycont", action="store_true")
    p.add_argument("--sync-every", type=int, default=SYNC_EVERY)
    return p.parse_args()


def files_for(var, input_dir):
    files = sorted(glob.glob(str(input_dir / f"{var}.*.nc")))
    if not files:
        raise FileNotFoundError(f"No raw files found for {var} in {input_dir}")
    return files


def available_variables(input_dir):
    return {
        re.sub(r"([._]).*$", "", path.name)
        for path in Path(input_dir).iterdir()
        if path.is_file() and path.suffix == ".nc"
    }


def extract_year(path):
    match = YEAR_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse year from {path}")
    return int(match.group(1))


def expected_steps(year):
    return (366 if pd.Timestamp(year=year, month=1, day=1).is_leap_year else 365) * 8


def year_meta(year):
    if year not in YEAR_META:
        n_steps = expected_steps(year)
        time_index = pd.date_range(f"{year}-01-01 00:00:00", periods=n_steps, freq="3h")
        months = time_index.month.to_numpy()
        month_days = time_index.strftime("%m-%d").to_numpy()
        month_steps = np.array([(months == month).sum() for month in range(1, 13)], dtype=int)
        month_offsets = np.concatenate(([0], np.cumsum(month_steps)))
        month_slices = {month: slice(month_offsets[month - 1], month_offsets[month]) for month in range(1, 13)}
        day_breaks = np.flatnonzero(month_days[1:] != month_days[:-1]) + 1
        day_starts = np.concatenate(([0], day_breaks))
        day_stops = np.concatenate((day_breaks, [n_steps]))
        day_keys = [month_days[start] for start in day_starts]
        day_positions = np.array([DAY_INDEX[key] for key in day_keys], dtype=int)
        rise = np.arange(1, n_steps + 1, dtype="float64") / n_steps
        desc = np.arange(n_steps - 1, -1, -1, dtype="float64") / n_steps
        YEAR_META[year] = {
            "n_steps": n_steps,
            "rise": rise,
            "desc": desc,
            "month_slices": month_slices,
            "day_positions": day_positions,
        }
    return YEAR_META[year]


def year_file_map(var, input_dir):
    files = files_for(var, input_dir)
    year_paths = {extract_year(path): Path(path) for path in files}
    years = np.array(sorted(year_paths), dtype=int)
    if len(years) < 3 or not np.all(np.diff(years) == 1):
        raise ValueError(f"{var}: expected at least three consecutive raw years")
    return years, year_paths


def open_template(var, input_dir):
    return xr.open_dataset(files_for(var, input_dir)[0])


def read_raw_year(path, var, year):
    with nc.Dataset(path) as ds:
        variable = ds.variables[var]
        variable.set_auto_maskandscale(False)
        data = np.asarray(variable[:], dtype="float32")
        fill_value = getattr(variable, "_FillValue", None)
    if fill_value is not None:
        data = data.copy()
        data[data == np.float32(fill_value)] = np.nan
    if data.shape[0] != expected_steps(year):
        raise ValueError(f"{var}: expected {expected_steps(year)} steps in {year}, found {data.shape[0]}")
    return data


def make_shifted(raw, next_first):
    shifted = np.empty_like(raw, dtype="float32")
    shifted[:-1] = raw[1:]
    if next_first is None:
        shifted[-1].fill(0.0)
    else:
        shifted[-1] = next_first
    return shifted


def weighted_sum(weights, data):
    return np.tensordot(weights, data, axes=(0, 0))


def weighted_daily_groups(data, weights, year):
    meta = year_meta(year)
    n_days = int(meta["n_steps"]) // 8
    reshaped = np.asarray(data, dtype="float64").reshape((n_days, 8) + data.shape[1:])
    weight_shape = (n_days, 8) + (1,) * (data.ndim - 1)
    weights_reshaped = np.asarray(weights, dtype="float64").reshape(weight_shape)
    nonfinite = ~np.isfinite(reshaped)
    bad = nonfinite & (weights_reshaped != 0)
    if np.any(bad):
        raise RuntimeError(f"daily weighted grouping found non-finite values with nonzero weight in target year {year}")
    safe = np.where(weights_reshaped == 0, 0.0, reshaped * weights_reshaped)
    weighted = safe.sum(axis=1)
    out = np.zeros((len(DAY_KEYS),) + data.shape[1:], dtype="float64")
    out[np.asarray(meta["day_positions"], dtype=int)] = weighted
    return out


def monthly_block(shift_curr, shift_next, year):
    meta_curr = year_meta(year)
    meta_next = year_meta(year + 1)
    parts = np.empty((12,) + shift_curr.shape[1:], dtype="float64")
    for month in range(1, 13):
        sl_curr = meta_curr["month_slices"][month]
        sl_next = meta_next["month_slices"][month]
        parts[month - 1] = (
            weighted_sum(meta_curr["rise"][sl_curr], shift_curr[sl_curr])
            + weighted_sum(meta_next["desc"][sl_next], shift_next[sl_next])
        )
    return np.concatenate((parts.sum(axis=0, keepdims=True), parts), axis=0)


def daily_block(shift_curr, shift_next, year):
    meta_curr = year_meta(year)
    meta_next = year_meta(year + 1)
    parts = (
        weighted_daily_groups(shift_curr, meta_curr["rise"], year)
        + weighted_daily_groups(shift_next, meta_next["desc"], year + 1)
    )
    return np.concatenate((parts.sum(axis=0, keepdims=True), parts), axis=0)


def seasonal_block(shift_prev, shift_curr, shift_next, climate_year):
    meta_prev = year_meta(climate_year - 1)
    meta_curr = year_meta(climate_year)
    meta_next = year_meta(climate_year + 1)
    season_index = {month: idx for idx, (_, months, _) in enumerate(SEASONS) for month in months}
    climate_months = (12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    n_curr = meta_curr["n_steps"]
    n_next = meta_next["n_steps"]
    rise = np.arange(1, n_curr + 1, dtype="float64") / n_curr
    desc = np.arange(n_next - 1, -1, -1, dtype="float64") / n_next
    parts = np.zeros((4,) + shift_curr.shape[1:], dtype="float64")

    offset = 0
    for month in climate_months:
        sl = meta_prev["month_slices"][12] if month == 12 else meta_curr["month_slices"][month]
        data = shift_prev[sl] if month == 12 else shift_curr[sl]
        length = sl.stop - sl.start
        parts[season_index[month]] += weighted_sum(rise[offset : offset + length], data)
        offset += length

    offset = 0
    for month in climate_months:
        sl = meta_curr["month_slices"][12] if month == 12 else meta_next["month_slices"][month]
        data = shift_curr[sl] if month == 12 else shift_next[sl]
        length = sl.stop - sl.start
        parts[season_index[month]] += weighted_sum(desc[offset : offset + length], data)
        offset += length

    return np.concatenate((parts.sum(axis=0, keepdims=True), parts), axis=0)


def residual(label, lhs, rhs):
    diff = np.abs(lhs - rhs)
    logging.info("%s: mean=%.3e max=%.3e", label, float(diff.mean()), float(diff.max()))


def time_values(times):
    times = pd.DatetimeIndex(times)
    ref = pd.Timestamp(year=int(times[0].year), month=1, day=1)
    vals = np.asarray((times - ref) / np.timedelta64(1, "D"), dtype="int64")
    return ref, vals


def daily_period_stamp(years):
    return f"{int(years[0]):04d}0101", f"{int(years[-1]):04d}1231"


class OutputWriter:
    def __init__(self, template, var, mode, times, path, shift_minus_one, season_year=None):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        self.path = path
        self.tmp = path.with_suffix(path.suffix + ".tmp")
        if self.tmp.exists():
            self.tmp.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)

        times = pd.DatetimeIndex(times)
        ref, time_var = time_values(times)
        self.ds = nc.Dataset(self.tmp, "w", format="NETCDF4")
        self.ds.createDimension("time", None)
        for dim in SPATIAL_DIMS:
            self.ds.createDimension(dim, template.sizes[dim])

        for key, value in template.attrs.items():
            setattr(self.ds, key, value)
        self.ds.CreationDate = datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S %Y")
        self.ds.title = f"{template.attrs.get('title', var)} ({mode.lower()} exact yearly contribution)"
        self.ds.comment = (
            f"{template.attrs.get('comment', '')} | exact yearly contribution from raw 3-hourly increments; "
            f"grouped by {'calendar month' if mode == 'Monthly' else 'season in climate years (Dec-Nov)'}; float64 on disk."
        ).strip()
        self.ds.ShiftMinusOneApplied = int(shift_minus_one)

        tvar = self.ds.createVariable("time", "i8", ("time",))
        tvar.units = f"days since {ref:%Y-%m-%d %H:%M:%S}"
        tvar.calendar = "proleptic_gregorian"
        tvar[:] = time_var

        for name in ("rlon", "rlat", "height", "lon", "lat"):
            src = template[name]
            out = self.ds.createVariable(name, "f8", src.dims, fill_value=np.nan)
            out[:] = src.values.astype("float64")
            for key, value in src.attrs.items():
                setattr(out, key, value)

        if season_year is not None:
            out = self.ds.createVariable("season_year", "i4", ("time",))
            out[:] = np.asarray(season_year, dtype="int32")
            out.long_name = "climate year"
            out.comment = "Climate year runs from December of the previous calendar year through November."

        time_chunk = min(12 if mode == "Monthly" else 4, len(times))
        self.var = self.ds.createVariable(
            var,
            "f8",
            ("time",) + SPATIAL_DIMS,
            fill_value=np.nan,
            chunksizes=(time_chunk, template.sizes["height"], template.sizes["rlat"], template.sizes["rlon"]),
        )
        for key, value in template[var].attrs.items():
            setattr(self.var, key, value)
        self.var.cell_methods = "time: exact yearly contribution from raw 3-hourly increments"
        self.var.coordinates = "lat lon" if mode == "Monthly" else "lat lon season_year"
        self.var.contribution_definition = "Weights are (i+1)/N_y in year y and (N_{y+1}-1-i)/N_{y+1} in year y+1."

    def write_block(self, start, values):
        stop = start + values.shape[0]
        self.var[start:stop, :, :, :] = np.asarray(values, dtype="float64")

    def sync(self):
        self.ds.sync()

    def close(self, ok=True):
        self.ds.close()
        if ok:
            self.tmp.replace(self.path)
        elif self.tmp.exists():
            self.tmp.unlink()


class DailyOutputWriter:
    def __init__(self, template, var, years, path, shift_minus_one, comment):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        self.path = path
        self.tmp = path.with_suffix(path.suffix + ".tmp")
        if self.tmp.exists():
            self.tmp.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)

        self.space_dims = tuple(dim for dim in template[var].dims if dim != "time")
        self.ds = nc.Dataset(self.tmp, "w", format="NETCDF4")
        self.ds.createDimension("year", len(years))
        self.ds.createDimension("month_day", len(DAY_KEYS))
        for dim in self.space_dims:
            self.ds.createDimension(dim, template.sizes[dim])

        for key, value in template.attrs.items():
            setattr(self.ds, key, value)
        self.ds.CreationDate = datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S %Y")
        self.ds.title = f"{template.attrs.get('title', var)} (daily exact yearly contribution)"
        self.ds.comment = comment
        self.ds.ShiftMinusOneApplied = int(shift_minus_one)

        year_var = self.ds.createVariable("year", "i4", ("year",))
        year_var[:] = np.asarray(years, dtype="int32")
        year_var.long_name = "target calendar year"

        month_day_var = self.ds.createVariable("month_day", str, ("month_day",))
        month_day_var[:] = np.asarray(DAY_KEYS, dtype=object)
        month_day_var.long_name = "calendar day label"
        month_day_var.comment = "Month-day grouping key; February 29 is retained as its own label."

        for name, src in template.variables.items():
            if name in (var, "time"):
                continue
            if "time" in src.dims:
                continue
            if not all(dim in self.space_dims for dim in src.dims):
                continue
            if np.issubdtype(src.dtype, np.floating):
                dtype = "f8"
                fill_value = np.nan
                values = np.asarray(src.values, dtype="float64")
            elif np.issubdtype(src.dtype, np.integer):
                dtype = "i4"
                fill_value = None
                values = np.asarray(src.values, dtype="int32")
            else:
                continue
            out = self.ds.createVariable(name, dtype, src.dims, fill_value=fill_value)
            out[:] = values
            for key, value in src.attrs.items():
                setattr(out, key, value)

        var_dims = ("year", "month_day") + self.space_dims
        chunksizes = (1, len(DAY_KEYS)) + tuple(template.sizes[dim] for dim in self.space_dims)
        self.var = self.ds.createVariable(var, "f8", var_dims, fill_value=np.nan, chunksizes=chunksizes)
        for key, value in template[var].attrs.items():
            setattr(self.var, key, value)
        coord_parts = [name for name in ("lat", "lon", "latitude", "longitude", "rlat", "rlon", "height") if name in self.ds.variables]
        coord_parts.extend(["year", "month_day"])
        self.var.coordinates = " ".join(coord_parts)
        self.var.cell_methods = "year,month_day: exact yearly contribution from raw 3-hourly increments"
        self.var.contribution_definition = (
            "Weights are (i+1)/N_y in year y and (N_{y+1}-1-i)/N_{y+1} in year y+1, grouped by calendar day label."
        )

    def write_year(self, index, values):
        self.var[index, ...] = np.asarray(values, dtype="float64")

    def sync(self):
        self.ds.sync()

    def close(self, ok=True):
        self.ds.close()
        if ok:
            self.tmp.replace(self.path)
        elif self.tmp.exists():
            self.tmp.unlink()


def output_times(years, mode):
    if mode == "Monthly":
        return pd.to_datetime(
            {"year": np.repeat(years, 12), "month": np.tile(np.arange(1, 13), len(years)), "day": 1}
        )
    anchors = [anchor for _, _, anchor in SEASONS]
    return pd.to_datetime({"year": np.repeat(years, 4), "month": np.tile(anchors, len(years)), "day": 1})


def output_path(template, var, times, mode, out_dir):
    if mode == "Daily":
        years = np.asarray(times, dtype=int)
        start, end = daily_period_stamp(years)
        return out_dir / f"{var}_dailycont_{template.attrs['Domain']}_{template.attrs['Experiment']}_{start}_{end}.nc"
    tag = "monthlycont" if mode == "Monthly" else "seasonalcont"
    times = pd.DatetimeIndex(times)
    return out_dir / f"{var}_{tag}_{template.attrs['Domain']}_{template.attrs['Experiment']}_{times[0]:%Y%m}_{times[-1]:%Y%m}.nc"


def init_closure(shape, calendar_years, climate_years, args, requested):
    components = {comp for comps in PROCESS_GROUPS.values() for comp in comps}
    if not components.issubset(requested):
        missing = sorted(components - requested)
        logging.warning("closure-group check skipped because requested variables are missing components: %s", ", ".join(missing))
        return None
    track_calendar = not args.daily_only
    track_climate = not args.daily_only
    return {
        "calendar_years": calendar_years,
        "climate_years": climate_years,
        "sum_daily": np.zeros((len(calendar_years),) + shape, dtype="float64"),
        "sum_calendar": np.zeros((len(calendar_years),) + shape, dtype="float64") if track_calendar else None,
        "sum_climate": np.zeros((len(climate_years),) + shape, dtype="float64") if track_climate else None,
        "tendtot_daily": np.zeros((len(calendar_years),) + shape, dtype="float64") if args.check_total_var in requested else None,
        "tendtot_calendar": np.zeros((len(calendar_years),) + shape, dtype="float64") if track_calendar and args.check_total_var in requested else None,
        "tendtot_climate": np.zeros((len(climate_years),) + shape, dtype="float64") if track_climate and args.check_total_var in requested else None,
        "components": components,
    }


def year_means(var, input_dir, calendar_years, climate_years):
    raw_years, year_paths = year_file_map(var, input_dir)
    template = open_template(var, input_dir)
    shape = tuple(template.sizes[dim] for dim in SPATIAL_DIMS)
    template.close()

    calendar_out = np.empty((len(calendar_years),) + shape, dtype="float64")
    climate_out = np.empty((len(climate_years),) + shape, dtype="float64")

    needed_calendar = sorted(set(calendar_years.tolist()) | set((calendar_years + 1).tolist()))
    previous_mean = None
    previous_year = None
    calendar_lookup = {int(year): idx for idx, year in enumerate(calendar_years)}
    for year in needed_calendar:
        raw = read_raw_year(year_paths[year], var, year)
        mean = raw.mean(axis=0, dtype="float64")
        if previous_mean is not None and previous_year in calendar_lookup:
            calendar_out[calendar_lookup[previous_year]] = mean - previous_mean
        previous_mean = mean
        previous_year = year

    climate_lookup = {int(year): idx for idx, year in enumerate(climate_years)}
    previous_climate_mean = None
    previous_climate_year = None
    raw_prev = read_raw_year(year_paths[int(raw_years[0])], var, int(raw_years[0]))
    for climate_year in range(int(climate_years[0]), int(climate_years[-1]) + 2):
        raw_curr = read_raw_year(year_paths[climate_year], var, climate_year)
        n_steps = expected_steps(climate_year)
        climate_mean = (
            raw_prev[-DEC_STEPS:].sum(axis=0, dtype="float64")
            + raw_curr[: n_steps - DEC_STEPS].sum(axis=0, dtype="float64")
        ) / n_steps
        if previous_climate_mean is not None and previous_climate_year in climate_lookup:
            climate_out[climate_lookup[previous_climate_year]] = climate_mean - previous_climate_mean
        previous_climate_mean = climate_mean
        previous_climate_year = climate_year
        raw_prev = raw_curr

    return calendar_out, climate_out


def verify_group_closure(state, args):
    if state is None:
        return
    logging.info("closure groups used: %s", ", ".join(PROCESS_GROUPS))
    if state["tendtot_daily"] is not None:
        residual("closure-group sum vs tendtot (daily groups)", state["sum_daily"], state["tendtot_daily"])
    if state["tendtot_calendar"] is not None:
        residual("closure-group sum vs tendtot (calendar years)", state["sum_calendar"], state["tendtot_calendar"])
    if state["tendtot_climate"] is not None:
        residual("closure-group sum vs tendtot (climate years)", state["sum_climate"], state["tendtot_climate"])
    if args.check_temp_var:
        dtemp_calendar, dtemp_climate = year_means(args.check_temp_var, args.input_dir, state["calendar_years"], state["climate_years"])
        residual(f"closure-group sum vs {args.check_temp_var} (daily groups)", state["sum_daily"], dtemp_calendar)
        if state["sum_calendar"] is not None:
            residual(f"closure-group sum vs {args.check_temp_var} (calendar years)", state["sum_calendar"], dtemp_calendar)
        if state["sum_climate"] is not None:
            residual(f"closure-group sum vs {args.check_temp_var} (climate years)", state["sum_climate"], dtemp_climate)


def process(var, args, closure_state):
    logging.info("processing %s", var)
    raw_years, year_paths = year_file_map(var, args.input_dir)
    calendar_years = raw_years[:-1]
    climate_years = raw_years[1:-1]
    template = open_template(var, args.input_dir)
    daily_writer = None
    monthly_writer = None
    seasonal_writer = None

    raw_cache = {}
    shift_cache = {}

    def ensure_raw(year):
        if year not in raw_cache:
            raw_cache[year] = read_raw_year(year_paths[year], var, year)
        return raw_cache[year]

    def ensure_shift(year):
        if year not in shift_cache:
            next_first = ensure_raw(year + 1)[0] if (year + 1) in year_paths else None
            shift_cache[year] = make_shifted(ensure_raw(year), next_first if args.shift_minus_one else None) if args.shift_minus_one else ensure_raw(year)
        return shift_cache[year]

    try:
        shape = tuple(template.sizes[dim] for dim in SPATIAL_DIMS)
        if closure_state["state"] is None:
            closure_state["state"] = init_closure(shape, calendar_years, climate_years, args, set(args.variables))
        elif closure_state["state"] is not None:
            if not np.array_equal(closure_state["state"]["calendar_years"], calendar_years):
                raise ValueError(f"{var}: calendar target years differ from previous variable")
            if not np.array_equal(closure_state["state"]["climate_years"], climate_years):
                raise ValueError(f"{var}: climate target years differ from previous variable")

        daily_path = output_path(template, var, calendar_years, "Daily", args.daily_dir)
        if args.daily_only and daily_path.exists():
            logging.info("  skip existing %s", daily_path)
            return
        monthly_times = output_times(calendar_years, "Monthly")
        seasonal_times = output_times(climate_years, "Seasonal")
        monthly_path = output_path(template, var, monthly_times, "Monthly", args.monthly_dir)
        seasonal_path = output_path(template, var, seasonal_times, "Seasonal", args.seasonal_dir)
        daily_writer = DailyOutputWriter(
            template,
            var,
            calendar_years,
            daily_path,
            args.shift_minus_one,
            (
                f"{template.attrs.get('comment', '')} | exact yearly contribution from raw 3-hourly increments; "
                "grouped by calendar day label (month-day); February 29 retained as its own group; float64 on disk."
            ).strip(),
        )
        if not args.daily_only:
            monthly_writer = OutputWriter(template, var, "Monthly", monthly_times, monthly_path, args.shift_minus_one)
            seasonal_writer = OutputWriter(
                template,
                var,
                "Seasonal",
                seasonal_times,
                seasonal_path,
                args.shift_minus_one,
                season_year=np.repeat(climate_years, 4),
            )
        logging.info("  output files prepared")

        climate_start = int(climate_years[0])
        for i, year in enumerate(calendar_years):
            shift_curr = ensure_shift(int(year))
            shift_next = ensure_shift(int(year + 1))
            block_day = daily_block(shift_curr, shift_next, int(year))
            if not np.isfinite(block_day).all():
                raise RuntimeError(f"{var}: daily output contains non-finite values for target year {year}")
            total_day = block_day[0]
            parts_day = block_day[1:]
            err_day = float(np.abs(parts_day.sum(axis=0) - total_day).max())
            if err_day > TOL:
                raise RuntimeError(f"{var}: daily internal closure failed for {year} (max={err_day:.3e})")
            daily_writer.write_year(i, parts_day)
            if (i + 1) % args.sync_every == 0:
                daily_writer.sync()
            if i < 3 or (i + 1) % 8 == 0 or i + 1 == len(calendar_years):
                logging.info("  daily target year %d (%d/%d)", int(year), i + 1, len(calendar_years))

            if not args.daily_only:
                block_month = monthly_block(shift_curr, shift_next, int(year))
                if not np.isfinite(block_month).all():
                    raise RuntimeError(f"{var}: monthly output contains non-finite values for target year {year}")
                total_month = block_month[0]
                parts_month = block_month[1:]
                err_month = float(np.abs(parts_month.sum(axis=0) - total_month).max())
                if err_month > TOL:
                    raise RuntimeError(f"{var}: monthly internal closure failed for {year} (max={err_month:.3e})")
                monthly_writer.write_block(i * 12, parts_month)
                if (i + 1) % args.sync_every == 0:
                    monthly_writer.sync()
                if i < 3 or (i + 1) % 8 == 0 or i + 1 == len(calendar_years):
                    logging.info("  monthly target year %d (%d/%d)", int(year), i + 1, len(calendar_years))

            if closure_state["state"] is not None and var in closure_state["state"]["components"]:
                closure_state["state"]["sum_daily"][i] += total_day
                if not args.daily_only:
                    closure_state["state"]["sum_calendar"][i] += total_month
            if closure_state["state"] is not None and var == args.check_total_var and closure_state["state"]["tendtot_daily"] is not None:
                closure_state["state"]["tendtot_daily"][i] = total_day
                if not args.daily_only and closure_state["state"]["tendtot_calendar"] is not None:
                    closure_state["state"]["tendtot_calendar"][i] = total_month

            if (not args.daily_only) and year >= climate_start:
                climate_index = int(year - climate_start)
                block_season = seasonal_block(ensure_shift(int(year - 1)), shift_curr, shift_next, int(year))
                if not np.isfinite(block_season).all():
                    raise RuntimeError(f"{var}: seasonal output contains non-finite values for climate year {year}")
                total_season = block_season[0]
                parts_season = block_season[1:]
                err_season = float(np.abs(parts_season.sum(axis=0) - total_season).max())
                if err_season > TOL:
                    raise RuntimeError(f"{var}: seasonal internal closure failed for climate year {year} (max={err_season:.3e})")
                seasonal_writer.write_block(climate_index * 4, parts_season)
                if (climate_index + 1) % args.sync_every == 0:
                    seasonal_writer.sync()
                if climate_index < 2 or (climate_index + 1) % 8 == 0 or climate_index + 1 == len(climate_years):
                    logging.info("  seasonal climate year %d (%d/%d)", int(year), climate_index + 1, len(climate_years))

                if closure_state["state"] is not None and var in closure_state["state"]["components"]:
                    closure_state["state"]["sum_climate"][climate_index] += total_season
                if closure_state["state"] is not None and var == args.check_total_var and closure_state["state"]["tendtot_climate"] is not None:
                    closure_state["state"]["tendtot_climate"][climate_index] = total_season

                shift_cache.pop(int(year - 1), None)

            raw_cache.pop(int(year), None)

        if monthly_writer is not None:
            monthly_writer.sync()
        if seasonal_writer is not None:
            seasonal_writer.sync()
        daily_writer.sync()
        logging.info("  wrote %s", daily_path)
        if monthly_writer is not None:
            logging.info("  wrote %s", monthly_path)
        if seasonal_writer is not None:
            logging.info("  wrote %s", seasonal_path)
    except Exception:
        if daily_writer is not None:
            daily_writer.close(ok=False)
        if monthly_writer is not None:
            monthly_writer.close(ok=False)
        if seasonal_writer is not None:
            seasonal_writer.close(ok=False)
        raise
    else:
        daily_writer.close(ok=True)
        if monthly_writer is not None:
            monthly_writer.close(ok=True)
        if seasonal_writer is not None:
            seasonal_writer.close(ok=True)
    finally:
        template.close()


class GridTendtotStore:
    def __init__(self, year_paths):
        self.year_paths = year_paths

    def available_years(self):
        years = []
        for year in sorted(self.year_paths):
            with xr.open_dataset(self.year_paths[int(year)]) as ds:
                if ds["tendtot"].sizes["time"] == expected_steps(int(year)):
                    years.append(int(year))
        return np.asarray(years, dtype=int)

    def template(self):
        return xr.open_dataset(self.year_paths[int(self.available_years()[0])])

    def read_year(self, year):
        with xr.open_dataset(self.year_paths[int(year)]) as ds:
            return np.asarray(ds["tendtot"].values, dtype="float32")


class StationTendtotStore:
    def __init__(self, path):
        self.path = path

    def available_years(self):
        with xr.open_dataset(self.path) as ds:
            time_index = pd.DatetimeIndex(ds["time"].values)
        years = []
        for year in range(int(time_index.year.min()), int(time_index.year.max()) + 1):
            year_time = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 21:00:00", freq="3h")
            if len(year_time.intersection(time_index)) == len(year_time):
                data = self.read_year(year)
                if np.isfinite(data).all():
                    years.append(year)
        return np.asarray(years, dtype=int)

    def template(self):
        return xr.open_dataset(self.path)

    def read_year(self, year):
        year_time = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 21:00:00", freq="3h")
        with xr.open_dataset(self.path) as ds:
            return np.asarray(ds["tendtot"].reindex(time=year_time).values, dtype="float32")


def year_paths_from_glob(directory, pattern):
    year_paths = {}
    for path in sorted(Path(directory).glob(pattern)):
        match = YEAR_RE.search(path.name)
        if match is not None:
            year_paths[int(match.group(1))] = path
    return year_paths


def process_temp_daily_store(store, out_path, label):
    years = store.available_years()
    year_set = set(int(year) for year in years)
    target_years = np.asarray([int(year) for year in years if int(year) + 1 in year_set], dtype=int)
    if target_years.size == 0:
        logging.warning("skip dailycont for %s: need at least two complete years", label)
        return

    writer = None
    template = store.template()
    try:
        writer = DailyOutputWriter(
            template,
            "tendtot",
            target_years,
            out_path,
            False,
            (
                f"{template.attrs.get('comment', '')} | exact yearly contribution from three-hourly tendtot values; "
                "grouped by calendar day label (month-day); February 29 retained as its own group; float64 on disk."
            ).strip(),
        )
        cache = {}

        def ensure(year):
            if year not in cache:
                cache[year] = store.read_year(year)
            return cache[year]

        for idx, year in enumerate(target_years):
            block = daily_block(ensure(int(year)), ensure(int(year + 1)), int(year))
            if not np.isfinite(block).all():
                raise RuntimeError(f"{label}: dailycont output contains non-finite values for target year {year}")
            err = float(np.abs(block[1:].sum(axis=0) - block[0]).max())
            if err > TOL:
                raise RuntimeError(f"{label}: dailycont internal closure failed for {year} (max={err:.3e})")
            writer.write_year(idx, block[1:])
            if (idx + 1) % SYNC_EVERY == 0:
                writer.sync()
            cache.pop(int(year - 1), None)
        writer.sync()
        logging.info("  wrote %s", out_path)
    except Exception:
        if writer is not None:
            writer.close(ok=False)
        raise
    else:
        writer.close(ok=True)
    finally:
        template.close()


def process_temperature_dailycont():
    for spec in TEMP_GRID_SPECS:
        year_paths = year_paths_from_glob(spec["input_dir"], spec["glob"])
        if not year_paths:
            logging.warning("skip temp dailycont for %s: no three-hourly temp files found in %s", spec["name"], spec["input_dir"])
            continue
        store = GridTendtotStore(year_paths)
        years = store.available_years()
        year_set = set(int(year) for year in years)
        target_years = np.asarray([int(year) for year in years if int(year) + 1 in year_set], dtype=int)
        if target_years.size == 0:
            logging.warning("skip temp dailycont for %s: need at least two complete years", spec["name"])
            continue
        start, end = daily_period_stamp(target_years)
        out_path = spec["daily_dir"] / spec["pattern"].format(start=start, end=end)
        if out_path.exists():
            logging.info("skip existing %s", out_path)
            continue
        logging.info("processing temp dailycont for %s", spec["name"])
        process_temp_daily_store(store, out_path, spec["name"])

    for path in sorted(KNMI_THREEHOURLY_DIR.glob("temp_3hourly_KNMI_*.nc")):
        station_tag = path.stem.split("_")[3]
        store = StationTendtotStore(path)
        years = store.available_years()
        year_set = set(int(year) for year in years)
        target_years = np.asarray([int(year) for year in years if int(year) + 1 in year_set], dtype=int)
        if target_years.size == 0:
            logging.warning("skip temp dailycont for KNMI %s: need at least two complete years", station_tag)
            continue
        start, end = daily_period_stamp(target_years)
        out_path = KNMI_DAILY_DIR / f"temp_dailycont_KNMI_{station_tag}_{start}_{end}.nc"
        if out_path.exists():
            logging.info("skip existing %s", out_path)
            continue
        logging.info("processing temp dailycont for KNMI %s", station_tag)
        process_temp_daily_store(store, out_path, f"KNMI {station_tag}")


def main():
    args = parse_args()
    bad = sorted(set(args.variables) - set(TENDENCY_VARS))
    if bad:
        raise ValueError(f"Only tendency variables are allowed here: {bad}")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    available = available_variables(args.input_dir)
    missing = [var for var in args.variables if var not in available]
    args.variables = [var for var in args.variables if var in available]
    if missing:
        logging.warning("skipping missing variables: %s", ", ".join(missing))
    if not args.variables:
        raise ValueError(f"No requested tendency variables were found in {args.input_dir}")
    logging.info("shift(time=-1)=%s", args.shift_minus_one)
    logging.info("daily-only=%s", args.daily_only)
    logging.info("direct yearly/day reader active")

    closure_state = {"state": None}
    for var in args.variables:
        process(var, args, closure_state)
    verify_group_closure(closure_state["state"], args)
    if not args.skip_temp_dailycont:
        process_temperature_dailycont()


if __name__ == "__main__":
    main()
