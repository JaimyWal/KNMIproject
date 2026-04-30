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
MONTHLY_DIR = Path("/nobackup_1/users/walj/racmo24/Monthly")
SEASONAL_DIR = Path("/nobackup_1/users/walj/racmo24/Seasonal")
DEFAULT_VARS = [
    "tendtot",
    "dyntot",
    "radtot",
    "numbnd", "numdif", "numtot",
    "consens", "vdfsens", "senstot",
    "confric", "vdffric", "frictot",
    "conphase", "vdfphase", "lscld", "phasetot",
]
RAW_TENDENCY_VARS = (
    "adiadj adicomp confric conphase consens contot dyntot horadv lcbotdn lctopdn "
    "lcbotup lctopup lcnet lscld lwbotdn lwbotup lwnet lwtopdn lwtopup numbnd numdif "
    "orography phystot radtot scbotdn scbotup scnet sctopdn sctopup swbotdn swbotup "
    "swnet swtopdn swtopup tendtot tendtotpr udtdx vdffric vdfphase vdfsens vdftot "
    "vdtdy vertadv"
).split()
ALLOWED_VARS = sorted(set(RAW_TENDENCY_VARS) | set(DEFAULT_VARS))
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
YEAR_RE = re.compile(r"KNMI-(\d{4})\.")
SYNC_EVERY = 8
TOL = 1e-10
YEAR_META = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build exact same-month and same-season yearly B contributions from raw 3-hourly tendencies."
    )
    parser.add_argument("--variables", nargs="+", default=DEFAULT_VARS)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--monthly-dir", type=Path, default=MONTHLY_DIR)
    parser.add_argument("--seasonal-dir", type=Path, default=SEASONAL_DIR)
    parser.add_argument("--shift-minus-one", action="store_true")
    parser.add_argument("--check-total-var", default="tendtot")
    parser.add_argument("--check-temp-var", default=None)
    parser.add_argument("--sync-every", type=int, default=SYNC_EVERY)
    return parser.parse_args()


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
        month_steps = np.array([(months == month).sum() for month in range(1, 13)], dtype=int)
        month_offsets = np.concatenate(([0], np.cumsum(month_steps)))
        month_slices = {month: slice(month_offsets[month - 1], month_offsets[month]) for month in range(1, 13)}
        YEAR_META[year] = {
            "n_steps": n_steps,
            "month_slices": month_slices,
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


def build_year_offsets(years):
    offsets = {}
    offset = 0
    for year in sorted(int(year) for year in years):
        offsets[year] = offset
        offset += year_meta(year)["n_steps"]
    return offsets


def month_segments_for_year(year, month):
    return [(int(year), year_meta(int(year))["month_slices"][int(month)])]


def season_segments_for_climate_year(climate_year, season_name):
    climate_year = int(climate_year)
    if season_name == "DJF":
        return [
            (climate_year - 1, year_meta(climate_year - 1)["month_slices"][12]),
            (climate_year, year_meta(climate_year)["month_slices"][1]),
            (climate_year, year_meta(climate_year)["month_slices"][2]),
        ]
    if season_name == "MAM":
        return [(climate_year, year_meta(climate_year)["month_slices"][month]) for month in (3, 4, 5)]
    if season_name == "JJA":
        return [(climate_year, year_meta(climate_year)["month_slices"][month]) for month in (6, 7, 8)]
    if season_name == "SON":
        return [(climate_year, year_meta(climate_year)["month_slices"][month]) for month in (9, 10, 11)]
    raise ValueError(f"Unknown season: {season_name}")


def interval_info(segments, year_offsets):
    start = year_offsets[segments[0][0]] + segments[0][1].start
    stop = year_offsets[segments[-1][0]] + segments[-1][1].stop
    n_steps = int(sum(segment.stop - segment.start for _, segment in segments))
    if stop - start != n_steps:
        raise ValueError("Interval segments are not contiguous on the raw timeline")
    return {"segments": segments, "start": start, "stop": stop, "N": n_steps}


def concat_interval(data_cache, segments):
    pieces = [np.asarray(data_cache[year][segment], dtype="float64") for year, segment in segments]
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)


def sum_global_slice(data_cache, year_offsets, start, stop):
    total = None
    sample = None
    for year in sorted(year_offsets):
        sample = data_cache[year]
        year_start = year_offsets[year]
        year_stop = year_start + year_meta(year)["n_steps"]
        lo = max(start, year_start)
        hi = min(stop, year_stop)
        if lo >= hi:
            continue
        part = np.asarray(data_cache[year][lo - year_start : hi - year_start], dtype="float64")
        summed = part.sum(axis=0, dtype="float64")
        total = summed if total is None else total + summed
    if total is None:
        if sample is None:
            raise ValueError("sum_global_slice received an empty cache")
        return np.zeros(sample.shape[1:], dtype="float64")
    return total


def weighted_interval(interval, label):
    interval = np.asarray(interval, dtype="float64")
    n_steps = interval.shape[0]
    weights = np.arange(n_steps - 1, -1, -1, dtype="float64") / n_steps
    bad = ~np.isfinite(interval)
    if bad.any():
        zero_rows = weights == 0.0
        if bad[~zero_rows].any():
            raise RuntimeError(f"{label}: non-finite values appear with non-zero within-interval weights")
        shape = (n_steps,) + (1,) * (interval.ndim - 1)
        interval = np.where(
            zero_rows.reshape(shape),
            np.nan_to_num(interval, nan=0.0, posinf=0.0, neginf=0.0),
            interval,
        )
    return weighted_sum(weights, interval)


def monthly_yearly_block(shift_curr, shift_next, year):
    data_cache = {int(year): shift_curr, int(year + 1): shift_next}
    year_offsets = build_year_offsets(data_cache)
    shape = shift_curr.shape[1:]
    total_block = np.empty((12,) + shape, dtype="float64")
    between_block = np.empty_like(total_block)
    within_block = np.empty_like(total_block)

    for month in range(1, 13):
        curr = interval_info(month_segments_for_year(year, month), year_offsets)
        nxt = interval_info(month_segments_for_year(year + 1, month), year_offsets)
        between = sum_global_slice(data_cache, year_offsets, curr["start"], nxt["start"])
        within = (
            weighted_interval(concat_interval(data_cache, nxt["segments"]), f"month {month} in {year + 1}")
            - weighted_interval(concat_interval(data_cache, curr["segments"]), f"month {month} in {year}")
        )
        between_block[month - 1] = between
        within_block[month - 1] = within
        total_block[month - 1] = between + within

    return total_block, between_block, within_block


def seasonal_yearly_block(shift_prev, shift_curr, shift_next, climate_year):
    data_cache = {int(climate_year - 1): shift_prev, int(climate_year): shift_curr, int(climate_year + 1): shift_next}
    year_offsets = build_year_offsets(data_cache)
    shape = shift_curr.shape[1:]
    total_block = np.empty((4,) + shape, dtype="float64")
    between_block = np.empty_like(total_block)
    within_block = np.empty_like(total_block)

    for idx, (season_name, _, _) in enumerate(SEASONS):
        curr = interval_info(season_segments_for_climate_year(climate_year, season_name), year_offsets)
        nxt = interval_info(season_segments_for_climate_year(climate_year + 1, season_name), year_offsets)
        between = sum_global_slice(data_cache, year_offsets, curr["start"], nxt["start"])
        within = (
            weighted_interval(concat_interval(data_cache, nxt["segments"]), f"{season_name} in climate year {climate_year + 1}")
            - weighted_interval(concat_interval(data_cache, curr["segments"]), f"{season_name} in climate year {climate_year}")
        )
        between_block[idx] = between
        within_block[idx] = within
        total_block[idx] = between + within

    return total_block, between_block, within_block


def monthly_temperature_block(raw_curr, raw_next, year):
    data_cache = {int(year): raw_curr, int(year + 1): raw_next}
    year_offsets = build_year_offsets(data_cache)
    shape = raw_curr.shape[1:]
    total_block = np.empty((12,) + shape, dtype="float64")

    for month in range(1, 13):
        curr = interval_info(month_segments_for_year(year, month), year_offsets)
        nxt = interval_info(month_segments_for_year(year + 1, month), year_offsets)
        total_block[month - 1] = (
            concat_interval(data_cache, nxt["segments"]).mean(axis=0, dtype="float64")
            - concat_interval(data_cache, curr["segments"]).mean(axis=0, dtype="float64")
        )

    return total_block


def seasonal_temperature_block(raw_prev, raw_curr, raw_next, climate_year):
    data_cache = {int(climate_year - 1): raw_prev, int(climate_year): raw_curr, int(climate_year + 1): raw_next}
    shape = raw_curr.shape[1:]
    total_block = np.empty((4,) + shape, dtype="float64")

    for idx, (season_name, _, _) in enumerate(SEASONS):
        curr = concat_interval(data_cache, season_segments_for_climate_year(climate_year, season_name))
        nxt = concat_interval(data_cache, season_segments_for_climate_year(climate_year + 1, season_name))
        total_block[idx] = nxt.mean(axis=0, dtype="float64") - curr.mean(axis=0, dtype="float64")

    return total_block


def residual(label, lhs, rhs):
    diff = np.abs(np.asarray(lhs, dtype="float64") - np.asarray(rhs, dtype="float64"))
    logging.info("%s: mean=%.3e max=%.3e", label, float(diff.mean()), float(diff.max()))


def time_values(times):
    times = pd.DatetimeIndex(times)
    ref = pd.Timestamp(year=int(times[0].year), month=1, day=1)
    vals = np.asarray((times - ref) / np.timedelta64(1, "D"), dtype="int64")
    return ref, vals


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
        self.ds.title = f"{template.attrs.get('title', var)} ({mode.lower()} same-interval yearly B contribution)"
        self.ds.comment = (
            f"{template.attrs.get('comment', '')} | exact same-interval yearly difference from raw 3-hourly increments; "
            f"{'same calendar month' if mode == 'Monthly' else 'same climate-year season'}; float64 on disk."
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
        self.var = self._create_data_var(template, var, var, time_chunk, mode)
        self.between = self._create_data_var(template, var, "between_interval", time_chunk, mode)
        self.within = self._create_data_var(template, var, "within_interval", time_chunk, mode)

    def _create_data_var(self, template, source_var, name, time_chunk, mode):
        out = self.ds.createVariable(
            name,
            "f8",
            ("time",) + SPATIAL_DIMS,
            fill_value=np.nan,
            chunksizes=(time_chunk, template.sizes["height"], template.sizes["rlat"], template.sizes["rlon"]),
        )
        for key, value in template[source_var].attrs.items():
            setattr(out, key, value)
        out.coordinates = "lat lon" if mode == "Monthly" else "lat lon season_year"
        if name == source_var:
            out.long_name = f"{getattr(out, 'long_name', source_var)} same-interval yearly contribution"
        elif name == "between_interval":
            out.long_name = f"{source_var} start-to-start between-interval term"
            out.comment = "Start-to-start accumulated raw tendency term."
        else:
            out.long_name = f"{source_var} within-interval correction term"
            out.comment = "Weighted within-interval correction term with w_j=(N-1-j)/N."
        return out

    def write_block(self, start, total, between, within):
        stop = start + total.shape[0]
        self.var[start:stop, :, :, :] = np.asarray(total, dtype="float64")
        self.between[start:stop, :, :, :] = np.asarray(between, dtype="float64")
        self.within[start:stop, :, :, :] = np.asarray(within, dtype="float64")

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
    tag = "monthlyyearlyB" if mode == "Monthly" else "seasonalyearlyB"
    times = pd.DatetimeIndex(times)
    return out_dir / f"{var}_{tag}_{template.attrs['Domain']}_{template.attrs['Experiment']}_{times[0]:%Y%m}_{times[-1]:%Y%m}.nc"


def init_closure(shape, calendar_years, climate_years, args, requested):
    components = {component for components in PROCESS_GROUPS.values() for component in components}
    sum_enabled = components.issubset(requested)
    if not sum_enabled:
        missing = sorted(components - requested)
        logging.warning("grouped-process check skipped because requested variables are missing components: %s", ", ".join(missing))

    track_total = args.check_total_var in requested
    return {
        "calendar_years": calendar_years,
        "climate_years": climate_years,
        "sum_monthly": np.zeros((len(calendar_years), 12) + shape, dtype="float64") if sum_enabled else None,
        "sum_seasonal": np.zeros((len(climate_years), 4) + shape, dtype="float64") if sum_enabled else None,
        "tendtot_monthly": np.zeros((len(calendar_years), 12) + shape, dtype="float64") if track_total else None,
        "tendtot_seasonal": np.zeros((len(climate_years), 4) + shape, dtype="float64") if track_total else None,
        "components": components,
    }


def same_interval_temp_truth(var, input_dir, calendar_years, climate_years):
    raw_years, year_paths = year_file_map(var, input_dir)
    template = open_template(var, input_dir)
    shape = tuple(template.sizes[dim] for dim in SPATIAL_DIMS)
    template.close()

    monthly_out = np.empty((len(calendar_years), 12) + shape, dtype="float64")
    seasonal_out = np.empty((len(climate_years), 4) + shape, dtype="float64")
    raw_cache = {}

    def ensure_raw(year):
        if year not in raw_cache:
            raw_cache[year] = np.asarray(read_raw_year(year_paths[year], var, year), dtype="float64")
        return raw_cache[year]

    for idx, year in enumerate(calendar_years):
        monthly_out[idx] = monthly_temperature_block(ensure_raw(int(year)), ensure_raw(int(year + 1)), int(year))
        if idx > 0:
            raw_cache.pop(int(year - 1), None)

    raw_cache.clear()
    for idx, climate_year in enumerate(climate_years):
        seasonal_out[idx] = seasonal_temperature_block(
            ensure_raw(int(climate_year - 1)),
            ensure_raw(int(climate_year)),
            ensure_raw(int(climate_year + 1)),
            int(climate_year),
        )
        raw_cache.pop(int(climate_year - 1), None)

    return monthly_out, seasonal_out


def verify_closure(state, args):
    if state is None:
        return

    if state["sum_monthly"] is not None and state["tendtot_monthly"] is not None:
        residual("grouped-process sum vs tendtot (monthly same-month yearly)", state["sum_monthly"], state["tendtot_monthly"])
        residual("grouped-process sum vs tendtot (seasonal same-season yearly)", state["sum_seasonal"], state["tendtot_seasonal"])

    if args.check_temp_var is not None and state["tendtot_monthly"] is not None:
        temp_monthly, temp_seasonal = same_interval_temp_truth(
            args.check_temp_var,
            args.input_dir,
            state["calendar_years"],
            state["climate_years"],
        )
        residual(f"{args.check_total_var} vs {args.check_temp_var} (monthly same-month yearly)", state["tendtot_monthly"], temp_monthly)
        residual(f"{args.check_total_var} vs {args.check_temp_var} (seasonal same-season yearly)", state["tendtot_seasonal"], temp_seasonal)
        if state["sum_monthly"] is not None:
            residual(f"grouped-process sum vs {args.check_temp_var} (monthly same-month yearly)", state["sum_monthly"], temp_monthly)
            residual(f"grouped-process sum vs {args.check_temp_var} (seasonal same-season yearly)", state["sum_seasonal"], temp_seasonal)


def process(var, args, closure_state):
    logging.info("processing %s", var)
    raw_years, year_paths = year_file_map(var, args.input_dir)
    calendar_years = raw_years[:-1]
    climate_years = raw_years[1:-1]
    template = open_template(var, args.input_dir)
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
            if args.shift_minus_one:
                next_first = ensure_raw(year + 1)[0] if (year + 1) in year_paths else None
                shift_cache[year] = make_shifted(ensure_raw(year), next_first)
            else:
                shift_cache[year] = ensure_raw(year)
        return shift_cache[year]

    try:
        shape = tuple(template.sizes[dim] for dim in SPATIAL_DIMS)
        if closure_state["state"] is None:
            closure_state["state"] = init_closure(shape, calendar_years, climate_years, args, set(args.variables))
        else:
            if not np.array_equal(closure_state["state"]["calendar_years"], calendar_years):
                raise ValueError(f"{var}: calendar target years differ from previous variable")
            if not np.array_equal(closure_state["state"]["climate_years"], climate_years):
                raise ValueError(f"{var}: climate target years differ from previous variable")

        monthly_times = output_times(calendar_years, "Monthly")
        seasonal_times = output_times(climate_years, "Seasonal")
        monthly_path = output_path(template, var, monthly_times, "Monthly", args.monthly_dir)
        seasonal_path = output_path(template, var, seasonal_times, "Seasonal", args.seasonal_dir)

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

        for idx, year in enumerate(calendar_years):
            total_block, between_block, within_block = monthly_yearly_block(
                ensure_shift(int(year)),
                ensure_shift(int(year + 1)),
                int(year),
            )
            if not np.isfinite(total_block).all():
                raise RuntimeError(f"{var}: monthly total block contains non-finite values for {year}")
            if not np.isfinite(between_block).all():
                raise RuntimeError(f"{var}: monthly between block contains non-finite values for {year}")
            if not np.isfinite(within_block).all():
                raise RuntimeError(f"{var}: monthly within block contains non-finite values for {year}")
            err = float(np.max(np.abs(total_block - (between_block + within_block))))
            if err > TOL:
                raise RuntimeError(f"{var}: monthly block identity failed for {year} (max={err:.3e})")

            monthly_writer.write_block(idx * 12, total_block, between_block, within_block)
            if (idx + 1) % args.sync_every == 0:
                monthly_writer.sync()
            if idx < 3 or (idx + 1) % 8 == 0 or idx + 1 == len(calendar_years):
                logging.info("  monthly target year %d (%d/%d)", int(year), idx + 1, len(calendar_years))

            if closure_state["state"]["sum_monthly"] is not None and var in closure_state["state"]["components"]:
                closure_state["state"]["sum_monthly"][idx] += total_block
            if closure_state["state"]["tendtot_monthly"] is not None and var == args.check_total_var:
                closure_state["state"]["tendtot_monthly"][idx] = total_block

            raw_cache.pop(int(year - 1), None)
            shift_cache.pop(int(year - 1), None)

        raw_cache.clear()
        shift_cache.clear()

        for idx, climate_year in enumerate(climate_years):
            total_block, between_block, within_block = seasonal_yearly_block(
                ensure_shift(int(climate_year - 1)),
                ensure_shift(int(climate_year)),
                ensure_shift(int(climate_year + 1)),
                int(climate_year),
            )
            if not np.isfinite(total_block).all():
                raise RuntimeError(f"{var}: seasonal total block contains non-finite values for climate year {climate_year}")
            if not np.isfinite(between_block).all():
                raise RuntimeError(f"{var}: seasonal between block contains non-finite values for climate year {climate_year}")
            if not np.isfinite(within_block).all():
                raise RuntimeError(f"{var}: seasonal within block contains non-finite values for climate year {climate_year}")
            err = float(np.max(np.abs(total_block - (between_block + within_block))))
            if err > TOL:
                raise RuntimeError(f"{var}: seasonal block identity failed for climate year {climate_year} (max={err:.3e})")

            seasonal_writer.write_block(idx * 4, total_block, between_block, within_block)
            if (idx + 1) % args.sync_every == 0:
                seasonal_writer.sync()
            if idx < 2 or (idx + 1) % 8 == 0 or idx + 1 == len(climate_years):
                logging.info("  seasonal climate year %d (%d/%d)", int(climate_year), idx + 1, len(climate_years))

            if closure_state["state"]["sum_seasonal"] is not None and var in closure_state["state"]["components"]:
                closure_state["state"]["sum_seasonal"][idx] += total_block
            if closure_state["state"]["tendtot_seasonal"] is not None and var == args.check_total_var:
                closure_state["state"]["tendtot_seasonal"][idx] = total_block

            raw_cache.pop(int(climate_year - 2), None)
            shift_cache.pop(int(climate_year - 2), None)

        monthly_writer.sync()
        seasonal_writer.sync()
        logging.info("  wrote %s", monthly_path)
        logging.info("  wrote %s", seasonal_path)
    except Exception:
        if monthly_writer is not None:
            monthly_writer.close(ok=False)
        if seasonal_writer is not None:
            seasonal_writer.close(ok=False)
        raise
    else:
        monthly_writer.close(ok=True)
        seasonal_writer.close(ok=True)
    finally:
        template.close()


def main():
    args = parse_args()
    bad = sorted(set(args.variables) - set(ALLOWED_VARS))
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
    logging.info("exact B yearly writer active")

    closure_state = {"state": None}
    for var in args.variables:
        process(var, args, closure_state)
    verify_closure(closure_state["state"], args)


if __name__ == "__main__":
    main()
