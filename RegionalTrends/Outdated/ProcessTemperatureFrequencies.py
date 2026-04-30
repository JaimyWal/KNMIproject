#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr


SEASONS = (
    ("DJF", (12, 1, 2), 1),
    ("MAM", (3, 4, 5), 4),
    ("JJA", (6, 7, 8), 7),
    ("SON", (9, 10, 11), 10),
)
COMP_LEVEL = 4
YEAR_RE = re.compile(r"-(\d{4})\.")

KNMI_ROOT = Path("/nobackup/users/walj/knmi")
KNMI_RAW_DIR = KNMI_ROOT / "Raw"
KNMI_HOURLY_DIR = KNMI_ROOT / "Hourly"
KNMI_THREEHOURLY_DIR = KNMI_ROOT / "Threehourly"
KNMI_DAILY_DIR = KNMI_ROOT / "Daily"
KNMI_MONTHLY_DIR = KNMI_ROOT / "Monthly"
KNMI_SEASONAL_DIR = KNMI_ROOT / "Seasonal"

ERA5_ROOT = Path("/nobackup/users/walj/era5")
ERA5_THREEHOURLY_DIR = ERA5_ROOT / "Threehourly"
ERA5_DAILY_DIR = ERA5_ROOT / "Daily"
ERA5_MONTHLY_DIR = ERA5_ROOT / "Monthly"
ERA5_SEASONAL_DIR = ERA5_ROOT / "Seasonal"

RACMO_V7_ROOT = Path("/nobackup_1/users/walj/racmo24")
RACMO_V5_ROOT = Path("/nobackup/users/walj/racmo24")

KNMI_STATIONS = {
    235: {"tag": "Kooy", "name": "De Kooy", "lat": 52.924172463538795, "lon": 4.779336630180403},
    260: {"tag": "Bilt", "name": "De Bilt", "lat": 52.098872302947974, "lon": 5.179442289152804},
    280: {"tag": "Eelde", "name": "Eelde", "lat": 53.12385846866912, "lon": 6.584799434350561},
    310: {"tag": "Vlissingen", "name": "Vlissingen", "lat": 51.441328455552586, "lon": 3.5958610840956884},
    380: {"tag": "Maastricht", "name": "Maastricht", "lat": 50.90548320406765, "lon": 5.761839846736004},
}

YEAR_META: dict[int, dict[str, object]] = {}


@dataclass(frozen=True)
class TemplateInfo:
    global_attrs: dict[str, object]
    var_attrs: dict[str, object]
    aux_vars: dict[str, xr.DataArray]
    space_dims: tuple[str, ...]


@dataclass(frozen=True)
class GridSpec:
    name: str
    source_root: Path
    input_dir: Path
    threehourly_dir: Path
    daily_dir: Path
    monthly_dir: Path
    seasonal_dir: Path
    input_glob: str
    raw_output_glob: str
    input_var: str
    input_time_name: str
    domain: str
    experiment: str
    raw_output_pattern: str
    daily_output_pattern: str
    monthly_output_pattern: str
    seasonal_output_pattern: str
    monthlycont_output_pattern: str
    seasonalcont_output_pattern: str
    monthlyb_output_pattern: str
    seasonalb_output_pattern: str


ERA5_SPEC = GridSpec(
    name="ERA5",
    source_root=ERA5_ROOT,
    input_dir=ERA5_THREEHOURLY_DIR,
    threehourly_dir=ERA5_THREEHOURLY_DIR,
    daily_dir=ERA5_DAILY_DIR,
    monthly_dir=ERA5_MONTHLY_DIR,
    seasonal_dir=ERA5_SEASONAL_DIR,
    input_glob="era5_t2m_europe_3hourly_*.nc",
    raw_output_glob="temp.ERA5-*.EU.3H.nc",
    input_var="t2m",
    input_time_name="valid_time",
    domain="EU",
    experiment="ERA5",
    raw_output_pattern="temp.ERA5-{year:04d}.EU.3H.nc",
    daily_output_pattern="temp_dailyA_EU_ERA5_{start}_{end}.nc",
    monthly_output_pattern="temp_monthlyA_EU_ERA5_{start}_{end}.nc",
    seasonal_output_pattern="temp_seasonalA_EU_ERA5_{start}_{end}.nc",
    monthlycont_output_pattern="temp_monthlycont_EU_ERA5_{start}_{end}.nc",
    seasonalcont_output_pattern="temp_seasonalcont_EU_ERA5_{start}_{end}.nc",
    monthlyb_output_pattern="temp_monthlyyearlyB_EU_ERA5_{start}_{end}.nc",
    seasonalb_output_pattern="temp_seasonalyearlyB_EU_ERA5_{start}_{end}.nc",
)

def build_racmo_spec(name: str, source_root: Path, experiment: str) -> GridSpec:
    hourly_dir = source_root / "Hourly"
    threehourly_dir = source_root / "Threehourly"
    daily_dir = source_root / "Daily"
    monthly_dir = source_root / "Monthly"
    seasonal_dir = source_root / "Seasonal"
    return GridSpec(
        name=name,
        source_root=source_root,
        input_dir=hourly_dir,
        threehourly_dir=threehourly_dir,
        daily_dir=daily_dir,
        monthly_dir=monthly_dir,
        seasonal_dir=seasonal_dir,
        input_glob=f"tas.KNMI-*.KEXT12.{experiment}.1H.nc",
        raw_output_glob=f"temp.KNMI-*.KEXT12.{experiment}.3H.nc",
        input_var="tas",
        input_time_name="time",
        domain="KEXT12",
        experiment=experiment,
        raw_output_pattern=f"temp.KNMI-{{year:04d}}.KEXT12.{experiment}.3H.nc",
        daily_output_pattern=f"temp_dailyA_KEXT12_{experiment}_{{start}}_{{end}}.nc",
        monthly_output_pattern=f"temp_monthlyA_KEXT12_{experiment}_{{start}}_{{end}}.nc",
        seasonal_output_pattern=f"temp_seasonalA_KEXT12_{experiment}_{{start}}_{{end}}.nc",
        monthlycont_output_pattern=f"temp_monthlycont_KEXT12_{experiment}_{{start}}_{{end}}.nc",
        seasonalcont_output_pattern=f"temp_seasonalcont_KEXT12_{experiment}_{{start}}_{{end}}.nc",
        monthlyb_output_pattern=f"temp_monthlyyearlyB_KEXT12_{experiment}_{{start}}_{{end}}.nc",
        seasonalb_output_pattern=f"temp_seasonalyearlyB_KEXT12_{experiment}_{{start}}_{{end}}.nc",
    )


RACMO_V7_SPEC = build_racmo_spec(
    name="RACMO24",
    source_root=RACMO_V7_ROOT,
    experiment="RACMO24p1v7_FINAL_enbud_fix",
)

RACMO_V5_SPEC = build_racmo_spec(
    name="RACMO2.4p1_v5",
    source_root=RACMO_V5_ROOT,
    experiment="RACMO2.4p1_v5_trends_bugfixes",
)


def now_utc_string() -> str:
    return datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S %Y")


def full_interval_time(time_coord: xr.DataArray | pd.DatetimeIndex, interval: str) -> pd.DatetimeIndex:
    time_index = pd.DatetimeIndex(time_coord.values if hasattr(time_coord, "values") else time_coord)
    if time_index.size == 0:
        return pd.DatetimeIndex([])

    start_year = int(time_index.year.min())
    end_year = int(time_index.year.max())

    if interval == "Daily":
        return pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    if interval == "Monthly":
        return pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    if interval == "Seasonal":
        return pd.DatetimeIndex(
            [pd.Timestamp(year=year, month=month, day=1) for year in range(start_year, end_year + 1) for _, _, month in SEASONS]
        )
    raise ValueError(f"Unsupported interval: {interval}")


def next_time_values(time_index: pd.DatetimeIndex) -> np.ndarray:
    values = np.empty(time_index.size, dtype="datetime64[ns]")
    values[:-1] = time_index[1:].values
    values[-1] = np.datetime64("NaT")
    return values


def expected_steps(year: int) -> int:
    return (366 if pd.Timestamp(year=year, month=1, day=1).is_leap_year else 365) * 8


def year_meta(year: int) -> dict[str, object]:
    if year not in YEAR_META:
        n_steps = expected_steps(year)
        time_index = pd.date_range(f"{year}-01-01 00:00:00", periods=n_steps, freq="3h")
        months = time_index.month.to_numpy()
        month_steps = np.array([(months == month).sum() for month in range(1, 13)], dtype=int)
        month_offsets = np.concatenate(([0], np.cumsum(month_steps)))
        month_slices = {month: slice(month_offsets[month - 1], month_offsets[month]) for month in range(1, 13)}
        YEAR_META[year] = {
            "n_steps": n_steps,
            "rise": np.arange(1, n_steps + 1, dtype="float64") / n_steps,
            "desc": np.arange(n_steps - 1, -1, -1, dtype="float64") / n_steps,
            "month_slices": month_slices,
        }
    return YEAR_META[year]


def find_aux_vars(ds: xr.Dataset, var_name: str, time_name: str) -> tuple[tuple[str, ...], dict[str, xr.DataArray]]:
    space_dims = tuple(dim for dim in ds[var_name].dims if dim != time_name)
    names: list[str] = []

    for dim in space_dims:
        if dim in ds.variables and dim not in names:
            names.append(dim)

    for candidate in ("lat", "lon", "latitude", "longitude", "rlat", "rlon", "height"):
        if candidate in ds.variables and candidate not in names:
            if time_name not in ds[candidate].dims:
                names.append(candidate)

    aux_vars = {name: ds[name].copy(deep=True) for name in names}
    return space_dims, aux_vars


def make_template_info(ds: xr.Dataset, var_name: str, time_name: str, attr_overrides: dict[str, object] | None = None) -> TemplateInfo:
    global_attrs = dict(ds.attrs)
    if attr_overrides:
        global_attrs.update(attr_overrides)
    space_dims, aux_vars = find_aux_vars(ds, var_name, time_name)
    return TemplateInfo(global_attrs=global_attrs, var_attrs=dict(ds[var_name].attrs), aux_vars=aux_vars, space_dims=space_dims)


def build_station_template(station_id: int, station_meta: dict[str, object]) -> TemplateInfo:
    aux_vars = {
        "lat": xr.DataArray(np.asarray(station_meta["lat"], dtype="float64"), attrs={"units": "degrees_north", "long_name": "latitude"}),
        "lon": xr.DataArray(np.asarray(station_meta["lon"], dtype="float64"), attrs={"units": "degrees_east", "long_name": "longitude"}),
    }
    return TemplateInfo(
        global_attrs={
            "Conventions": "CF-1.8",
            "source": "KNMI",
            "institution": "Royal Netherlands Meteorological Institute (KNMI)",
            "station_id": int(station_id),
            "station_name": str(station_meta["name"]),
            "title": "Air temperature",
            "comment": "Hourly station temperature derived from KNMI raw hourly observations.",
        },
        var_attrs={
            "standard_name": "air_temperature",
            "long_name": "Air temperature",
            "units": "degrees_Celsius",
        },
        aux_vars=aux_vars,
        space_dims=(),
    )


def coords_string(aux_vars: dict[str, xr.DataArray], include_interval_next: bool = False, include_season_year: bool = False) -> str:
    parts = []
    for name in ("lat", "lon", "latitude", "longitude"):
        if name in aux_vars:
            parts.append(name)
    if include_interval_next:
        parts.append("interval_next")
    if include_season_year:
        parts.append("season_year")
    return " ".join(parts)


def temp_attributes(template: TemplateInfo, frequency: str, coordinates_attr: str) -> dict[str, object]:
    attrs = dict(template.var_attrs)
    attrs["long_name"] = "Temperature"
    if frequency == "Hourly":
        attrs["cell_methods"] = "time: instantaneous values"
    elif frequency == "Threehourly":
        attrs["cell_methods"] = "time: 3-hourly instantaneous values"
    else:
        attrs["cell_methods"] = "time: mean of 3-hourly instantaneous values"
    if coordinates_attr:
        attrs["coordinates"] = coordinates_attr
    return attrs


def tendtot_attributes(template: TemplateInfo, frequency: str, coordinates_attr: str) -> dict[str, object]:
    attrs = {
        "long_name": "Temperature difference to next interval",
        "units": template.var_attrs.get("units", ""),
        "cell_methods": f"time: difference to next {frequency.lower()} interval",
    }
    if coordinates_attr:
        attrs["coordinates"] = coordinates_attr
    return attrs


def build_output_comment(template: TemplateInfo, message: str) -> str:
    base = str(template.global_attrs.get("comment", "")).strip()
    if not base:
        return message
    return f"{base} | {message}"


def attach_aux_vars(ds_out: xr.Dataset, template: TemplateInfo) -> xr.Dataset:
    coords = {name: var for name, var in template.aux_vars.items()}
    return ds_out.assign_coords(coords)


def build_series_dataset(
    temp_da: xr.DataArray,
    template: TemplateInfo,
    frequency: str,
    title: str,
    comment: str,
    include_tendtot: bool,
    include_interval_next: bool,
    include_season_year: bool,
) -> xr.Dataset:
    time_index = pd.DatetimeIndex(temp_da["time"].values)
    coord_string = coords_string(template.aux_vars, include_interval_next=include_interval_next, include_season_year=include_season_year)

    ds_out = xr.Dataset(coords={"time": temp_da["time"]})
    ds_out = attach_aux_vars(ds_out, template)
    ds_out["temp"] = temp_da.astype("float32")
    ds_out["temp"].attrs = temp_attributes(template, frequency, coord_string)

    if include_tendtot:
        ds_out["tendtot"] = (temp_da.shift(time=-1) - temp_da).astype("float32")
        ds_out["tendtot"].attrs = tendtot_attributes(template, frequency, coord_string)

    if include_interval_next:
        ds_out["interval_next"] = xr.DataArray(
            next_time_values(time_index),
            dims=("time",),
            coords={"time": temp_da["time"]},
        )
        ds_out["interval_next"].attrs = {"long_name": "start time of next interval"}

    if include_season_year:
        ds_out["season_year"] = xr.DataArray(time_index.year.astype("int32"), dims=("time",), coords={"time": temp_da["time"]})
        ds_out["season_year"].attrs = {
            "long_name": "climate year",
            "comment": "Climate year runs from December of the previous calendar year through November.",
        }

    ds_out.attrs = dict(template.global_attrs)
    ds_out.attrs["CreationDate"] = now_utc_string()
    ds_out.attrs["title"] = title
    ds_out.attrs["comment"] = comment
    return ds_out


def write_dataset(ds_out: xr.Dataset, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        logging.info("skip existing %s", out_path)
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    encoding: dict[str, dict[str, object]] = {}
    time_chunk = min(ds_out.sizes.get("time", 1), 124)
    for name, var in ds_out.data_vars.items():
        if np.issubdtype(var.dtype, np.floating):
            if var.ndim == 1:
                chunksizes = (time_chunk,)
            else:
                chunksizes = (time_chunk,) + tuple(int(size) for size in var.shape[1:])
            encoding[name] = {
                "zlib": True,
                "complevel": COMP_LEVEL,
                "shuffle": True,
                "chunksizes": chunksizes,
            }

    ds_out.to_netcdf(tmp_path, engine="netcdf4", encoding=encoding, unlimited_dims=["time"])
    tmp_path.replace(out_path)
    logging.info("wrote %s", out_path)


def complete_resample_mean(temp_da: xr.DataArray, rule: str, interval: str) -> xr.DataArray:
    averaged = temp_da.resample(time=rule).mean(skipna=False)
    count = temp_da.resample(time=rule).count()
    if interval == "Daily":
        expected = xr.DataArray(np.full(averaged.sizes["time"], 8, dtype="int16"), dims=("time",), coords={"time": averaged["time"]})
    elif interval == "Monthly":
        expected = (averaged["time"].dt.days_in_month * 8).astype("int32")
    else:
        raise ValueError(f"Unsupported interval for mean: {interval}")
    averaged = averaged.where(count == expected)
    return averaged.reindex(time=full_interval_time(temp_da["time"], interval=interval))


def seasonal_average(monthly_da: xr.DataArray) -> xr.DataArray:
    full_time = full_interval_time(monthly_da["time"], interval="Seasonal")
    seasonal_parts = []

    for _, months, anchor_month in SEASONS:
        month_arr = np.asarray(months, dtype=int)
        output_time = full_time[full_time.month == anchor_month]
        output_years = output_time.year.values

        season_da = monthly_da.where(monthly_da["time"].dt.month.isin(month_arr), drop=True)
        month_vals = season_da["time"].dt.month
        year_vals = season_da["time"].dt.year

        if month_arr[0] > month_arr[-1]:
            season_year = xr.where(month_vals >= month_arr[0], year_vals + 1, year_vals)
        else:
            season_year = year_vals

        season_da = season_da.assign_coords(season_year=("time", season_year.values))
        season_da = season_da.where(season_da["season_year"].isin(output_years), drop=True)

        year_months = pd.DataFrame(
            {"season_year": season_da["season_year"].values, "month": season_da["time"].dt.month.values}
        )
        complete_years = year_months.groupby("season_year")["month"].nunique()
        complete_years = complete_years[complete_years == len(month_arr)].index.values

        month_weights = season_da["time"].dt.days_in_month.astype("float32")
        month_weights = month_weights.assign_coords(season_year=season_da["season_year"])

        weighted_sum = (season_da * month_weights).groupby("season_year").sum("time", skipna=False)
        weight_sum = month_weights.groupby("season_year").sum("time")
        season_yearly = (weighted_sum / weight_sum).reindex(season_year=output_years).astype("float32")

        complete_mask = xr.DataArray(
            np.isin(output_years, complete_years),
            dims=("season_year",),
            coords={"season_year": output_years},
        )
        season_yearly = season_yearly.where(complete_mask)
        season_yearly = season_yearly.rename({"season_year": "time"}).assign_coords(
            time=("time", output_time),
        )
        seasonal_parts.append(season_yearly)

    return xr.concat(seasonal_parts, dim="time").sortby("time").astype("float32")


def output_path_station(directory: Path, pattern: str, station_tag: str, start: str, end: str) -> Path:
    return directory / pattern.format(station=station_tag, start=start, end=end)


def output_path_grid(directory: Path, pattern: str, start: str, end: str) -> Path:
    return directory / pattern.format(start=start, end=end)


def monthly_stamp(time_index: pd.DatetimeIndex) -> tuple[str, str]:
    return time_index[0].strftime("%Y%m"), time_index[-1].strftime("%Y%m")


def daily_stamp(time_index: pd.DatetimeIndex) -> tuple[str, str]:
    return time_index[0].strftime("%Y%m%d"), time_index[-1].strftime("%Y%m%d")


def read_knmi_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            text = fh.read().decode("latin1")

    lines = text.splitlines()
    header_idx = next(idx for idx, line in enumerate(lines) if line.startswith("# STN"))
    header = [col.strip() for col in lines[header_idx][2:].split(",")]
    data = "\n".join(lines[header_idx + 1 :])

    return pd.read_csv(
        io.StringIO(data),
        header=None,
        names=header,
        usecols=["STN", "YYYYMMDD", "HH", "T"],
        na_values=["", "     "],
        skipinitialspace=True,
        low_memory=False,
    )


def knmi_station_series(station_id: int) -> pd.Series:
    frames = [read_knmi_zip(path) for path in sorted(KNMI_RAW_DIR.glob(f"uurgeg_{station_id}_*.zip"))]
    if not frames:
        raise FileNotFoundError(f"No KNMI raw files found for station {station_id}")

    df = pd.concat(frames, ignore_index=True)
    df["STN"] = pd.to_numeric(df["STN"], errors="coerce")
    df = df[df["STN"] == station_id].copy()
    df["YYYYMMDD"] = pd.to_numeric(df["YYYYMMDD"], errors="coerce").astype("Int64")
    df["HH"] = pd.to_numeric(df["HH"], errors="coerce").astype("Int64")
    df["T"] = pd.to_numeric(df["T"], errors="coerce") * 0.1

    base_time = pd.to_datetime(df["YYYYMMDD"].astype(str), format="%Y%m%d")
    hours = df["HH"].to_numpy(dtype="float64")
    times = base_time + pd.to_timedelta(np.where(hours == 24, 0, hours), unit="h")
    times = times + pd.to_timedelta(np.where(hours == 24, 1, 0), unit="D")

    series = pd.Series(df["T"].to_numpy(dtype="float32"), index=pd.DatetimeIndex(times), name="temp")
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]

    start = series.index.min().floor("D")
    end = series.index.max().ceil("h")
    full_index = pd.date_range(start, end, freq="h")
    return series.reindex(full_index)


def build_hourly_station_dataset(series: pd.Series, template: TemplateInfo) -> xr.Dataset:
    temp_da = xr.DataArray(series.to_numpy(dtype="float32"), dims=("time",), coords={"time": series.index}, name="temp")
    return build_series_dataset(
        temp_da=temp_da,
        template=template,
        frequency="Hourly",
        title="Temperature",
        comment=build_output_comment(template, "clean hourly temperature values."),
        include_tendtot=False,
        include_interval_next=False,
        include_season_year=False,
    )


def build_threehourly_station_dataset(series: pd.Series, template: TemplateInfo) -> xr.Dataset:
    series_3h = series[series.index.hour % 3 == 0]
    temp_da = xr.DataArray(series_3h.to_numpy(dtype="float32"), dims=("time",), coords={"time": series_3h.index}, name="temp")
    return build_series_dataset(
        temp_da=temp_da,
        template=template,
        frequency="Threehourly",
        title="Temperature",
        comment=build_output_comment(template, "three-hourly instantaneous temperature values sampled from the hourly record."),
        include_tendtot=True,
        include_interval_next=True,
        include_season_year=False,
    )


def normalized_source(source_path: Path, var_name: str, time_name: str, attr_overrides: dict[str, object] | None = None) -> tuple[xr.DataArray, TemplateInfo]:
    with xr.open_dataset(source_path) as ds:
        template = make_template_info(ds, var_name=var_name, time_name=time_name, attr_overrides=attr_overrides)
        temp_da = ds[var_name].rename({time_name: "time"}).astype("float32").load()
    temp_da = temp_da.rename("temp").sortby("time")
    return temp_da, template


def raw_grid_output_dataset(
    temp_da: xr.DataArray,
    next_first: xr.DataArray | None,
    template: TemplateInfo,
    source_label: str,
    comment_suffix: str,
) -> xr.Dataset:
    arr = np.asarray(temp_da.values, dtype="float32")
    shifted = np.empty_like(arr)
    shifted[:-1] = arr[1:]
    if next_first is None:
        shifted[-1] = np.nan
    else:
        shifted[-1] = np.asarray(next_first.values, dtype="float32")

    tendtot = shifted - arr
    time_index = pd.DatetimeIndex(temp_da["time"].values)
    coord_string = coords_string(template.aux_vars, include_interval_next=True)

    ds_out = xr.Dataset(coords={"time": temp_da["time"]})
    ds_out = attach_aux_vars(ds_out, template)
    ds_out["temp"] = temp_da.astype("float32")
    ds_out["temp"].attrs = temp_attributes(template, "Threehourly", coord_string)
    ds_out["tendtot"] = xr.DataArray(tendtot, dims=temp_da.dims, coords=temp_da.coords)
    ds_out["tendtot"].attrs = tendtot_attributes(template, "Threehourly", coord_string)
    ds_out["interval_next"] = xr.DataArray(
        next_time_values(time_index),
        dims=("time",),
        coords={"time": temp_da["time"]},
    )
    ds_out["interval_next"].attrs = {"long_name": "start time of next interval"}
    ds_out.attrs = dict(template.global_attrs)
    ds_out.attrs["CreationDate"] = now_utc_string()
    ds_out.attrs["title"] = "Temperature"
    ds_out.attrs["comment"] = build_output_comment(template, comment_suffix)
    ds_out.attrs["derived_source"] = source_label
    return ds_out


def existing_year_file_map(directory: Path, pattern: str) -> dict[int, Path]:
    files = {}
    for path in sorted(directory.glob(pattern)):
        match = YEAR_RE.search(path.name)
        if match is None:
            continue
        files[int(match.group(1))] = path
    return files


def process_era5_threehourly() -> dict[int, Path]:
    input_paths = {int(path.stem.rsplit("_", 1)[-1]): path for path in sorted(ERA5_SPEC.input_dir.glob(ERA5_SPEC.input_glob))}
    output_paths: dict[int, Path] = {}

    for year in sorted(input_paths):
        out_path = ERA5_SPEC.threehourly_dir / ERA5_SPEC.raw_output_pattern.format(year=year)
        output_paths[year] = out_path
        if out_path.exists():
            logging.info("skip existing %s", out_path)
            continue

        temp_da, template = normalized_source(
            input_paths[year],
            var_name=ERA5_SPEC.input_var,
            time_name=ERA5_SPEC.input_time_name,
            attr_overrides={"source": "ERA5", "Domain": ERA5_SPEC.domain, "Experiment": ERA5_SPEC.experiment},
        )
        next_first = None
        if (year + 1) in input_paths:
            next_first, _ = normalized_source(
                input_paths[year + 1],
                var_name=ERA5_SPEC.input_var,
                time_name=ERA5_SPEC.input_time_name,
                attr_overrides={"source": "ERA5", "Domain": ERA5_SPEC.domain, "Experiment": ERA5_SPEC.experiment},
            )
            next_first = next_first.isel(time=0)

        ds_out = raw_grid_output_dataset(
            temp_da=temp_da,
            next_first=next_first,
            template=template,
            source_label="ERA5",
            comment_suffix="clean three-hourly temperature values with interval differences.",
        )
        write_dataset(ds_out, out_path)

    return output_paths


def process_racmo_threehourly(spec: GridSpec) -> dict[int, Path]:
    input_paths = existing_year_file_map(spec.input_dir, spec.input_glob)
    output_paths: dict[int, Path] = {}

    for year in sorted(input_paths):
        out_path = spec.threehourly_dir / spec.raw_output_pattern.format(year=year)
        output_paths[year] = out_path
        if out_path.exists():
            logging.info("skip existing %s", out_path)
            continue

        temp_da, template = normalized_source(
            input_paths[year],
            var_name=spec.input_var,
            time_name=spec.input_time_name,
            attr_overrides={"Domain": spec.domain, "Experiment": spec.experiment},
        )
        mask = pd.DatetimeIndex(temp_da["time"].values).hour % 3 == 0
        temp_da = temp_da.isel(time=np.where(mask)[0])

        next_first = None
        if (year + 1) in input_paths:
            next_year_da, _ = normalized_source(
                input_paths[year + 1],
                var_name=spec.input_var,
                time_name=spec.input_time_name,
                attr_overrides={"Domain": spec.domain, "Experiment": spec.experiment},
            )
            next_mask = pd.DatetimeIndex(next_year_da["time"].values).hour % 3 == 0
            next_year_da = next_year_da.isel(time=np.where(next_mask)[0])
            next_first = next_year_da.isel(time=0)

        ds_out = raw_grid_output_dataset(
            temp_da=temp_da,
            next_first=next_first,
            template=template,
            source_label=f"{spec.experiment} hourly tas",
            comment_suffix="clean three-hourly instantaneous temperature values sampled from the hourly record, with interval differences.",
        )
        write_dataset(ds_out, out_path)

    return output_paths


def build_frequency_products_from_threehourly(
    temp_da: xr.DataArray,
    template: TemplateInfo,
    daily_path: Path,
    monthly_path: Path,
    seasonal_path: Path,
) -> None:
    temp_da = temp_da.sortby("time")
    daily_temp = complete_resample_mean(temp_da, rule="1D", interval="Daily")
    monthly_temp = complete_resample_mean(temp_da, rule="MS", interval="Monthly")
    seasonal_temp = seasonal_average(monthly_temp)

    daily_ds = build_series_dataset(
        temp_da=daily_temp,
        template=template,
        frequency="Daily",
        title="Temperature",
        comment=build_output_comment(template, "daily averages computed from three-hourly instantaneous temperature values."),
        include_tendtot=True,
        include_interval_next=True,
        include_season_year=False,
    )
    write_dataset(daily_ds, daily_path)

    monthly_ds = build_series_dataset(
        temp_da=monthly_temp,
        template=template,
        frequency="Monthly",
        title="Temperature",
        comment=build_output_comment(template, "monthly averages computed from three-hourly instantaneous temperature values."),
        include_tendtot=True,
        include_interval_next=True,
        include_season_year=False,
    )
    write_dataset(monthly_ds, monthly_path)

    seasonal_ds = build_series_dataset(
        temp_da=seasonal_temp,
        template=template,
        frequency="Seasonal",
        title="Temperature",
        comment=build_output_comment(template, "seasonal averages computed from three-hourly instantaneous temperature values."),
        include_tendtot=True,
        include_interval_next=True,
        include_season_year=True,
    )
    write_dataset(seasonal_ds, seasonal_path)


def process_grid_frequency_products(spec: GridSpec, raw_paths: dict[int, Path]) -> None:
    if not raw_paths:
        logging.warning("no raw three-hourly files found for %s", spec.name)
        return

    first_path = raw_paths[min(raw_paths)]
    with xr.open_dataset(first_path) as ds:
        template = make_template_info(ds, var_name="temp", time_name="time")

    paths = [raw_paths[year] for year in sorted(raw_paths)]
    ds = xr.open_mfdataset(
        paths,
        combine="nested",
        concat_dim="time",
        chunks={"time": 248},
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=False,
        preprocess=lambda x: x[["temp"]],
    ).sortby("time")

    try:
        temp_da = ds["temp"].astype("float32")
        daily_time = full_interval_time(temp_da["time"], interval="Daily")
        monthly_time = full_interval_time(temp_da["time"], interval="Monthly")
        seasonal_time = full_interval_time(temp_da["time"], interval="Seasonal")
        daily_start, daily_end = daily_stamp(daily_time)
        monthly_start, monthly_end = monthly_stamp(monthly_time)
        seasonal_start, seasonal_end = monthly_stamp(seasonal_time)

        build_frequency_products_from_threehourly(
            temp_da=temp_da,
            template=template,
            daily_path=output_path_grid(spec.daily_dir, spec.daily_output_pattern, daily_start, daily_end),
            monthly_path=output_path_grid(spec.monthly_dir, spec.monthly_output_pattern, monthly_start, monthly_end),
            seasonal_path=output_path_grid(spec.seasonal_dir, spec.seasonal_output_pattern, seasonal_start, seasonal_end),
        )
    finally:
        ds.close()


def process_knmi_station_frequency_products(station_id: int, station_meta: dict[str, object], threehourly_path: Path) -> None:
    with xr.open_dataset(threehourly_path) as ds:
        template = build_station_template(station_id, station_meta)
        temp_da = ds["temp"].astype("float32").load()

    daily_time = full_interval_time(temp_da["time"], interval="Daily")
    monthly_time = full_interval_time(temp_da["time"], interval="Monthly")
    seasonal_time = full_interval_time(temp_da["time"], interval="Seasonal")
    daily_start, daily_end = daily_stamp(daily_time)
    monthly_start, monthly_end = monthly_stamp(monthly_time)
    seasonal_start, seasonal_end = monthly_stamp(seasonal_time)

    build_frequency_products_from_threehourly(
        temp_da=temp_da,
        template=template,
        daily_path=output_path_station(
            KNMI_DAILY_DIR,
            "temp_dailyA_KNMI_{station}_{start}_{end}.nc",
            station_meta["tag"],
            daily_start,
            daily_end,
        ),
        monthly_path=output_path_station(
            KNMI_MONTHLY_DIR,
            "temp_monthlyA_KNMI_{station}_{start}_{end}.nc",
            station_meta["tag"],
            monthly_start,
            monthly_end,
        ),
        seasonal_path=output_path_station(
            KNMI_SEASONAL_DIR,
            "temp_seasonalA_KNMI_{station}_{start}_{end}.nc",
            station_meta["tag"],
            seasonal_start,
            seasonal_end,
        ),
    )


def month_segments_for_year(year: int, month: int) -> list[tuple[int, slice]]:
    return [(int(year), year_meta(int(year))["month_slices"][int(month)])]


def season_segments_for_climate_year(climate_year: int, season_name: str) -> list[tuple[int, slice]]:
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
    raise ValueError(f"Unknown season {season_name}")


def build_year_offsets(years: dict[int, np.ndarray]) -> dict[int, int]:
    offsets: dict[int, int] = {}
    offset = 0
    for year in sorted(years):
        offsets[year] = offset
        offset += int(year_meta(year)["n_steps"])
    return offsets


def interval_info(segments: list[tuple[int, slice]], year_offsets: dict[int, int]) -> dict[str, object]:
    start = year_offsets[segments[0][0]] + segments[0][1].start
    stop = year_offsets[segments[-1][0]] + segments[-1][1].stop
    n_steps = int(sum(segment.stop - segment.start for _, segment in segments))
    return {"segments": segments, "start": start, "stop": stop, "N": n_steps}


def concat_interval(data_cache: dict[int, np.ndarray], segments: list[tuple[int, slice]]) -> np.ndarray:
    pieces = [np.asarray(data_cache[year][segment], dtype="float64") for year, segment in segments]
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)


def weighted_sum_strict(weights: np.ndarray, data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype="float64")
    weights = np.asarray(weights, dtype="float64")
    shape = (weights.size,) + (1,) * (data.ndim - 1)
    zero_mask = weights.reshape(shape) == 0.0
    valid_mask = np.isfinite(data)
    ok = np.all(valid_mask | zero_mask, axis=0)
    safe = np.where(valid_mask, data, 0.0)
    out = np.tensordot(weights, safe, axes=(0, 0))
    return np.where(ok, out, np.nan)


def weighted_interval(interval: np.ndarray) -> np.ndarray:
    interval = np.asarray(interval, dtype="float64")
    n_steps = interval.shape[0]
    weights = np.arange(n_steps - 1, -1, -1, dtype="float64") / n_steps
    return weighted_sum_strict(weights, interval)


def sum_global_slice(data_cache: dict[int, np.ndarray], year_offsets: dict[int, int], start: int, stop: int) -> np.ndarray:
    sample = next(iter(data_cache.values()))
    total = np.zeros(sample.shape[1:], dtype="float64")
    valid = np.ones(sample.shape[1:], dtype=bool)

    for year in sorted(year_offsets):
        year_start = year_offsets[year]
        year_stop = year_start + int(year_meta(year)["n_steps"])
        lo = max(start, year_start)
        hi = min(stop, year_stop)
        if lo >= hi:
            continue
        part = np.asarray(data_cache[year][lo - year_start : hi - year_start], dtype="float64")
        valid &= np.all(np.isfinite(part), axis=0)
        total += np.where(np.isfinite(part), part, 0.0).sum(axis=0, dtype="float64")

    return np.where(valid, total, np.nan)


def monthly_cont_block(shift_curr: np.ndarray, shift_next: np.ndarray, year: int) -> np.ndarray:
    meta_curr = year_meta(year)
    meta_next = year_meta(year + 1)
    shape = shift_curr.shape[1:]
    parts = np.empty((12,) + shape, dtype="float64")

    for month in range(1, 13):
        sl_curr = meta_curr["month_slices"][month]
        sl_next = meta_next["month_slices"][month]
        parts[month - 1] = (
            weighted_sum_strict(meta_curr["rise"][sl_curr], shift_curr[sl_curr])
            + weighted_sum_strict(meta_next["desc"][sl_next], shift_next[sl_next])
        )

    return parts


def seasonal_cont_block(shift_prev: np.ndarray, shift_curr: np.ndarray, shift_next: np.ndarray, climate_year: int) -> np.ndarray:
    meta_prev = year_meta(climate_year - 1)
    meta_curr = year_meta(climate_year)
    meta_next = year_meta(climate_year + 1)
    season_index = {month: idx for idx, (_, months, _) in enumerate(SEASONS) for month in months}
    climate_months = (12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    n_curr = int(meta_curr["n_steps"])
    n_next = int(meta_next["n_steps"])
    rise = np.arange(1, n_curr + 1, dtype="float64") / n_curr
    desc = np.arange(n_next - 1, -1, -1, dtype="float64") / n_next
    parts = np.zeros((4,) + shift_curr.shape[1:], dtype="float64")

    offset = 0
    for month in climate_months:
        sl = meta_prev["month_slices"][12] if month == 12 else meta_curr["month_slices"][month]
        data = shift_prev[sl] if month == 12 else shift_curr[sl]
        length = sl.stop - sl.start
        parts[season_index[month]] += weighted_sum_strict(rise[offset : offset + length], data)
        offset += length

    offset = 0
    for month in climate_months:
        sl = meta_curr["month_slices"][12] if month == 12 else meta_next["month_slices"][month]
        data = shift_curr[sl] if month == 12 else shift_next[sl]
        length = sl.stop - sl.start
        parts[season_index[month]] += weighted_sum_strict(desc[offset : offset + length], data)
        offset += length

    return parts


def monthly_yearlyb_block(shift_curr: np.ndarray, shift_next: np.ndarray, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_cache = {int(year): shift_curr, int(year + 1): shift_next}
    year_offsets = build_year_offsets(data_cache)
    shape = shift_curr.shape[1:]
    total_block = np.empty((12,) + shape, dtype="float64")
    between_block = np.empty_like(total_block)
    within_block = np.empty_like(total_block)

    for month in range(1, 13):
        curr = interval_info(month_segments_for_year(year, month), year_offsets)
        nxt = interval_info(month_segments_for_year(year + 1, month), year_offsets)
        between = sum_global_slice(data_cache, year_offsets, int(curr["start"]), int(nxt["start"]))
        within = weighted_interval(concat_interval(data_cache, nxt["segments"])) - weighted_interval(
            concat_interval(data_cache, curr["segments"])
        )
        between_block[month - 1] = between
        within_block[month - 1] = within
        total_block[month - 1] = between + within

    return total_block, between_block, within_block


def seasonal_yearlyb_block(
    shift_prev: np.ndarray,
    shift_curr: np.ndarray,
    shift_next: np.ndarray,
    climate_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_cache = {int(climate_year - 1): shift_prev, int(climate_year): shift_curr, int(climate_year + 1): shift_next}
    year_offsets = build_year_offsets(data_cache)
    shape = shift_curr.shape[1:]
    total_block = np.empty((4,) + shape, dtype="float64")
    between_block = np.empty_like(total_block)
    within_block = np.empty_like(total_block)

    for idx, (season_name, _, _) in enumerate(SEASONS):
        curr = interval_info(season_segments_for_climate_year(climate_year, season_name), year_offsets)
        nxt = interval_info(season_segments_for_climate_year(climate_year + 1, season_name), year_offsets)
        between = sum_global_slice(data_cache, year_offsets, int(curr["start"]), int(nxt["start"]))
        within = weighted_interval(concat_interval(data_cache, nxt["segments"])) - weighted_interval(
            concat_interval(data_cache, curr["segments"])
        )
        between_block[idx] = between
        within_block[idx] = within
        total_block[idx] = between + within

    return total_block, between_block, within_block


def time_values(times: pd.DatetimeIndex) -> tuple[pd.Timestamp, np.ndarray]:
    ref = pd.Timestamp(year=int(times[0].year), month=1, day=1)
    vals = np.asarray((times - ref) / np.timedelta64(1, "D"), dtype="int64")
    return ref, vals


def output_times(years: np.ndarray, mode: str) -> pd.DatetimeIndex:
    if mode == "Monthly":
        return pd.DatetimeIndex(
            pd.to_datetime({"year": np.repeat(years, 12), "month": np.tile(np.arange(1, 13), len(years)), "day": 1})
        )
    anchors = [anchor for _, _, anchor in SEASONS]
    return pd.DatetimeIndex(
        pd.to_datetime({"year": np.repeat(years, 4), "month": np.tile(anchors, len(years)), "day": 1})
    )


class ContributionWriter:
    def __init__(
        self,
        template: TemplateInfo,
        times: pd.DatetimeIndex,
        out_path: Path,
        mode: str,
        global_title: str,
        global_comment: str,
        include_parts: bool,
    ) -> None:
        self.out_path = out_path
        self.tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        self.template = template
        self.mode = mode
        self.include_parts = include_parts
        self.space_dims = template.space_dims
        self.coord_string = coords_string(template.aux_vars, include_season_year=(mode == "Seasonal"))

        if out_path.exists():
            raise FileExistsError(f"Refusing to overwrite {out_path}")
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ref, time_var = time_values(times)
        self.ds = nc.Dataset(self.tmp_path, "w", format="NETCDF4")
        self.ds.createDimension("time", None)
        for dim in self.space_dims:
            self.ds.createDimension(dim, int(next(aux.sizes[dim] for aux in template.aux_vars.values() if dim in aux.sizes)))

        for key, value in template.global_attrs.items():
            setattr(self.ds, key, value)
        self.ds.CreationDate = now_utc_string()
        self.ds.title = global_title
        self.ds.comment = global_comment
        self.ds.ShiftMinusOneApplied = 0

        time_out = self.ds.createVariable("time", "i8", ("time",))
        time_out.units = f"days since {ref:%Y-%m-%d %H:%M:%S}"
        time_out.calendar = "proleptic_gregorian"
        time_out[:] = time_var

        for name, src in template.aux_vars.items():
            dtype = "f8" if np.issubdtype(src.dtype, np.floating) else "i4"
            fill_value = np.nan if dtype == "f8" else None
            out = self.ds.createVariable(name, dtype, src.dims, fill_value=fill_value)
            out[:] = np.asarray(src.values, dtype=np.float64 if dtype == "f8" else np.int32)
            for key, value in src.attrs.items():
                setattr(out, key, value)

        if mode == "Seasonal":
            season_year = times.year.astype("int32")
            out = self.ds.createVariable("season_year", "i4", ("time",))
            out[:] = season_year
            out.long_name = "climate year"
            out.comment = "Climate year runs from December of the previous calendar year through November."

        var_dims = ("time",) + self.space_dims
        if len(var_dims) == 1:
            chunksizes = (min(len(times), 124),)
        else:
            chunksizes = (min(len(times), 124),) + tuple(
                int(next(aux.sizes[dim] for aux in template.aux_vars.values() if dim in aux.sizes)) for dim in self.space_dims
            )

        self.main = self._create_var("tendtot", var_dims, chunksizes)
        if include_parts:
            self.between = self._create_var("between_interval", var_dims, chunksizes)
            self.within = self._create_var("within_interval", var_dims, chunksizes)
        else:
            self.between = None
            self.within = None

    def _create_var(self, name: str, dims: tuple[str, ...], chunksizes: tuple[int, ...]) -> nc.Variable:
        out = self.ds.createVariable(
            name,
            "f8",
            dims,
            fill_value=np.nan,
            zlib=True,
            complevel=COMP_LEVEL,
            shuffle=True,
            chunksizes=chunksizes,
        )
        if name == "tendtot":
            out.long_name = "Temperature tendency"
            out.units = self.template.var_attrs.get("units", "")
            if self.mode == "Monthly":
                out.coordinates = self.coord_string
                out.cell_methods = "time: exact yearly contribution from raw 3-hourly temperature differences"
                out.contribution_definition = "Weights are (i+1)/N_y in year y and (N_{y+1}-1-i)/N_{y+1} in year y+1."
            else:
                out.coordinates = self.coord_string
                out.cell_methods = "time: exact yearly contribution from raw 3-hourly temperature differences"
                out.contribution_definition = "Weights are (i+1)/N_y in year y and (N_{y+1}-1-i)/N_{y+1} in year y+1."
        elif name == "between_interval":
            out.long_name = "Temperature start-to-start between-interval term"
            out.units = self.template.var_attrs.get("units", "")
            out.coordinates = self.coord_string
            out.comment = "Start-to-start accumulated raw tendency term."
        else:
            out.long_name = "Temperature within-interval correction term"
            out.units = self.template.var_attrs.get("units", "")
            out.coordinates = self.coord_string
            out.comment = "Weighted within-interval correction term with w_j=(N-1-j)/N."
        return out

    def write_block(self, start: int, total: np.ndarray, between: np.ndarray | None = None, within: np.ndarray | None = None) -> None:
        stop = start + total.shape[0]
        self.main[start:stop, ...] = np.asarray(total, dtype="float64")
        if self.include_parts and between is not None and within is not None:
            self.between[start:stop, ...] = np.asarray(between, dtype="float64")
            self.within[start:stop, ...] = np.asarray(within, dtype="float64")

    def sync(self) -> None:
        self.ds.sync()

    def close(self, ok: bool) -> None:
        self.ds.close()
        if ok:
            self.tmp_path.replace(self.out_path)
            logging.info("wrote %s", self.out_path)
        elif self.tmp_path.exists():
            self.tmp_path.unlink()


class YearStore:
    def available_years(self) -> np.ndarray:
        raise NotImplementedError

    def template(self, var_name: str) -> TemplateInfo:
        raise NotImplementedError

    def read_year(self, year: int, var_name: str) -> np.ndarray:
        raise NotImplementedError


class GridYearStore(YearStore):
    def __init__(self, raw_paths: dict[int, Path]) -> None:
        self.raw_paths = raw_paths

    def available_years(self) -> np.ndarray:
        years = []
        for year in sorted(self.raw_paths):
            year_time = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 21:00:00", freq="3h")
            with xr.open_dataset(self.raw_paths[int(year)]) as ds:
                time_index = pd.DatetimeIndex(ds["time"].values)
            if len(year_time.intersection(time_index)) == len(year_time):
                years.append(int(year))
        return np.asarray(years, dtype=int)

    def template(self, var_name: str) -> TemplateInfo:
        first_year = int(self.available_years()[0])
        with xr.open_dataset(self.raw_paths[first_year]) as ds:
            return make_template_info(ds, var_name=var_name, time_name="time")

    def read_year(self, year: int, var_name: str) -> np.ndarray:
        with xr.open_dataset(self.raw_paths[int(year)]) as ds:
            return np.asarray(ds[var_name].values, dtype="float32")


class StationYearStore(YearStore):
    def __init__(self, threehourly_path: Path, template: TemplateInfo) -> None:
        self.path = threehourly_path
        self._template = template

    def available_years(self) -> np.ndarray:
        with xr.open_dataset(self.path) as ds:
            time_index = pd.DatetimeIndex(ds["time"].values)

        years = []
        for year in range(int(time_index.year.min()), int(time_index.year.max()) + 1):
            year_index = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 21:00:00", freq="3h")
            if len(year_index.intersection(time_index)) == len(year_index):
                years.append(year)
        return np.asarray(years, dtype=int)

    def template(self, var_name: str) -> TemplateInfo:
        return self._template

    def read_year(self, year: int, var_name: str) -> np.ndarray:
        year_time = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 21:00:00", freq="3h")
        with xr.open_dataset(self.path) as ds:
            da = ds[var_name].reindex(time=year_time)
            return np.asarray(da.values, dtype="float32")


def contribution_paths_grid(spec: GridSpec, calendar_years: np.ndarray, climate_years: np.ndarray) -> dict[str, Path]:
    monthly_times = output_times(calendar_years, "Monthly")
    seasonal_times = output_times(climate_years, "Seasonal")
    monthly_start, monthly_end = monthly_stamp(monthly_times)
    seasonal_start, seasonal_end = monthly_stamp(seasonal_times)
    return {
        "monthlycont": output_path_grid(spec.monthly_dir, spec.monthlycont_output_pattern, monthly_start, monthly_end),
        "seasonalcont": output_path_grid(spec.seasonal_dir, spec.seasonalcont_output_pattern, seasonal_start, seasonal_end),
        "monthlyb": output_path_grid(spec.monthly_dir, spec.monthlyb_output_pattern, monthly_start, monthly_end),
        "seasonalb": output_path_grid(spec.seasonal_dir, spec.seasonalb_output_pattern, seasonal_start, seasonal_end),
    }


def contribution_paths_station(station_tag: str, calendar_years: np.ndarray, climate_years: np.ndarray) -> dict[str, Path]:
    monthly_times = output_times(calendar_years, "Monthly")
    seasonal_times = output_times(climate_years, "Seasonal")
    monthly_start, monthly_end = monthly_stamp(monthly_times)
    seasonal_start, seasonal_end = monthly_stamp(seasonal_times)
    return {
        "monthlycont": output_path_station(
            KNMI_MONTHLY_DIR, "temp_monthlycont_KNMI_{station}_{start}_{end}.nc", station_tag, monthly_start, monthly_end
        ),
        "seasonalcont": output_path_station(
            KNMI_SEASONAL_DIR, "temp_seasonalcont_KNMI_{station}_{start}_{end}.nc", station_tag, seasonal_start, seasonal_end
        ),
        "monthlyb": output_path_station(
            KNMI_MONTHLY_DIR, "temp_monthlyyearlyB_KNMI_{station}_{start}_{end}.nc", station_tag, monthly_start, monthly_end
        ),
        "seasonalb": output_path_station(
            KNMI_SEASONAL_DIR, "temp_seasonalyearlyB_KNMI_{station}_{start}_{end}.nc", station_tag, seasonal_start, seasonal_end
        ),
    }


def process_contribution_outputs(store: YearStore, paths: dict[str, Path], label: str) -> None:
    raw_years = store.available_years()
    if raw_years.size < 3:
        logging.warning("skip contribution outputs for %s: need at least three complete years", label)
        return

    calendar_years = raw_years[:-1]
    climate_years = raw_years[1:-1]
    template = store.template("tendtot")

    if not paths["monthlycont"].exists():
        writer = None
        try:
            times = output_times(calendar_years, "Monthly")
            writer = ContributionWriter(
                template=template,
                times=times,
                out_path=paths["monthlycont"],
                mode="Monthly",
                global_title="Temperature tendency (monthly exact yearly contribution)",
                global_comment=build_output_comment(
                    template, "exact yearly contribution from raw 3-hourly temperature differences grouped by calendar month; float64 on disk."
                ),
                include_parts=False,
            )
            cache: dict[int, np.ndarray] = {}

            def ensure(year: int) -> np.ndarray:
                if year not in cache:
                    cache[year] = store.read_year(year, "tendtot")
                return cache[year]

            for idx, year in enumerate(calendar_years):
                block = monthly_cont_block(ensure(int(year)), ensure(int(year + 1)), int(year))
                writer.write_block(idx * 12, block)
                if (idx + 1) % 8 == 0:
                    writer.sync()
                cache.pop(int(year - 1), None)
        except Exception:
            if writer is not None:
                writer.close(ok=False)
            raise
        else:
            writer.close(ok=True)
    else:
        logging.info("skip existing %s", paths["monthlycont"])

    if climate_years.size and not paths["seasonalcont"].exists():
        writer = None
        try:
            times = output_times(climate_years, "Seasonal")
            writer = ContributionWriter(
                template=template,
                times=times,
                out_path=paths["seasonalcont"],
                mode="Seasonal",
                global_title="Temperature tendency (seasonal exact yearly contribution)",
                global_comment=build_output_comment(
                    template, "exact yearly contribution from raw 3-hourly temperature differences grouped by climate-year season; float64 on disk."
                ),
                include_parts=False,
            )
            cache = {}

            def ensure(year: int) -> np.ndarray:
                if year not in cache:
                    cache[year] = store.read_year(year, "tendtot")
                return cache[year]

            for idx, year in enumerate(climate_years):
                block = seasonal_cont_block(ensure(int(year - 1)), ensure(int(year)), ensure(int(year + 1)), int(year))
                writer.write_block(idx * 4, block)
                if (idx + 1) % 8 == 0:
                    writer.sync()
                cache.pop(int(year - 2), None)
        except Exception:
            if writer is not None:
                writer.close(ok=False)
            raise
        else:
            writer.close(ok=True)
    elif climate_years.size:
        logging.info("skip existing %s", paths["seasonalcont"])

    if not paths["monthlyb"].exists():
        writer = None
        try:
            times = output_times(calendar_years, "Monthly")
            writer = ContributionWriter(
                template=template,
                times=times,
                out_path=paths["monthlyb"],
                mode="Monthly",
                global_title="Temperature tendency (monthly same-interval yearly B contribution)",
                global_comment=build_output_comment(
                    template, "exact same-interval yearly difference from raw 3-hourly temperature differences for the same calendar month; float64 on disk."
                ),
                include_parts=True,
            )
            cache = {}

            def ensure(year: int) -> np.ndarray:
                if year not in cache:
                    cache[year] = store.read_year(year, "tendtot")
                return cache[year]

            for idx, year in enumerate(calendar_years):
                total, between, within = monthly_yearlyb_block(ensure(int(year)), ensure(int(year + 1)), int(year))
                writer.write_block(idx * 12, total, between, within)
                if (idx + 1) % 8 == 0:
                    writer.sync()
                cache.pop(int(year - 1), None)
        except Exception:
            if writer is not None:
                writer.close(ok=False)
            raise
        else:
            writer.close(ok=True)
    else:
        logging.info("skip existing %s", paths["monthlyb"])

    if climate_years.size and not paths["seasonalb"].exists():
        writer = None
        try:
            times = output_times(climate_years, "Seasonal")
            writer = ContributionWriter(
                template=template,
                times=times,
                out_path=paths["seasonalb"],
                mode="Seasonal",
                global_title="Temperature tendency (seasonal same-interval yearly B contribution)",
                global_comment=build_output_comment(
                    template, "exact same-interval yearly difference from raw 3-hourly temperature differences for the same climate-year season; float64 on disk."
                ),
                include_parts=True,
            )
            cache = {}

            def ensure(year: int) -> np.ndarray:
                if year not in cache:
                    cache[year] = store.read_year(year, "tendtot")
                return cache[year]

            for idx, year in enumerate(climate_years):
                total, between, within = seasonal_yearlyb_block(
                    ensure(int(year - 1)), ensure(int(year)), ensure(int(year + 1)), int(year)
                )
                writer.write_block(idx * 4, total, between, within)
                if (idx + 1) % 8 == 0:
                    writer.sync()
                cache.pop(int(year - 2), None)
        except Exception:
            if writer is not None:
                writer.close(ok=False)
            raise
        else:
            writer.close(ok=True)
    elif climate_years.size:
        logging.info("skip existing %s", paths["seasonalb"])


def process_knmi_station(station_id: int, station_meta: dict[str, object], do_frequency: bool, do_contrib: bool) -> None:
    logging.info("processing KNMI station %s (%s)", station_id, station_meta["name"])
    template = build_station_template(station_id, station_meta)
    series = knmi_station_series(station_id)

    hourly_start = series.index[0].strftime("%Y%m%d%H")
    hourly_end = series.index[-1].strftime("%Y%m%d%H")
    hourly_path = output_path_station(
        KNMI_HOURLY_DIR,
        "temp_hourly_KNMI_{station}_{start}_{end}.nc",
        station_meta["tag"],
        hourly_start,
        hourly_end,
    )
    if not hourly_path.exists():
        hourly_ds = build_hourly_station_dataset(series, template)
        write_dataset(hourly_ds, hourly_path)
    else:
        logging.info("skip existing %s", hourly_path)

    series_3h = series[series.index.hour % 3 == 0]
    three_start = series_3h.index[0].strftime("%Y%m%d%H")
    three_end = series_3h.index[-1].strftime("%Y%m%d%H")
    threehourly_path = output_path_station(
        KNMI_THREEHOURLY_DIR,
        "temp_3hourly_KNMI_{station}_{start}_{end}.nc",
        station_meta["tag"],
        three_start,
        three_end,
    )
    if not threehourly_path.exists():
        threehourly_ds = build_threehourly_station_dataset(series, template)
        write_dataset(threehourly_ds, threehourly_path)
    else:
        logging.info("skip existing %s", threehourly_path)

    if do_frequency:
        process_knmi_station_frequency_products(station_id, station_meta, threehourly_path)

    if do_contrib:
        store = StationYearStore(threehourly_path, template)
        raw_years = store.available_years()
        if raw_years.size < 3:
            logging.warning("skip contribution outputs for KNMI %s: need at least three complete years", station_meta["tag"])
        else:
            paths = contribution_paths_station(str(station_meta["tag"]), raw_years[:-1], raw_years[1:-1])
            process_contribution_outputs(store, paths, f"KNMI {station_meta['tag']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean temperature frequency and tendency products for KNMI, ERA5, and RACMO datasets.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["knmi", "era5", "racmo", "racmo_v7", "racmo_v5"],
        default=["knmi", "era5", "racmo"],
    )
    parser.add_argument("--stations", nargs="+", choices=sorted(meta["tag"] for meta in KNMI_STATIONS.values()), default=None)
    parser.add_argument("--skip-frequency-products", action="store_true")
    parser.add_argument("--skip-contribution-products", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if "knmi" in args.sources:
        selected = {meta["tag"] for meta in KNMI_STATIONS.values()} if args.stations is None else set(args.stations)
        KNMI_HOURLY_DIR.mkdir(parents=True, exist_ok=True)
        for station_id, station_meta in KNMI_STATIONS.items():
            if station_meta["tag"] not in selected:
                continue
            process_knmi_station(
                station_id=station_id,
                station_meta=station_meta,
                do_frequency=not args.skip_frequency_products,
                do_contrib=not args.skip_contribution_products,
            )

    if "era5" in args.sources:
        logging.info("processing ERA5 three-hourly temperature")
        raw_paths = process_era5_threehourly()
        if not args.skip_frequency_products:
            process_grid_frequency_products(ERA5_SPEC, raw_paths)
        if not args.skip_contribution_products:
            store = GridYearStore(raw_paths)
            raw_years = store.available_years()
            if raw_years.size < 3:
                logging.warning("skip contribution outputs for ERA5: need at least three complete years")
            else:
                process_contribution_outputs(store, contribution_paths_grid(ERA5_SPEC, raw_years[:-1], raw_years[1:-1]), "ERA5")

    racmo_specs: list[GridSpec] = []
    if "racmo" in args.sources or "racmo_v7" in args.sources:
        racmo_specs.append(RACMO_V7_SPEC)
    if "racmo_v5" in args.sources:
        racmo_specs.append(RACMO_V5_SPEC)

    for spec in racmo_specs:
        logging.info("processing %s hourly tas from %s", spec.experiment, spec.input_dir)
        raw_paths = process_racmo_threehourly(spec)
        if not args.skip_frequency_products:
            process_grid_frequency_products(spec, raw_paths)
        if not args.skip_contribution_products:
            store = GridYearStore(raw_paths)
            raw_years = store.available_years()
            if raw_years.size < 3:
                logging.warning("skip contribution outputs for %s: need at least three complete years", spec.experiment)
            else:
                process_contribution_outputs(
                    store,
                    contribution_paths_grid(spec, raw_years[:-1], raw_years[1:-1]),
                    spec.experiment,
                )


if __name__ == "__main__":
    main()
