from __future__ import annotations

import os
from pathlib import Path

import xarray as xr

PAT = os.getenv(
    "EDH_PAT",
    "edh_pat_43c156abac92ae2538677a4918e4d58bcb47bdacbaef600a1d56bdaadc02560511fb87f53810b46341507f76e50bcb7b",
)

DATASET_URL = (
    f"https://edh:{PAT}@data.earthdatahub.destine.eu/"
    "era5/reanalysis-era5-single-levels-v0.zarr"
)
OUTPUT_DIR = Path(
    os.getenv("ERA5_OUTPUT_DIR", "/nobackup/users/walj/era5/Hourly")
)
VAR_NAME = "t2m"
FILE_TEMPLATE = "era5_t2m_europe_3hourly_{year}.nc"

EUROPE_BOUNDS = {
    "north": 85.0,
    "west": -45.0,
    "south": 20.0,
    "east": 65.0,
}

START_YEAR = int(os.getenv("ERA5_START_YEAR", "1950"))
END_YEAR = int(os.getenv("ERA5_END_YEAR", "2025"))
TIME_START = f"{START_YEAR}-01-01T00:00:00"
TIME_END = f"{END_YEAR}-12-31T23:00:00"
WRITE_TIME_CHUNK = int(os.getenv("ERA5_WRITE_TIME_CHUNK", "124"))


def _output_path(year: int) -> Path:
    return OUTPUT_DIR / FILE_TEMPLATE.format(year=year)


def _partial_output_path(year: int) -> Path:
    return OUTPUT_DIR / f"{FILE_TEMPLATE.format(year=year)}.part"


def _choose_netcdf_engine() -> str:
    try:
        import netCDF4  # noqa: F401

        return "netcdf4"
    except ImportError:
        try:
            import h5netcdf  # noqa: F401

            return "h5netcdf"
        except ImportError as exc:
            raise RuntimeError(
                "Writing NetCDF needs either 'netCDF4' or 'h5netcdf' installed."
            ) from exc


def _subset_large_europe(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sel(
        valid_time=slice(TIME_START, TIME_END),
        latitude=slice(EUROPE_BOUNDS["north"], EUROPE_BOUNDS["south"]),
    )

    west_chunk = ds.sel(longitude=slice(360 + EUROPE_BOUNDS["west"], None))
    east_chunk = ds.sel(longitude=slice(0, EUROPE_BOUNDS["east"]))
    ds = xr.concat([west_chunk, east_chunk], dim="longitude")

    longitude = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
    return ds.assign_coords(longitude=longitude)


def load_t2m_europe_3hourly() -> xr.Dataset:
    ds = xr.open_dataset(
        DATASET_URL,
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
    )[[VAR_NAME]]

    ds = _subset_large_europe(ds)
    return ds.isel(valid_time=slice(None, None, 3))


def save_t2m_year(year: int, ds: xr.Dataset, engine: str) -> Path:
    output_path = _output_path(year)
    partial_path = _partial_output_path(year)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[skip] {year}: {output_path} already exists")
        return output_path

    if partial_path.exists():
        print(f"[cleanup] {year}: removing stale partial file {partial_path}")
        partial_path.unlink()

    year_ds = ds.sel(
        valid_time=slice(f"{year}-01-01T00:00:00", f"{year}-12-31T23:00:00")
    )
    year_ds = year_ds.chunk({"valid_time": WRITE_TIME_CHUNK})
    print(
        f"[start] {year}: writing {dict(year_ds.sizes)} to {partial_path}",
        flush=True,
    )

    encoding = {
        VAR_NAME: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
            "chunksizes": (
                min(WRITE_TIME_CHUNK, year_ds.sizes["valid_time"]),
                year_ds.sizes["latitude"],
                year_ds.sizes["longitude"],
            ),
        }
    }
    year_ds.to_netcdf(partial_path, engine=engine, encoding=encoding)
    partial_path.replace(output_path)

    print(
        f"[done] {year}: saved {output_path} ({output_path.stat().st_size} bytes)",
        flush=True,
    )
    return output_path


def export_all_years() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = _choose_netcdf_engine()
    ds = load_t2m_europe_3hourly()

    print(f"NetCDF engine: {engine}", flush=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    print(
        "Dataset summary: "
        f"sizes={dict(ds.sizes)} "
        f"time=({ds.valid_time.values[0]}, {ds.valid_time.values[-1]}) "
        f"lat=({float(ds.latitude.max())}, {float(ds.latitude.min())}) "
        f"lon=({float(ds.longitude.min())}, {float(ds.longitude.max())})",
        flush=True,
    )

    for year in range(START_YEAR, END_YEAR + 1):
        save_t2m_year(year, ds, engine)


if __name__ == "__main__":
    export_all_years()
