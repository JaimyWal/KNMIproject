import glob
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs

xr.set_options(use_new_combine_kwarg_defaults=True)

def open_dataset(path):

    files = sorted(glob.glob(path))

    try:
        ds = xr.open_mfdataset(
            files,
            combine='by_coords',
            chunks='auto',
            decode_times=True,
        )

        return ds

    except Exception:

        ds = xr.open_mfdataset(
            files,
            combine='by_coords',
            chunks='auto',
            decode_times=False,
        )

        n_time = ds.sizes['time']
        new_time = pd.date_range(start='1979-01-01', periods=n_time, freq='MS')

        ds = ds.assign_coords(time=new_time)

        ds['time'].attrs = {}

        return ds


def is_monthly_time(time):

    year = time.dt.year.values
    month = time.dt.month.values
    unique_pairs = np.unique(np.stack([year, month], axis=1), axis=0)

    return len(unique_pairs) == len(time)


def align_month_to_start(time):

    if not is_monthly_time(time):
        return time

    # day offset: 0 for day 1, 1 for day 2, ..., 30 for day 31
    day_offset = time.dt.day - 1

    # convert offset in days to a timedelta and subtract it
    offset = day_offset*np.timedelta64(1, 'D')

    return time - offset


def rect_sel(lats, lons, rotpole, n_edge=1e6):

    lat_min, lat_max = sorted(lats)
    lon_min, lon_max = sorted(lons)

    if rotpole == ccrs.PlateCarree():
        return lat_min, lat_max, lon_min, lon_max
    
    n_edge = int(n_edge)

    lon_bottom = np.linspace(lon_min, lon_max, n_edge)
    lat_bottom = np.full_like(lon_bottom, lat_min)

    lon_top = np.linspace(lon_min, lon_max, n_edge)
    lat_top = np.full_like(lon_top, lat_max)

    lat_left = np.linspace(lat_min, lat_max, n_edge)
    lon_left = np.full_like(lat_left, lon_min)

    lat_right = np.linspace(lat_min, lat_max, n_edge)
    lon_right = np.full_like(lat_right, lon_max)

    edge_lon = np.concatenate([lon_bottom, lon_top, lon_left, lon_right])
    edge_lat = np.concatenate([lat_bottom, lat_top, lat_left, lat_right])

    pts = rotpole.transform_points(ccrs.PlateCarree(), edge_lon, edge_lat)

    rlon = pts[..., 0]
    rlat = pts[..., 1]

    rlat_min = float(rlat.min())
    rlat_max = float(rlat.max())
    rlon_min = float(rlon.min())
    rlon_max = float(rlon.max())

    return rlat_min, rlat_max, rlon_min, rlon_max


def subset_space(da, 
                 lat2d, 
                 lon2d, 
                 lats, 
                 lons, 
                 dim_lat, 
                 dim_lon, 
                 rotpole_sel=ccrs.PlateCarree(),
                 rotpole_native=ccrs.PlateCarree()):

    if lats is None or lons is None:
        return da

    if isinstance(lats, (list, tuple)) and len(lats) == 2 and \
       isinstance(lons, (list, tuple)) and len(lons) == 2:

        lat_min, lat_max = sorted(lats)
        lon_min, lon_max = sorted(lons)

        if rotpole_sel == ccrs.PlateCarree():

            mask = (
                (lat2d >= lat_min) & (lat2d <= lat_max) &
                (lon2d >= lon_min) & (lon2d <= lon_max)
            )
        
        elif rotpole_sel != ccrs.PlateCarree(): 

            if rotpole_sel == rotpole_native:
                rlat1d = da[dim_lat]
                rlon1d = da[dim_lon]
                rlat2d, rlon2d = xr.broadcast(rlat1d, rlon1d)

            elif rotpole_sel != rotpole_native:
                pts = rotpole_sel.transform_points(ccrs.PlateCarree(), lon2d.values, lat2d.values)
                rlon2d = xr.DataArray(pts[..., 0], coords=lat2d.coords, dims=lat2d.dims)
                rlat2d = xr.DataArray(pts[..., 1], coords=lat2d.coords, dims=lat2d.dims)

            rlat_min, rlat_max, rlon_min, rlon_max = rect_sel(lats, lons, rotpole_sel)

            mask = (
                (rlat2d >= rlat_min) & (rlat2d <= rlat_max) &
                (rlon2d >= rlon_min) & (rlon2d <= rlon_max)
            )
            
        da_sel = da.where(mask.compute(), drop=True)

        return da_sel

    if isinstance(lats, (int, float)) and isinstance(lons, (int, float)):
        target_lat = float(lats)
        target_lon = float(lons)

        lat_rad = np.deg2rad(lat2d)
        lon_rad = np.deg2rad(lon2d)
        tlat = np.deg2rad(target_lat)
        tlon = np.deg2rad(target_lon)

        dlat = lat_rad - tlat
        dlon = lon_rad - tlon

        a = (
            np.sin(dlat / 2.0)**2
            + np.cos(tlat)*np.cos(lat_rad)*np.sin(dlon / 2.0)**2
        )

        R = 6371.0
        dist = 2.0*R*np.arcsin(np.sqrt(a))

        ii, jj = np.unravel_index(np.nanargmin(dist.values), dist.shape)
        return da.isel({dim_lat: [ii], dim_lon: [jj]})

    return da


def subset_time(time, months=None, years=None):

    tsel = time
    if months is not None:
        tsel = tsel.where(tsel.dt.month.isin(months), drop=True)

    if years is not None and isinstance(years, (list, tuple)):
        if len(years) == 2:
            start, end = years
            if start is not None:
                tsel = tsel.where(tsel.dt.year >= start, drop=True)
            if end is not None:
                tsel = tsel.where(tsel.dt.year <= end, drop=True)
        else:
            tsel = tsel.where(tsel.dt.year.isin(years), drop=True)

    return tsel


def make_landmask(
    land_file,
    land_var,
    lake_file=None,
    lake_var=None,
    land_thresh=0.5,
    lake_thresh=0.0):

    ds_land = xr.open_dataset(land_file)
    land = ds_land[land_var]

    mask = land >= land_thresh

    if lake_file is None or lake_var is None:
        return mask.astype(bool)

    ds_lake = xr.open_dataset(lake_file)
    lake = ds_lake[lake_var]

    mask = mask | (lake >= lake_thresh)

    return mask.astype(bool)


def preprocess_netcdf(
    source,
    file_path,
    var_name,
    months=None,
    years=None,
    lats=None,
    lons=None,
    land_only=False,
    trim_border=None,
    rotpole_sel=ccrs.PlateCarree(),
    rotpole_native=ccrs.PlateCarree(),
    chunks_time=180,
    chunks_lat=200,
    chunks_lon=200
):
    
    src = source.upper()

    # 1. read data and fix time if needed
    ds = open_dataset(file_path)

    # 2. rename coords to a common convention
    rename = {}
    if 'valid_time' in ds.coords:
        rename['valid_time'] = 'time'
    if 'lat' in ds.coords:
        rename['lat'] = 'latitude'
    if 'lon' in ds.coords:
        rename['lon'] = 'longitude'
    if rename:
        ds = ds.rename(rename)

    if 'time' not in ds.coords:
        raise ValueError('No time coordinate found in dataset')

    # 3. align monthly timestamps (RACMO2.4 mid month, etc.) to day 1
    ds = ds.assign_coords(time=align_month_to_start(ds['time']))

    # 4. select variable and convert units where needed
    da = ds[var_name].astype('float32')

    if land_only:

        if 'ERA5' in src:
            landmask = make_landmask(
                land_file='/nobackup/users/walj/landmasks/era5_landmask.nc',
                land_var='lsm',
                land_thresh=0.5
            ).squeeze()
            da = da.where(landmask)
        
        elif 'RACMO2.3' in src:
            landmask = make_landmask(
                land_file='/nobackup/users/walj/landmasks/lsm_racmo2.3_hxeur12.nc',
                land_var='var81',
                lake_file='/nobackup/users/walj/landmasks/lakefrac_racmo2.3_hxeur12.nc',
                lake_var='clake',
                land_thresh=0.5,
                lake_thresh=1.0
            ).squeeze()
            landmask = landmask.isel(rlat=slice(16, -16), rlon=slice(16, -16))
            landmask = landmask.assign_coords(rlat=da['rlat'], rlon=da['rlon'])
            da = da.where(landmask)

        elif 'RACMO2.4' in src:
            landmask = make_landmask(
                land_file='/nobackup/users/walj/landmasks/lsm_racmo2.4_kext06.nc',
                land_var='var81',
                lake_file='/nobackup/users/walj/landmasks/lakefrac_racmo2.4_kext06.nc',
                lake_var='var26',
                land_thresh=0.5,
                lake_thresh=1.0
            ).squeeze()
            da = da.where(landmask)

    if 'RACMO' in src:
        if var_name == 't2m' or var_name == 'tas':
            da = da - 273.15
            da.attrs['units'] = 'degC'
        elif var_name == 'precip' or (var_name == 'pr' and not is_monthly_time(da['time'])):
            da = da*86400.0
            da.attrs['units'] = 'mm/day'
        elif var_name == 'pr' and is_monthly_time(da['time']):
            days_in_month = da['time'].dt.days_in_month
            da = da / days_in_month
            da.attrs['units'] = 'mm/day'
        if var_name == 'sund':
            if ('2.4' in src) and is_monthly_time(da['time']):
                scale_to_seconds = 1e-5
                days_in_month = da['time'].dt.days_in_month
                da = scale_to_seconds*da / (days_in_month*3600.0)
                da.attrs['units'] = 'hours/day'
            else:
                da = da / 3600.0
                da.attrs['units'] = 'hours/day'
        if var_name == 'rsds':
            if ('2.4' in src) and is_monthly_time(da['time']):
                days_in_month = da['time'].dt.days_in_month
                da = da / (days_in_month*86400.0)
                da.attrs['units'] = 'W/m2'

    if 'ERA5' in src:
        if var_name == 't2m':
            da = da - 273.15
            da.attrs['units'] = 'degC'
        elif var_name in ['tp', 'precip']:
            da = da*1000.0
            da.attrs['units'] = 'mm/day'
        elif var_name == 'ssrd':
            da = da / 86400.0
            da.attrs['units'] = 'W/m2'

    # 5. determine spatial dims on the raw grid
    if 'rlat' in da.dims and 'rlon' in da.dims:
        dim_lat, dim_lon = 'rlat', 'rlon'
    else:
        dim_lat, dim_lon = 'latitude', 'longitude'

    # 5a. ensure 1D latitude is ascending (only for 1D lat)
    if 'latitude' in da.coords and da['latitude'].ndim == 1:
        lat1d = da['latitude']
        if lat1d[0] > lat1d[-1]:
            da = da.isel(latitude=slice(None, None, -1))

    # 5b. optionally remove n grid cells from each spatial border on the raw field
    if trim_border is not None:
        n = int(trim_border)
        da = da.isel({dim_lat: slice(n, -n), dim_lon: slice(n, -n)})

    # 6. build lat2d / lon2d on the trimmed grid and subset in space
    if 'rlat' in da.dims and 'rlon' in da.dims:
        # rotated grid already has 2D lat/lon
        lat2d = da['latitude']
        lon2d = da['longitude']
    else:
        # regular lat/lon grid, build 2D lat/lon by broadcasting
        lat1d = da['latitude']
        lon1d = da['longitude']
        lat2d, lon2d = xr.broadcast(lat1d, lon1d)

    da = subset_space(
        da, 
        lat2d, 
        lon2d, 
        lats, 
        lons, 
        dim_lat, 
        dim_lon, 
        rotpole_sel=rotpole_sel, 
        rotpole_native=rotpole_native
    )

    # 7. decide if resampling is needed
    #    If time spacing is monthly already, do not resample.
    #    If not monthly (daily / subdaily), resample to monthly.
    # if is_monthly_time(da['time']):
    #     da_month = da
    # else:
    #     da_month = da.resample(time='MS').mean('time')

    # 8. select months and years on the monthly time axis
    tsel = subset_time(da['time'], months=months, years=years)
    out = da.sel(time=tsel)

    # 9. chunk
    chunk_dict = {'time': chunks_time}
    for dim in out.dims:
        if dim in ['latitude', 'rlat']:
            chunk_dict[dim] = chunks_lat
        elif dim in ['longitude', 'rlon']:
            chunk_dict[dim] = chunks_lon

    out = out.chunk(chunk_dict).persist()

    return out