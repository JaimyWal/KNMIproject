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

    year  = time.dt.year.values
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


def subset_space(da, 
                 lat2d, 
                 lon2d, 
                 lats, 
                 lons, 
                 dim_lat, 
                 dim_lon, 
                 rect_sel=True, 
                 rotpole=False, 
                 native_rotpole=False):

    if lats is None or lons is None:
        return da

    if isinstance(lats, (list, tuple)) and len(lats) == 2 and \
       isinstance(lons, (list, tuple)) and len(lons) == 2:
        
        lat_min, lat_max = sorted(lats)
        lon_min, lon_max = sorted(lons)

        mask_geo = (
            (lat2d >= lat_min) & (lat2d <= lat_max) &
            (lon2d >= lon_min) & (lon2d <= lon_max)
        )

        if rotpole == ccrs.PlateCarree():
            use_rotpole = False
        else:
            use_rotpole = True

        if rect_sel == False or use_rotpole == False:
            return da.where(mask_geo.compute(), drop=True)
        
        elif rect_sel == True and use_rotpole == True and native_rotpole == False:
            plate = ccrs.PlateCarree()
            pts = rotpole.transform_points(plate, lon2d.values, lat2d.values)
            rlon2d = xr.DataArray(pts[..., 0], coords=lat2d.coords, dims=lat2d.dims)
            rlat2d = xr.DataArray(pts[..., 1], coords=lat2d.coords, dims=lat2d.dims)

            rlon_masked = rlon2d.where(mask_geo)
            rlat_masked = rlat2d.where(mask_geo)

            rlat_min = np.nanmin(rlat_masked.values)
            rlat_max = np.nanmax(rlat_masked.values)
            rlon_min = np.nanmin(rlon_masked.values)
            rlon_max = np.nanmax(rlon_masked.values)

            mask_rot = (
                (rlat2d >= rlat_min) & (rlat2d <= rlat_max) &
                (rlon2d >= rlon_min) & (rlon2d <= rlon_max)
            )

            return da.where(mask_rot, drop=True)

        elif rect_sel == True and use_rotpole == True and native_rotpole == True:
            valid_lat = mask_geo.any(dim=dim_lon).compute()
            valid_lon = mask_geo.any(dim=dim_lat).compute()
        
        return da.isel({dim_lat: valid_lat, dim_lon: valid_lon})

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

def preprocess_netcdf_monthly(
    source,
    file_path,
    var_name,
    months=None,
    years=None,
    lats=None,
    lons=None,
    trim_border=None,
    rect_sel=True,
    rotpole=None,
    native_rotpole=False,
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

    if 'ERA5' in src:
        if var_name == 't2m':
            da = da - 273.15
            da.attrs['units'] = 'degC'
        elif var_name in ['tp', 'precip']:
            da = da*1000.0
            da.attrs['units'] = 'mm/day'

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
        rect_sel=rect_sel, 
        rotpole=rotpole, 
        native_rotpole=native_rotpole
    )

    # 7. decide if resampling is needed
    #    If time spacing is monthly already, do not resample.
    #    If not monthly (daily / subdaily), resample to monthly.
    if is_monthly_time(da['time']):
        da_month = da
    else:
        da_month = da.resample(time='MS').mean('time')

    # 8. select months and years on the monthly time axis
    tsel = subset_time(da_month['time'], months=months, years=years)
    out = da_month.sel(time=tsel)

    # 9. chunk
    chunk_dict = {'time': chunks_time}
    for dim in out.dims:
        if dim in ['latitude', 'rlat']:
            chunk_dict[dim] = chunks_lat
        elif dim in ['longitude', 'rlon']:
            chunk_dict[dim] = chunks_lon

    out = out.chunk(chunk_dict).persist()

    return out