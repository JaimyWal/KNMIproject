import os
import glob
import numpy as np
import xarray as xr
import pandas as pd

#%%

xr.set_options(use_new_combine_kwarg_defaults=True)

def preprocess_racmo_monthly(
    dir_path,
    var_name,
    months=None,
    years=None,
    lats=None,
    lons=None,
    chunks_time=180,
    chunks_lat=200,
    chunks_lon=200,
):
    file_pattern = '*.nc'

    file_list = sorted(glob.glob(os.path.join(dir_path, file_pattern)))

    data_raw = xr.open_mfdataset(
        file_list,
        combine='by_coords',
        chunks='auto'
    )

    data_raw = data_raw.rename({'lat': 'latitude', 'lon': 'longitude'})

    rotpole_coord = data_raw['rotated_pole']

    # Select variable and cast to float32
    da = data_raw[var_name].astype('float32')

    lat2d = data_raw['latitude']   # dims: (rlat, rlon)
    lon2d = data_raw['longitude']   # dims: (rlat, rlon)

    # spatial selection
    if isinstance(lats, (list, tuple)) and len(lats) == 2 and \
       isinstance(lons, (list, tuple)) and len(lons) == 2:

        lat_min, lat_max = sorted(lats)
        lon_min, lon_max = sorted(lons)

        mask = (
            (lat2d >= lat_min) & (lat2d <= lat_max) &
            (lon2d >= lon_min) & (lon2d <= lon_max)
        ).compute()

        # valid_rlat = mask.any('rlon')
        # valid_rlon = mask.any('rlat')

        # da = da.sel(rlat=valid_rlat, rlon=valid_rlon)

        da = da.where(mask, drop=True)

    # nearest grid point to a single (lat, lon) 
    elif isinstance(lats, (int, float)) and \
         isinstance(lons, (int, float)):

        target_lat = float(lats)
        target_lon = float(lons)

        # squared distance in lat lon space
        dist2 = (lat2d - target_lat)**2 + (lon2d - target_lon)**2

        # find indices of minimum distance
        ii, jj = np.unravel_index(
            np.argmin(dist2.values),
            dist2.shape
        )

        da = da.isel(rlat=[ii], rlon=[jj])

    if var_name == 't2m':
        da = da - 273.15
        da.attrs['units'] = 'degC'
    elif var_name == 'precip':
        da = da * 86400.0
        da.attrs['units'] = 'mm/day'

    data_monthly_full = (
        da
        .resample(time='MS')
        .mean('time')
    )

    time_sel = data_monthly_full.time

    # Month selection
    if months is not None:
        time_sel = time_sel.where(time_sel.dt.month.isin(months), drop=True)

    # Year selection
    if years is not None and isinstance(years, (list, tuple)):
        if len(years) == 2:
            start, end = years
            if start is not None:
                time_sel = time_sel.where(time_sel.dt.year >= start, drop=True)
            if end is not None:
                time_sel = time_sel.where(time_sel.dt.year <= end, drop=True)
        else:
            time_sel = time_sel.where(time_sel.dt.year.isin(years), drop=True)

    data_monthly = (
        data_monthly_full
        .sel(time=time_sel)
        .assign_coords(rotated_pole=rotpole_coord)
        .chunk({'time': chunks_time,
                'rlat': chunks_lat,
                'rlon': chunks_lon})
        .persist()
    )

    return data_monthly

# racmo = preprocess_racmo_monthly('/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/precip', 
#                                 'precip',
#                                 months=[6,7,8],
#                                 years=[2000, 2020],
#                                 lats=[35, 72],
#                                 lons=[-12, 35])


# #%%

# import cartopy.crs as ccrs
# from importlib import reload

# import PlotMaps
# reload(PlotMaps)          
# from PlotMaps import plot_map 

# rp = racmo['rotated_pole']
# pole_lat = rp.grid_north_pole_latitude
# pole_lon = rp.grid_north_pole_longitude
# central_rlon = 18.0

# rotpole = ccrs.RotatedPole(
#     pole_latitude=pole_lat,
#     pole_longitude=pole_lon,
#     central_rotated_longitude=central_rlon,
# )

# plot_map(
#     data=racmo.isel(time=0).squeeze().values,
#     lon=racmo.lon.values,
#     lat=racmo.lat.values,
#     label='T2m (°C)',
#     proj=ccrs.PlateCarree(),
#     rotated_grid=False,
#     x_ticks_num=False,
#     x_ticks=15,
#     y_ticks_num=False,
#     y_ticks=5,
#     figsize=(14, 12),
#     extent=[-12, 35, 35, 72]
# )
