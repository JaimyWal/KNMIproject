import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from importlib import reload

from RegionalTrends.Helpers import ProcessNetCDF
reload(ProcessNetCDF)          
from RegionalTrends.Helpers.ProcessNetCDF import preprocess_netcdf

from RegionalTrends.Helpers import ProcessStation
reload(ProcessStation)          
from RegionalTrends.Helpers.ProcessStation import preprocess_station


def process_source(
    data_source,
    var,
    data_sources,
    station_sources,
    var_name_cfg,
    file_cfg,
    proj_cfg,
    months=None,
    years=None,
    lats=None,
    lons=None,
    land_only=False,
    trim_border=None,
    rotpole_sel=ccrs.PlateCarree(),
    rolling_mean_var=False,
    fit_against_gmst=False,
    rolling_mean_years=1,
    min_periods=1,
    return_items=('raw', 'monthly', 'yearly', 'fit', 'avg'),
):

    if months is None:
        months_local = np.arange(1, 13, dtype=int)
    else:
        months_local = np.asarray(months, dtype=int)

    if years is None:
        years_req = None
        years_load = None
    else:
        years_req = list(years)
        years_load = list(years_req)

        month_start = int(months_local[0])
        month_end = int(months_local[-1])

        if month_start > month_end:
            years_load[0] = years_req[0] - 1

    trim_local = trim_border
    if (data_source == 'RACMO2.4') and (trim_border is None):
        trim_local = 8

    # Load raw
    if data_source in station_sources:
        var_name = var_name_cfg['Station'][var]
        file_path = file_cfg['Station'][data_source]

        data = preprocess_station(
            file_path=file_path,
            var_name=var_name,
            months=months_local,
            years=years_load,
        ).squeeze()

    else:
        file_key = next(src for src in data_sources if src in data_source)
        var_name = var_name_cfg[file_key][var]
        file_path = file_cfg[data_source][var]
        proj_native = proj_cfg.get(file_key, ccrs.PlateCarree())

        data = preprocess_netcdf(
            source=data_source,
            file_path=file_path,
            var_name=var_name,
            months=months_local,
            years=years_load,
            lats=lats,
            lons=lons,
            land_only=land_only,
            trim_border=trim_local,
            rotpole_sel=rotpole_sel,
            rotpole_native=proj_native,
        ).squeeze()

    # If only raw requested, return early
    want_raw = 'raw' in return_items
    want_monthly = ('monthly' in return_items) or ('yearly' in return_items) or ('fit' in return_items) or ('avg' in return_items)
    want_yearly = ('yearly' in return_items) or ('fit' in return_items) or ('avg' in return_items)
    want_fit = 'fit' in return_items
    want_avg = 'avg' in return_items

    month_start = int(months_local[0])
    month_end = int(months_local[-1])
    y0, y1 = years_req[0], years_req[-1] if years_req is not None else (None, None)

    # Assign clim_year to raw if it is requested or needed for later filtering
    if want_raw or want_monthly:
        month_r = data['time'].dt.month
        year_r = data['time'].dt.year
        if month_start <= month_end:
            clim_year_r = year_r
        else:
            clim_year_r = xr.where(month_r >= month_start, year_r + 1, year_r)
        data = data.assign_coords(clim_year=clim_year_r)
        if years_req is not None:
            data = data.where((data['clim_year'] >= y0) & (data['clim_year'] <= y1), drop=True)

    # Monthly
    data_monthly = None
    if want_monthly:
        data_monthly = data.resample(time='MS').mean('time')
        data_monthly = data_monthly.where(data_monthly['time'].dt.month.isin(months_local), drop=True)

        month_m = data_monthly['time'].dt.month
        year_m = data_monthly['time'].dt.year
        if month_start <= month_end:
            clim_year_m = year_m
        else:
            clim_year_m = xr.where(month_m >= month_start, year_m + 1, year_m)

        data_monthly = data_monthly.assign_coords(clim_year=clim_year_m)
        if years_req is not None:
            data_monthly = data_monthly.where((data_monthly['clim_year'] >= y0) & (data_monthly['clim_year'] <= y1), drop=True)

    # Yearly (by clim_year)
    data_yearly = None
    data_year = None
    data_year_time = None
    if want_yearly:
        data_year = data_monthly.groupby('clim_year').mean('time')
        
        if years_req is not None:
            data_year = data_year.sel(clim_year=slice(y0, y1))

        # Convert clim_year coordinate to time for plotting / rolling
        time_coord = pd.to_datetime(data_year['clim_year'].astype(str))
        data_year_time = data_year.assign_coords(time=('clim_year', time_coord)).swap_dims({'clim_year': 'time'})
        data_yearly = data_year_time.copy()

    # Avg
    data_avg = None
    if want_avg:
        data_avg = data_year.mean(dim='clim_year').astype('float32')

    # Rolling mean for fit
    if want_fit:
        if rolling_mean_var:
            data_year_time = data_year_time.rolling(
                time=rolling_mean_years,
                center=True,
                min_periods=min_periods,
            ).mean()

        if fit_against_gmst:
            file_GMST = '/nobackup/users/walj/era5/era5_gmst_anom.nc'
            gmst = xr.open_dataset(file_GMST)['GMST']

            gmst_roll = gmst.rolling(
                time=rolling_mean_years,
                center=True,
                min_periods=min_periods,
            ).mean()

            fit_coord = gmst_roll.sel(time=data_year_time['time']).astype(float)
        else:
            fit_coord = data_year_time['clim_year'].astype(float)

        data_fit = (
            data_year_time
            .rename({'time': 'fit_against'})
            .assign_coords(fit_against=('fit_against', fit_coord.values))
        ).astype('float32')
    else:
        data_fit = None

    out = {}
    if 'raw' in return_items:
        out['raw'] = data
    if 'monthly' in return_items:
        out['monthly'] = data_monthly
    if 'yearly' in return_items:
        out['yearly'] = data_yearly
    if 'fit' in return_items:
        out['fit'] = data_fit
    if 'avg' in return_items:
        out['avg'] = data_avg

    return out





# def process_source(data_source,
#                    var,
#                    data_sources=data_sources,
#                    station_sources=station_sources,
#                    var_name_cfg=var_name_cfg,
#                    file_cfg=file_cfg,
#                    proj_cfg=proj_cfg,
#                    months=None,
#                    years=None,
#                    lats=None,
#                    lons=None,
#                    land_only=False,
#                    trim_border=None,
#                    rotpole_sel=ccrs.PlateCarree(),
#                    rolling_mean_var=False,
#                    fit_against_gmst=False,
#                    rolling_mean_years=1,
#                    min_periods=1):

#     if months is None:
#         months_local = np.arange(1, 13)
#     else:
#         months_local = np.asarray(months, dtype=int)

#     if years is None:
#         years_req = None
#         years_load = None
#     else:
#         years_req = list(years)
#         years_load = list(years_req)

#         month_start = months_local[0]
#         month_end = months_local[-1]

#         if month_start > month_end:
#             years_load[0] = years_req[0] - 1

#     trim_local = trim_border
#     if data_source == 'RACMO2.4' and trim_border is None:
#         trim_local = 8

#     if data_source not in station_sources:

#         file_key = next(src for src in data_sources if src in data_source)
#         var_name = var_name_cfg[file_key][var]
#         file = file_cfg[data_source][var]
#         proj_native = proj_cfg.get(file_key, ccrs.PlateCarree())

#         data = preprocess_netcdf(
#             source=data_source,
#             file_path=file,
#             var_name=var_name,
#             months=months_local,
#             years=years_load,
#             lats=lats,
#             lons=lons,
#             land_only=land_only,
#             trim_border=trim_local,
#             rotpole_sel=rotpole_sel,
#             rotpole_native=proj_native
#         ).squeeze()

#     elif data_source in station_sources:
        
#         var_name = var_name_cfg['Station'][var]
#         file = file_cfg['Station'][data_source]

#         data = preprocess_station(
#             file_path=file,
#             var_name=var_name,
#             months=months_local,
#             years=years_load,
#         ).squeeze()

#     month_d = data['time'].dt.month
#     year_d = data['time'].dt.year

#     month_start = months_local[0]
#     month_end = months_local[-1]

#     if month_start <= month_end:
#         clim_year_d = year_d
#     else:
#         clim_year_d = xr.where(month_d >= month_start, year_d + 1, year_d)

#     data = data.assign_coords(clim_year=clim_year_d)

#     data_monthly = data.resample(time='MS').mean('time')
#     existing = pd.DatetimeIndex(data['time'].values).to_period('M').unique().to_timestamp()
#     data_monthly = data_monthly.sel(time=existing)

#     month_m = data_monthly['time'].dt.month
#     year_m = data_monthly['time'].dt.year

#     if month_start <= month_end:
#         clim_year_m = year_m
#     else:
#         clim_year_m = xr.where(month_m >= month_start, year_m + 1, year_m)

#     data_monthly = data_monthly.assign_coords(clim_year=clim_year_m)

#     data_year = data_monthly.groupby('clim_year').mean('time')

#     if years_req is not None:
#         y0, y1 = years_req[0], years_req[-1]

#         data = data.where((data['clim_year'] >= y0) & (data['clim_year'] <= y1), drop=True)
#         data_monthly = data_monthly.where(
#             (data_monthly['clim_year'] >= y0) & (data_monthly['clim_year'] <= y1),
#             drop=True
#         )

#         data_year = data_year.sel(clim_year=slice(y0, y1))

#     data_avg = data_year.mean(dim='clim_year').astype('float32')

#     time_coord = pd.to_datetime(data_year['clim_year'].astype(str))
#     data_year_time = data_year.assign_coords(time=('clim_year', time_coord)).swap_dims({'clim_year': 'time'})
#     data_yearly = data_year_time.copy()

#     if rolling_mean_var:
#         data_year_time = data_year_time.rolling(
#             time=rolling_mean_years,
#             center=True,
#             min_periods=min_periods
#         ).mean()

#     if fit_against_gmst:
#         file_GMST = '/nobackup/users/walj/era5/era5_gmst_anom.nc'
#         data_GMST = xr.open_dataset(file_GMST)

#         gmst_roll = data_GMST.rolling(
#             time=rolling_mean_years,
#             center=True,
#             min_periods=min_periods
#         ).mean()

#         gmst_full = gmst_roll['GMST']

#         gmst_sel = gmst_full.sel(time=data_year_time['time'])
#         fit_coord = gmst_sel.astype(float)

#     else:
#         fit_coord = data_year_time['clim_year'].astype(float)

#     data_fit = (
#         data_year_time
#         .rename({'time': 'fit_against'})
#         .assign_coords(fit_against=('fit_against', fit_coord.values))
#     ).astype('float32')

#     return data, data_monthly, data_yearly, data_fit, data_avg