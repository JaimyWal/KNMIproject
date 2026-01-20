import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from importlib import reload

from RegionalTrends.Helpers import ProcessVar
reload(ProcessVar)
from RegionalTrends.Helpers.ProcessVar import load_var


def process_source(
    data_source,
    var,
    data_sources,
    station_sources,
    file_freq,
    var_name_cfg,
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
    if (data_source == 'RACMO2.4_KEXT06') and (trim_border is None):
        trim_local = 8
    elif (data_source == 'RACMO2.4_KEXT12') and (trim_border is None):
        trim_local = 4

    load_args = dict(
        data_source=data_source,
        data_sources=data_sources,
        station_sources=station_sources,
        file_freq=file_freq,
        var_name_cfg=var_name_cfg,
        proj_cfg=proj_cfg,
        months_local=months_local,
        years_load=years_load,
        lats=lats,
        lons=lons,
        land_only=land_only,
        trim_local=trim_local,
        rotpole_sel=rotpole_sel,
    )

    data = load_var(var=var, **load_args)

    use_max = (var == 'Tmaxmax')

    # If only raw requested, return early
    want_raw = 'raw' in return_items
    want_monthly = ('monthly' in return_items) or ('yearly' in return_items) or ('fit' in return_items) or ('avg' in return_items)
    want_yearly = ('yearly' in return_items) or ('fit' in return_items) or ('avg' in return_items)
    want_fit = 'fit' in return_items
    want_avg = 'avg' in return_items

    month_start = int(months_local[0])
    month_end = int(months_local[-1])

    if years_req is None:
        y0, y1 = None, None
    else:
        y0, y1 = years_req[0], years_req[-1]

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
        if use_max:
            data_monthly = data.resample(time='MS').max('time')
        else:
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
        if use_max:
            data_year = data_monthly.groupby('clim_year').max('time')
        else:
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
            gmst = xr.open_dataset(file_GMST)['__xarray_dataarray_variable__']

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
