import numpy as np
import cartopy.crs as ccrs
import xarray as xr
from importlib import reload

from RegionalTrends.Helpers import ProcessNetCDF
reload(ProcessNetCDF)          
from RegionalTrends.Helpers.ProcessNetCDF import preprocess_netcdf, is_monthly_time

from RegionalTrends.Helpers import ProcessStation
reload(ProcessStation)          
from RegionalTrends.Helpers.ProcessStation import preprocess_station

import RegionalTrends.Helpers.Config.Paths as Paths
reload(Paths)
from RegionalTrends.Helpers.Config.Paths import build_file_cfg, freq_tags


DEFAULT_TRIM_BORDERS = {
    'RACMO2.4_KEXT06': 8,
    'RACMO2.4_KEXT12': 4,
    'RACMO2.4': 4,
}


def load_single_var(
    var_to_load,
    data_source,
    data_sources,
    station_sources,
    file_freq,
    var_file_cfg,
    proj_cfg,
    months=None,
    years=None,
    lats=None,
    lons=None,
    land_only=False,
    trim_border=None,
    rotpole_sel=ccrs.PlateCarree(),
    station_coords=None,
):

    freq_str, racmo24_sep, racmo23_base = freq_tags(file_freq)
    file_cfg = build_file_cfg(freq_str, racmo24_sep, racmo23_base)
    
    if trim_border is None:
        trim_border = DEFAULT_TRIM_BORDERS.get(data_source, None)
    
    if data_source in station_sources:
        var_name = var_file_cfg['Station'][var_to_load]
        file_path = file_cfg['Station'][data_source]
        data = preprocess_station(
            file_path=file_path,
            var_name=var_name,
            months=months,
            years=years,
        ).squeeze()

    else:
        file_key = next(src for src in data_sources if src in data_source)
        var_name = var_file_cfg[file_key][var_to_load]
        file_path = file_cfg[data_source][var_to_load]
        proj_native = proj_cfg.get(file_key, ccrs.PlateCarree())
        data = preprocess_netcdf(
            source=data_source,
            file_path=file_path,
            var_name=var_name,
            months=months,
            years=years,
            lats=lats,
            lons=lons,
            land_only=land_only,
            trim_border=trim_border,
            rotpole_sel=rotpole_sel,
            rotpole_native=proj_native,
            station_coords=station_coords,
        ).squeeze()

    if file_freq == 'Monthly' and not is_monthly_time(data['time']):
        data = data.resample(time='MS').mean('time')          

    return data


def teten_es(temp_c):
    return 6.112*np.exp((17.67 * temp_c) / (temp_c + 243.5))


def specifc_humidity(vapor_pressure, total_pressure):
    return 0.622*vapor_pressure / (total_pressure - 0.378*vapor_pressure)


def RH_proxy(**kwargs):
    T2m = load_single_var(var_to_load='Tg', **kwargs)
    Tdew2m = load_single_var(var_to_load='Tdew', **kwargs)
    es_tdew = teten_es(Tdew2m)
    es_t = teten_es(T2m)
    return ((es_tdew / es_t) * 100.0).clip(0, 100)


def Bowen(**kwargs):
    SHF = load_single_var(var_to_load='SHF', **kwargs)
    LHF = load_single_var(var_to_load='LHF', **kwargs)
    LHF_safe = LHF.where(np.abs(LHF) > 5)
    return SHF / LHF_safe


def Albedo(**kwargs):
    SWin = load_single_var(var_to_load='SWin', **kwargs)
    SWnet = load_single_var(var_to_load='SWnet', **kwargs)
    SWout = SWin - SWnet
    SWin_safe = SWin.where(np.abs(SWin) > 5)
    return SWout / SWin_safe


def Q_from_era(**kwargs):
    Tdew= load_single_var(var_to_load='Tdew', **kwargs)
    Ps = load_single_var(var_to_load='Ps', **kwargs)

    vapor_pressure = teten_es(Tdew)
    q = specifc_humidity(vapor_pressure, Ps)*1000
    return q


def Q_from_obs(**kwargs):
    T2m = load_single_var(var_to_load='Tg', **kwargs)
    Psl = load_single_var(var_to_load='Psl', **kwargs)
    RH = load_single_var(var_to_load='RH', **kwargs)

    T2m, Psl, RH = xr.align(T2m, Psl, RH, join='override')

    es = teten_es(T2m)
    vapor_pressure = (RH / 100.0)*es
    q = specifc_humidity(vapor_pressure, Psl)*1000
    return q


def P_rel(**kwargs):
    precip = load_single_var(var_to_load='P', **kwargs)
    precip_clim = precip.groupby('time.month').mean('time')
    p_rel = (precip.groupby('time.month') / precip_clim) * 100
    return p_rel


def Rnet(**kwargs):
    SWnet = load_single_var(var_to_load='SWnet', **kwargs)
    LWnet = load_single_var(var_to_load='LWnet', **kwargs)
    return SWnet + LWnet


# Also add advection here for example as well. However, this requires differentiation in space
# and therefore also need to convert from rotated grid to normal grid...

DERIVED_VARS = {
    'RH_proxy': RH_proxy,
    'Bowen': Bowen,
    'Albedo': Albedo,
    'Q_era': Q_from_era,
    'Q_obs': Q_from_obs,
    'P_rel': P_rel,
    'Rnet': Rnet,
}


FALLBACK_VARS = {
    'Q_all': ['Q', 'Q_era', 'Q_obs'],
    'RH_all': ['RH', 'RH_proxy'],
}


def load_var(
    var,
    data_source,
    data_sources,
    station_sources,
    file_freq,
    var_file_cfg,
    proj_cfg,
    months=None,
    years=None,
    lats=None,
    lons=None,
    land_only=False,
    trim_border=None,
    rotpole_sel=ccrs.PlateCarree(),
    station_coords=None,
):

    load_args = dict(
        data_source=data_source,
        data_sources=data_sources,
        station_sources=station_sources,
        file_freq=file_freq,
        var_file_cfg=var_file_cfg,
        proj_cfg=proj_cfg,
        months=months,
        years=years,
        lats=lats,
        lons=lons,
        land_only=land_only,
        trim_border=trim_border,
        rotpole_sel=rotpole_sel,
        station_coords=station_coords,
    )

    # Handle fallback variables (try each in order until one works)
    if var in FALLBACK_VARS:
        fallback_list = FALLBACK_VARS[var]
        errors = []
        
        for fallback_var in fallback_list:
            try:
                result = load_var(fallback_var, **load_args)

                if result is not None and result.size > 0:
                    return result
            except Exception as e:
                errors.append(f'{fallback_var}: {e}')
                continue  # Try next fallback
        
        raise ValueError(
            f"All fallback variables for '{var}' failed for data source '{data_source}':\n" +
            "\n".join(errors)
        )
            
    if var in DERIVED_VARS:
        return DERIVED_VARS[var](**load_args)
    
    return load_single_var(var_to_load=var, **load_args)
