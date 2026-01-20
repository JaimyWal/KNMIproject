import numpy as np
import cartopy.crs as ccrs
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


def load_single_var(
    var_to_load,
    data_source,
    data_sources,
    station_sources,
    file_freq,
    var_name_cfg,
    proj_cfg,
    months_local,
    years_load,
    lats,
    lons,
    land_only,
    trim_local,
    rotpole_sel,
):
    
    freq_str, racmo24_sep = freq_tags(file_freq)
    file_cfg = build_file_cfg(freq_str, racmo24_sep)
    
    if data_source in station_sources:
        var_name = var_name_cfg['Station'][var_to_load]
        file_path = file_cfg['Station'][data_source]
        data = preprocess_station(
            file_path=file_path,
            var_name=var_name,
            months=months_local,
            years=years_load,
        ).squeeze()

    else:
        file_key = next(src for src in data_sources if src in data_source)
        var_name = var_name_cfg[file_key][var_to_load]
        file_path = file_cfg[data_source][var_to_load]
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

    if file_freq == 'Monthly' and not is_monthly_time(data['time']):
        data = data.resample(time='MS').mean('time')          

    return data


def RH_proxy(**kwargs):

    def teten_es(temp_c):
        return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))

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



# Also add advection here for example as well. However, this requires differentiation in space
# and therefore also need to convert from rotated grid to normal grid...

DERIVED_VARS = {
    'RH_proxy': RH_proxy,
    'Bowen': Bowen,
}


def load_var(var, **load_args):
    if var in DERIVED_VARS:
        return DERIVED_VARS[var](**load_args)
    return load_single_var(var_to_load=var, **load_args)
