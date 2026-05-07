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
from RegionalTrends.Helpers.Config.Paths import build_file_cfg

import RegionalTrends.Helpers.Config.Constants as Constants
reload(Constants)

enbud_vars = Constants.ENBUD_VARS

DEFAULT_TRIM_BORDERS = {
    'RACMO2.4_KEXT06': 8,
    'RACMO2.4_KEXT12': 4,
    'RACMO2.4': 4,
    'RACMO2.4A': 4
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

    file_cfg = build_file_cfg(file_freq)
    
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

    if file_freq == 'Monthly' and not is_monthly_time(data['time']) and var_to_load not in enbud_vars:
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


def Albedocs(**kwargs):
    SWin = load_single_var(var_to_load='SWincs', **kwargs)
    SWnet = load_single_var(var_to_load='SWnetcs', **kwargs)
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


def senstot(**kwargs):
    consens = load_single_var(var_to_load='consens', **kwargs)
    vdfsens = load_single_var(var_to_load='vdfsens', **kwargs)
    return consens + vdfsens


def phasetot(**kwargs):
    conphase = load_single_var(var_to_load='conphase', **kwargs)
    vdfphase = load_single_var(var_to_load='vdfphase', **kwargs)
    lscld = load_single_var(var_to_load='lscld', **kwargs)
    return conphase + vdfphase + lscld


def frictot(**kwargs):
    confric = load_single_var(var_to_load='confric', **kwargs)
    vdffric = load_single_var(var_to_load='vdffric', **kwargs)
    return confric + vdffric


def numtot(**kwargs):
    numbnd = load_single_var(var_to_load='numbnd', **kwargs)
    numdif = load_single_var(var_to_load='numdif', **kwargs)
    return numbnd + numdif


def adiadj(**kwargs):
    orography = load_single_var(var_to_load='orography', **kwargs)
    adicomp = load_single_var(var_to_load='adicomp', **kwargs)
    return orography + adicomp


def swtop(**kwargs):
    swtopdn = load_single_var(var_to_load='swtopdn', **kwargs)
    swtopup = load_single_var(var_to_load='swtopup', **kwargs)
    return swtopdn + swtopup


def swbot(**kwargs):
    swbotdn = load_single_var(var_to_load='swbotdn', **kwargs)
    swbotup = load_single_var(var_to_load='swbotup', **kwargs)
    return swbotdn + swbotup


def lwtop(**kwargs):
    lwtopdn = load_single_var(var_to_load='lwtopdn', **kwargs)
    lwtopup = load_single_var(var_to_load='lwtopup', **kwargs)
    return lwtopdn + lwtopup


def lwbot(**kwargs):
    lwbotdn = load_single_var(var_to_load='lwbotdn', **kwargs)
    lwbotup = load_single_var(var_to_load='lwbotup', **kwargs)
    return lwbotdn + lwbotup


def sctop(**kwargs):
    sctopdn = load_single_var(var_to_load='sctopdn', **kwargs)
    sctopup = load_single_var(var_to_load='sctopup', **kwargs)
    return sctopdn + sctopup


def scbot(**kwargs):
    scbotdn = load_single_var(var_to_load='scbotdn', **kwargs)
    scbotup = load_single_var(var_to_load='scbotup', **kwargs)
    return scbotdn + scbotup


def lctop(**kwargs):
    lctopdn = load_single_var(var_to_load='lctopdn', **kwargs)
    lctopup = load_single_var(var_to_load='lctopup', **kwargs)
    return lctopdn + lctopup


def lcbot(**kwargs):
    lcbotdn = load_single_var(var_to_load='lcbotdn', **kwargs)
    lcbotup = load_single_var(var_to_load='lcbotup', **kwargs)
    return lcbotdn + lcbotup


def diabatic(**kwargs):
    sens = senstot(**kwargs)
    phase = phasetot(**kwargs)
    fric = frictot(**kwargs)
    rad = load_single_var(var_to_load='radtot', **kwargs)
    return sens + phase + fric + rad


DERIVED_VARS = {
    'RH_proxy': RH_proxy,
    'Bowen': Bowen,
    'Albedo': Albedo,
    'Albedocs': Albedocs,
    'Q_era': Q_from_era,
    'Q_obs': Q_from_obs,
    'P_rel': P_rel,
    'Rnet': Rnet,
    'senstot': senstot,
    'phasetot': phasetot,
    'frictot': frictot,
    'numtot': numtot,
    'adiadj': adiadj,
    'swtop': swtop,
    'swbot': swbot,
    'lwtop': lwtop,
    'lwbot': lwbot,
    'sctop': sctop,
    'scbot': scbot,
    'lctop': lctop,
    'lcbot': lcbot,
    'diabatic': diabatic,
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
