import os


def freq_tags(monthly_or_daily):
    if monthly_or_daily == 'Daily':
        return 'Daily', '.', '_ERIK'
    elif monthly_or_daily == 'Monthly':
        return 'Monthly', '_', '_JAIMY'



BASE_DIR_CFG = {
    'Eobs': '/nobackup/users/walj/eobs',
    'ERA5': '/nobackup/users/walj/era5',
    'RACMO2.3_JAIMY': '/nobackup/users/walj/racmo23',
    'RACMO2.3_CHRISTIAAN': '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5',
    'RACMO2.3_ERIK': '/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5',
    'RACMO2.4_KEXT06': '/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning',
    'RACMO2.4_KEXT12': '/nobackup/users/walj/racmo24',
    'Station': '/nobackup/users/walj/knmi',
}


def build_file_cfg(freq_str, racmo24_sep, racmo23_base):

    def connect_paths(key, file_path):
        return os.path.join(BASE_DIR_CFG[key], file_path)

    cfg = {
        'Eobs': {
            'Tg':  connect_paths('Eobs', f'{freq_str}/tg_ens_mean_0.1deg_reg_v31.0e*.nc'),
            'P': connect_paths('Eobs', f'{freq_str}/rr_ens_mean_0.1deg_reg_v31.0e*.nc'),
            'SWin': connect_paths('Eobs', f'{freq_str}/qq_ens_mean_0.1deg_reg_v31.0e*.nc'),
            'Tmax': connect_paths('Eobs', f'{freq_str}/tx_ens_mean_0.1deg_reg_v31.0e*.nc'),
            'Tmin': connect_paths('Eobs', f'{freq_str}/tn_ens_mean_0.1deg_reg_v31.0e*.nc'),
            'RH': connect_paths('Eobs', f'{freq_str}/hu_ens_mean_0.1deg_reg_v31.0e*.nc'),
            'Psl': connect_paths('Eobs', f'{freq_str}/pp_ens_mean_0.1deg_reg_v32.0e*.nc'),
        },

        'Eobs_coarse': {
            'Tg': connect_paths('Eobs', f'{freq_str}/tg_ens_mean_0.25deg_reg_v31.0e*.nc'),
            'P': connect_paths('Eobs', f'{freq_str}/rr_ens_mean_0.25deg_reg_v31.0e*.nc'),
        },
    }

    if freq_str == 'Daily':
        cfg['ERA5L'] = {}
        cfg['ERA5'] = {
            'Psl': connect_paths('ERA5', 'Daily/era5_msl_daily_eu.nc'),
            'Ps': connect_paths('ERA5', 'Daily/era5_sp_daily_eu.nc'),
            'Tmax': connect_paths('ERA5', 'Daily/era5_tmax_daily_eu.nc'),
            'Tmin': connect_paths('ERA5', 'Daily/era5_tmin_daily_eu.nc'),
        }

    elif freq_str == 'Monthly':
        cfg['ERA5L'] = {
            'Tg': connect_paths('ERA5', 'Monthly/era5_land_gen.nc'),
            'P': connect_paths('ERA5', 'Monthly/era5_land_gen.nc'),
            'Ps': connect_paths('ERA5', 'Monthly/era5_land_gen.nc'),
            'Tdew': connect_paths('ERA5', 'Monthly/era5_land_gen.nc'),
            'Ts': connect_paths('ERA5', 'Monthly/era5_land_soil.nc'),
            'SWin': connect_paths('ERA5', 'Monthly/era5_land_fluxes.nc'),
            'SWnet': connect_paths('ERA5', 'Monthly/era5_land_fluxes.nc'),
            'LWin': connect_paths('ERA5', 'Monthly/era5_land_fluxes.nc'),
            'LWnet': connect_paths('ERA5', 'Monthly/era5_land_fluxes.nc'),
            'SHF': connect_paths('ERA5', 'Monthly/era5_land_fluxes.nc'),
            'LHF': connect_paths('ERA5', 'Monthly/era5_land_fluxes.nc'),
            'SWC': connect_paths('ERA5', 'Monthly/era5_land_soil.nc'),
            'SWVL1': connect_paths('ERA5', 'Monthly/era5_land_soil.nc'),
            'SWVL2': connect_paths('ERA5', 'Monthly/era5_land_soil.nc'),
            'SWVL3': connect_paths('ERA5', 'Monthly/era5_land_soil.nc'),
            'SWVL4': connect_paths('ERA5', 'Monthly/era5_land_soil.nc'),
        }
        cfg['ERA5'] = {
            'Tg': connect_paths('ERA5', 'Monthly/era5_coarse_t2m_ps.nc'),
            'Tdew': connect_paths('ERA5', 'Monthly/era5_coarse_dew_wind.nc'),
            'P': connect_paths('ERA5', 'Monthly/era5_coarse_tp.nc'),
            'Ps': connect_paths('ERA5', 'Monthly/era5_coarse_t2m_ps.nc'),
            'Psl': connect_paths('ERA5', 'Monthly/era5_coarse_t2m_ps.nc'),
            'SWin': connect_paths('ERA5', 'Monthly/era5_coarse_sw_shf.nc'),
            'SWnet': connect_paths('ERA5', 'Monthly/era5_coarse_sw_shf.nc'),
            'SWincs': connect_paths('ERA5', 'Monthly/era5_coarse_sw_shf.nc'),
            'SWnetcs': connect_paths('ERA5', 'Monthly/era5_coarse_sw_shf.nc'),
            'LWin': connect_paths('ERA5', 'Monthly/era5_coarse_lw_lhf.nc'),
            'LWnet': connect_paths('ERA5', 'Monthly/era5_coarse_lw_lhf.nc'),
            'LWincs': connect_paths('ERA5', 'Monthly/era5_coarse_lw_lhf.nc'),
            'LWnetcs': connect_paths('ERA5', 'Monthly/era5_coarse_lw_lhf.nc'),
            'SHF': connect_paths('ERA5', 'Monthly/era5_coarse_sw_shf.nc'),
            'LHF': connect_paths('ERA5', 'Monthly/era5_coarse_lw_lhf.nc'),
            'CloudLow': connect_paths('ERA5', 'Monthly/era5_coarse_clouds.nc'),
            'CloudMid': connect_paths('ERA5', 'Monthly/era5_coarse_clouds.nc'),
            'CloudHigh': connect_paths('ERA5', 'Monthly/era5_coarse_clouds.nc'),
            'CloudTotal': connect_paths('ERA5', 'Monthly/era5_coarse_clouds.nc'),
            'LWP': connect_paths('ERA5', 'Monthly/era5_coarse_clouds.nc'),
            'IWP': connect_paths('ERA5', 'Monthly/era5_coarse_clouds.nc'),
            'Ts': connect_paths('ERA5', 'Monthly/era5_coarse_soil_skin.nc'),
            'SWC': connect_paths('ERA5', 'Monthly/era5_coarse_soil_skin.nc'),
            'SWVL1': connect_paths('ERA5', 'Monthly/era5_coarse_soil_skin.nc'),
            'SWVL2': connect_paths('ERA5', 'Monthly/era5_coarse_soil_skin.nc'),
            'SWVL3': connect_paths('ERA5', 'Monthly/era5_coarse_soil_skin.nc'),
            'SWVL4': connect_paths('ERA5', 'Monthly/era5_coarse_soil_skin.nc'),
            'TWC': connect_paths('ERA5', 'Monthly/era5_coarse_water.nc'),
            'TWV': connect_paths('ERA5', 'Monthly/era5_coarse_water.nc'),
            'TSWin': connect_paths('ERA5', 'Monthly/era5_coarse_toa.nc'),
            'TSWnet': connect_paths('ERA5', 'Monthly/era5_coarse_toa.nc'),
            'TLWnet': connect_paths('ERA5', 'Monthly/era5_coarse_toa.nc'),
            'TSWnetcs': connect_paths('ERA5', 'Monthly/era5_coarse_toa.nc'),
            'TLWnetcs': connect_paths('ERA5', 'Monthly/era5_coarse_toa.nc'),
        }

    racmo23_key = f'RACMO2.3{racmo23_base}'
    cfg['RACMO2.3'] = {
        'Tg': connect_paths('RACMO2.3_CHRISTIAAN', f'{freq_str}_data/t2m/*.nc'),
        'P': connect_paths('RACMO2.3_CHRISTIAAN', f'{freq_str}_data/precip/*.nc'),
        'Sq': connect_paths('RACMO2.3_CHRISTIAAN', f'{freq_str}_data/sund/*.nc'),
        'SWin': connect_paths(racmo23_key, f'{freq_str}_data/swsd/*.nc'),
        'SWnet': connect_paths(racmo23_key, f'{freq_str}_data/swsn/*.nc'),
        'SWincs': connect_paths(racmo23_key, f'{freq_str}_data/swsdcs/*.nc'),
        'SWnetcs': connect_paths(racmo23_key, f'{freq_str}_data/swsncs/*.nc'),
        'LWin': connect_paths(racmo23_key, f'{freq_str}_data/lwsd/*.nc'),
        'LWnet': connect_paths(racmo23_key, f'{freq_str}_data/lwsn/*.nc'),
        'LWincs': connect_paths(racmo23_key, f'{freq_str}_data/lwsdcs/*.nc'),
        'LWnetcs': connect_paths(racmo23_key, f'{freq_str}_data/lwsncs/*.nc'),
        'SHF': connect_paths(racmo23_key, f'{freq_str}_data/senf/*.nc'),
        'LHF': connect_paths(racmo23_key, f'{freq_str}_data/latf/*.nc'),
        'CloudLow': connect_paths(racmo23_key, f'{freq_str}_data/aclcovL/*.nc'),
        'CloudMid': connect_paths(racmo23_key, f'{freq_str}_data/aclcovM/*.nc'),
        'CloudHigh': connect_paths(racmo23_key, f'{freq_str}_data/aclcovH/*.nc'),
        'CloudTotal': connect_paths(racmo23_key, f'{freq_str}_data/aclcov/*.nc'),
        'LWP': connect_paths(racmo23_key, f'{freq_str}_data/qli/*.nc'),
        'IWP': connect_paths(racmo23_key, f'{freq_str}_data/qii/*.nc'),
    }

    cfg['RACMO2.4_KEXT06'] = {
        'Tg': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/tas{racmo24_sep}*.nc'),
        'Tdew': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/tdew2m{racmo24_sep}*.nc'),
        'P': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/pr{racmo24_sep}*.nc'),
        'Sq': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/sund{racmo24_sep}*.nc'),
        'SWin': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/rsds{racmo24_sep}*.nc'),
        'SWnet': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/ssr{racmo24_sep}*.nc'),
        'SWincs': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/rsdscs{racmo24_sep}*.nc'),
        'SWnetcs': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/ssrc{racmo24_sep}*.nc'),
        'LWin': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/rlds{racmo24_sep}*.nc'),
        'LWnet': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/str{racmo24_sep}*.nc'),
        'LWincs': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/rldscs{racmo24_sep}*.nc'),
        'LWnetcs': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/strc{racmo24_sep}*.nc'),
        'SHF': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/hfss{racmo24_sep}*.nc'),
        'LHF': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/hfls{racmo24_sep}*.nc'),
        'CloudLow': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/cll{racmo24_sep}*.nc'),
        'CloudMid': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/clm{racmo24_sep}*.nc'),
        'CloudHigh': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/clh{racmo24_sep}*.nc'),
        'CloudTotal': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/clt{racmo24_sep}*.nc'),
        'LWP': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/clwvi{racmo24_sep}*.nc'),
        'IWP': connect_paths('RACMO2.4_KEXT06', f'{freq_str}/clivi{racmo24_sep}*.nc'),
    }

    cfg['RACMO2.4'] = {
        'Tg': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/tas{racmo24_sep}*.nc'),
        'Tmax': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/tasmax{racmo24_sep}*.nc'),
        'Tmin': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/tasmin{racmo24_sep}*.nc'),
        'P': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/pr{racmo24_sep}*.nc'),
        'Ps': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/ps{racmo24_sep}*.nc'),
        'Psl': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/psl{racmo24_sep}*.nc'),
        'RH': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/hurs{racmo24_sep}*.nc'),
        'Tdew': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/tdew2m{racmo24_sep}*.nc'),
        'Sq': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/sund{racmo24_sep}*.nc'),
        'SWin': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/rsds{racmo24_sep}*.nc'),
        'SWnet': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/ssr{racmo24_sep}*.nc'),
        'SWincs': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/rsdscs{racmo24_sep}*.nc'),
        'SWnetcs': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/ssrc{racmo24_sep}*.nc'),
        'LWin': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/rlds{racmo24_sep}*.nc'),
        'LWnet': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/str{racmo24_sep}*.nc'),
        'LWincs': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/rldscs{racmo24_sep}*.nc'),
        'LWnetcs': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/strc{racmo24_sep}*.nc'),
        'SHF': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/hfss{racmo24_sep}*.nc'),
        'LHF': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/hfls{racmo24_sep}*.nc'),
        'CloudLow': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/cll{racmo24_sep}*.nc'),
        'CloudMid': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/clm{racmo24_sep}*.nc'),
        'CloudHigh': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/clh{racmo24_sep}*.nc'),
        'CloudTotal': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/clt{racmo24_sep}*.nc'),
        'LWP': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/clwvi{racmo24_sep}*.nc'),
        'IWP': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/clivi{racmo24_sep}*.nc'),
        'Q': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/huss{racmo24_sep}*.nc'),
        'Ts': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/ts{racmo24_sep}*.nc'),
        'SWC': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/wskin{racmo24_sep}*.nc'),
        'SWVL1': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/swvl1{racmo24_sep}*.nc'),
        'SWVL2': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/swvl2{racmo24_sep}*.nc'),
        'SWVL3': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/swvl3{racmo24_sep}*.nc'),
        'SWVL4': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/swvl4{racmo24_sep}*.nc'),
        'TWC': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/tcw{racmo24_sep}*.nc'),
        'TWV': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/prw{racmo24_sep}*.nc'),
        'TSWin': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/rsdt{racmo24_sep}*.nc'),
        'TSWnet': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/tsr{racmo24_sep}*.nc'),
        'TLWnet': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/toptr{racmo24_sep}*.nc'),
        'TSWnetcs': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/tsrc{racmo24_sep}*.nc'),
        'TLWnetcs': connect_paths('RACMO2.4_KEXT12', f'{freq_str}/ttrc{racmo24_sep}*.nc'),
    }

    freq_str = 'Daily'
    cfg['Station'] = {
        'Bilt': connect_paths('Station', f'{freq_str}/KNMI_Bilt.txt'),
        'Cabauw': connect_paths('Station', f'{freq_str}/KNMI_Cabauw.txt'),
        'Eelde': connect_paths('Station', f'{freq_str}/KNMI_Eelde.txt'),
        'Maastricht': connect_paths('Station', f'{freq_str}/KNMI_Maastricht.txt'),
        'Vlissingen': connect_paths('Station', f'{freq_str}/KNMI_Vlissingen.txt'),
        'Kooy': connect_paths('Station', f'{freq_str}/KNMI_Kooy.txt'),
    }

    return cfg
