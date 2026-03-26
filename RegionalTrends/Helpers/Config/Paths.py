import os


BASE_DIR_CFG = {
    'Eobs': '/nobackup/users/walj/eobs',
    'ERA5': '/nobackup/users/walj/era5',
    'RACMO2.3_JAIMY': '/nobackup/users/walj/racmo23',
    'RACMO2.3_CHRISTIAAN': '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5',
    'RACMO2.3_ERIK': '/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5',
    'RACMO2.4_KEXT06': '/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning',
    'RACMO2.4_KEXT12': '/nobackup/users/walj/racmo24',
    'RACMO2.4A_1H': '/nobackup_1/users/walj/racmo24/Hourly',
    'RACMO2.4A_6H': '/nobackup_1/users/walj/racmo24/Sixhourly',
    'RACMO2.4A_3H': '/nobackup_1/users/walj/racmo24/Threehourly',
    'RACMO2.4A_Daily': '/nobackup_1/users/walj/racmo24/Daily',
    'RACMO2.4A_Monthly': '/nobackup_1/users/walj/racmo24/Monthly',
    'RACMO2.4A_Seasonal': '/nobackup_1/users/walj/racmo24/Seasonal',
    'Station': '/nobackup/users/walj/knmi',
}


def build_file_cfg(freq_str):

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

    if freq_str == 'Daily':
        racmo23_base = '_ERIK'
    else:
        racmo23_base = '_JAIMY'
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

    if freq_str == 'Monthly' or freq_str == 'Seasonal':
        racmo24_sep = '_'
    else:
        racmo24_sep = '.'
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
    
    if freq_str == 'Daily':
        racmo24_1honly = 'RACMO2.4A_Daily'
        racmo24_1hboth = 'RACMO2.4A_Daily'
        racmo24_6honly = 'RACMO2.4A_Daily'
        racmo24_6hboth = 'RACMO2.4A_Daily'
        racmo24_dailyonly = 'RACMO2.4A_Daily'
        racmo24_enbud = 'RACMO2.4A_Daily'
        racmo24_enbudsep = '_'
    elif freq_str == 'Monthly':
        racmo24_1honly = 'RACMO2.4A_Monthly'
        racmo24_1hboth = 'RACMO2.4A_Monthly'
        racmo24_6honly = 'RACMO2.4A_Monthly'
        racmo24_6hboth = 'RACMO2.4A_Monthly'
        racmo24_dailyonly = 'RACMO2.4A_Monthly'
        racmo24_enbud = 'RACMO2.4A_Monthly'
        racmo24_enbudsep = '_'
    elif freq_str == 'Seasonal':
        racmo24_1honly = 'RACMO2.4A_Seasonal'
        racmo24_1hboth = 'RACMO2.4A_Seasonal'
        racmo24_6honly = 'RACMO2.4A_Seasonal'
        racmo24_6hboth = 'RACMO2.4A_Seasonal'
        racmo24_dailyonly = 'RACMO2.4A_Seasonal'
        racmo24_enbud = 'RACMO2.4A_Seasonal'
        racmo24_enbudsep = '_'
    else:
        racmo24_1honly = 'RACMO2.4A_Daily'
        racmo24_1hboth = 'RACMO2.4A_Daily'
        racmo24_6honly = 'RACMO2.4A_Daily'
        racmo24_6hboth = 'RACMO2.4A_Daily'
        racmo24_dailyonly = 'RACMO2.4A_Daily'
        racmo24_enbud = 'RACMO2.4A_3H'
        racmo24_enbudsep = '.'
    cfg['RACMO2.4A'] = {
        'Tg': connect_paths(racmo24_1hboth, f'tas{racmo24_sep}*.nc'),
        'Tmax': connect_paths(racmo24_dailyonly, f'tasmax{racmo24_sep}*.nc'),
        'Tmin': connect_paths(racmo24_dailyonly, f'tasmin{racmo24_sep}*.nc'),
        'P': connect_paths(racmo24_1hboth, f'pr{racmo24_sep}*.nc'),
        'Ps': connect_paths(racmo24_1hboth, f'ps{racmo24_sep}*.nc'),
        'Psl': connect_paths(racmo24_1hboth, f'psl{racmo24_sep}*.nc'),
        'RH': connect_paths(racmo24_1hboth, f'hurs{racmo24_sep}*.nc'),
        'Tdew': connect_paths(racmo24_dailyonly, f'tdew2m{racmo24_sep}*.nc'),
        'Sq': connect_paths(racmo24_dailyonly, f'sund{racmo24_sep}*.nc'),
        'SWin': connect_paths(racmo24_1hboth, f'rsds{racmo24_sep}*.nc'),
        'SWnet': connect_paths(racmo24_dailyonly, f'ssr{racmo24_sep}*.nc'),
        'SWincs': connect_paths(racmo24_6hboth, f'rsdscs{racmo24_sep}*.nc'),
        'SWnetcs': connect_paths(racmo24_dailyonly, f'ssrc{racmo24_sep}*.nc'),
        'LWin': connect_paths(racmo24_1hboth, f'rlds{racmo24_sep}*.nc'),
        'LWnet': connect_paths(racmo24_dailyonly, f'str{racmo24_sep}*.nc'),
        'LWincs': connect_paths(racmo24_6hboth, f'rldscs{racmo24_sep}*.nc'),
        'LWnetcs': connect_paths(racmo24_dailyonly, f'strc{racmo24_sep}*.nc'),
        'SHF': connect_paths(racmo24_1hboth, f'hfss{racmo24_sep}*.nc'),
        'LHF': connect_paths(racmo24_1hboth, f'hfls{racmo24_sep}*.nc'),
        'CloudLow': connect_paths(racmo24_6hboth, f'cll{racmo24_sep}*.nc'),
        'CloudMid': connect_paths(racmo24_6hboth, f'clm{racmo24_sep}*.nc'),
        'CloudHigh': connect_paths(racmo24_6hboth, f'clh{racmo24_sep}*.nc'),
        'CloudTotal': connect_paths(racmo24_1hboth, f'clt{racmo24_sep}*.nc'),
        'LWP': connect_paths(racmo24_1hboth, f'clwvi{racmo24_sep}*.nc'),
        'IWP': connect_paths(racmo24_1hboth, f'clivi{racmo24_sep}*.nc'),
        'Q': connect_paths(racmo24_1hboth, f'huss{racmo24_sep}*.nc'),
        'Ts': connect_paths(racmo24_1hboth, f'ts{racmo24_sep}*.nc'),
        'SWC': connect_paths(racmo24_dailyonly, f'wskin{racmo24_sep}*.nc'),
        'SWVL1': connect_paths(racmo24_6hboth, f'swvl1{racmo24_sep}*.nc'),
        'SWVL2': connect_paths(racmo24_6hboth, f'swvl2{racmo24_sep}*.nc'),
        'SWVL3': connect_paths(racmo24_6hboth, f'swvl3{racmo24_sep}*.nc'),
        'SWVL4': connect_paths(racmo24_6hboth, f'swvl4{racmo24_sep}*.nc'),
        'TWC': connect_paths(racmo24_1hboth, f'tcw{racmo24_sep}*.nc'),
        'TWV': connect_paths(racmo24_1hboth, f'prw{racmo24_sep}*.nc'),
        'TSWin': connect_paths(racmo24_dailyonly, f'rsdt{racmo24_sep}*.nc'),
        'TSWnet': connect_paths(racmo24_dailyonly, f'tsr{racmo24_sep}*.nc'),
        'TLWnet': connect_paths(racmo24_dailyonly, f'toptr{racmo24_sep}*.nc'),
        'TSWnetcs': connect_paths(racmo24_dailyonly, f'tsrc{racmo24_sep}*.nc'),
        'TLWnetcs': connect_paths(racmo24_dailyonly, f'ttrc{racmo24_sep}*.nc'),

        'adicomp': connect_paths(racmo24_enbud, f'adicomp{racmo24_enbudsep}*.nc'),
        'confric': connect_paths(racmo24_enbud, f'confric{racmo24_enbudsep}*.nc'),
        'conphase': connect_paths(racmo24_enbud, f'conphase{racmo24_enbudsep}*.nc'),
        'consens': connect_paths(racmo24_enbud, f'consens{racmo24_enbudsep}*.nc'),
        'contot': connect_paths(racmo24_enbud, f'contot{racmo24_enbudsep}*.nc'),
        'dyntot': connect_paths(racmo24_enbud, f'dyntot{racmo24_enbudsep}*.nc'),
        'horadv': connect_paths(racmo24_enbud, f'horadv{racmo24_enbudsep}*.nc'),
        'lcbotdn': connect_paths(racmo24_enbud, f'lcbotdn{racmo24_enbudsep}*.nc'),
        'lcbotup': connect_paths(racmo24_enbud, f'lcbotup{racmo24_enbudsep}*.nc'),
        'lcnet': connect_paths(racmo24_enbud, f'lcnet{racmo24_enbudsep}*.nc'),
        'lctopdn': connect_paths(racmo24_enbud, f'lctopdn{racmo24_enbudsep}*.nc'),
        'lctopup': connect_paths(racmo24_enbud, f'lctopup{racmo24_enbudsep}*.nc'),
        'lscld': connect_paths(racmo24_enbud, f'lscld{racmo24_enbudsep}*.nc'),
        'lwbotdn': connect_paths(racmo24_enbud, f'lwbotdn{racmo24_enbudsep}*.nc'),
        'lwbotup': connect_paths(racmo24_enbud, f'lwbotup{racmo24_enbudsep}*.nc'),
        'lwnet': connect_paths(racmo24_enbud, f'lwnet{racmo24_enbudsep}*.nc'),
        'lwtopdn': connect_paths(racmo24_enbud, f'lwtopdn{racmo24_enbudsep}*.nc'),
        'lwtopup': connect_paths(racmo24_enbud, f'lwtopup{racmo24_enbudsep}*.nc'),
        'mlmid': connect_paths(racmo24_enbud, f'mlmid{racmo24_enbudsep}*.nc'),
        'mltop': connect_paths(racmo24_enbud, f'mltop{racmo24_enbudsep}*.nc'),
        'numbnd': connect_paths(racmo24_enbud, f'numbnd{racmo24_enbudsep}*.nc'),
        'numdif': connect_paths(racmo24_enbud, f'numdif{racmo24_enbudsep}*.nc'),
        'orography': connect_paths(racmo24_enbud, f'orography{racmo24_enbudsep}*.nc'),
        'phystot': connect_paths(racmo24_enbud, f'phystot{racmo24_enbudsep}*.nc'),
        'radtot': connect_paths(racmo24_enbud, f'radtot{racmo24_enbudsep}*.nc'),
        'scbotdn': connect_paths(racmo24_enbud, f'scbotdn{racmo24_enbudsep}*.nc'),
        'scbotup': connect_paths(racmo24_enbud, f'scbotup{racmo24_enbudsep}*.nc'),
        'scnet': connect_paths(racmo24_enbud, f'scnet{racmo24_enbudsep}*.nc'),
        'sctopdn': connect_paths(racmo24_enbud, f'sctopdn{racmo24_enbudsep}*.nc'),
        'sctopup': connect_paths(racmo24_enbud, f'sctopup{racmo24_enbudsep}*.nc'),
        'swbotdn': connect_paths(racmo24_enbud, f'swbotdn{racmo24_enbudsep}*.nc'),
        'swbotup': connect_paths(racmo24_enbud, f'swbotup{racmo24_enbudsep}*.nc'),
        'swnet': connect_paths(racmo24_enbud, f'swnet{racmo24_enbudsep}*.nc'),
        'swtopdn': connect_paths(racmo24_enbud, f'swtopdn{racmo24_enbudsep}*.nc'),
        'swtopup': connect_paths(racmo24_enbud, f'swtopup{racmo24_enbudsep}*.nc'),
        'templ1': connect_paths(racmo24_enbud, f'templ1{racmo24_enbudsep}*.nc'),
        'templ1s': connect_paths(racmo24_enbud, f'templ1s{racmo24_enbudsep}*.nc'),
        'templ1spr': connect_paths(racmo24_enbud, f'templ1spr{racmo24_enbudsep}*.nc'),
        'tendtot': connect_paths(racmo24_enbud, f'tendtot{racmo24_enbudsep}*.nc'),
        'tendtotpr': connect_paths(racmo24_enbud, f'tendtotpr{racmo24_enbudsep}*.nc'),
        'udtdx': connect_paths(racmo24_enbud, f'udtdx{racmo24_enbudsep}*.nc'),
        'vdffric': connect_paths(racmo24_enbud, f'vdffric{racmo24_enbudsep}*.nc'),
        'vdfphase': connect_paths(racmo24_enbud, f'vdfphase{racmo24_enbudsep}*.nc'),
        'vdfsens': connect_paths(racmo24_enbud, f'vdfsens{racmo24_enbudsep}*.nc'),
        'vdftot': connect_paths(racmo24_enbud, f'vdftot{racmo24_enbudsep}*.nc'),
        'vdtdy': connect_paths(racmo24_enbud, f'vdtdy{racmo24_enbudsep}*.nc'),
        'vertadv': connect_paths(racmo24_enbud, f'vertadv{racmo24_enbudsep}*.nc'),
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
