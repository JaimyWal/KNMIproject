import xarray as xr
import cartopy.crs as ccrs
import os


DATA_SOURCES = ['Eobs', 'ERA5', 'RACMO2.3', 'RACMO2.4']
STATION_SOURCES = ['Bilt', 'Cabauw', 'Eelde', 'Maastricht', 'Vlissingen', 'Kooy']

VAR_FILE_CFG = {
    'Eobs': {'Tg': 'tg', 'P': 'rr', 'SWin': 'qq', 'Tmax': 'tx', 'Tmin': 'tn', 
             'RH': 'hu', 'Psl': 'pp', 'Tmaxmax': 'tx', 'Tminmin': 'tn'},
    'ERA5': {
        'Tg': 't2m', 'P': 'tp', 'Tmax': 'tmax', 'Tmin': 'tmin', 'Tdew': 'd2m',
        'SWin': 'avg_sdswrf', 'SWnet': 'avg_snswrf', 'SWincs': 'avg_sdswrfcs', 'SWnetcs': 'avg_snswrfcs',
        'LWin': 'avg_sdlwrf', 'LWnet': 'avg_snlwrf', 'LWincs': 'avg_sdlwrfcs', 'LWnetcs': 'avg_snlwrfcs',
        'SHF': 'avg_ishf', 'LHF': 'avg_slhtf',
        'CloudLow': 'lcc', 'CloudMid': 'mcc', 'CloudHigh': 'hcc', 'CloudTotal': 'tcc',
        'LWP': 'tclw', 'IWP': 'tciw',
        'Ps': 'sp', 'Psl': 'msl', 'Tmaxmax': 'tmax', 'Tminmin': 'tmin',
    },
    'RACMO2.3': {
        'Tg': 't2m', 'P': 'precip', 'Sq': 'sund',
        'SWin': 'swsd', 'SWnet': 'swsn', 'SWincs': 'swsdcs', 'SWnetcs': 'swsncs',
        'LWin': 'lwsd', 'LWnet': 'lwsn', 'LWincs': 'lwsdcs', 'LWnetcs': 'lwsncs',
        'SHF': 'senf', 'LHF': 'latf',
        'CloudLow': 'aclcovL', 'CloudMid': 'aclcovM', 'CloudHigh': 'aclcovH', 'CloudTotal': 'aclcov',
        'LWP': 'qli', 'IWP': 'qii',
    },
    'RACMO2.4': {
        'Tg': 'tas', 'P': 'pr', 'Sq': 'sund', 'Tmax': 'tasmax', 'Tmin': 'tasmin', 'RH': 'hurs', 'Tdew': 'tdew2m',
        'SWin': 'rsds', 'SWnet': 'ssr', 'SWincs': 'rsdscs', 'SWnetcs': 'ssrc',
        'LWin': 'rlds', 'LWnet': 'str', 'LWincs': 'rldscs', 'LWnetcs': 'strc',
        'SHF': 'hfss', 'LHF': 'hfls',
        'CloudLow': 'cll', 'CloudMid': 'clm', 'CloudHigh': 'clh', 'CloudTotal': 'clt',
        'LWP': 'clwvi', 'IWP': 'clivi',
        'Ps': 'ps', 'Psl': 'psl', 'Q': 'huss', 'Tmaxmax': 'tasmax', 'Tminmin': 'tasmin',
    },
    'Station': {'Tg': 'TG', 'P': 'RH', 'Sq': 'SQ', 'SWin': 'Q', 'Tmax': 'TX', 'Tmin': 'TN', 
                'RH': 'UG', 'Psl': 'PG', 'Tmaxmax': 'TX', 'Tminmin': 'TN'},
}

STATION_COORD_CFG = {
    'Bilt': {'latitude': 52.098872302947974, 'longitude': 5.179442289152804},
    'Cabauw': {'latitude': 51.970212171384865, 'longitude': 4.926283190645085},
    'Eelde': {'latitude': 53.12385846866912, 'longitude': 6.584799434350561},
    'Maastricht': {'latitude': 50.90548320406765, 'longitude': 5.761839846736004},
    'Vlissingen': {'latitude': 51.441328455552586, 'longitude': 3.5958610840956884},
    'Kooy': {'latitude': 52.924172463538795, 'longitude': 4.779336630180403},
}

VAR_NAME_CFG = {
    'Tg': 'Temperature',
    'Tmax': 'Max Temperature',
    'Tmin': 'Min Temperature',
    'RH': 'Relative Humidity',
    'Tdew': 'Dew Point',
    'P': 'Precipitation',
    'Sq': 'Sunshine',
    'SWin': r'SW$_{\text{in}}$',
    'SWnet': r'SW$_{\text{net}}$',
    'SWincs': r'SW$_{\text{in,cs}}$',
    'SWnetcs': r'SW$_{\text{net,cs}}$',
    'LWin': r'LW$_{\text{in}}$',
    'LWincs': r'LW$_{\text{in,cs}}$',
    'LWnet': r'LW$_{\text{net}}$',
    'LWnetcs': r'LW$_{\text{net,cs}}$',
    'SHF': 'SHF',
    'LHF': 'LHF',
    'CloudLow': 'Low Cloud',
    'CloudMid': 'Mid Cloud',
    'CloudHigh': 'High Cloud',
    'CloudTotal': 'Total Cloud',
    'LWP': 'LWP',
    'IWP': 'IWP',
    'Ps': 'Surface Pressure',
    'Psl': 'Sea Level Pressure',
    'Q': 'Specific Humidity',

    'Tmaxmax': 'Annual maximum Temperature',
    'Tminmin': 'Annual minimum Temperature',
    'RH_proxy': 'Relative Humidity',
    'Bowen': 'Bowen Ratio',
    'Albedo': 'Albedo',
    'Q_era': 'Specific Humidity',
    'Q_obs': 'Specific Humidity',
}

VAR_SYMBOL_CFG = {
    'Tg': 'T',
    'Tmax': r'T$_{\text{max}}$',
    'Tmin': r'T$_{\text{min}}$',
    'RH': 'RH',
    'Tdew': r'T$_{\text{dew}}$',
    'P': 'P',
    'Sq': 'Sund.',
    'SWin': r'SW$_{\text{in}}$',
    'SWnet': r'SW$_{\text{net}}$',
    'SWincs': r'SW$_{\text{in,cs}}$',
    'SWnetcs': r'SW$_{\text{net,cs}}$',
    'LWin': r'LW$_{\text{in}}$',
    'LWincs': r'LW$_{\text{in,cs}}$',
    'LWnet': r'LW$_{\text{net}}$',
    'LWnetcs': r'LW$_{\text{net,cs}}$',
    'SHF': 'SHF',
    'LHF': 'LHF',
    'CloudLow': r'C$_{\text{low}}$',
    'CloudMid': r'C$_{\text{mid}}$',
    'CloudHigh': r'C$_{\text{high}}$',
    'CloudTotal': r'C$_{\text{total}}$',
    'LWP': 'LWP',
    'IWP': 'IWP',
    'Ps': r'p$_{s}$',
    'Psl': r'p$_{msl}$',
    'Q': 'q',

    'Tmaxmax': r'T$_{\text{xx}}$',
    'Tminmin': r'T$_{\text{nn}}$',
    'RH_proxy': 'RH',
    'Bowen': 'Bowen',
    'Albedo': r'$\alpha$',
    'Q_era': 'q',
    'Q_obs': 'q',
}

VAR_UNIT_CFG = {
    'Tg': '°C',
    'Tmax': '°C',
    'Tmin': '°C',
    'RH': '%',
    'Tdew': '°C',
    'P': 'mm',
    'Sq': 'hours/day',
    'SWin': r'W/m$^2$',
    'SWnet': r'W/m$^2$',
    'SWincs': r'W/m$^2$',
    'SWnetcs': r'W/m$^2$',
    'LWin': r'W/m$^2$',
    'LWincs': r'W/m$^2$',
    'LWnet': r'W/m$^2$',
    'LWnetcs': r'W/m$^2$',
    'SHF': r'W/m$^2$',
    'LHF': r'W/m$^2$',
    'CloudLow': '%',
    'CloudMid': '%',
    'CloudHigh': '%',
    'CloudTotal': '%',
    'LWP': r'g/m$^2$',
    'IWP': r'g/m$^2$',
    'Ps': 'hPa',
    'Psl': 'hPa',
    'Q': 'g/kg',

    'Tmaxmax': '°C',
    'Tminmin': '°C',
    'RH_proxy': '%',
    'Bowen': '',
    'Albedo': '',
    'Q_era': 'g/kg',
    'Q_obs': 'g/kg',
}


def load_rotpole(rotpole_dir, rotpole_file):

    ds = xr.open_dataset(os.path.join(rotpole_dir, rotpole_file))

    rp = ds['rotated_pole']
    pole_lat = rp.grid_north_pole_latitude
    pole_lon = rp.grid_north_pole_longitude

    rotpole = ccrs.RotatedPole(
        pole_latitude=pole_lat,
        pole_longitude=pole_lon,
    )

    return rotpole


rotpole23 = load_rotpole(
    '/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/precip',
    'precip.KNMI-2000.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc'
)

rotpole24_kext06 = load_rotpole(
    '/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly',
    'pr_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc'
)

rotpole24_kext12 = load_rotpole(
    '/nobackup/users/walj/TestRacmo24/Monthly',
    'pr_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc'
)

PROJ_CFG = {
    'RACMO2.3': rotpole23,
    'RACMO2.4': rotpole24_kext12,
    'RACMO2.4_KEXT06': rotpole24_kext06,
    'RACMO2.4_KEXT12': rotpole24_kext12
}

# However, rotpole24_kext06 == rotpole24_kext12