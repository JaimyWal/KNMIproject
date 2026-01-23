import xarray as xr
import cartopy.crs as ccrs
import os


DATA_SOURCES = ['Eobs', 'ERA5', 'RACMO2.3', 'RACMO2.4']
STATION_SOURCES = ['Bilt', 'Cabauw', 'Eelde', 'Maastricht', 'Vlissingen', 'Kooy']

VAR_NAME_CFG = {
    'Eobs': {'Tg': 'tg', 'P': 'rr', 'SWin': 'qq', 'Tmax': 'tx', 'Tmin': 'tn', 
             'RH': 'hu', 'Psl': 'pp'},
    'ERA5': {
        'Tg': 't2m', 'P': 'tp', 'Tmax': 'tmax', 'Tmin': 'tmin', 'Tdew': 'd2m',
        'SWin': 'avg_sdswrf', 'SWnet': 'avg_snswrf', 'SWincs': 'avg_sdswrfcs', 'SWnetcs': 'avg_snswrfcs',
        'LWin': 'avg_sdlwrf', 'LWnet': 'avg_snlwrf', 'LWincs': 'avg_sdlwrfcs', 'LWnetcs': 'avg_snlwrfcs',
        'SHF': 'avg_ishf', 'LHF': 'avg_slhtf',
        'CloudLow': 'lcc', 'CloudMid': 'mcc', 'CloudHigh': 'hcc', 'CloudTotal': 'tcc',
        'LWP': 'tclw', 'IWP': 'tciw',
        'Ps': 'sp', 'Psl': 'msl',
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
        'Ps': 'ps', 'Psl': 'psl', 'Q': 'huss',
    },
    'Station': {'Tg': 'TG', 'P': 'RH', 'Sq': 'SQ', 'SWin': 'Q', 'Tmax': 'TX', 'Tmin': 'TN', 
                'RH': 'UG', 'Psl': 'PG'},
}

STATION_COORD_CFG = {
    'Bilt': {'latitude': 52.098872302947974, 'longitude': 5.179442289152804},
    'Cabauw': {'latitude': 51.970212171384865, 'longitude': 4.926283190645085},
    'Eelde': {'latitude': 53.12385846866912, 'longitude': 6.584799434350561},
    'Maastricht': {'latitude': 50.90548320406765, 'longitude': 5.761839846736004},
    'Vlissingen': {'latitude': 51.441328455552586, 'longitude': 3.5958610840956884},
    'Kooy': {'latitude': 52.924172463538795, 'longitude': 4.779336630180403},
}

LABEL_PLOT_CFG = {
    'Tg': 'Temperature (°C)',
    'Tmax': 'Max Temperature (°C)',
    'Tmin': 'Min Temperature (°C)',
    'RH': 'Relative Humidity (%)',
    'Tdew': 'Dew Point Temperature (°C)',
    'P': 'Precipitation (mm)',
    'Sq': 'Sund. (hours/day)',
    'SWin': r'SW$_{\text{in}}$ (W/m$^2$)',
    'SWnet': r'SW$_{\text{net}}$ (W/m$^2$)',
    'SWincs': r'SW$_{\text{in,cs}}$ (W/m$^2$)',
    'SWnetcs': r'SW$_{\text{net,cs}}$ (W/m$^2$)',
    'LWin': r'LW$_{\text{in}}$ (W/m$^2$)',
    'LWincs': r'LW$_{\text{in,cs}}$ (W/m$^2$)',
    'LWnet': r'LW$_{\text{net}}$ (W/m$^2$)',
    'LWnetcs': r'LW$_{\text{net,cs}}$ (W/m$^2$)',
    'SHF': r'SHF (W/m$^2$)',
    'LHF': r'LHF (W/m$^2$)',
    'CloudLow': 'Low cloud (%)',
    'CloudMid': 'Mid cloud (%)',
    'CloudHigh': 'High cloud (%)',
    'CloudTotal': 'Total cloud (%)',
    'LWP': r'LWP (g/m$^2$)',
    'IWP': r'IWP (g/m$^2$)',
    'Ps': 'Surface Pressure (hPa)',
    'Psl': 'Sea Level Pressure (hPa)',
    'Q': 'Specific Humidity (g/kg)',

    'RH_proxy': 'Relative Humidity (%)',
    'Bowen': 'Bowen Ratio',
    'Albedo': 'Albedo',
    'Q_era': 'Specific Humidity (g/kg)',
    'Q_obs': 'Specific Humidity (g/kg)',
}

VAR_PHYS_CFG = {
    'Tg': 'Temperature',
    'Tmax': 'Max Temperature',
    'Tmin': 'Min Temperature',
    'RH': 'Relative Humidity',
    'Tdew': 'Dew Point',
    'P': 'Precipitation',
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
    'CloudLow': 'Low cloud',
    'CloudMid': 'Mid cloud',
    'CloudHigh': 'High cloud',
    'CloudTotal': 'Total cloud',
    'LWP': 'LWP',
    'IWP': 'IWP',
    'Ps': 'Surface Pressure',
    'Psl': 'Sea Level Pressure',
    'Q': 'Specific Humidity',

    'RH_proxy': 'Relative Humidity',
    'Bowen': 'Bowen Ratio',
    'Albedo': 'Albedo',
    'Q_era': 'Specific Humidity',
    'Q_obs': 'Specific Humidity',
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