import xarray as xr
import numpy as np


def teten_formula(temp_c):
    return 6.112*np.exp((17.67*temp_c) / (temp_c + 243.5))


era5_temp = xr.open_dataset('/nobackup/users/walj/era5/era5_coarse_t2m_ps.nc')
# era5_tdew = xr.open_dataset('/nobackup/users/walj/era5/era5_coarse_dew_wind.nc')

eobs_rh = xr.open_dataset('/nobackup/users/walj/eobs/hu_ens_mean_0.1deg_reg_v31.0e.nc')

# racmo24_temp_kext12 = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/tas_monthlyA_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
# racmo24_tdew_kext12 = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/td_monthlyA_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
racmo24_rh_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly2017/hurs_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_201712.nc')
