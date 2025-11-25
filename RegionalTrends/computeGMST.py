import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from importlib import reload

import ProcessERA5
reload(ProcessERA5)          
from ProcessERA5 import preprocess_era5 

def area_weights(lat_val, lon_val): 
    lat_180 = lat_val + 90
    dlat = lat_val[0] - lat_val[1]
    area_weight =  np.zeros((len(lat_180), len(lon_val)))
    for ii in range(len(lat_180)):
        area_weight[ii, :] = np.cos((lat_180[ii] - dlat/2)*np.pi/180) - \
                             np.cos((lat_180[ii] + dlat/2)*np.pi/180)
    return area_weight

def weighted_mean(variable, weights, n_time=None):
    if n_time is not None:
        mean_var = np.zeros(n_time)
        for ii in range(n_time):
            mean_var[ii] = np.nanmean(variable[ii, :, :]*weights) / np.nanmean(weights)
    else:
        mean_var = np.nanmean(variable*weights) / np.nanmean(weights)
    return mean_var

t2m = preprocess_era5('/nobackup/users/walj/era5/era5_coarse_full_t2m.nc',
                      't2m')

t2m_yearly = t2m.resample(time='YS').mean()

lat = t2m_yearly['latitude'].values
lon = t2m_yearly['longitude'].values

weights = area_weights(lat, lon)

t2m_yearly_np = t2m_yearly.values

ntime = t2m_yearly_np.shape[0]
gmst_yearly = weighted_mean(t2m_yearly_np, weights, n_time=ntime)

gmst_yearly_da = xr.DataArray(
    gmst_yearly,
    coords={'time': t2m_yearly['time']},
    dims=['time'],
    name='GMST'
)

ref = gmst_yearly_da.sel(time=slice('1940', '1970')).mean()
gmst_anom = gmst_yearly_da - ref

#%% plotting

gmst_roll5 = gmst_anom.rolling(time=5, center=False).mean()

fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(
    gmst_anom['time'],
    gmst_anom.values,
    c='xkcd:black',
    linewidth=2,
    label='Yearly GMST'
)

ax.plot(
    gmst_roll5['time'],
    gmst_roll5.values,
    c='xkcd:red',
    linewidth=3,
    label='5 year rolling mean'
)

ax.set_xlabel('Year', fontsize=28)
ax.set_ylabel('ΔGMST (°C)', fontsize=28)
ax.tick_params(axis='both', labelsize=20, length=6)
ax.grid()

leg = ax.legend(fontsize=18, handlelength=1.5, handletextpad=0.4, loc='best')
for line in leg.get_lines():
    line.set_linewidth(4.0)

#%%

save_path = '/nobackup/users/walj/era5/era5_gmst_anom.nc'

if save_path is not None:
        gmst_anom.to_netcdf(save_path)



# Reference temperature??
# Wellicht echte GMST voor observed data..!!!!