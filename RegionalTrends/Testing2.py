import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
from importlib import reload

import PlotMaps
reload(PlotMaps)          
from PlotMaps import plot_map, shared_colorbar

#%%

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

rotpole24 = load_rotpole(
    '/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly',
    'pr_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc'
)

proj_cfg = {
    'RACMO2.3': rotpole23,
    'RACMO2.4': rotpole24,
}

proj_plot = proj_cfg.get('RACMO2.4', ccrs.PlateCarree())

#%%


test_daily = xr.open_dataset('/nobackup/users/walj/TestRacmo/Daily/NC_DEFAULT/tas.KNMI-1980.KEXT12.RACMO2.4p1_v5.DD.nc')
test_hourly = xr.open_dataset('/nobackup/users/walj/TestRacmo/Hourly/tas.KNMI-1980.KEXT12.RACMO2.4p1_v5.1H.nc')

test_avg = test_daily['tas'].mean(dim='time').squeeze()
test_avg_hourly = test_hourly['tas'].mean(dim='time').squeeze()

fig, ax = plt.subplots(
    1, figsize=(14, 12), 
    constrained_layout=True,
    subplot_kw={'projection': proj_plot}
)

plot_map(
    fig, ax,
    test_avg, 
    test_avg['lon'], 
    test_avg['lat'],  
    show_x_ticks=True,
    show_y_ticks=True,
    y_ticks_num=False,
    y_ticks=5,
    x_ticks_num=False,
    x_ticks=10,
    extent=[-20,40, 30,75],
    proj=proj_plot
)



#%%

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt

# Correlation bounds
bounds = np.arange(-1.0, 1.01, 0.1)

# Colors chosen to resemble the shown plot
colors = [
    '#3b007a', '#2a1dbb', '#1536e6', '#0a4cff', '#0066cc',
    '#008080', '#1fa187', '#4ac16d', '#a0da39',
    '#d9f0a3',
    '#ffffff',
    '#fff59d', '#ffe066', '#ffc300', '#ff9f1c',
    '#ff7a00', '#e94f08', '#c62828', '#8e0000', '#5a0000'
]

cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)

# Example colorbar
fig, ax = plt.subplots(figsize=(6, 1))
cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax,
    orientation='horizontal',
    ticks=bounds
)
cb.set_label('Correlation')


#%%

import numpy as np
import matplotlib.colors as clr

colors = [
    "#570088", "#3700b3", "#1d00d7", "#0300f6", "#0231be",
    "#056775", "#079d2c", "#35c13b", "#80d883",
    "#ffffff", "#ffffff", "#ffffff",
    "#fff400", "#ffe400", "#ffc900", "#ffad00",
    "#ff8200", "#ff5500", "#ff2800", "#a30e03", "#6b0902"
]

cmap = clr.ListedColormap(colors)
bounds = np.linspace(-1.0, 1.0, len(colors) + 1)
norm = clr.BoundaryNorm(bounds, cmap.N)

ticks = np.arange(-1.0, 1.01, 0.2)

fig, ax = plt.subplots(figsize=(6, 1))
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # keeps some matplotlib versions happy

cbar = fig.colorbar(
    sm,
    ax=ax,
    ticks=ticks,
    orientation='horizontal',
    label='Correlation'
)

ax.remove()  # optional, hides the empty axes so only the colorbar remains
plt.show()

# corr_colors = [
#     "#570088", "#3700b3", "#1d00d7", "#0300f6", "#0231be",
#     "#056775", "#079d2c", "#35c13b", "#80d883",
#     "#ffffff", "#ffffff",
#     "#fff400", "#ffe400", "#ffc900", "#ffad00",
#     "#ff8200", "#ff5500", "#ff2800", "#a30e03", "#6b0902"
# ]
# cmap_corr = LinearSegmentedColormap.from_list('correlation', corr_colors, N=len(corr_colors))
# corr_extreme = ("#570088", "#6b0902")


