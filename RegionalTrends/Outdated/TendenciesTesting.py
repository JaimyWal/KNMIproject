#%% Imports

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import dask
from pathlib import Path
import sys
from importlib import reload

# Custom libraries
PROJECT_ROOT = Path.home() / 'KNMIproject'
sys.path.insert(0, str(PROJECT_ROOT))

from RegionalTrends.Helpers import ProcessVar
reload(ProcessVar)
from RegionalTrends.Helpers.ProcessVar import load_var

from RegionalTrends.Helpers import ComputeTendencies_old
reload(ComputeTendencies_old)
from RegionalTrends.Helpers.ComputeTendencies import construct_tendency

import RegionalTrends.Helpers.Config.Constants as Constants
reload(Constants)


plt.rcParams['axes.unicode_minus'] = False
dask.config.set(scheduler='threads', num_workers=12)

#%% User inputs

# Main arguments
var = 'tendtot'
file_freq = 'Raw'

# Data selection arguments
years = np.arange(1980, 2024 + 1)
months = [6,7,8]
lats = [37.7, 63.3]
lons = [-13.3, 22.3]
proj_sel = 'RACMO2.4'
land_only = False
trim_border = None

#%% Dataset configurations

data_sources = Constants.DATA_SOURCES
station_sources = Constants.STATION_SOURCES
station_coord_cfg = Constants.STATION_COORD_CFG
enbud_vars = Constants.ENBUD_VARS
tendency_vars = Constants.TENDENCY_VARS
var_file_cfg = Constants.VAR_FILE_CFG
var_name_cfg = Constants.VAR_NAME_CFG
var_units_cfg = Constants.VAR_UNIT_CFG
var_symbol_cfg = Constants.VAR_SYMBOL_CFG
proj_cfg = Constants.PROJ_CFG

# Setup projections
proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())

#%% 

load_args = dict(
    data_source='RACMO2.4A',
    data_sources=data_sources,
    station_sources=station_sources,
    file_freq=file_freq,
    var_file_cfg=var_file_cfg,
    proj_cfg=proj_cfg,
    months=None,
    years=years,
    lats=lats,
    lons=lons,
    land_only=land_only,
    trim_border=trim_border,
    rotpole_sel=proj_sel,
    station_coords=station_coord_cfg
)

# adicomp = load_var(var='adicomp', **load_args)
# confric = load_var(var='confric', **load_args)
# conphase = load_var(var='conphase', **load_args)
# consens = load_var(var='consens', **load_args)
# contot = load_var(var='contot', **load_args)
# dyntot = load_var(var='dyntot', **load_args)
# horadv = load_var(var='horadv', **load_args)
# lcbotdn = load_var(var='lcbotdn', **load_args)
# lcbotup = load_var(var='lcbotup', **load_args)
# lcnet = load_var(var='lcnet', **load_args)
# lctopdn = load_var(var='lctopdn', **load_args)
# lctopup = load_var(var='lctopup', **load_args)
# lscld = load_var(var='lscld', **load_args)
# lwbotdn = load_var(var='lwbotdn', **load_args)
# lwbotup = load_var(var='lwbotup', **load_args)
# lwnet = load_var(var='lwnet', **load_args)
# lwtopdn = load_var(var='lwtopdn', **load_args)
# lwtopup = load_var(var='lwtopup', **load_args)
# mlmid = load_var(var='mlmid', **load_args)
# mltop = load_var(var='mltop', **load_args)
# numbnd = load_var(var='numbnd', **load_args)
# numdif = load_var(var='numdif', **load_args)
orography = load_var(var='orography', **load_args)
# phystot = load_var(var='phystot', **load_args)
# radtot = load_var(var='radtot', **load_args)
# scbotdn = load_var(var='scbotdn', **load_args)
# scbotup = load_var(var='scbotup', **load_args)
# scnet = load_var(var='scnet', **load_args)
# sctopdn = load_var(var='sctopdn', **load_args)
# sctopup = load_var(var='sctopup', **load_args)
# swbotdn = load_var(var='swbotdn', **load_args)
# swbotup = load_var(var='swbotup', **load_args)
# swnet = load_var(var='swnet', **load_args)
# swtopdn = load_var(var='swtopdn', **load_args)
# swtopup = load_var(var='swtopup', **load_args)
templ1 = load_var(var='templ1', **load_args)
# templ1s = load_var(var='templ1s', **load_args)
# templ1spr = load_var(var='templ1spr', **load_args)
tendtot = load_var(var='tendtot', **load_args)
# tendtotpr = load_var(var='tendtotpr', **load_args)
# udtdx = load_var(var='udtdx', **load_args)
# vdffric = load_var(var='vdffric', **load_args)
# vdfphase = load_var(var='vdfphase', **load_args)
# vdfsens = load_var(var='vdfsens', **load_args)
# vdftot = load_var(var='vdftot', **load_args)
# vdtdy = load_var(var='vdtdy', **load_args)
# vertadv = load_var(var='vertadv', **load_args)

# senstot = load_var(var='senstot', **load_args)
# phasetot = load_var(var='phasetot', **load_args)
# frictot = load_var(var='frictot', **load_args)

#%% Compute tendencies

load_args = dict(
    data_source='RACMO2.4A',
    data_sources=data_sources,
    station_sources=station_sources,
    var_file_cfg=var_file_cfg,
    proj_cfg=proj_cfg,
    months=None,
    years=years,
    lats=lats,
    lons=lons,
    land_only=land_only,
    trim_border=trim_border,
    rotpole_sel=proj_sel,
    station_coords=station_coord_cfg
)

orography_daily = load_var(var='orography', **load_args, file_freq='Daily')
orography_monthly = load_var(var='orography', **load_args, file_freq='Monthly')
orography_seasonal = load_var(var='orography', **load_args, file_freq='Seasonal')

orography_daily_raw = construct_tendency(
    orography, 
    interval='Daily', 
    relation='Adjacent', 
    return_intermediates=False
)

orography_monthly_raw = construct_tendency(
    orography, 
    interval='Monthly', 
    relation='Adjacent', 
    return_intermediates=False
)

orography_seasonal_raw = construct_tendency(
    orography, 
    interval='Seasonal', 
    relation='Adjacent', 
    return_intermediates=False
)

#%%

tendtot_seasonal_raw = construct_tendency(
    tendtot, 
    interval='Seasonal', 
    relation='Adjacent', 
    return_intermediates=False
)

months_arr = np.asarray(months, dtype=int)
templ1_season = templ1.where(templ1['time'].dt.month.isin(months_arr), drop=True)
clim_year = templ1_season['time'].dt.year +\
     ((templ1_season['time'].dt.month >= months_arr[0]) if months_arr[0] > months_arr[-1] else 0)
templ1_season = templ1_season.assign_coords(clim_year=('time', clim_year.data))
templ1_seasonal = templ1_season.groupby('clim_year').mean('time')

templ1_seasonal = templ1_seasonal.shift(clim_year=-1) - templ1_seasonal
templ1_seasonal = templ1_seasonal.dropna('clim_year')

#%%

templ1_seasonal_file = load_var(var='templ1', **load_args, file_freq='Seasonal')
templ1_seasonal_fileA = templ1_seasonal_file.shift(season_year=-1) - templ1_seasonal_file
templ1_seasonal_fileA = templ1_seasonal_fileA.dropna('season_year')


#%%

# fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# tendtot_seasonal_raw.plot(ax=ax, x='season_year', label='from tendencies', color='red')
# templ1_seasonal.plot(ax=ax, x='season_year', label='true templ1', color='black')


#%%

# Zijn templ1 en Tg voor racmo2.4a niet per ongeluk hetzelfde???
# Work with seasonal or monthly?????
# Kijken naar andere processen
# Kijken naar aftrekken van seizoenale cyclus
# outputten van raw tendencies en seizoenale tendencies en dagelijkse/maandelijkse templ1
