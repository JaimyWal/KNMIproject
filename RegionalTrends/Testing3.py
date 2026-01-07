import numpy as np
import xarray as xr
from importlib import reload

import ProcessNetCDF
reload(ProcessNetCDF)          
from ProcessNetCDF import preprocess_netcdf


racmo24_lowcloud = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/cll.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')
racmo24_midcloud = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/clm.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')
racmo24_highcloud = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/clh.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')

racmo24_swallnet = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/ssr.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')
racmo24_swclearnet = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/ssrc.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')
racmo24_swalldown = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/rsds.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')
racmo24_swcleardown = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/rsdscs.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')

racmo24_lwallnet = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/str.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc') 
racmo24_lwclearnet = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/strc.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')
racmo24_lwalldown = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/rlds.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')
racmo24_lwcleardown = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/rldscs.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')

racmo24_shf = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/hfss.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')
racmo24_lhf = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/hfls.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')

racmo24_lwp = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/clwvi.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')
racmo24_iwp = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/clivi.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')


racmo23_lowcloud = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/aclcovL/aclcovL.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')
racmo23_midcloud = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/aclcovM/aclcovM.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')
racmo23_highcloud = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/aclcovH/aclcovH.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')

racmo23_swallnet = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/swsn/swsn.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')
racmo23_swclearnet = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/swsncs/swsncs.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')
racmo23_swalldown = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/swsd/swsd.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')
racmo23_swcleardown = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/swsdcs/swsdcs.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')

racmo23_lwallnet = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/lwsn/lwsn.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc') 
racmo23_lwclearnet = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/lwsncs/lwsncs.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')
racmo23_lwalldown = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/lwsd/lwsd.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')
racmo23_lwcleardown = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/lwsdcs/lwsdcs.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')

racmo23_shf = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/senf/senf.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')
racmo23_lhf = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/latf/latf.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')

racmo23_lwp = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/qli/qli.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')
racmo23_iwp = xr.open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/qii/qii.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')