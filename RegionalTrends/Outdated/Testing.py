import xarray as xr

#%% racmo24 daily

racmo24_temp = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/tas.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')
racmo24_precip = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/pr.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')
racmo24_sund = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Daily/sund.KNMI-2016.KEXT06.RACMO2.4p1_v5_nocloudtuning.DD.nc')

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

#%% racmo24 monthly

racmo24_temp_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/tas_monthlyA_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
racmo24_precip_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/pr_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
racmo24_sund_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/sund_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')

racmo24_lowcloud_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/cll_monthlyA_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
racmo24_midcloud_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/clm_monthlyA_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
racmo24_highcloud_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/clh_monthlyA_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')

racmo24_swallnet_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/ssr_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
racmo24_swclearnet_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/ssrc_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
racmo24_swalldown_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/rsds_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
racmo24_swcleardown_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/rsdscs_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')

racmo24_lwallnet_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/str_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc') 
racmo24_lwclearnet_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/strc_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
racmo24_lwalldown_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/rlds_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
racmo24_lwcleardown_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/rldscs_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')

racmo24_shf_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/hfss_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
racmo24_lhf_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/hfls_monthlyS_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')

racmo24_lwp_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/clwvi_monthlyA_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')
racmo24_iwp_monthly = xr.open_dataset('/net/pc200010/nobackup/users/dalum/RACMO2.4/RACMO_output/KEXT06/RACMO2.4p1_v5_nocloudtuning/Monthly/clivi_monthlyA_KEXT06_RACMO2.4p1_v5_nocloudtuning_201501_202412.nc')

#%% racmo23 daily

racmo23_temp = xr.open_dataset('/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/t2m/t2m.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')
racmo23_precip = xr.open_dataset('/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/precip/precip.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')
racmo23_sund = xr.open_dataset('/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/sund/sund.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')

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

#%% racmo23 monthly

racmo23_temp_monthly = xr.open_dataset('/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Monthly_data/t2m/t2m.KNMI-2011-2020.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)
racmo23_precip_monthly = xr.open_dataset('/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Monthly_data/precip/precip.KNMI-2011-2020.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)
racmo23_sund_monthly = xr.open_dataset('/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Monthly_data/sund/sund.KNMI-2011-2020.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)

racmo23_lowcloud_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/aclcovL/aclcovL.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)
racmo23_midcloud_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/aclcovM/aclcovM.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)
racmo23_highcloud_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/aclcovH/aclcovH.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)

racmo23_swallnet_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/swsn/swsn.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)
racmo23_swclearnet_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/swsncs/swsncs.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)
racmo23_swalldown_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/swsd/swsd.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)
racmo23_swcleardown_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/swsdcs/swsdcs.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)  

racmo23_lwallnet_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/lwsn/lwsn.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False) 
racmo23_lwclearnet_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/lwsncs/lwsncs.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)
racmo23_lwalldown_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/lwsd/lwsd.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)
racmo23_lwcleardown_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/lwsdcs/lwsdcs.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)

racmo23_shf_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/senf/senf.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)
racmo23_lhf_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/latf/latf.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)  
racmo23_lwp_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/qli/qli.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)
racmo23_iwp_monthly = xr.open_dataset('/nobackup/users/walj/racmo23/Monthly_data/qii/qii.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc', decode_times=False)


#%% era5 monthly

era5_temp = xr.open_dataset('/nobackup/users/walj/era5/Monthly/era5_coarse_t2m_ps.nc')
era5_precip = xr.open_dataset('/nobackup/users/walj/era5/Monthly/era5_coarse_tp.nc')
era5_sw_shf = xr.open_dataset('/nobackup/users/walj/era5/Monthly/era5_coarse_sw_shf.nc')
era5_lw_lhf = xr.open_dataset('/nobackup/users/walj/era5/Monthly/era5_coarse_lw_lhf.nc')
era5_clouds = xr.open_dataset('/nobackup/users/walj/era5/Monthly/era5_coarse_clouds.nc')

#%% era5 daily

# era5_tmax = xr.open_dataset('/nobackup/users/walj/era5/era5_tmax_daily_eu.nc')
# era5_tmin = xr.open_dataset('/nobackup/users/walj/era5/era5_tmin_daily_eu.nc')

#%% eobs daily

eobs_temp = xr.open_dataset('/nobackup/users/walj/eobs/Daily/tg_ens_mean_0.1deg_reg_v31.0e.nc')
eobs_precip = xr.open_dataset('/nobackup/users/walj/eobs/Daily/rr_ens_mean_0.1deg_reg_v31.0e.nc')
eobs_swin = xr.open_dataset('/nobackup/users/walj/eobs/Daily/qq_ens_mean_0.1deg_reg_v31.0e.nc')
eobs_tmax = xr.open_dataset('/nobackup/users/walj/eobs/Daily/tx_ens_mean_0.1deg_reg_v31.0e.nc')
eobs_tmin = xr.open_dataset('/nobackup/users/walj/eobs/Daily/tn_ens_mean_0.1deg_reg_v31.0e.nc')
eobs_pres = xr.open_dataset('/nobackup/users/walj/eobs/Daily/pp_ens_mean_0.1deg_reg_v32.0e.nc')

#%% eobs monthly

eobs_temp_monthly = xr.open_dataset('/nobackup/users/walj/eobs/Monthly/tg_ens_mean_0.1deg_reg_v31.0e_monthly.nc')
eobs_precip_monthly = xr.open_dataset('/nobackup/users/walj/eobs/Monthly/rr_ens_mean_0.1deg_reg_v31.0e_monthly.nc')
eobs_swin_monthly = xr.open_dataset('/nobackup/users/walj/eobs/Monthly/qq_ens_mean_0.1deg_reg_v31.0e_monthly.nc')
eobs_tmax_monthly = xr.open_dataset('/nobackup/users/walj/eobs/Monthly/tx_ens_mean_0.1deg_reg_v31.0e_monthly.nc')
eobs_tmin_monthly = xr.open_dataset('/nobackup/users/walj/eobs/Monthly/tn_ens_mean_0.1deg_reg_v31.0e_monthly.nc')
eobs_pres_monthly = xr.open_dataset('/nobackup/users/walj/eobs/Monthly/pp_ens_mean_0.1deg_reg_v32.0e_monthly.nc')

#%% racmo24 daily kext12

racmo24_temp_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/tas.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_precip_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/pr.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_sund_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/sund.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

racmo24_lowcloud_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/cll.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_midcloud_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/clm.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_highcloud_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/clh.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

racmo24_swallnet_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/ssr.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_swclearnet_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/ssrc.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_swalldown_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/rsds.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_swcleardown_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/rsdscs.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

racmo24_lwallnet_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/str.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc') 
racmo24_lwclearnet_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/strc.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_lwalldown_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/rlds.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_lwcleardown_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/rldscs.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

racmo24_shf_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/hfss.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_lhf_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/hfls.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

racmo24_lwp_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/clwvi.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_iwp_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Daily/clivi.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

racmo24_ps_kext12 = xr.open_dataset('/nobackup/users/walj/racmo24/Daily/ps.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_psl_kext12 = xr.open_dataset('/nobackup/users/walj/racmo24/Daily/psl.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')
racmo24_q_kext12 = xr.open_dataset('/nobackup/users/walj/racmo24/Daily/huss.KNMI-1975.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

#%% racmo24 monthly kext12

racmo24_temp_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/tas_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')
racmo24_precip_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/pr_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')
racmo24_sund_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/sund_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')

racmo24_lowcloud_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/cll_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')
racmo24_midcloud_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/clm_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')
racmo24_highcloud_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/clh_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')

racmo24_swallnet_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/ssr_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')
racmo24_swclearnet_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/ssrc_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')
racmo24_swalldown_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/rsds_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')
racmo24_swcleardown_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/rsdscs_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')

racmo24_lwallnet_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/str_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc') 
racmo24_lwclearnet_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/strc_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')
racmo24_lwalldown_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/rlds_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')
racmo24_lwcleardown_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/rldscs_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')

racmo24_shf_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/hfss_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')
racmo24_lhf_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/hfls_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')

racmo24_lwp_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/clwvi_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')
racmo24_iwp_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/TestRacmo24/Monthly/clivi_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_197712.nc')

racmo24_ps_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/racmo24/Monthly/ps_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_202509.nc')
racmo24_psl_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/racmo24/Monthly/psl_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_202509.nc')
racmo24_q_monthly_kext12 = xr.open_dataset('/nobackup/users/walj/racmo24/Monthly/huss_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_202509.nc')

#%% New

era5_soil_skin = xr.open_dataset('/nobackup/users/walj/era5/Monthly/era5_coarse_soil_skin.nc')
era5_water = xr.open_dataset('/nobackup/users/walj/era5/Monthly/era5_coarse_water.nc')
era5_toa = xr.open_dataset('/nobackup/users/walj/era5/Monthly/era5_coarse_toa.nc')

racmo24_swvl1_monthly = xr.open_dataset('/nobackup/users/walj/racmo24/Monthly/swvl1_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_202509.nc')
racmo24_swvl1 = xr.open_dataset('/nobackup/users/walj/racmo24/Daily/swvl1.KNMI-2010.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

racmo24_tcw_monthly = xr.open_dataset('/nobackup/users/walj/racmo24/Monthly/tcw_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_202509.nc')
racmo24_tcw = xr.open_dataset('/nobackup/users/walj/racmo24/Daily/tcw.KNMI-2010.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

racmo24_twv_monthly = xr.open_dataset('/nobackup/users/walj/racmo24/Monthly/prw_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_202509.nc')
racmo24_twv = xr.open_dataset('/nobackup/users/walj/racmo24/Daily/prw.KNMI-2010.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

racmo24_wskin_monthly = xr.open_dataset('/nobackup/users/walj/racmo24/Monthly/wskin_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_202509.nc')
racmo24_wskin = xr.open_dataset('/nobackup/users/walj/racmo24/Daily/wskin.KNMI-2010.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

racmo24_ts_monthly = xr.open_dataset('/nobackup/users/walj/racmo24/Monthly/ts_monthlyA_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_202509.nc')
racmo24_ts = xr.open_dataset('/nobackup/users/walj/racmo24/Daily/ts.KNMI-2010.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

racmo24_tswin_monthly = xr.open_dataset('/nobackup/users/walj/racmo24/Monthly/rsdt_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_202509.nc')
racmo24_tswin = xr.open_dataset('/nobackup/users/walj/racmo24/Daily/rsdt.KNMI-2010.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

racmo24_tswnet_monthly = xr.open_dataset('/nobackup/users/walj/racmo24/Monthly/tsr_monthlyS_KEXT12_RACMO2.4p1_v5_trends_bugfixes_197206_202509.nc')
racmo24_tswnet = xr.open_dataset('/nobackup/users/walj/racmo24/Daily/tsr.KNMI-2010.KEXT12.RACMO2.4p1_v5_trends_bugfixes.DD.nc')

era5_land = xr.open_dataset('/nobackup/users/walj/era5/Monthly/era5_land.nc')
era5_land_fluxes = xr.open_dataset('/nobackup/users/walj/era5/Monthly/era5_land_fluxes.nc')
era5_land_soil = xr.open_dataset('/nobackup/users/walj/era5/Monthly/era5_land_soil.nc')
era5_land_general = xr.open_dataset('/nobackup/users/walj/era5/Monthly/era5_land_gen.nc')

#%%

# import pandas as pd
# import glob

# def open_dataset(path):

#     files = sorted(glob.glob(path))

#     try:
#         ds = xr.open_mfdataset(
#             files,
#             combine='by_coords',
#             chunks='auto',
#             decode_times=True,
#         )
#         print('hoi1')

#         return ds

#     except Exception:

#         ds = xr.open_mfdataset(
#             files,
#             combine='by_coords',
#             chunks='auto',
#             decode_times=False,
#         )

#         n_time = ds.sizes['time']
#         new_time = pd.date_range(start='1979-01-01', periods=n_time, freq='MS')
#         print(new_time)
#         print('hoi2')

#         ds = ds.assign_coords(time=new_time)

#         ds['time'].attrs = {}

#         return ds


# test1 = open_dataset('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data/swsn/swsn.KNMI-2016.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc')
# test2 = open_dataset('/nobackup/users/walj/racmo23/Monthly/aclcovL/aclcovL.KNMI-2015.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc')
# test3 = open_dataset('/net/pc230066/nobackup/users/dalum/RACMO2.3/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Monthly_data/t2m/t2m.KNMI-2011-2020.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.MM.nc')

