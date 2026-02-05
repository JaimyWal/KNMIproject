#%% Imports

import xarray as xr
import numpy as np
from pathlib import Path
import os

#%% Configuration

# Input path with daily RACMO2.3 data
input_base = Path('/net/pc230066/nobackup_1/users/vanmeijg/CORDEX_CMIP6_ROOT/HXEUR12/eR2v3-v578rev-LU2015-MERRA2-fERA5/Daily_data')

# Output path for monthly data
output_base = Path('/nobackup/users/walj/racmo23/Monthly')

# Create output directory if it doesn't exist
output_base.mkdir(parents=True, exist_ok=True)

# Only process the flux variables that failed before
variables = ['latf', 'lwsd', 'lwsdcs', 'lwsn', 'lwsncs', 'senf', 'swsd', 'swsdcs', 'swsn', 'swsncs']

# Variables that should be summed instead of averaged - EMPTY since we want mean for all
sum_vars = []  # Using mean for everything

print(f"Processing variables: {variables}")

#%% Process each variable

for var in variables:
    print(f"\n{'='*60}")
    print(f"Processing variable: {var}")
    print(f"{'='*60}")
    
    var_input_dir = input_base / var
    var_output_dir = output_base / var
    var_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all daily files for this variable
    daily_files = sorted(var_input_dir.glob(f'{var}.KNMI-*.DD.nc'))
    
    if not daily_files:
        print(f"  No daily files found for {var}, skipping...")
        continue
    
    print(f"  Found {len(daily_files)} daily files")
    
    # Process each year separately to manage memory
    for daily_file in daily_files:
        # Extract year from filename (e.g., aclcov.KNMI-1979.HXEUR12.eR2v3-v578rev-LU2015-MERRA2-fERA5.DD.nc)
        filename = daily_file.name
        year = filename.split('.')[1].split('-')[1]  # Extract '1979' from 'KNMI-1979'
        
        # Construct output filename (replace DD with MM)
        out_filename = filename.replace('.DD.nc', '.MM.nc')
        out_file = var_output_dir / out_filename
        
        # Skip if already processed
        if out_file.exists():
            print(f"  Year {year}: Already exists, skipping...")
            continue
        
        print(f"  Year {year}: Processing...")
        
        try:
            # Open daily data
            ds = xr.open_dataset(daily_file)
            
            # Resample to monthly - use sum or mean depending on variable
            if var in sum_vars:
                ds_monthly = ds.resample(time='ME').sum(skipna=True)
                print(f"    Using SUM aggregation")
            else:
                ds_monthly = ds.resample(time='ME').mean(skipna=True)
                print(f"    Using MEAN aggregation")
            
            # Update attributes to reflect monthly data
            ds_monthly.attrs['frequency'] = 'monthly'
            ds_monthly.attrs['source_frequency'] = 'daily'
            ds_monthly.attrs['aggregation_method'] = 'sum' if var in sum_vars else 'mean'
            
            # Save to NetCDF with compression
            encoding = {}
            for data_var in ds_monthly.data_vars:
                encoding[data_var] = {
                    'zlib': True,
                    'complevel': 4,
                    'dtype': 'float32'
                }
            
            ds_monthly.to_netcdf(out_file, encoding=encoding)
            print(f"    Saved to: {out_file}")
            
            # Close datasets to free memory
            ds.close()
            ds_monthly.close()
            
        except Exception as e:
            print(f"    ERROR processing {year}: {e}")
            continue

print(f"\n{'='*60}")
print("All processing complete!")
print(f"Monthly data saved to: {output_base}")
print(f"{'='*60}")

#%% E-OBS Configuration

# Input path with daily E-OBS data
eobs_input_base = Path('/nobackup/users/walj/eobs/Daily')

# Output path for monthly data
eobs_output_base = Path('/nobackup/users/walj/eobs/Monthly')

# Create output directory if it doesn't exist
eobs_output_base.mkdir(parents=True, exist_ok=True)

# Find all E-OBS files
eobs_files = sorted(eobs_input_base.glob('*.nc'))

print(f"\n{'='*60}")
print("Processing E-OBS data")
print(f"{'='*60}")
print(f"Found {len(eobs_files)} E-OBS files")

#%% Process each E-OBS file

for eobs_file in eobs_files:
    filename = eobs_file.name
    
    # Extract variable name (first part before _ens)
    var = filename.split('_ens')[0]
    
    # Output filename - replace nothing, just put in Monthly folder
    out_filename = filename.replace('.nc', '_monthly.nc')
    out_file = eobs_output_base / out_filename
    
    # Skip if already processed
    if out_file.exists():
        print(f"  {var}: Already exists, skipping...")
        continue
    
    print(f"\n  Processing: {filename}")
    print(f"  Variable: {var}")
    
    try:
        # Open daily data with chunking for large files
        ds = xr.open_dataset(eobs_file, chunks={'time': 365})
        
        # Resample to monthly mean
        print(f"    Resampling to monthly (using MEAN)...")
        ds_monthly = ds.resample(time='ME').mean(skipna=True)
        
        # Update attributes
        ds_monthly.attrs['frequency'] = 'monthly'
        ds_monthly.attrs['source_frequency'] = 'daily'
        ds_monthly.attrs['aggregation_method'] = 'mean'
        
        # Save to NetCDF with compression
        encoding = {}
        for data_var in ds_monthly.data_vars:
            encoding[data_var] = {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32'
            }
        
        print(f"    Computing and saving...")
        ds_monthly.to_netcdf(out_file, encoding=encoding)
        print(f"    Saved to: {out_file}")
        
        # Close datasets
        ds.close()
        
    except Exception as e:
        print(f"    ERROR processing {filename}: {e}")
        continue

print(f"\n{'='*60}")
print("E-OBS processing complete!")
print(f"Monthly data saved to: {eobs_output_base}")
print(f"{'='*60}")

#%% KNMI Configuration

# Input path with daily KNMI text files
knmi_input_base = Path('/nobackup/users/walj/knmi/Daily')

# Output path for monthly data
knmi_output_base = Path('/nobackup/users/walj/knmi/Monthly')

# Create output directory if it doesn't exist
knmi_output_base.mkdir(parents=True, exist_ok=True)

# Find all KNMI text files
knmi_files = sorted(knmi_input_base.glob('*.txt'))

print(f"\n{'='*60}")
print("Processing KNMI station data")
print(f"{'='*60}")
print(f"Found {len(knmi_files)} KNMI files")

#%% Process each KNMI file

for knmi_file in knmi_files:
    filename = knmi_file.name
    station_name = filename.replace('KNMI_', '').replace('.txt', '')
    
    # Output filename
    out_filename = filename.replace('.txt', '_monthly.nc')
    out_file = knmi_output_base / out_filename
    
    # Skip if already processed
    if out_file.exists():
        print(f"  {station_name}: Already exists, skipping...")
        continue
    
    print(f"\n  Processing: {filename}")
    print(f"  Station: {station_name}")
    
    try:
        # Read the text file, skipping comment lines
        # Header line starts with "# STN,YYYYMMDD,..."
        import pandas as pd
        
        # Read file and find header line
        with open(knmi_file, 'r') as f:
            lines = f.readlines()
        
        # Find the header line (starts with "# STN")
        header_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('# STN'):
                header_idx = i
                break
        
        if header_idx is None:
            print(f"    ERROR: Could not find header line")
            continue
        
        # Parse header - remove "# " prefix and split by comma
        header_line = lines[header_idx].strip()[2:]  # Remove "# "
        columns = [col.strip() for col in header_line.split(',')]
        
        # Read data starting after header
        df = pd.read_csv(
            knmi_file,
            skiprows=header_idx + 1,
            header=None,
            names=columns,
            na_values=['     ', ''],
            skipinitialspace=True
        )
        
        # Parse date
        df['time'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
        df = df.drop(columns=['STN', 'YYYYMMDD'])
        df = df.set_index('time')
        
        # Convert to xarray Dataset
        ds = xr.Dataset.from_dataframe(df)
        
        # Add station coordinate
        ds = ds.expand_dims({'station': [station_name]})
        
        # Resample to monthly mean
        print(f"    Resampling to monthly (using MEAN)...")
        ds_monthly = ds.resample(time='ME').mean(skipna=True)
        
        # Update attributes
        ds_monthly.attrs['frequency'] = 'monthly'
        ds_monthly.attrs['source_frequency'] = 'daily'
        ds_monthly.attrs['aggregation_method'] = 'mean'
        ds_monthly.attrs['station'] = station_name
        ds_monthly.attrs['source'] = 'KNMI'
        
        # Save to NetCDF with compression
        encoding = {}
        for data_var in ds_monthly.data_vars:
            encoding[data_var] = {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32'
            }
        
        print(f"    Saving...")
        ds_monthly.to_netcdf(out_file, encoding=encoding)
        print(f"    Saved to: {out_file}")
        
    except Exception as e:
        print(f"    ERROR processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*60}")
print("KNMI processing complete!")
print(f"Monthly data saved to: {knmi_output_base}")
print(f"{'='*60}")
