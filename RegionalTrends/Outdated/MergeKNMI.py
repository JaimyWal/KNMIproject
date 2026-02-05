#%% Imports

import pandas as pd
from pathlib import Path

#%% Configuration

# Input path with daily KNMI text files
input_base = Path('/nobackup/users/walj/knmi/Daily')

# Output path for monthly data
output_base = Path('/nobackup/users/walj/knmi/Monthly')

# Create output directory if it doesn't exist
output_base.mkdir(parents=True, exist_ok=True)

# Find all KNMI text files
knmi_files = sorted(input_base.glob('*.txt'))

print(f"Found {len(knmi_files)} KNMI files")

#%% Process each KNMI file

for knmi_file in knmi_files:
    filename = knmi_file.name
    station_name = filename.replace('KNMI_', '').replace('.txt', '')
    
    # Output filename
    out_file = output_base / filename
    
    # Skip if already processed
    if out_file.exists():
        print(f"  {station_name}: Already exists, skipping...")
        continue
    
    print(f"\nProcessing: {filename}")
    
    try:
        # Read file and extract header
        with open(knmi_file, 'r') as f:
            header_lines = []
            header_idx = None
            for i, line in enumerate(f):
                if line.strip().startswith('# STN'):
                    header_idx = i
                    header_line = line
                    break
                header_lines.append(line)
        
        if header_idx is None:
            print(f"  ERROR: Could not find header line")
            continue
        
        # Parse header - remove "# " prefix and split by comma
        columns = [col.strip() for col in header_line[2:].strip().split(',')]
        
        # Read data
        df = pd.read_csv(
            knmi_file,
            skiprows=header_idx + 1,
            header=None,
            names=columns,
            na_values=['     ', ''],
            skipinitialspace=True
        )
        
        # Parse date and set as index
        df['time'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
        stn = df['STN'].iloc[0]  # Save station number
        df = df.drop(columns=['STN', 'YYYYMMDD'])
        df = df.set_index('time')
        
        # Resample to monthly mean
        df_monthly = df.resample('MS').mean()
        
        # Reset index and recreate YYYYMMDD column (first day of month)
        df_monthly = df_monthly.reset_index()
        df_monthly['YYYYMMDD'] = df_monthly['time'].dt.strftime('%Y%m%d').astype(int)
        df_monthly['STN'] = stn
        df_monthly = df_monthly.drop(columns=['time'])
        
        # Reorder columns to match original
        df_monthly = df_monthly[columns]
        
        # Write output file
        with open(out_file, 'w') as f:
            # Write original header lines (comments)
            for line in header_lines:
                f.write(line)
            # Write column header
            f.write(header_line)
            # Write data
            for _, row in df_monthly.iterrows():
                values = []
                for col in columns:
                    val = row[col]
                    if pd.isna(val):
                        values.append('     ')
                    elif col in ['STN', 'YYYYMMDD']:
                        values.append(f'{int(val):>5}')
                    else:
                        values.append(f'{val:>18g}')
                f.write(','.join(values) + '\n')
        
        print(f"  Saved to: {out_file}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\nDone! Monthly data saved to: {output_base}")
