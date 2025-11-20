import numpy as np
import xarray as xr
import pandas as pd

def preprocess_knmi_monthly(
    file_path,
    var_name,
    months=None,
    years=None,
    scale_factor=0.1):
    
    # Find header line and its line number
    with open(file_path, 'r') as f:
        header = None
        header_line_no = None
        for ii, line in enumerate(f):
            if line.startswith('# STN'):
                header = line[2:].strip()  # drop '# '
                header_line_no = ii
                break

    col_names = [c.strip() for c in header.split(',')]

    # Read only the data lines after the header
    df = pd.read_csv(
        file_path,
        header=None,
        names=col_names,
        usecols=['YYYYMMDD', var_name],
        skiprows=header_line_no + 1,   # skip everything up to and including the header
        na_values=[''],
        skipinitialspace=True,
        low_memory=False,
    )

    df[var_name] = pd.to_numeric(df[var_name], errors='coerce')

    df.loc[df[var_name] == -1, var_name] = 0

    # Parse time index
    df['time'] = pd.to_datetime(df['YYYYMMDD'].astype(str), format='%Y%m%d')
    df = df.set_index('time')

    # Apply scale factor (eg 0.1 for TG, RH)
    series = df[var_name].astype('float32')*scale_factor

    # Select months
    if months is not None:
        series = series[series.index.month.isin(months)]

    # Select years
    if years is not None and isinstance(years, (list, tuple)):
        if len(years) == 2:
            start, end = years
            if start is not None:
                series = series[series.index.year >= start]
            if end is not None:
                series = series[series.index.year <= end]
        else:
            series = series[series.index.year.isin(years)]

    # Aggregate to monthly means (start of month, like 'MS')
    series_monthly = series.resample('MS').mean()

    data_monthly = xr.DataArray(
        series_monthly.values.astype('float32'),
        coords={'time': series_monthly.index},
        dims=['time'],
        name=var_name
    )

    return data_monthly