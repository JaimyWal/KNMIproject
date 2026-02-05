import numpy as np
import xarray as xr
import pandas as pd

def preprocess_station(
    file_path,
    var_name,
    months=None,
    years=None):

    with open(file_path, 'r') as f:
        header = None
        header_line_no = None
        for ii, line in enumerate(f):
            if line.startswith('# STN'):
                header = line[2:].strip()
                header_line_no = ii
                break

    col_names = [c.strip() for c in header.split(',')]

    df = pd.read_csv(
        file_path,
        header=None,
        names=col_names,
        usecols=['YYYYMMDD', var_name],
        skiprows=header_line_no + 1,
        na_values=[''],
        skipinitialspace=True,
        low_memory=False,
    )

    if var_name in ['Q', 'UG', 'NG']:
        sf = 1.0
    else:
        sf = 0.1

    df[var_name] = pd.to_numeric(df[var_name], errors='coerce')

    if var_name in ['RH', 'SQ']:
        df.loc[df[var_name] == -1, var_name] = 0

    df['time'] = pd.to_datetime(df['YYYYMMDD'].astype(str), format='%Y%m%d')
    df = df.set_index('time')

    series = df[var_name].astype('float32') * sf

    if var_name == 'Q':
        series = series * 1e4 / 86400.0

    if var_name == 'NG':
        series = series.where(series != 9)
        series = series * 100.0 / 8.0

    if months is not None:
        series = series[series.index.month.isin(months)]

    if years is not None and isinstance(years, (list, tuple)):
        if len(years) == 2:
            start, end = years
            if start is not None:
                series = series[series.index.year >= start]
            if end is not None:
                series = series[series.index.year <= end]
        else:
            series = series[series.index.year.isin(years)]

    data = xr.DataArray(
        series.values.astype('float32'),
        coords={'time': series.index},
        dims=['time'],
        name=var_name
    )

    return data
