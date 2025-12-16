#%% Station NaN diagnostics and daily plots

# Standard libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from importlib import reload

import ProcessStation
reload(ProcessStation)          
from ProcessStation import preprocess_station_monthly

#%% User inputs

var = 'RH'
stations = ['Bilt', 'Eelde', 'Maastricht', 'Vlissingen', 'Kooy']
months = [12, 1, 2]
years = [1980, 2024] 

#%%

base_dir_station = '/nobackup/users/walj/knmi'
file_cfg_station = {
    'Bilt': 'KNMI_Bilt.txt',
    'Cabauw': 'KNMI_Cabauw.txt',
    'Eelde': 'KNMI_Eelde.txt',
    'Maastricht': 'KNMI_Maastricht.txt',
    'Vlissingen': 'KNMI_Vlissingen.txt',
    'Kooy': 'KNMI_Kooy.txt'
}

#%% Load daily station data for selection

daily = {}
orig_index = {}

for stn in stations:
    fp = os.path.join(base_dir_station, file_cfg_station[stn])

    da = preprocess_station_monthly(
        file_path=fp,
        var_name=var,
        months=months,
        years=years,
        aggregate=False
    )

    s = da.to_pandas()
    s.name = stn
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    daily[stn] = s
    orig_index[stn] = s.index

df_daily = pd.concat(daily.values(), axis=1).sort_index()

#%% NaN counting for selected period

def count_nans_by_month_selected(series_daily, months=None, years=None):
    s = series_daily.copy()

    if months is not None:
        s = s[s.index.month.isin(months)]

    if years is not None and isinstance(years, (list, tuple)):
        if len(years) == 2:
            start, end = years
            if start is not None:
                s = s[s.index.year >= start]
            if end is not None:
                s = s[s.index.year <= end]
        else:
            s = s[s.index.year.isin(years)]

    return s.isna().groupby(s.index.month).sum().astype(int).reindex(range(1, 13), fill_value=0)

def count_nans_by_year_selected(series_daily, months=None, years=None):
    s = series_daily.copy()

    if months is not None:
        s = s[s.index.month.isin(months)]

    if years is not None and isinstance(years, (list, tuple)):
        if len(years) == 2:
            start, end = years
            if start is not None:
                s = s[s.index.year >= start]
            if end is not None:
                s = s[s.index.year <= end]
        else:
            s = s[s.index.year.isin(years)]

    return s.isna().groupby(s.index.year).sum().astype(int)

df_nan_month = {}
df_nan_year = {}

for stn in df_daily.columns:
    s = df_daily[stn].reindex(orig_index[stn])
    df_nan_month[stn] = count_nans_by_month_selected(s, months=months, years=years)
    df_nan_year[stn] = count_nans_by_year_selected(s, months=months, years=years)

df_nan_month = pd.DataFrame(df_nan_month)
df_nan_year = pd.DataFrame(df_nan_year)

#%% Plots

fig1 = plot_nan_bars(
    df_nan_month,
    title=f'NaNs per month for {col_name}',
    xlabel='Month',
    ylabel='Number of missing daily values',
    ncols=1,
    figsize=(14, 12)
)

fig2 = plot_nan_bars(
    df_nan_year,
    title=f'NaNs per year for {col_name}',
    xlabel='Year',
    ylabel='Number of missing daily values',
    ncols=1,
    figsize=(14, 12)
)

ylabel_map = {
    'TG': 'Temperature',
    'RH': 'Precipitation',
    'SQ': 'Sunshine duration',
    'Q':  'Shortwave radiation (W/m²)'
}

fig3 = plot_daily_values(
    df_daily,
    title=f'Daily station values for {var}',
    ylabel=ylabel_map.get(var, var),
    figsize=(16, 7)
)

plt.show()


#%% Yearly means (from selected months) + linear fit per station

fit_scaling = 10.0  # per decade, set 1.0 for per year
trend_unit = f'per {int(fit_scaling)}y' if fit_scaling != 1.0 else 'per year'

ylabel_map = {
    'TG': 'Temperature',
    'RH': 'Precipitation',
    'SQ': 'Sunshine duration',
    'Q':  'Shortwave radiation (W/m²)'
}

def yearly_mean_selected(series_daily, months=None, years=None):
    s = series_daily.copy()

    if months is not None:
        s = s[s.index.month.isin(months)]

    if years is not None and isinstance(years, (list, tuple)):
        if len(years) == 2:
            start, end = years
            if start is not None:
                s = s[s.index.year >= start]
            if end is not None:
                s = s[s.index.year <= end]
        else:
            s = s[s.index.year.isin(years)]

    out = s.groupby(s.index.year).mean()
    out.index = out.index.astype(int)
    return out

# build yearly series per station (avoid union-index artefact)
yearly = {}
for stn in df_daily.columns:
    s = df_daily[stn].reindex(orig_index[stn])
    yearly[stn] = yearly_mean_selected(s, months=months, years=years)

df_yearly = pd.DataFrame(yearly).sort_index()

# fit stats per station
trend_stats = {}

for stn in df_yearly.columns:
    y = df_yearly[stn].values.astype(float)
    x = df_yearly.index.values.astype(float)

    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if x_clean.size < 2:
        continue

    X = sm.add_constant(x_clean)
    model = sm.OLS(y_clean, X).fit()

    slope = float(model.params[1])
    slope_std = float(model.bse[1])

    trend_stats[stn] = {
        'model': model,
        'x_clean': x_clean,
        'y_clean': y_clean,
        'slope': slope,
        'intercept': float(model.params[0]),
        'slope_trend': slope * fit_scaling,
        'slope_trend_std': slope_std * fit_scaling,
    }

def plot_yearly_with_fit(
    df_yearly,
    trend_stats,
    title,
    ylabel,
    fit_scaling=10.0,
    trend_unit='per decade',
    figsize=(12, 8)
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for stn in df_yearly.columns:
        y = df_yearly[stn].values.astype(float)
        x = df_yearly.index.values.astype(float)

        m = np.isfinite(x) & np.isfinite(y)
        x_plot = x[m]
        y_plot = y[m]

        if x_plot.size == 0:
            continue

        # build label with trend if available
        if stn in trend_stats:
            slope_trend = trend_stats[stn]['slope_trend']
            slope_trend_std = trend_stats[stn]['slope_trend_std']
            label = f'{stn} (trend: {slope_trend:.3f} ± {slope_trend_std:.3f} {trend_unit})'
        else:
            label = stn

        # plot data, capture line color
        line = ax.plot(
            x_plot,
            y_plot,
            linewidth=2.5,
            marker='o',
            ms=7,
            linestyle='--',
            label=label,
            zorder=10
        )[0]

        color = line.get_color()

        # fit and confidence band, same color
        if stn in trend_stats:
            model = trend_stats[stn]['model']
            x_clean = trend_stats[stn]['x_clean']

            order = np.argsort(x_clean)
            x_sorted = x_clean[order]

            X_sorted = sm.add_constant(x_sorted)
            pred = model.get_prediction(X_sorted)
            frame = pred.summary_frame(alpha=0.05)

            y_fit = frame['mean'].values
            y_lo = frame['mean_ci_lower'].values
            y_hi = frame['mean_ci_upper'].values

            ax.plot(
                x_sorted,
                y_fit,
                linewidth=3.0,
                color=color,
                zorder=15
            )

            ax.fill_between(
                x_sorted,
                y_lo,
                y_hi,
                color=color,
                alpha=0.15,
                linewidth=0,
                zorder=5
            )

    ax.grid(True)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Year', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=12, length=6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    leg = ax.legend(fontsize=11, handlelength=1.5, handletextpad=0.4, loc='best', ncol=1)
    for l in leg.get_lines():
        l.set_linewidth(4.0)
    leg.set_zorder(20)

    fig.tight_layout()
    return fig

fig4 = plot_yearly_with_fit(
    df_yearly,
    trend_stats,
    title=f'Yearly mean for {var} using months {months}',
    ylabel=ylabel_map.get(var, var),
    fit_scaling=fit_scaling,
    trend_unit=trend_unit,
    figsize=(12, 8)
)

