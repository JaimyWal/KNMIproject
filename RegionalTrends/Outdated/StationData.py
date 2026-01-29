"""Station NaN diagnostics and daily plots.

Loads daily KNMI station series, reports missing values (NaNs) per month and per
year (per station), and plots daily values plus yearly means with a linear trend.

COMPLETELY WRITTEN BY CHATGPT!!! THIS WAS DONE PURELY TO GET AN IDEA ABOUT THE AMOUNT OF NANS
IN THE STATION DATASETS. DO NOT USE THIS SCRIPT FOR ANY SERIOUS ANALYSIS WITHOUT
THOROUGH REVIEW AND TESTING.
"""

#%% Imports

# Standard libraries
from __future__ import annotations

from importlib import reload
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm


# Custom libraries (match Main script style)
PROJECT_ROOT = Path.home() / 'KNMIproject'
sys.path.insert(0, str(PROJECT_ROOT))

from RegionalTrends.Helpers import ProcessStation
reload(ProcessStation)
from RegionalTrends.Helpers.ProcessStation import preprocess_station

plt.rcParams['axes.unicode_minus'] = False

#%% User inputs

var = 'UG'
stations = ['Bilt', 'Eelde', 'Maastricht', 'Vlissingen', 'Kooy']

# Applied to NaN counts + trend calculation. Use `None` for full year.
months = None

# Either [start_year, end_year] or a list of explicit years.
years = [1974, 2024]

#%%

base_dir_station = Path('/nobackup/users/walj/knmi')
file_cfg_station = {
    'Bilt': 'KNMI_Bilt.txt',
    'Cabauw': 'KNMI_Cabauw.txt',
    'Eelde': 'KNMI_Eelde.txt',
    'Maastricht': 'KNMI_Maastricht.txt',
    'Vlissingen': 'KNMI_Vlissingen.txt',
    'Kooy': 'KNMI_Kooy.txt'
}

#%% Helpers

def _parse_years(years_sel: list[int] | tuple[int, ...] | None):
    if years_sel is None:
        return None, None, None

    if not isinstance(years_sel, (list, tuple)):
        raise TypeError('years must be a list/tuple like [start, end] or [y1, y2, ...]')

    if len(years_sel) == 2:
        start, end = years_sel
        if start is None or end is None:
            raise ValueError('years=[start, end] must both be non-None')
        return int(start), int(end), None

    year_list = [int(y) for y in years_sel]
    if not year_list:
        raise ValueError('years list cannot be empty')
    return min(year_list), max(year_list), set(year_list)


def build_daily_index(years_sel, months_sel=None) -> pd.DatetimeIndex:
    start_year, end_year, year_whitelist = _parse_years(years_sel)
    if start_year is None or end_year is None:
        raise ValueError('years must be provided to build a daily index')

    idx = pd.date_range(
        start=pd.Timestamp(start_year, 1, 1),
        end=pd.Timestamp(end_year, 12, 31),
        freq='D'
    )

    if year_whitelist is not None:
        idx = idx[idx.year.isin(year_whitelist)]

    if months_sel is not None:
        idx = idx[idx.month.isin(months_sel)]

    return idx


def count_nans_by_month(series_daily: pd.Series) -> pd.Series:
    return (
        series_daily.isna()
        .groupby(series_daily.index.month)
        .sum()
        .astype(int)
        .reindex(range(1, 13), fill_value=0)
    )


def count_nans_by_year(series_daily: pd.Series) -> pd.Series:
    return series_daily.isna().groupby(series_daily.index.year).sum().astype(int)


def yearly_mean_selected(series_daily: pd.Series, months_sel=None, years_sel=None) -> pd.Series:
    s = series_daily.copy()

    if months_sel is not None:
        s = s[s.index.month.isin(months_sel)]

    if years_sel is not None and isinstance(years_sel, (list, tuple)):
        if len(years_sel) == 2:
            start, end = years_sel
            s = s[(s.index.year >= start) & (s.index.year <= end)]
        else:
            s = s[s.index.year.isin(years_sel)]

    out = s.groupby(s.index.year).mean()
    out.index = out.index.astype(int)
    return out


def plot_nan_bars(
    df_nan: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    ncols: int = 2,
    figsize=(14, 10),
):
    stations = list(df_nan.columns)
    n_panels = len(stations)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    x = df_nan.index.values
    for ii, stn in enumerate(stations):
        ax = axes[ii]
        ax.bar(x, df_nan[stn].values, color='k', alpha=0.8)
        ax.set_title(stn, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=10, length=5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for jj in range(n_panels, len(axes)):
        axes[jj].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.supxlabel(xlabel, fontsize=12)
    fig.supylabel(ylabel, fontsize=12)
    return fig


def plot_daily_values(
    df_daily: pd.DataFrame,
    title: str,
    ylabel: str,
    figsize=(16, 7),
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for stn in df_daily.columns:
        ax.plot(
            df_daily.index,
            df_daily[stn].values,
            lw=1.2,
            alpha=0.85,
            label=stn,
        )

    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', labelsize=11, length=6)
    ax.legend(fontsize=10, ncol=1, loc='best')
    fig.tight_layout()
    return fig


def plot_yearly_with_fit(
    df_yearly: pd.DataFrame,
    trend_stats: dict,
    title: str,
    ylabel: str,
    trend_unit: str,
    figsize=(12, 8),
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

        if stn in trend_stats:
            slope_trend = trend_stats[stn]['slope_trend']
            slope_trend_std = trend_stats[stn]['slope_trend_std']
            label = f'{stn} (trend: {slope_trend:.3f} ± {slope_trend_std:.3f} {trend_unit})'
        else:
            label = stn

        line = ax.plot(
            x_plot,
            y_plot,
            linewidth=2.5,
            marker='o',
            ms=7,
            linestyle='--',
            label=label,
            zorder=10,
        )[0]

        color = line.get_color()

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

            ax.plot(x_sorted, y_fit, linewidth=3.0, color=color, zorder=15)
            ax.fill_between(
                x_sorted,
                y_lo,
                y_hi,
                color=color,
                alpha=0.15,
                linewidth=0,
                zorder=5,
            )

    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', labelsize=11, length=6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    leg = ax.legend(fontsize=10, handlelength=1.5, handletextpad=0.4, loc='best', ncol=1)
    for l in leg.get_lines():
        l.set_linewidth(4.0)
    leg.set_zorder(20)

    fig.tight_layout()
    return fig


#%% Load daily station data for selection

full_index = build_daily_index(years, months_sel=months)

daily = {}
for stn in stations:
    fp = base_dir_station / file_cfg_station[stn]
    if not fp.exists():
        raise FileNotFoundError(f'Missing station file: {fp}')

    da = preprocess_station(
        file_path=str(fp),
        var_name=var,
        months=months,
        years=years,
    )

    s = da.to_pandas()
    s.name = stn
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    # Ensure missing days are represented as NaN (important for NaN diagnostics)
    s = s.reindex(full_index)
    daily[stn] = s

df_daily = pd.DataFrame(daily, index=full_index)

#%% NaN counting (per station)

nan_month = {}
nan_year = {}

for stn in df_daily.columns:
    s = df_daily[stn]
    nan_month[stn] = count_nans_by_month(s)
    nan_year[stn] = count_nans_by_year(s)

df_nan_month = pd.DataFrame(nan_month)
df_nan_year = pd.DataFrame(nan_year).sort_index()

#%% Plots: NaNs and daily values

ylabel_map = {
    'TG': 'Temperature',
    'RH': 'Precipitation',
    'SQ': 'Sunshine duration',
    'Q': 'Shortwave radiation (W/m²)',
}

fig1 = plot_nan_bars(
    df_nan_month,
    title=f'NaNs per month ({var}) for selected stations',
    xlabel='Month',
    ylabel='Number of missing daily values',
    ncols=2,
    figsize=(14, 10),
)

fig2 = plot_nan_bars(
    df_nan_year,
    title=f'NaNs per year ({var}) for selected stations',
    xlabel='Year',
    ylabel='Number of missing daily values',
    ncols=2,
    figsize=(14, 10),
)

fig3 = plot_daily_values(
    df_daily,
    title=f'Daily station values ({var})',
    ylabel=ylabel_map.get(var, var),
    figsize=(16, 7),
)


#%% Yearly means (from selected months) + linear fit per station

fit_scaling = 10.0  # per decade, set 1.0 for per year
trend_unit = f'per {int(fit_scaling)}y' if fit_scaling != 1.0 else 'per year'

yearly = {}
for stn in df_daily.columns:
    yearly[stn] = yearly_mean_selected(df_daily[stn], months_sel=months, years_sel=years)

df_yearly = pd.DataFrame(yearly).sort_index()

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

fig4 = plot_yearly_with_fit(
    df_yearly,
    trend_stats,
    title=f'Yearly mean ({var}) using months {months}',
    ylabel=ylabel_map.get(var, var),
    trend_unit=trend_unit,
    figsize=(12, 8),
)

plt.show()

