#%% Imports

# Standard libraries
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import statsmodels.api as sm
from pathlib import Path
import sys
from importlib import reload

# Custom libraries
PROJECT_ROOT = Path.home() / 'KNMIproject'
sys.path.insert(0, str(PROJECT_ROOT))

from RegionalTrends.Helpers import ProcessSource
reload(ProcessSource)
from RegionalTrends.Helpers.ProcessSource import process_source

from RegionalTrends.Helpers import AreaWeights
reload(AreaWeights)
from RegionalTrends.Helpers.AreaWeights import area_weights, area_weighted_mean

# Data config custom libraries
import RegionalTrends.Helpers.Config.Constants as Constants
reload(Constants)


plt.rcParams['axes.unicode_minus'] = False
dask.config.set(scheduler='threads', num_workers=12)

#%% User inputs

# Main arguments
var = 'Psl'
file_freq = 'Monthly'
data_area = ['Eobs_fine','ERA5_coarse', 'RACMO2.4_KEXT12']
stations = ['Bilt', 'Eelde', 'Vlissingen', 'Maastricht', 'Kooy']

# Data selection arguments
months_dict = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
years = [1980, 2020]
lats = [50.7, 53.6]
lons = [3.25, 7.35]
proj_sel = 'RACMO2.4'
land_only = True
trim_border = None

# Plotting arguments
fig_places = (2, 2)
panel_width = 6
panel_height = 4
fit_range = None
uncertainty_band = False
shared_labels = True
save_name = None

# Other arguments
fit_against_gmst = False # Werkt niet denk ik
rolling_mean_var = False
rolling_mean_years = 3
min_periods = 1
relative_precip = False

#%% Dataset configurations

data_sources = Constants.DATA_SOURCES
station_sources = Constants.STATION_SOURCES
station_coord_cfg = Constants.STATION_COORD_CFG
var_file_cfg = Constants.VAR_FILE_CFG
var_name_cfg = Constants.VAR_NAME_CFG
var_units_cfg = Constants.VAR_UNIT_CFG
var_symbol_cfg = Constants.VAR_SYMBOL_CFG
proj_cfg = Constants.PROJ_CFG

if fit_against_gmst:
    fit_unit = '°C GMST'
    fit_scaling = 1
    fit_x_label = r'$\Delta$GMST (°C)'
else:
    fit_unit = 'decade'
    fit_scaling = 10
    fit_x_label = 'Year'

if var == 'P' and relative_precip:
    label_unit = '% / ' + fit_unit
elif var_units_cfg[var] == '':
    label_unit = 'per' + fit_unit
else:
    label_unit = var_units_cfg[var] + ' / ' + fit_unit

proj_sel = proj_cfg.get(proj_sel, ccrs.PlateCarree())

#%% Loading data for chosen area

def combine_lists(a, b):
    if a is None and b is None:
        return None
    return (a or []) + (b or [])

data_area_all = combine_lists(data_area, stations)

# Determine if we need to extend the year range for non-monotonic months (e.g., DJF)
all_months = set()
needs_prev_year = False
for months in months_dict.values():
    all_months.update(months)
    # Check if months list is non-monotonic (like [12, 1, 2])
    if months != sorted(months):
        needs_prev_year = True

years_extended = [years[0] - 1, years[1]] if needs_prev_year else years

# Pre-load all data once per source (with all months)
print('Pre-loading data for all sources...')
source_data_cache = {}
weights_cache = {}

for src in data_area_all:
    print(f'  Loading {src}...')
    
    is_station = src in station_sources

    if not is_station:
        if isinstance(lats, str) or isinstance(lons, str):
            station_name = lats if isinstance(lats, str) else lons
            lat_sel = station_coord_cfg[station_name]['latitude']
            lon_sel = station_coord_cfg[station_name]['longitude']
        else:
            lat_sel = lats
            lon_sel = lons
    else:
        lat_sel = None
        lon_sel = None

    # Load with all months and extended years
    data_res = process_source(
        src,
        var,
        data_sources,
        station_sources,
        file_freq,
        var_file_cfg,
        proj_cfg,
        months=None,  # Load all months
        years=years_extended,
        lats=lat_sel,
        lons=lon_sel,
        land_only=land_only,
        trim_border=trim_border,
        rotpole_sel=proj_sel,
        rolling_mean_var=rolling_mean_var,
        fit_against_gmst=fit_against_gmst,
        rolling_mean_years=rolling_mean_years,
        min_periods=min_periods,
        return_items=('avg', 'monthly'),
    )
    
    # Compute and cache weights
    weights = area_weights(data_res['avg'], rotpole_native=proj_cfg.get(src, ccrs.PlateCarree()))
    weights_cache[src] = weights.compute() if hasattr(weights, 'compute') else weights
    
    # Compute and cache the data arrays
    source_data_cache[src] = {
        'avg': data_res['avg'].compute() if hasattr(data_res['avg'], 'compute') else data_res['avg'],
        'monthly': data_res['monthly'].compute() if hasattr(data_res['monthly'], 'compute') else data_res['monthly'],    
    }

print('Data pre-loading complete.')

# Now process each month selection using cached data
all_results = {}

for month_key, months in months_dict.items():
    
    print(f'Processing months: {month_key}')

    if data_area_all is not None:

        data_area_avg = {}
        data_area_monthly = {}
        data_area_fit = {}
        
        # Determine climate year logic for this month selection
        month_start = months[0]
        month_end = months[-1]
        spans_year_boundary = month_start > month_end  # e.g., [12, 1, 2]

        for src in data_area_all:

            cached = source_data_cache[src]
            weights = weights_cache[src]
            
            # Filter monthly data to selected months
            monthly_all = cached['monthly']
            monthly_sel = monthly_all.sel(time=monthly_all['time.month'].isin(months))
            
            # Assign climate year to monthly data
            month_m = monthly_sel['time'].dt.month
            year_m = monthly_sel['time'].dt.year
            if spans_year_boundary:
                # For DJF: December gets next year, Jan/Feb keep their year
                clim_year = xr.where(month_m >= month_start, year_m + 1, year_m)
            else:
                clim_year = year_m
            
            monthly_sel = monthly_sel.assign_coords(clim_year=clim_year)
            
            # Filter to requested climate years
            monthly_for_fit = monthly_sel.where(
                (monthly_sel['clim_year'] >= years[0]) & (monthly_sel['clim_year'] <= years[1]),
                drop=True
            )
            
            # Group by climate year to get yearly means
            yearly_fit = monthly_for_fit.groupby('clim_year').mean('time')
            yearly_fit = yearly_fit.rename({'clim_year': 'fit_against'})
            
            # Compute area-weighted means
            avg_val = area_weighted_mean(cached['avg'], weights=weights).compute()
            monthly_val = area_weighted_mean(monthly_sel, weights=weights).compute()
            yearly_val = area_weighted_mean(yearly_fit, weights=weights).compute()

            if var == 'P' and relative_precip:
                data_area_monthly[src] = 100*monthly_val / avg_val
                data_area_fit[src] = 100*yearly_val / avg_val
            else:
                data_area_avg[src] = avg_val
                data_area_monthly[src] = monthly_val
                data_area_fit[src] = yearly_val

        station_keys = [k for k in data_area_avg if k in station_sources]

        if station_keys:
            data_area_avg['Stations'] = xr.concat(
                [data_area_avg[k] for k in station_keys],
                dim='station'
            ).mean(dim='station').compute()

            data_area_monthly['Stations'] = xr.concat(
                [data_area_monthly[k] for k in station_keys],
                dim='station'
            ).mean(dim='station').compute()

            data_area_fit['Stations'] = xr.concat(
                [data_area_fit[k] for k in station_keys],
                dim='station'
            ).mean(dim='station').compute()
        
        all_results[month_key] = {
            'data_area_avg': data_area_avg,
            'data_area_monthly': data_area_monthly,
            'data_area_fit': data_area_fit
        }

# Clear cache to free memory
del source_data_cache
del weights_cache

#%% Fit statistics for area

data_area_sources = (
    ['Stations'] + (data_area or [])
    if stations is not None
    else data_area
)

all_trend_stats = {}

for month_key in months_dict.keys():
    
    if data_area_sources is not None:

        trend_stats = {}
        data_area_fit = all_results[month_key]['data_area_fit']

        for src in data_area_sources:

            fit_data = data_area_fit[src]
            
            x_arr = fit_data['fit_against'].values
            y_arr = fit_data.values

            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            x_clean = x_arr[mask]
            y_clean = y_arr[mask]

            X = sm.add_constant(x_clean)

            # lags = np.ceil(len(x_clean)**(1/4)).astype(int)
            # model = sm.OLS(y_clean, X).fit(cov_type='HAC', cov_kwds={'maxlags':3})
            model = sm.OLS(y_clean, X).fit()

            slope = model.params[1]
            intercept = model.params[0]

            slope_std = model.bse[1]
            slope_trend = slope*fit_scaling
            slope_trend_std = slope_std*fit_scaling

            trend_stats[src] = {
                'model': model,
                'x_clean': x_clean,
                'y_clean': y_clean,
                'slope': slope,
                'intercept': intercept,
                'slope_trend': slope_trend,
                'slope_trend_std': slope_trend_std,
            }
        
        all_trend_stats[month_key] = trend_stats

#%% Temporal plotting for area

if data_area_sources is not None:

    colors = ['#000000', '#DB2525', '#0168DE', '#00A236', "#CA721B", '#7B2CBF']

    month_keys = list(months_dict.keys())
    n_months = len(month_keys)
    n_rows, n_cols = fig_places
    
    # Calculate figure size based on panel dimensions
    fig_width = panel_width * n_cols
    fig_height = panel_height * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        sharex=True,
        sharey=False
    )
    
    # Handle axes array shape
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Determine fit_range type
    fit_range_is_dict = isinstance(fit_range, dict)

    for idx, month_key in enumerate(month_keys):
        
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]
        
        trend_stats = all_trend_stats[month_key]

        for ii, src in enumerate(data_area_sources):

            stats = trend_stats[src]

            model = stats['model']
            x_clean = stats['x_clean']
            y_clean = stats['y_clean']

            slope_trend = stats['slope_trend']
            slope_trend_std = stats['slope_trend_std']

            color = colors[ii % len(colors)]

            if src == 'Stations':
                base_name = 'Stations'
            else:
                base_name = next(key for key in data_sources if key in src)

            if base_name == 'Eobs':
                base_name = 'E-OBS'

            label = (
                rf'{base_name} ({slope_trend:.2f} $\pm$ {slope_trend_std:.2f} '
                f'{label_unit})'
            )

            order = np.argsort(x_clean)
            x_sorted = x_clean[order]
            y_sorted = y_clean[order]
            
            X_sorted = sm.add_constant(x_sorted)
            pred = model.get_prediction(X_sorted)
            frame = pred.summary_frame(alpha=0.05)

            y_trend = frame['mean'].values
            y_lo = frame['mean_ci_lower'].values
            y_hi = frame['mean_ci_upper'].values

            ax.plot(
                    x_sorted,
                    y_sorted,
                    c=color,
                    linewidth=2,
                    zorder=10,
                    ms=6,
                    marker='o',
                    linestyle='--',
            )

            ax.plot(
                x_sorted,
                y_trend,
                c=color,
                linewidth=2.5,
                alpha=1,
                label=label,
                zorder=15
            )

            if uncertainty_band:
                ax.fill_between(
                    x_sorted,
                    y_lo,
                    y_hi,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )

        ax.grid()
        ax.set_title(month_key, fontsize=max(18, int(panel_height * 5)), fontweight='bold')
        ax.tick_params(axis='both', labelsize=max(12, int(panel_height * 3)), length=5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Set fit range for this month
        if fit_range_is_dict:
            range_this = fit_range.get(month_key, None)
        else:
            range_this = fit_range
        
        if range_this is not None:
            ax.set_ylim(*range_this)
        
        leg = ax.legend(fontsize=max(10, int(panel_height * 2.5)), handlelength=1.5, handletextpad=0.4, loc='best')
        for line in leg.get_lines():
            line.set_linewidth(3.0)
        leg.set_zorder(20)

        # X-axis and Y-axis labels
        if not shared_labels:
            # Per-subplot labels (only on edges)
            if row_idx == n_rows - 1:
                ax.set_xlabel(fit_x_label, fontsize=max(14, int(panel_height * 4)))
            if col_idx == 0:
                ax.set_ylabel(var_name_cfg[var] + ' (' + var_units_cfg[var] + ')', fontsize=max(14, int(panel_height * 4)))

    # Shared labels for entire figure (centered on plot area)
    if shared_labels:
        label_fontsize = max(38, int(panel_height * 10))
        # X-label: centered horizontally, below all subplots
        fig.text(0.5, -0.01, fit_x_label, ha='center', va='top', 
                 fontsize=label_fontsize)
        # Y-label: centered vertically, left of all subplots
        fig.text(-0.005, 0.5, var_name_cfg[var] + ' (' + var_units_cfg[var] + ')', 
                 ha='right', va='center', rotation=90, 
                 fontsize=label_fontsize)

    # Hide any unused subplots
    for idx in range(n_months, n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx, col_idx].set_visible(False)

    if save_name is not None:
        out = Path.home() / 'KNMIproject' / 'RegionalTrends' / 'Main' / 'figuresproposal' / (save_name + '.pdf')
        plt.savefig(str(out), 
                    format='pdf', 
                    bbox_inches='tight')
    
    plt.show()

    #%%


