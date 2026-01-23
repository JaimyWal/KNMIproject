import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as clr
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import colormaps as cmaps
import cmocean


def fit_settings(fit_against_gmst):
    if fit_against_gmst:
        return '°C GMST', 1, 'ΔGMST (°C)'
    return 'decade', 10, 'Year'


def precip_labels(relative_precip, fit_unit):
    if relative_precip:
        return {
            'precip_trend_label': 'Relative trend (% / ' + fit_unit + ')',
            'precip_ylabel': 'Precipitation (% of climatology)',
            'precip_trend_unit': '% / ' + fit_unit,
        }
    return {
        'precip_trend_label': 'Trend (mm / ' + fit_unit + ')',
        'precip_ylabel': 'Precipitation (mm)',
        'precip_trend_unit': 'mm / ' + fit_unit,
    }


def build_cmap_sun():
    sun_colors = [
        '#2b0a3d',
        '#5c1a1b',
        '#8b2f1c',
        '#c45a1a',
        '#e39b2d',
        '#f4e27a',
    ]
    return LinearSegmentedColormap.from_list('sunshine', sun_colors, N=256)


def build_corr_cmap(corr_type):
    if corr_type == 'Diverging':
        colors = [
            '#570088', '#3700b3', '#1d00d7', '#0300f6', '#0231be',
            '#056775', '#079d2c', '#35c13b', '#80d883',
            '#ffffff', '#ffffff', '#ffffff',
            '#fff400', '#ffe400', '#ffc900', '#ffad00',
            '#ff8200', '#ff5500', '#ff2800', '#a30e03', '#6b0902',
        ]
        return {'corr_cmap': clr.ListedColormap(colors), 'corr_extreme': (None, None)}

    elif corr_type == 'Sequential':
        corr_colors = [
            '#ffffff',
            '#fff7bc',
            '#fee391',
            '#fec44f',
            '#fe9929',
            '#ec7014',
            '#cc4c02',
            '#993404',
            '#662506',
            '#3d1f0f',
        ]
        return {'corr_cmap': clr.ListedColormap(corr_colors), 'corr_extreme': ('#999898', None)}

    else:
        return {'corr_cmap': 'jet', 'corr_extreme': ("#790A77", "#510301")}


def build_plot_cfg(
    avg_crange,
    trend_crange,
    fit_unit,
    fit_range,
    relative_precip=False
):
    
    precip_labels_cfg = precip_labels(relative_precip, fit_unit)
    precip_trend_label = precip_labels_cfg['precip_trend_label']
    precip_ylabel = precip_labels_cfg['precip_ylabel']
    precip_trend_unit = precip_labels_cfg['precip_trend_unit']

    cmap_sun = build_cmap_sun()
    cmap_trend_default = ListedColormap(cmaps.cmp_b2r(np.linspace(0, 1, 20)))
    extreme_trend_default = ('#1B1C70', '#7e060c')

    cmap_sw_mean = cmocean.cm.solar
    extreme_sw_mean = (None, '#fff3b2')

    cmap_lw_mean = cmocean.cm.balance
    extreme_lw_mean = ('#0a0a86', '#700c0c')

    cmap_turb_mean = cmocean.cm.balance
    extreme_turb_mean = ('#0a0a86', '#700c0c')

    cmap_cloud_mean = cmocean.cm.gray
    extreme_cloud_mean = ('#000000', '#ffffff')

    cmap_wp_mean = cmocean.cm.matter
    extreme_wp_mean = ('#fff3b2', '#0a0a86')

    cmap_rh_mean = cmocean.cm.haline
    extreme_rh_mean = ('#0a0a86', "#fef7cd")

    cmap_default_mean = 'turbo'
    extreme_default_mean = ("#400074", "#4b0000")


    plot_cfg = {
        'Tg': {
            'label_mean': 'Temperature (°C)',
            'label_trend': 'Trend (°C / ' + fit_unit + ')',
            'cmap_mean': 'Spectral_r',
            'cmap_trend': cmaps.temp_19lev,
            'crange_mean': avg_crange,
            'crange_trend': trend_crange,
            'extreme_mean': ('#0a0a86', '#700c0c'),
            'extreme_trend': ('#000020', '#350000'),
            'label_plot': 'Temperature (°C)',
            'trend_unit': '°C / ' + fit_unit,
            'ylim_fit': fit_range,
        },
        'P': {
            'label_mean': 'Precipitation (mm)',
            'label_trend': precip_trend_label,
            'cmap_mean': cmocean.cm.rain,
            'cmap_trend': plt.get_cmap('BrBG', 20),
            'crange_mean': avg_crange,
            'crange_trend': trend_crange,
            'extreme_mean': (None, '#040812'),
            'extreme_trend': ('#271500', '#001f1f'),
            'label_plot': precip_ylabel,
            'trend_unit': precip_trend_unit,
            'ylim_fit': fit_range,
        },
        'Sq': {
            'label_mean': 'Sund. (hours/day)',
            'label_trend': 'Trend (hours/day / ' + fit_unit + ')',
            'cmap_mean': cmap_sun,
            'cmap_trend': cmap_trend_default,
            'crange_mean': avg_crange,
            'crange_trend': trend_crange,
            'extreme_mean': (None, '#fff3b2'),
            'extreme_trend': ('#1B1C70', '#7e060c'),
            'label_plot': 'Sund. (hours/day)',
            'trend_unit': 'hours/day / ' + fit_unit,
            'ylim_fit': fit_range,
        },
    }

    temp_base = plot_cfg['Tg'].copy()

    for k, lab in {
        'Tmax': 'Maximum temperature (°C)',
        'Tmin': 'Minimum temperature (°C)',
        'Tmaxmax': 'Maximum of Tmax (°C)',
        'Tminmin': 'Minimum of Tmin (°C)',
    }.items():
        plot_cfg[k] = temp_base.copy()
        plot_cfg[k]['label_mean'] = lab
        plot_cfg[k]['label_plot'] = lab

    def add_plot_cfg(
        label_mean,
        label_trend,
        cmap_mean,
        cmap_trend,
        extreme_mean,
        extreme_trend,
        label_plot,
        trend_unit,
        ylim_fit,
        crange_mean=avg_crange,
        crange_trend=trend_crange,
    ):
        return {
            'label_mean': label_mean,
            'label_trend': label_trend,
            'cmap_mean': cmap_mean,
            'cmap_trend': cmap_trend,
            'crange_mean': crange_mean,
            'crange_trend': crange_trend,
            'extreme_mean': extreme_mean,
            'extreme_trend': extreme_trend,
            'label_plot': label_plot,
            'trend_unit': trend_unit,
            'ylim_fit': ylim_fit,
        }

    sw_vars = {
        'SWin': r'SW$_{\text{in}}$ (W/m$^2$)',
        'SWnet': r'SW$_{\text{net}}$ (W/m$^2$)',
        'SWincs': r'SW$_{\text{in,cs}}$ (W/m$^2$)',
        'SWnetcs': r'SW$_{\text{net,cs}}$ (W/m$^2$)',
    }
    for v, lab in sw_vars.items():
        plot_cfg[v] = add_plot_cfg(
            label_mean=lab,
            label_trend=r'Trend (W/m$^2$ / ' + fit_unit + ')',
            cmap_mean=cmap_sw_mean,
            cmap_trend=cmap_trend_default,
            extreme_mean=extreme_sw_mean,
            extreme_trend=extreme_trend_default,
            label_plot=lab,
            trend_unit=r'W/m$^2$ / ' + fit_unit,
            ylim_fit=fit_range,
        )

    lw_vars = {
        'LWin': r'LW$_{\text{in}}$ (W/m$^2$)',
        'LWincs': r'LW$_{\text{in,cs}}$ (W/m$^2$)',
        'LWnet': r'LW$_{\text{net}}$ (W/m$^2$)',
        'LWnetcs': r'LW$_{\text{net,cs}}$ (W/m$^2$)',
    }
    for v, lab in lw_vars.items():
        plot_cfg[v] = add_plot_cfg(
            label_mean=lab,
            label_trend=r'Trend (W/m$^2$ / ' + fit_unit + ')',
            cmap_mean=cmap_lw_mean,
            cmap_trend=cmap_trend_default,
            extreme_mean=extreme_lw_mean,
            extreme_trend=extreme_trend_default,
            label_plot=lab,
            trend_unit=r'W/m$^2$ / ' + fit_unit,
            ylim_fit=fit_range,
        )

    turb_vars = {
        'SHF': r'SHF (W/m$^2$)',
        'LHF': r'LHF (W/m$^2$)',
    }
    for v, lab in turb_vars.items():
        plot_cfg[v] = add_plot_cfg(
            label_mean=lab,
            label_trend=r'Trend (W/m$^2$ / ' + fit_unit + ')',
            cmap_mean=cmap_turb_mean,
            cmap_trend=cmap_trend_default,
            extreme_mean=extreme_turb_mean,
            extreme_trend=extreme_trend_default,
            label_plot=lab,
            trend_unit=r'W/m$^2$ / ' + fit_unit,
            ylim_fit=fit_range,
        )

    cloud_vars = {
        'CloudLow': 'Low cloud (%)',
        'CloudMid': 'Mid cloud (%)',
        'CloudHigh': 'High cloud (%)',
        'CloudTotal': 'Total cloud (%)',
    }
    for v, lab in cloud_vars.items():
        plot_cfg[v] = add_plot_cfg(
            label_mean=lab,
            label_trend='Trend (% / ' + fit_unit + ')',
            cmap_mean=cmap_cloud_mean,
            cmap_trend=cmap_trend_default,
            extreme_mean=extreme_cloud_mean,
            extreme_trend=extreme_trend_default,
            label_plot=lab,
            trend_unit='% / ' + fit_unit,
            ylim_fit=fit_range,
        )

    wp_vars = {
        'LWP': 'LWP (g/m$^2$)',
        'IWP': 'IWP (g/m$^2$)',
    }
    for v, lab in wp_vars.items():
        plot_cfg[v] = add_plot_cfg(
            label_mean=lab,
            label_trend=r'Trend (g/m$^2$ / ' + fit_unit + ')',
            cmap_mean=cmap_wp_mean,
            cmap_trend=cmap_trend_default,
            extreme_mean=extreme_wp_mean,
            extreme_trend=extreme_trend_default,
            label_plot=lab,
            trend_unit=r'g/m$^2$ / ' + fit_unit,
            ylim_fit=fit_range,
        )

    rh_vars = {
        'RH': 'Relative Humidity (%)',
        'RH_proxy': 'Relative Humidity (%)',
    }
    for v, lab in rh_vars.items():
        plot_cfg[v] = add_plot_cfg(
            label_mean=lab,
            label_trend='Trend (% / ' + fit_unit + ')',
            cmap_mean=cmap_rh_mean,
            cmap_trend=cmap_trend_default,
            extreme_mean=extreme_rh_mean,
            extreme_trend=extreme_trend_default,
            label_plot=lab,
            trend_unit='% / ' + fit_unit,
            ylim_fit=fit_range,
        )

    dimless_vars = {
        'Bowen': 'Bowen Ratio',
        'Albedo': 'Albedo',
    }
    for v, lab in dimless_vars.items():
        plot_cfg[v] = add_plot_cfg(
            label_mean=lab,
            label_trend='Trend ( / ' + fit_unit + ')',
            cmap_mean=cmap_default_mean,
            cmap_trend=cmap_trend_default,
            extreme_mean=extreme_default_mean,
            extreme_trend=extreme_trend_default,
            label_plot=lab,
            trend_unit=' / ' + fit_unit,
            ylim_fit=fit_range,
        )

    p_vars = {
        'Ps': 'Surface Pressure (hPa)',
        'Psl': 'Sea Level Pressure (hPa)',
    }
    for v, lab in p_vars.items():
        plot_cfg[v] = add_plot_cfg(
            label_mean=lab,
            label_trend='Trend (hPa / ' + fit_unit + ')',
            cmap_mean=cmap_default_mean,
            cmap_trend=cmap_trend_default,
            extreme_mean=extreme_default_mean,
            extreme_trend=extreme_trend_default,
            label_plot=lab,
            trend_unit='hPa / ' + fit_unit,
            ylim_fit=fit_range,
        )

    q_vars = {
        'Q': 'Specific Humidity (g/kg)',
        'Q_era': 'Specific Humidity (g/kg)',
        'Q_obs': 'Specific Humidity (g/kg)',
    }
    for v, lab in q_vars.items():
        plot_cfg[v] = add_plot_cfg(
            label_mean=lab,
            label_trend='Trend (g/kg / ' + fit_unit + ')',
            cmap_mean=cmap_default_mean,
            cmap_trend=cmap_trend_default,
            extreme_mean=extreme_default_mean,
            extreme_trend=extreme_trend_default,
            label_plot=lab,
            trend_unit='g/kg / ' + fit_unit,
            ylim_fit=fit_range,
        )

    return plot_cfg


def plot_args(plot_cfg, var, data_compare):

    cfg_plot = plot_cfg[var].copy()
    if data_compare is not None:
        cfg_plot['cmap_mean'] = cfg_plot['cmap_trend']
        cfg_plot['extreme_mean'] = cfg_plot['extreme_trend']
        cfg_plot['label_mean'] = 'Difference in ' + cfg_plot['label_mean']
        cfg_plot['label_trend'] = 'Difference in ' + cfg_plot['label_trend']

    return cfg_plot