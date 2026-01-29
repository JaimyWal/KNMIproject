import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import colormaps as cmaps
import cmocean


# Hier optie voor discrete of continuous colormap toevoegen?

cmap_default_mean = 'turbo'
extreme_default_mean = ("#400074", "#4b0000")

cmap_trend_default = ListedColormap(cmaps.cmp_b2r(np.linspace(0, 1, 20)))
extreme_trend_default = ('#1B1C70', '#7e060c')


sun_colors = [
    '#2b0a3d',
    '#5c1a1b',
    '#8b2f1c',
    '#c45a1a',
    '#e39b2d',
    '#f4e27a',
]
cmap_sun = LinearSegmentedColormap.from_list('sunshine', sun_colors, N=256)


VAR_COLORS_CFG = {
    'Tg': {'cmap_mean': 'Spectral_r', 'extreme_mean': ('#0a0a86', '#700c0c'), 
           'cmap_trend': cmaps.temp_19lev, 'extreme_trend': ('#000020', '#350000')},
    'Tmax': {'cmap_mean': 'Spectral_r', 'extreme_mean': ('#0a0a86', '#700c0c'), 
             'cmap_trend': cmaps.temp_19lev, 'extreme_trend': ('#000020', '#350000')},
    'Tmin': {'cmap_mean': 'Spectral_r', 'extreme_mean': ('#0a0a86', '#700c0c'), 
             'cmap_trend': cmaps.temp_19lev, 'extreme_trend': ('#000020', '#350000')},
    'Tmaxmax': {'cmap_mean': 'Spectral_r', 'extreme_mean': ('#0a0a86', '#700c0c'), 
             'cmap_trend': cmaps.temp_19lev, 'extreme_trend': ('#000020', '#350000')},
    'Tminmin': {'cmap_mean': 'Spectral_r', 'extreme_mean': ('#0a0a86', '#700c0c'), 
             'cmap_trend': cmaps.temp_19lev, 'extreme_trend': ('#000020', '#350000')},
    'P': {'cmap_mean': cmocean.cm.rain, 'extreme_mean': (None, '#040812'), 
          'cmap_trend': plt.get_cmap('BrBG', 20), 'extreme_trend': ('#271500', '#001f1f')},
    'Sq': {'cmap_mean': cmap_sun, 'extreme_mean': ("#15031f", '#fff3b2')},
    'SWin': {'cmap_mean': cmocean.cm.solar, 'extreme_mean': ("#15031f", '#fff3b2')},
    'SWnet': {'cmap_mean': cmocean.cm.solar, 'extreme_mean': ("#15031f", '#fff3b2')},
    'SWincs': {'cmap_mean': cmocean.cm.solar, 'extreme_mean': ("#15031f", '#fff3b2')},
    'SWnetcs': {'cmap_mean': cmocean.cm.solar, 'extreme_mean': ("#15031f", '#fff3b2')},
    'LWin': {'cmap_mean': cmocean.cm.balance, 'extreme_mean': ('#0a0a86', '#700c0c')},
    'LWincs': {'cmap_mean': cmocean.cm.balance, 'extreme_mean': ('#0a0a86', '#700c0c')},
    'LWnet': {'cmap_mean': cmocean.cm.balance, 'extreme_mean': ('#0a0a86', '#700c0c')},
    'LWnetcs': {'cmap_mean': cmocean.cm.balance, 'extreme_mean': ('#0a0a86', '#700c0c')},
    'SHF': {'cmap_mean': cmocean.cm.balance, 'extreme_mean': ('#0a0a86', '#700c0c')},
    'LHF': {'cmap_mean': cmocean.cm.balance, 'extreme_mean': ('#0a0a86', '#700c0c')},
    'CloudLow': {'cmap_mean': cmocean.cm.gray, 'extreme_mean': ('#000000', '#ffffff')},
    'CloudMid': {'cmap_mean': cmocean.cm.gray, 'extreme_mean': ('#000000', '#ffffff')},
    'CloudHigh': {'cmap_mean': cmocean.cm.gray, 'extreme_mean': ('#000000', '#ffffff')},
    'CloudTotal': {'cmap_mean': cmocean.cm.gray, 'extreme_mean': ('#000000', '#ffffff')},
    'LWP': {'cmap_mean': cmocean.cm.matter, 'extreme_mean': ('#fff3b2', '#0a0a86')},
    'IWP': {'cmap_mean': cmocean.cm.matter, 'extreme_mean': ('#fff3b2', '#0a0a86')},
    'Default': {'cmap_mean': cmap_default_mean, 'extreme_mean': extreme_default_mean,
                'cmap_trend': cmap_trend_default, 'extreme_trend': extreme_trend_default},
}


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
        return {'corr_cmap': cmap_default_mean, 'corr_extreme': extreme_default_mean}


def convert_cmap(cmap, cmap_type, n_colors=20):
    if cmap_type is None:
        return cmap
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    if cmap_type == 'disc':
        if isinstance(cmap, ListedColormap):
            return cmap
        return ListedColormap(cmap(np.linspace(0, 1, n_colors)))
    elif cmap_type == 'cont':
        if isinstance(cmap, ListedColormap):
            return LinearSegmentedColormap.from_list('cont_cmap', cmap.colors, N=256)
        return cmap
    return cmap