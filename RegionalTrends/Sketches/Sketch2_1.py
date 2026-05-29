from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

RED = '#c62828'
BLUE = '#1565c0'
BLACK = '#111111'
GRID = '0.88'
YEAR_LINE = '0.55'
OUTPUT_DIR = Path('/nobackup/users/walj/Figure_sketches')


def draw_arrow(ax, start, end, color, lw=0.35, ms=5.0, zorder=4, alpha=0.8):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle='-|>',
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def seasonal_cycle(day):
    return -3.1 * np.cos(2 * np.pi * (day - 18) / 365.0) + 4.2


def build_data():
    x = np.arange(731)
    day = x % 365
    time_years = x / 365.0

    T = (
        seasonal_cycle(day)
        + 0.28 * time_years
        + 0.10 * time_years * np.cos(2 * np.pi * (day - 210) / 365.0)
    )

    A_tot = T[1:] - T[:-1]

    day_mid = day[:-1]

    # Red most positive in winter and most negative in summer
    # Blue exactly the opposite
    split = 0.22 * np.cos(2 * np.pi * (day_mid - 15) / 365.0)

    A_red = 0.5 * A_tot + split
    A_blue = 0.5 * A_tot - split

    np.testing.assert_allclose(A_red + A_blue, A_tot, rtol=0.0, atol=1.0e-12)

    return x, T, A_red, A_blue, A_tot


def draw_manual_grid(ax, x_grid, y_grid, y_axis, y_top):
    for xx in x_grid:
        ax.plot([xx, xx], [y_axis, y_top], color=GRID, lw=0.6, zorder=0)
    for yy in y_grid:
        ax.plot([0, 730], [yy, yy], color=GRID, lw=0.75, zorder=0)


def draw_background(
    ax,
    x,
    T,
    dot_size=6.0,
    curve_lw=0.65,
    alpha_line=0.50,
    alpha_dots=0.75,
):
    ax.plot(x[:365], T[:365], color=BLACK, lw=curve_lw, alpha=alpha_line, zorder=1)
    ax.plot(x[365:730], T[365:730], color=BLACK, lw=curve_lw, alpha=alpha_line, zorder=1)

    ax.scatter(
        x[:365], T[:365],
        s=dot_size,
        c=BLACK,
        edgecolors='none',
        alpha=alpha_dots,
        zorder=2,
    )
    ax.scatter(
        x[365:730], T[365:730],
        s=dot_size,
        c=BLACK,
        edgecolors='none',
        alpha=alpha_dots,
        zorder=2,
    )


def draw_total_arrows(ax, x, T, A_tot, indices, lw=0.18, ms=2.6, alpha=0.25, zorder=3):
    for i in indices:
        draw_arrow(
            ax,
            (x[i], T[i]),
            (x[i + 1], T[i] + A_tot[i]),
            BLACK,
            lw=lw,
            ms=ms,
            alpha=alpha,
            zorder=zorder,
        )


def draw_component_arrows_daily(
    ax,
    x,
    T,
    A_red,
    A_blue,
    indices,
    lw=0.70,
    ms=6.0,
    alpha=0.95,
    zorder=7,
):
    for i in indices:
        draw_arrow(
            ax,
            (x[i], T[i]),
            (x[i + 1], T[i] + A_red[i]),
            RED,
            lw=lw,
            ms=ms,
            alpha=alpha,
            zorder=zorder,
        )
        draw_arrow(
            ax,
            (x[i], T[i]),
            (x[i + 1], T[i] + A_blue[i]),
            BLUE,
            lw=lw,
            ms=ms,
            alpha=alpha,
            zorder=zorder,
        )


def draw_component_arrows_grouped_year(
    ax,
    x,
    T,
    A_red,
    A_blue,
    year_start,
    year_end,
    step=3,
    lw=0.42,
    ms=6.8,
    alpha=0.92,
    zorder=8,
):
    grouped_starts = np.arange(year_start, year_end - step + 1, step)

    for i in grouped_starts:
        i1 = i + step
        red_sum = A_red[i:i1].sum()
        blue_sum = A_blue[i:i1].sum()

        draw_arrow(
            ax,
            (x[i], T[i]),
            (x[i1], T[i] + red_sum),
            RED,
            lw=lw,
            ms=ms,
            alpha=alpha,
            zorder=zorder,
        )
        draw_arrow(
            ax,
            (x[i], T[i]),
            (x[i1], T[i] + blue_sum),
            BLUE,
            lw=lw,
            ms=ms,
            alpha=alpha,
            zorder=zorder,
        )


def zoom_ylim(T, A_red, A_blue, A_tot, i0, i1, pad=0.10):
    idx = np.arange(i0, i1 + 1)
    values = np.concatenate([
        T[i0:i1 + 2],
        T[idx] + A_red[idx],
        T[idx] + A_blue[idx],
        T[idx] + A_tot[idx],
    ])
    return values.min() - pad, values.max() + pad


def make_sketch(output_prefix=None):
    if output_prefix is None:
        output_prefix = OUTPUT_DIR / 'Sketch2_1'
    else:
        output_prefix = Path(output_prefix)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    x, T, A_red, A_blue, A_tot = build_data()

    y_axis = 0.0
    y_top = np.ceil((np.max(T) + 1.20) / 0.5) * 0.5
    y_arrow_top = y_top + 0.32

    fig, ax = plt.subplots(figsize=(13.0, 5.0))
    ax.set_xlim(-6.0, 741.0)
    ax.set_ylim(-0.72, y_arrow_top + 0.08)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    x_grid = np.r_[np.arange(0, 365, 40), 365 + np.arange(0, 365, 40)]
    y_grid = np.arange(y_axis, y_top + 0.001, 0.5)
    draw_manual_grid(ax, x_grid, y_grid, y_axis, y_top)

    for xx in [365, 730]:
        ax.plot([xx, xx], [y_axis, y_top], color=YEAR_LINE, lw=1.15, ls=(0, (5, 4)), zorder=0)

    ax.annotate(
        '',
        xy=(739.5, y_axis),
        xytext=(0.0, y_axis),
        arrowprops={'arrowstyle': '-|>', 'lw': 1.4, 'color': BLACK, 'shrinkA': 0, 'shrinkB': 0},
        annotation_clip=False,
        zorder=20,
    )
    ax.annotate(
        '',
        xy=(0.0, y_arrow_top),
        xytext=(0.0, y_axis),
        arrowprops={'arrowstyle': '-|>', 'lw': 1.4, 'color': BLACK, 'shrinkA': 0, 'shrinkB': 0},
        annotation_clip=False,
        zorder=20,
    )

    tick_days = np.arange(0, 321, 40)
    tick_labels = [str(int(v)) for v in tick_days]

    for offset in [0, 365]:
        for xx, label in zip(offset + tick_days, tick_labels):
            ax.plot([xx, xx], [y_axis, y_axis - 0.045], color=BLACK, lw=0.8, zorder=21)
            ax.text(
                xx,
                y_axis - 0.14,
                label,
                ha='center',
                va='top',
                fontsize=10,
                color=BLACK,
                zorder=21,
            )

    # Explicit 0 tick at the rightmost dashed line
    ax.plot([730, 730], [y_axis, y_axis - 0.045], color=BLACK, lw=0.8, zorder=21)
    ax.text(
        730,
        y_axis - 0.14,
        '0',
        ha='center',
        va='top',
        fontsize=10,
        color=BLACK,
        zorder=30,
        clip_on=False,
        bbox=dict(facecolor='white', edgecolor='none', pad=0.4, alpha=0.95),
    )

    draw_background(
        ax,
        x,
        T,
        dot_size=6.0,
        curve_lw=0.65,
        alpha_line=0.50,
        alpha_dots=0.75,
    )

    # Main black total arrows
    draw_total_arrows(
        ax,
        x,
        T,
        A_tot,
        indices=np.arange(730),
        lw=0.10,
        ms=2.1,
        alpha=0.10,
        zorder=3,
    )

    # Main colored arrows, grouped separately for each year
    draw_component_arrows_grouped_year(
        ax,
        x,
        T,
        A_red,
        A_blue,
        year_start=0,
        year_end=364,
        step=3,
        lw=0.42,
        ms=6.8,
        alpha=0.92,
        zorder=8,
    )
    draw_component_arrows_grouped_year(
        ax,
        x,
        T,
        A_red,
        A_blue,
        year_start=365,
        year_end=729,
        step=3,
        lw=0.42,
        ms=6.8,
        alpha=0.92,
        zorder=8,
    )

    ax.text(182, y_top + 0.22, r'Year $y$', ha='center', va='top', fontsize=28)
    ax.text(547, y_top + 0.22, r'Year $y\!+\!1$', ha='center', va='top', fontsize=28)
    ax.text(365, -0.56, 'Day of year', ha='center', va='top', fontsize=19)
    ax.text(-16, 0.5 * y_top, 'Temperature', ha='center', va='center', rotation=90, fontsize=19)

    # Inset
    zoom_vis0 = 362
    zoom_vis1 = 375
    zoom_draw0 = zoom_vis0 - 1
    zoom_draw1 = zoom_vis1

    axins = ax.inset_axes([0.39, 0.70, 0.22, 0.24])
    axins.set_facecolor('white')
    axins.set_xlim(x[zoom_vis0] - 0.18, x[zoom_vis1] + 0.18)
    axins.set_ylim(*zoom_ylim(T, A_red, A_blue, A_tot, zoom_draw0, zoom_draw1, pad=0.08))

    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('0.20')

    axins.grid(True, axis='both', color='0.92', lw=0.6)
    axins.set_xticks([])
    axins.set_yticks([])

    draw_background(
        axins,
        x,
        T,
        dot_size=18.0,
        curve_lw=0.55,
        alpha_line=0.30,
        alpha_dots=0.35,
    )

    # Colored arrows first
    draw_component_arrows_daily(
        axins,
        x,
        T,
        A_red,
        A_blue,
        indices=np.arange(zoom_draw0, zoom_draw1 + 1),
        lw=1.00,
        ms=7.0,
        alpha=0.98,
        zorder=8,
    )

    # Black total arrows on top
    draw_total_arrows(
        axins,
        x,
        T,
        A_tot,
        indices=np.arange(zoom_draw0, zoom_draw1 + 1),
        lw=0.85,
        ms=7.8,
        alpha=0.98,
        zorder=10,
    )

    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.35', lw=1.0)

    fig.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    make_sketch()