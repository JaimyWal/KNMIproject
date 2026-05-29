from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RED = '#c62828'
BLUE = '#1565c0'
BLACK = '#111111'
GRID = '0.84'
YEAR_LINE = '0.50'
OUTPUT_DIR = Path('/nobackup/users/walj/Figure_sketches')


def draw_arrow(ax, start, end, color, lw=1.15, ms=9, zorder=4):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle='-|>',
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        shrinkA=2.5,
        shrinkB=2.5,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def ar1_noise(rng, n, sigma=1.0, phi=0.60):
    noise = rng.normal(scale=sigma, size=n)
    out = np.empty(n)
    out[0] = noise[0]

    for k in range(1, n):
        out[k] = phi*out[k - 1] + noise[k]

    return out


def build_data():
    rng = np.random.default_rng(8)

    years = np.arange(1980, 2024)
    n_years = len(years)
    n_intervals = n_years - 1
    i = np.arange(n_intervals)

    noise_tot = ar1_noise(rng, n_intervals, sigma=0.035, phi=0.55)
    noise_phys = ar1_noise(rng, n_intervals, sigma=0.070, phi=0.65)

    # Natural-looking variable total yearly tendency
    Y_tot = (
        0.055
        + 0.040*np.sin(2*np.pi*(i - 2) / 8.5)
        + 0.020*np.sin(2*np.pi*(i + 1) / 4.2)
        + noise_tot
    )

    # Larger process tendencies, with smoother variability
    Y_phys = (
        0.46
        + 0.12*np.sin(2*np.pi*(i + 2) / 10.0)
        + 0.07*np.sin(2*np.pi*(i - 1) / 5.0)
        + noise_phys
    )

    # Keep physical warming clearly larger than the total tendency,
    # so the dynamical contribution remains cooling.
    Y_phys = np.maximum(Y_phys, Y_tot + 0.24)
    Y_phys = np.maximum(Y_phys, 0.16)

    Y_dyn = Y_tot - Y_phys

    assert np.all(Y_phys > 0.0)
    assert np.all(Y_dyn < 0.0)
    np.testing.assert_allclose(Y_phys + Y_dyn, Y_tot, rtol=0.0, atol=1.0e-12)

    T = np.empty(n_years)
    T[0] = 1.25
    T[1:] = T[0] + np.cumsum(Y_tot)

    return years, T, Y_phys, Y_dyn, Y_tot


def draw_manual_grid(ax, x_grid, y_grid, x_left, x_right, y_axis, y_top):
    for xx in x_grid:
        ax.plot([xx, xx], [y_axis, y_top], color=GRID, lw=0.8, zorder=0)

    for yy in y_grid:
        ax.plot([x_left, x_right], [yy, yy], color=GRID, lw=0.8, zorder=0)


def draw_tick(ax, x, y, length_points=8, lw=0.9):
    dpi = ax.figure.dpi

    start_disp = np.array(ax.transData.transform((x, y)), dtype=float)
    end_disp = start_disp + np.array([0, -length_points]) * dpi / 72.0

    end_data = ax.transData.inverted().transform(end_disp)

    ax.plot(
        [x, end_data[0]],
        [y, end_data[1]],
        color=BLACK,
        lw=lw,
        zorder=40,
        clip_on=False,
    )


def make_sketch(output_prefix=None):
    if output_prefix is None:
        output_prefix = OUTPUT_DIR / 'Sketch3_1'
    else:
        output_prefix = Path(output_prefix)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    years, T, Y_phys, Y_dyn, Y_tot = build_data()

    split_year = 2002

    x_left = 1979.3
    x_right = 2024.1
    y_axis = 0.0

    y_top = np.ceil((np.max(T[:-1] + Y_phys) + 0.50) / 0.5) * 0.5
    y_bottom = np.floor((np.min(T[:-1] + Y_dyn) - 0.40) / 0.5) * 0.5
    y_arrow_top = y_top + 0.24

    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_arrow_top + 0.08)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    x_grid = np.r_[np.arange(1980, 2024, 5), 2023]
    y_grid = np.arange(y_axis, y_top + 0.001, 0.5)
    draw_manual_grid(ax, x_grid, y_grid, 1980, 2023, y_axis, y_top)

    ax.plot(
        [split_year, split_year],
        [y_axis, y_top],
        color=YEAR_LINE,
        lw=1.2,
        ls=(0, (5, 4)),
        zorder=1,
    )

    ax.annotate(
        '',
        xy=(2023.75, y_axis),
        xytext=(1980.0, y_axis),
        arrowprops={'arrowstyle': '-|>', 'lw': 1.4, 'color': BLACK, 'shrinkA': 0, 'shrinkB': 0},
        annotation_clip=False,
        zorder=30,
    )
    ax.annotate(
        '',
        xy=(1980.0, y_arrow_top),
        xytext=(1980.0, y_axis),
        arrowprops={'arrowstyle': '-|>', 'lw': 1.4, 'color': BLACK, 'shrinkA': 0, 'shrinkB': 0},
        annotation_clip=False,
        zorder=30,
    )

    tick_years = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2023]

    tick_len_points = 3
    tick_label_offset_points = 6

    for xx in tick_years:
        draw_tick(ax, xx, y_axis, length_points=tick_len_points, lw=0.9)

        ax.annotate(
            str(xx),
            xy=(xx, y_axis),
            xytext=(0, -tick_label_offset_points),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=11,
            color=BLACK,
            annotation_clip=False,
            zorder=41,
        )

    ax.plot(years, T, color=BLACK, lw=0.9, alpha=0.55, zorder=2)

    # Process arrows for every yearly interval.
    # Color swap kept: physical is blue, dynamical is red.
    for k in range(len(years) - 1):
        start = (years[k], T[k])

        end_phys = (years[k + 1], T[k] + Y_phys[k])
        end_dyn = (years[k + 1], T[k] + Y_dyn[k])
        end_tot = (years[k + 1], T[k + 1])

        draw_arrow(ax, start, end_phys, BLUE, lw=1.05, ms=8, zorder=5)
        draw_arrow(ax, start, end_dyn, RED, lw=1.05, ms=8, zorder=5)
        draw_arrow(ax, start, end_tot, BLACK, lw=1.25, ms=8, zorder=8)

    ax.scatter(years, T, s=20, color=BLACK, alpha=0.55, zorder=32)

    ax.text(1990.5, y_top + 0.10, 'Period 1', ha='center', va='top', fontsize=29)
    ax.text(2012.5, y_top + 0.10, 'Period 2', ha='center', va='top', fontsize=29)
    ax.text(2001.5, y_bottom - 0.7, 'Year', ha='center', va='bottom', fontsize=22)
    ax.text(1978.7, 0.5*(y_bottom + y_top), 'Temperature', ha='center', va='center', rotation=90, fontsize=22)

    fig.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    make_sketch()