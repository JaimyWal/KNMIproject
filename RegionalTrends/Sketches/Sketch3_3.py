from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RED = '#c62828'    # dynamical cooling
BLUE = '#1565c0'   # physical warming
BLACK = '#111111'
GRID = '0.82'
YEAR_LINE = '0.48'
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


def place_label_perpendicular(
    ax,
    start,
    end,
    text,
    color,
    side='above',
    frac=0.5,
    offset_points=7.0,
    extra_for_shallow_points=1.2,
    fontsize=18,
    zorder=20,
):
    p0 = np.array(ax.transData.transform(start), dtype=float)
    p1 = np.array(ax.transData.transform(end), dtype=float)
    mid = p0 + frac * (p1 - p0)

    v = p1 - p0
    length = np.hypot(v[0], v[1])
    if length == 0:
        return

    n = np.array([-v[1], v[0]]) / length
    if side == 'below':
        n = -n

    steepness = abs(v[1]) / length
    offset = offset_points + extra_for_shallow_points * (1.0 - steepness)

    dpi = ax.figure.dpi
    pos_disp = mid + offset * dpi / 72.0 * n
    pos = ax.transData.inverted().transform(pos_disp)

    angle = np.degrees(np.arctan2(v[1], v[0]))

    ax.text(
        pos[0],
        pos[1],
        text,
        color=color,
        fontsize=fontsize,
        rotation=angle,
        rotation_mode='anchor',
        ha='center',
        va='center',
        zorder=zorder,
    )


def place_state_label(ax, x, y, text, dx_points=0.0, dy_points=0.0, fontsize=14):
    dpi = ax.figure.dpi
    point_disp = np.array(ax.transData.transform((x, y)), dtype=float)
    pos_disp = point_disp + np.array([dx_points, dy_points]) * dpi / 72.0
    pos = ax.transData.inverted().transform(pos_disp)

    ax.text(
        pos[0],
        pos[1],
        text,
        color=BLACK,
        fontsize=fontsize,
        ha='center',
        va='center',
        zorder=33,
    )


def build_data():
    x = np.array([1.0, 2.0])

    # Same initial dot as Sketch3_2
    T_ref = np.array([1.45, 1.45])

    mu_ref_tot = T_ref[1] - T_ref[0]
    mu_ref_dyn = -1.30
    mu_ref_phys = mu_ref_tot - mu_ref_dyn

    assert mu_ref_tot == 0.0
    assert mu_ref_phys > 0.0
    assert mu_ref_dyn < 0.0
    np.testing.assert_allclose(mu_ref_phys + mu_ref_dyn, mu_ref_tot, rtol=0.0, atol=1.0e-12)

    return x, T_ref, mu_ref_phys, mu_ref_dyn, mu_ref_tot


def draw_manual_grid(ax, x_grid, y_grid, y_axis, y_top):
    for xx in x_grid:
        ax.plot([xx, xx], [y_axis, y_top], color=GRID, lw=0.8, zorder=0)
    for yy in y_grid:
        ax.plot([0.5, 2.5], [yy, yy], color=GRID, lw=0.8, zorder=0)


def make_sketch(output_prefix=None):
    if output_prefix is None:
        output_prefix = OUTPUT_DIR / 'Sketch3_3'
    else:
        output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    x, T_ref, mu_ref_phys, mu_ref_dyn, mu_ref_tot = build_data()

    y_axis = 0.0
    y_top = np.ceil((max(T_ref[0] + mu_ref_phys, np.max(T_ref)) + 0.45) / 0.5) * 0.5
    y_bottom = np.floor((min(T_ref[0] + mu_ref_dyn, np.min(T_ref)) - 0.35) / 0.5) * 0.5
    y_arrow_top = y_top + 0.22

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.set_xlim(0.45, 2.605)
    ax.set_ylim(y_bottom, y_arrow_top + 0.10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    x_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    y_grid = np.arange(y_axis, y_top + 0.001, 0.5)
    draw_manual_grid(ax, x_grid, y_grid, y_axis, y_top)

    for xx in [1.5, 2.5]:
        ax.plot([xx, xx], [y_axis, y_top], color=YEAR_LINE, lw=1.2, ls=(0, (5, 4)), zorder=2)

    ax.annotate(
        '',
        xy=(2.58, y_axis),
        xytext=(0.5, y_axis),
        arrowprops={'arrowstyle': '-|>', 'lw': 1.4, 'color': BLACK, 'shrinkA': 0, 'shrinkB': 0},
        annotation_clip=False,
        zorder=30,
    )
    ax.annotate(
        '',
        xy=(0.5, y_arrow_top),
        xytext=(0.5, y_axis),
        arrowprops={'arrowstyle': '-|>', 'lw': 1.4, 'color': BLACK, 'shrinkA': 0, 'shrinkB': 0},
        annotation_clip=False,
        zorder=30,
    )

    tick_len = 0.045
    for xx, label in zip(x, [r'$P_1$', r'$P_2$']):
        ax.plot([xx, xx], [y_axis, y_axis - tick_len], color=BLACK, lw=0.8, zorder=31)
        ax.text(xx, y_axis - 0.14, label, ha='center', va='top', fontsize=16, color=BLACK, zorder=31)

    fig.canvas.draw()

    start = (x[0], T_ref[0])
    end_phys = (x[1], T_ref[0] + mu_ref_phys)
    end_dyn = (x[1], T_ref[0] + mu_ref_dyn)
    end_tot = (x[1], T_ref[1])

    draw_arrow(ax, start, end_phys, BLUE, lw=1.05, ms=8, zorder=5)
    draw_arrow(ax, start, end_dyn, RED, lw=1.05, ms=8, zorder=5)
    draw_arrow(ax, start, end_tot, BLACK, lw=1.25, ms=8, zorder=8)

    place_label_perpendicular(
        ax,
        start,
        end_phys,
        r'$\mu_{\mathrm{ref}}^{\mathrm{phys}}$',
        BLUE,
        side='above',
        frac=0.50,
        offset_points=12.0,
        fontsize=18,
        zorder=22,
    )

    place_label_perpendicular(
        ax,
        start,
        end_dyn,
        r'$\mu_{\mathrm{ref}}^{\mathrm{dyn}}$',
        RED,
        side='below',
        frac=0.50,
        offset_points=12.0,
        fontsize=18,
        zorder=22,
    )

    place_label_perpendicular(
        ax,
        start,
        end_tot,
        r'$\mu_{\mathrm{ref}}=\mu_{\mathrm{ref}}^{\mathrm{tot}}$',
        BLACK,
        side='below',
        frac=0.62,
        offset_points=10.0,
        fontsize=17,
        zorder=21,
    )

    ax.scatter(x, T_ref, s=28, color=BLACK, alpha=0.55, zorder=32)

    place_state_label(ax, x[0], T_ref[0], r'$\langle T\rangle_{P_1}$', dx_points=-20, dy_points=0, fontsize=15)
    place_state_label(ax, x[1], T_ref[1], r'$\langle T\rangle_{P_2}$', dx_points=26, dy_points=0, fontsize=14)

    ax.text(1.0, y_top + 0.10, 'Period 1', ha='center', va='top', fontsize=28)
    ax.text(2.0, y_top + 0.10, 'Period 2', ha='center', va='top', fontsize=28)
    ax.text(1.5, -0.45, 'Period', ha='center', va='top', fontsize=20)
    ax.text(0.36, 0.5 * (y_bottom + y_top), 'Temperature', ha='center', va='center', rotation=90, fontsize=20)

    fig.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    make_sketch()