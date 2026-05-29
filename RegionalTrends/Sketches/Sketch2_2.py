from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RED = '#c62828'
BLUE = '#1565c0'
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


def measure_text_width(ax, text, fontsize):
    renderer = ax.figure.canvas.get_renderer()
    temp = ax.text(0, 0, text, fontsize=fontsize, alpha=0.0)
    bbox = temp.get_window_extent(renderer=renderer)
    temp.remove()
    return bbox.width


def place_composite_label_perpendicular(
    ax,
    start,
    end,
    main_text,
    coord_text,
    color,
    side='above',
    frac=0.5,
    base_offset_points=4.6,
    extra_for_shallow_points=1.3,
    fontsize_main=7.0,
    fontsize_coord=5.8,
    gap_points=1.0,
    zorder=20,
):
    p0 = np.array(ax.transData.transform(start), dtype=float)
    p1 = np.array(ax.transData.transform(end), dtype=float)
    mid = p0 + frac * (p1 - p0)

    v = p1 - p0
    length = np.hypot(v[0], v[1])
    if length == 0:
        return

    u = v / length
    n = np.array([-v[1], v[0]]) / length
    if side == 'below':
        n = -n

    steepness = abs(v[1]) / length
    offset_points = base_offset_points + extra_for_shallow_points * (1.0 - steepness)

    dpi = ax.figure.dpi
    center_disp = mid + offset_points * dpi / 72.0 * n
    gap_pixels = gap_points * dpi / 72.0

    w1 = measure_text_width(ax, main_text, fontsize_main)
    w2 = measure_text_width(ax, coord_text, fontsize_coord)
    total_w = w1 + gap_pixels + w2

    pos1_disp = center_disp - 0.5 * total_w * u + 0.5 * w1 * u
    pos2_disp = center_disp + 0.5 * total_w * u - 0.5 * w2 * u

    pos1 = ax.transData.inverted().transform(pos1_disp)
    pos2 = ax.transData.inverted().transform(pos2_disp)

    angle = np.degrees(np.arctan2(v[1], v[0]))

    ax.text(
        pos1[0],
        pos1[1],
        main_text,
        color=color,
        fontsize=fontsize_main,
        rotation=angle,
        rotation_mode='anchor',
        ha='center',
        va='center',
        zorder=zorder,
    )
    ax.text(
        pos2[0],
        pos2[1],
        coord_text,
        color=color,
        fontsize=fontsize_coord,
        rotation=angle,
        rotation_mode='anchor',
        ha='center',
        va='center',
        zorder=zorder,
    )


def build_data():
    x = np.array([1.0, 2.0])
    T = np.array([1.45, 2.55])

    Y_tot = T[1] - T[0]
    Y_dyn = -1.40
    Y_phys = Y_tot - Y_dyn

    assert Y_phys > 0.0
    assert Y_dyn < 0.0
    np.testing.assert_allclose(Y_phys + Y_dyn, Y_tot, rtol=0.0, atol=1.0e-12)

    return x, T, Y_phys, Y_dyn, Y_tot


def label_parts(kind, year):
    main = rf'$Y^{{\mathrm{{{kind}}}}}$'
    coord = rf'$({year})$'
    return main, coord


def place_state_label(ax, x, y, text, dx_points=0.0, dy_points=0.0, fontsize=12.5):
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


def draw_manual_grid(ax, x_grid, y_grid, y_axis, y_top):
    for xx in x_grid:
        ax.plot([xx, xx], [y_axis, y_top], color=GRID, lw=0.8, zorder=0)
    for yy in y_grid:
        ax.plot([0.5, 2.5], [yy, yy], color=GRID, lw=0.8, zorder=0)


def make_sketch(output_prefix=None):
    if output_prefix is None:
        output_prefix = OUTPUT_DIR / 'Sketch2_2'
    else:
        output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    x, T, Y_phys, Y_dyn, Y_tot = build_data()

    y_axis = 0.0
    y_top = np.ceil((max(T[0] + Y_phys, np.max(T)) + 0.45) / 0.5) * 0.5
    y_arrow_top = y_top + 0.22

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.set_xlim(0.45, 2.605)
    ax.set_ylim(-0.62, y_arrow_top + 0.10)

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

    for xx, label in zip(x, [r'$y$', r'$y\!+\!1$']):
        ax.plot([xx, xx], [y_axis, y_axis - 0.045], color=BLACK, lw=0.8, zorder=31)
        ax.text(xx, y_axis - 0.14, label, ha='center', va='top', fontsize=16, color=BLACK, zorder=31)

    fig.canvas.draw()

    start = (x[0], T[0])
    end_phys = (x[1], T[0] + Y_phys)
    end_dyn = (x[1], T[0] + Y_dyn)
    end_tot = (x[1], T[1])

    draw_arrow(ax, start, end_phys, RED, lw=1.05, ms=8, zorder=5)
    draw_arrow(ax, start, end_dyn, BLUE, lw=1.05, ms=8, zorder=5)
    draw_arrow(ax, start, end_tot, BLACK, lw=1.25, ms=8, zorder=8)

    main, coord = label_parts('phys', 'y')
    place_composite_label_perpendicular(
        ax, start, end_phys, main, coord, RED,
        side='above', frac=0.50, base_offset_points=7.2, extra_for_shallow_points=1.3,
        fontsize_main=18, fontsize_coord=15.8, gap_points=0.10, zorder=22
    )

    main, coord = label_parts('dyn', 'y')
    place_composite_label_perpendicular(
        ax, start, end_dyn, main, coord, BLUE,
        side='below', frac=0.50, base_offset_points=11.2, extra_for_shallow_points=1.3,
        fontsize_main=18, fontsize_coord=15.8, gap_points=0.10, zorder=22
    )

    main, coord = label_parts('tot', 'y')
    place_composite_label_perpendicular(
        ax, start, end_tot, main, coord, BLACK,
        side='below', frac=0.50, base_offset_points=10.8, extra_for_shallow_points=1.1,
        fontsize_main=18, fontsize_coord=15.8, gap_points=0.10, zorder=21
    )

    ax.scatter(x, T, s=28, color=BLACK, alpha=0.55, zorder=32)

    place_state_label(ax, x[0], T[0], r'$\langle T(y)\rangle$', dx_points=-26, dy_points=0, fontsize=15)
    place_state_label(ax, x[1], T[1], r'$\langle T(y\!+\!1)\rangle$', dx_points=0, dy_points=10, fontsize=15)

    ax.text(1.0, y_top + 0.1, r'Year $y$', ha='center', va='top', fontsize=28)
    ax.text(2.0, y_top + 0.1, r'Year $y\!+\!1$', ha='center', va='top', fontsize=28)
    ax.text(1.5, -0.45, 'Year', ha='center', va='top', fontsize=20)
    ax.text(0.36, 0.5 * y_top, 'Temperature', ha='center', va='center', rotation=90, fontsize=20)

    fig.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    make_sketch()