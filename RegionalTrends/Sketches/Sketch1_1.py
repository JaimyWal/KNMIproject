
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RED = '#c62828'
BLUE = '#1565c0'
BLACK = '#111111'
GRID = '0.82'
DAY_LINE = '0.48'
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
    gap_points=-0.8,
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
    offset_pixels = offset_points * dpi / 72.0
    gap_pixels = gap_points * dpi / 72.0

    center_disp = mid + offset_pixels * n

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
    x = np.arange(0, 49, 3)

    T = np.array([
        0.60, 1.00, 1.50, 2.35, 3.35, 3.85, 3.05, 2.10, 1.25,
        1.55, 2.05, 2.65, 3.25, 3.75, 3.25, 2.45, 1.75,
    ]) + 0.55

    base_cooling = np.array([
        0.45, 0.40, 0.35, 0.30, 0.35, 1.05, 1.20, 1.15,
        0.50, 0.45, 0.45, 0.40, 0.40, 0.80, 1.05, 0.95,
    ])

    A_phys = -1.35 * base_cooling
    A_tot = T[1:] - T[:-1]
    A_dyn = A_tot - A_phys

    assert np.all(A_dyn > 0.0)
    assert np.all(A_phys < 0.0)
    np.testing.assert_allclose(A_dyn + A_phys, A_tot, rtol=0.0, atol=1.0e-12)

    return x, T, A_dyn, A_phys, A_tot


def label_parts(kind, x0):
    day_offset = int(x0 // 24)
    day = 'd' if day_offset == 0 else rf'd\!+\!{day_offset}'
    h = int((x0 % 24) // 3)
    main = rf'$A^{{\mathrm{{{kind}}}}}$'
    coord = rf'$({day},{h})$'
    return main, coord


def draw_manual_grid(ax, x_grid, y_grid, y_axis, y_top):
    for xx in x_grid:
        ax.plot([xx, xx], [y_axis, y_top], color=GRID, lw=0.8, zorder=0)
    for yy in y_grid:
        ax.plot([0, 48], [yy, yy], color=GRID, lw=0.8, zorder=0)


def make_sketch(output_prefix=None):
    if output_prefix is None:
        output_prefix = OUTPUT_DIR / Path(__file__).stem
    else:
        output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    x, T, A_dyn, A_phys, A_tot = build_data()

    max_dyn_end = np.max(T[:-1] + A_dyn)
    y_axis = 0.0
    y_top = np.ceil((max_dyn_end + 0.45) / 0.5) * 0.5
    y_arrow_top = y_top + 0.22

    fig, ax = plt.subplots(figsize=(11.2, 4.9))
    ax.set_xlim(-1.0, 49.4)
    ax.set_ylim(-0.62, y_arrow_top + 0.10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    y_grid = np.arange(y_axis, y_top + 0.001, 0.5)
    draw_manual_grid(ax, x, y_grid, y_axis, y_top)

    for xx in [24, 48]:
        ax.plot([xx, xx], [y_axis, y_top], color=DAY_LINE, lw=1.2, ls=(0, (5, 4)), zorder=2)

    ax.annotate(
        '',
        xy=(49.1, y_axis),
        xytext=(0.0, y_axis),
        arrowprops={'arrowstyle': '-|>', 'lw': 1.4, 'color': BLACK, 'shrinkA': 0, 'shrinkB': 0},
        annotation_clip=False,
        zorder=30,
    )
    ax.annotate(
        '',
        xy=(0.0, y_arrow_top),
        xytext=(0.0, y_axis),
        arrowprops={'arrowstyle': '-|>', 'lw': 1.4, 'color': BLACK, 'shrinkA': 0, 'shrinkB': 0},
        annotation_clip=False,
        zorder=30,
    )

    for xx in x:
        ax.plot([xx, xx], [y_axis, y_axis - 0.045], color=BLACK, lw=0.8, zorder=31)
        ax.text(xx, y_axis - 0.14, str(int(xx % 24)), ha='center', va='top', fontsize=11, color=BLACK, zorder=31)

    fig.canvas.draw()

    for i in range(len(A_tot)):
        x0 = x[i]
        x1 = x[i + 1]
        y0 = T[i]

        start = (x0, y0)
        end_dyn = (x1, y0 + A_dyn[i])
        end_phys = (x1, y0 + A_phys[i])
        end_tot = (x1, y0 + A_tot[i])

        draw_arrow(ax, start, end_dyn, RED, lw=1.05, ms=8, zorder=5)
        draw_arrow(ax, start, end_phys, BLUE, lw=1.05, ms=8, zorder=5)
        draw_arrow(ax, start, end_tot, BLACK, lw=1.25, ms=8, zorder=8)

        main, coord = label_parts('dyn', x0)
        place_composite_label_perpendicular(
            ax, start, end_dyn, main, coord, RED,
            side='above', frac=0.5, base_offset_points=4.6, extra_for_shallow_points=1.3,
            fontsize_main=7.0, fontsize_coord=5.6, gap_points=-0.8, zorder=22
        )

        main, coord = label_parts('phys', x0)

        place_composite_label_perpendicular(
            ax, start, end_phys, main, coord, BLUE,
            side='below', frac=0.5, base_offset_points=4.85, extra_for_shallow_points=1.3,
            fontsize_main=7.0, fontsize_coord=5.6, gap_points=-0.8, zorder=22
        )

        main, coord = label_parts('tot', x0)

        tot_side = 'below' if A_tot[i] >= 0 else 'above'
        tot_base_offset = 4.95 if tot_side == 'below' else 4.2

        place_composite_label_perpendicular(
            ax, start, end_tot, main, coord, BLACK,
            side=tot_side, frac=0.5, base_offset_points=tot_base_offset, extra_for_shallow_points=1.1,
            fontsize_main=6.8, fontsize_coord=5.4, gap_points=-0.8, zorder=21
        )

    ax.scatter(x, T, s=20, color=BLACK, alpha=0.55, zorder=32)
    ax.text(12, y_top+0.1, r'Day $d$', ha='center', va='top', fontsize=30)
    ax.text(36, y_top+0.1, r'Day $d\!+\!1$', ha='center', va='top', fontsize=30)
    ax.text(24, -0.50, 'Hour of day', ha='center', va='top', fontsize=20)
    ax.text(-1.5, 0.5 * y_top, 'Temperature', ha='center', va='center', rotation=90, fontsize=20)

    fig.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    make_sketch()
