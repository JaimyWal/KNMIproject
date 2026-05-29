import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %% Setup
T = 1.0
n = 1200
t = np.linspace(0, T, n)
dt = t[1] - t[0]

A = 1.0
pulse_width = 0.12
early_start = 0.12
late_start = 0.72

red = '#d7191c'
black = '#111111'
grid_color = '0.82'


def tendency_pulse(t, start, width, amplitude):
    return amplitude * ((t >= start) & (t <= start + width)).astype(float)


def temperature_from_tendency(tendency, T0=0.10):
    return T0 + np.r_[0, np.cumsum(0.5 * (tendency[1:] + tendency[:-1]) * dt)]


def hide_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_grid(ax, x0, x1, y0, y1, nx=9, ny=6):
    for xx in np.linspace(x0, x1, nx):
        ax.plot([xx, xx], [y0, y1], color=grid_color, lw=0.8, zorder=0)
    for yy in np.linspace(y0, y1, ny):
        ax.plot([x0, x1], [yy, yy], color=grid_color, lw=0.8, zorder=0)


def draw_axes(ax_left, ax_right, x0, x1, y0_left, y1_left, y0_right, y1_right):
    kw_left = {
        'arrowstyle': '-|>',
        'lw': 3,
        'color': black,
        'shrinkA': 0,
        'shrinkB': 0,
        'mutation_scale': 16,
    }
    kw_right = {
        'arrowstyle': '-|>',
        'lw': 3,
        'color': red,
        'shrinkA': 0,
        'shrinkB': 0,
        'mutation_scale': 16,
    }

    ax_left.plot([x0, x1], [y0_left, y0_left], color=black, lw=3, zorder=10, clip_on=False)

    ax_left.annotate(
        '',
        xy=(x0, y1_left),
        xytext=(x0, y0_left),
        arrowprops=kw_left,
        annotation_clip=False,
        zorder=10,
    )

    ax_right.annotate(
        '',
        xy=(x1, y1_right),
        xytext=(x1, y0_right),
        arrowprops=kw_right,
        annotation_clip=False,
        zorder=10,
    )


def place_labels(ax_left, ax_right, tendency_x, temperature_x, time_y=0.035,
                 time_fs=38, label_fs=38):
    ax_left.text(
        0.5,
        time_y,
        'Time',
        transform=ax_left.transAxes,
        ha='center',
        va='top',
        fontsize=time_fs,
    )
    ax_left.text(
        tendency_x,
        0.5,
        'Tendency',
        transform=ax_left.transAxes,
        ha='center',
        va='center',
        rotation=90,
        fontsize=label_fs,
        color=black,
        clip_on=False,
    )
    ax_right.text(
        temperature_x,
        0.5,
        'Temperature',
        transform=ax_right.transAxes,
        ha='center',
        va='center',
        rotation=270,
        fontsize=label_fs,
        color=red,
        clip_on=False,
    )


def draw_mean_line(ax, x0, x1, y, color):
    ax.plot(
        [x0, x1],
        [y, y],
        color=color,
        lw=3,
        ls='--',
        dashes=(4, 2),
        alpha=0.8,
        zorder=3,
        clip_on=True,
        dash_capstyle='butt',
    )


def style_panel(ax_left, ax_right, x0, x1, y0_left, y1_left, y0_right, y1_right,
                tendency_x, temperature_x):
    hide_axes(ax_left)
    hide_axes(ax_right)

    ax_left.set_xlim(x0 - 0.03, x1 + 0.03)
    ax_right.set_xlim(x0 - 0.03, x1 + 0.03)

    ax_left.set_ylim(
        y0_left - 0.08 * (y1_left - y0_left),
        y1_left + 0.08 * (y1_left - y0_left),
    )
    ax_right.set_ylim(
        y0_right - 0.08 * (y1_right - y0_right),
        y1_right + 0.08 * (y1_right - y0_right),
    )

    draw_grid(ax_left, x0, x1, y0_left, y1_left)
    draw_axes(ax_left, ax_right, x0, x1, y0_left, y1_left, y0_right, y1_right)
    place_labels(ax_left, ax_right, tendency_x, temperature_x)


# %% Data
tend_early = tendency_pulse(t, early_start, pulse_width, A)
tend_late = tendency_pulse(t, late_start, pulse_width, A)

temp_early = temperature_from_tendency(tend_early)
temp_late = temperature_from_tendency(tend_late)

mean_tend_early = tend_early.mean()
mean_tend_late = tend_late.mean()
mean_temp_early = temp_early.mean()
mean_temp_late = temp_late.mean()

# %% Limits
x0, x1 = 0.0, T
tend_y0, tend_y1 = -0.15, 1.25
temp_y0 = min(temp_early.min(), temp_late.min()) - 0.03
temp_y1 = max(temp_early.max(), temp_late.max()) + 0.05

# %% Plot
fig, axs = plt.subplots(1, 2, figsize=(14.0, 5.8))
fig.subplots_adjust(left=0.075, right=0.925, bottom=0.18, top=0.92, wspace=0.25)

tendencies = [tend_early, tend_late]
temperatures = [temp_early, temp_late]
mean_tendencies = [mean_tend_early, mean_tend_late]
mean_temperatures = [mean_temp_early, mean_temp_late]

label_positions = [
    (-0.04, 1.040),
    (-0.04, 1.040),
]

for ax, tend, temp, mean_tend, mean_temp, (tendency_x, temperature_x) in zip(
    axs,
    tendencies,
    temperatures,
    mean_tendencies,
    mean_temperatures,
    label_positions,
):
    ax2 = ax.twinx()

    style_panel(
        ax,
        ax2,
        x0,
        x1,
        tend_y0,
        tend_y1,
        temp_y0,
        temp_y1,
        tendency_x=tendency_x,
        temperature_x=temperature_x,
    )

    ax.step(t, tend, where='post', color=black, lw=2.6, zorder=4)
    draw_mean_line(ax, x0, x1, mean_tend, black)

    ax2.plot(t, temp, color=red, lw=2.7, zorder=5)
    draw_mean_line(ax2, x0, x1, mean_temp, red)

plt.show()

# %% Save figure
output_dir = Path('/nobackup/users/walj/Figure_sketches')
output_dir.mkdir(parents=True, exist_ok=True)

output_name = 'Sketch_tends_average'

fig.savefig(output_dir / f'{output_name}.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / f'{output_name}.pdf', bbox_inches='tight')

plt.show()