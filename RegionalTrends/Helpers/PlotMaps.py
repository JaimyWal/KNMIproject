import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as ticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import matplotlib.path as mpath
from shapely.geometry import Polygon
from shapely.ops import unary_union
import os
from importlib import reload

from RegionalTrends.Helpers import ProcessNetCDF
reload(ProcessNetCDF)
from RegionalTrends.Helpers.ProcessNetCDF import rect_sel


def cbar_extension(datasets, crange):
    dmin = min(float(np.nanmin(d)) for d in datasets)
    dmax = max(float(np.nanmax(d)) for d in datasets)

    vmin, vmax = crange
    if dmin < vmin and dmax > vmax:
        return 'both'
    elif dmin < vmin and dmax <= vmax:
        return 'min'
    elif dmin >= vmin and dmax > vmax:
        return 'max'
    else:
        return 'neither'


def plot_map(fig, ax, data, lon, lat,
             crange=None,
             label=None,
             cmap='viridis',
             interpolation=False,
             extent=None,
             cbar_orientation='vertical',
             c_ticks=10,
             c_ticks_num=True,
             x_ticks=5,
             show_x_ticks=True,
             show_x_labels=True,
             x_ticks_num=True,
             show_y_ticks=True,
             show_y_labels=True,
             y_labels_side='left',
             y_ticks=5,
             y_ticks_num=True,
             extreme_colors=False,
             clabel_size=32,
             ctick_size=28,
             tick_size=28,
             proj=ccrs.PlateCarree(),
             rotated_grid=False,
             save_name=None,
             add_colorbar=True,
             title=None,
             title_size=36,
             show_plot=True,
             lats_area=None,
             lons_area=None,
             proj_area=ccrs.PlateCarree(),
             area_facecolor='none',
             area_edgecolor='xkcd:black',
             area_linewidth=2.0,
             area_linestyle='-',
             area_alpha=1,
             mask_area=None,
             lat_b_area=None,
             lon_b_area=None,
             area_grid_facecolor='none',
             area_grid_edgecolor='xkcd:crimson',
             area_grid_linewidth=2.0,
             area_grid_linestyle='-',
             area_grid_alpha=1):

    plt.rcParams['axes.unicode_minus'] = False # Nicer minus signs

    if isinstance(cmap, str): # If provided colormap is a string, convert to object
        cmap = plt.get_cmap(cmap).copy()

    # Create 2D lon/lat arrays if needed
    if np.ndim(lon) == 1 and np.ndim(lat) == 1:
        Lon, Lat = np.meshgrid(lon, lat)
    else:
        Lon, Lat = lon, lat

    # Potentially use gouraud interpolation between grid cells
    shading = 'gouraud' if interpolation else 'auto'

    # Determine color range and extension
    if crange is not None:
        extension = cbar_extension([data], crange)
    else:
        crange = (np.nanmin(data), np.nanmax(data))
        extension = 'neither'

    # Handle extreme colors in colormap
    if extreme_colors is True:
        base_cmap = plt.get_cmap(cmap)
        basemap = base_cmap(np.linspace(0, 1, base_cmap.N))
        cmap = ListedColormap(basemap[1:-1])
        cmap.set_over(basemap[-1])
        cmap.set_under(basemap[0])
    elif isinstance(extreme_colors, (list, tuple, np.ndarray)):
        under_color, over_color = extreme_colors
        if under_color is not None:
            cmap.set_under(under_color)
        if over_color is not None:
            cmap.set_over(over_color)

    # Plotting
    ax.set_aspect('auto')

    ax.coastlines(resolution='10m', linewidth=1.5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', lw=1.5)
    ax.add_feature(cfeature.OCEAN, facecolor='#edf0f5')

    # Give plot boundaries (with small padding for ticklabels)
    eps = 1e-6
    if extent is not None:
        lon_min = extent[0] - eps
        lon_max = extent[1] + eps
        lat_min = extent[2] - eps
        lat_max = extent[3] + eps
    else:
        lat1d = np.sort(np.unique(np.asarray(lat)[np.isfinite(lat)]))
        lon1d = np.sort(np.unique(np.asarray(lon)[np.isfinite(lon)]))

        dlat = np.diff(lat1d).mean() if lat1d.size > 1 else 1.0
        dlon = np.diff(lon1d).mean() if lon1d.size > 1 else 1.0

        lat_min = lat1d[0]  - 0.5 * dlat - eps
        lat_max = lat1d[-1] + 0.5 * dlat + eps
        lon_min = lon1d[0]  - 0.5 * dlon - eps
        lon_max = lon1d[-1] + 0.5 * dlon + eps
        

    # gridlines: never let Cartopy draw labels, we do that manually
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        x_inline=False,
        y_inline=False,
        color='xkcd:black',
        alpha=0.4
    )

    xticks = None
    yticks = None

    if show_x_ticks:
        if x_ticks_num:
            xloc = ticker.MaxNLocator(nbins=x_ticks)
            xticks = xloc.tick_values(lon_min, lon_max)
        else:
            xloc = ticker.MultipleLocator(x_ticks)
            xticks = xloc.tick_values(lon_min, lon_max)
        # keep ticks strictly inside domain to avoid corner labels
        xticks = xticks[(xticks > lon_min) & (xticks < lon_max)]
        gl.xlocator = ticker.FixedLocator(xticks)
    else:
        gl.xlocator = ticker.NullLocator()

    if show_y_ticks:
        if y_ticks_num:
            yloc = ticker.MaxNLocator(nbins=y_ticks)
            yticks = yloc.tick_values(lat_min, lat_max)
        else:
            yloc = ticker.MultipleLocator(y_ticks)
            yticks = yloc.tick_values(lat_min, lat_max)
        yticks = yticks[(yticks > lat_min) & (yticks < lat_max)]
        gl.ylocator = ticker.FixedLocator(yticks)
    else:
        gl.ylocator = ticker.NullLocator()

    # helper formatters for manual labels
    def _fmt_lon(x):
        v = int(round(x))
        if v < 0:
            return f'{abs(v)}°W'
        if v > 0:
            return f'{v}°E'
        return '0°'

    def _fmt_lat(y):
        v = int(round(y))
        if v < 0:
            return f'{abs(v)}°S'
        if v > 0:
            return f'{v}°N'
        return '0°'

    # Rotated projections: manual bottom and left labels only, rotated along gridlines
    if rotated_grid:

        if extent is not None:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

            n = 400
            lons_bottom = np.linspace(lon_min, lon_max, n)
            lats_bottom = np.full_like(lons_bottom, lat_min)
            lons_right = np.full(n, lon_max)
            lats_right = np.linspace(lat_min, lat_max, n)
            lons_top = np.linspace(lon_max, lon_min, n)
            lats_top = np.full_like(lons_top, lat_max)
            lons_left = np.full(n, lon_min)
            lats_left = np.linspace(lat_max, lat_min, n)

            lons_all = np.concatenate([lons_bottom, lons_right, lons_top, lons_left])
            lats_all = np.concatenate([lats_bottom, lats_right, lats_top, lats_left])

            pts_all = proj.transform_points(ccrs.PlateCarree(), lons_all, lats_all)
            verts = pts_all[:, :2]
            boundary_path = mpath.Path(verts)
            ax.set_boundary(boundary_path)

        fig.canvas.draw() 

        # small step in degrees to estimate gridline orientation
        dlat = max((lat_max - lat_min) * 0.02, 0.1)
        dlon = max((lon_max - lon_min) * 0.02, 0.1)

        x_pad = 0.01*(lon_max - lon_min)
        y_pad = 0.02*(lat_max - lat_min)

        # bottom longitude labels, rotated along meridian then +90 degrees
        if show_x_labels and show_x_ticks and xticks is not None:
            for x in xticks:
                lons = np.array([x, x])
                lats = np.array([lat_min, lat_min + dlat])
                pts = proj.transform_points(ccrs.PlateCarree(), lons, lats)
                dx = pts[1, 0] - pts[0, 0]
                dy = pts[1, 1] - pts[0, 1]
                angle = np.degrees(np.arctan2(dy, dx))
                angle += 270 

                ax.text(
                    x, lat_min - y_pad,
                    _fmt_lon(x),
                    transform=ccrs.PlateCarree(),
                    fontsize=tick_size,
                    ha='center',
                    va='top',
                    rotation=angle,
                    rotation_mode='anchor',
                    clip_on=False
                )

        if show_y_labels and show_y_ticks and yticks is not None:
            for y in yticks:
                # Determine which side to place labels
                if y_labels_side == 'right':
                    lons = np.array([lon_max - dlon, lon_max])
                    lats = np.array([y, y])
                    pts = proj.transform_points(ccrs.PlateCarree(), lons, lats)
                    dx = pts[1, 0] - pts[0, 0]
                    dy = pts[1, 1] - pts[0, 1]
                    angle = np.degrees(np.arctan2(dy, dx))
                    
                    ax.text(
                        lon_max + x_pad, y,
                        _fmt_lat(y),
                        transform=ccrs.PlateCarree(),
                        fontsize=tick_size,
                        ha='left',
                        va='center',
                        rotation=angle,
                        rotation_mode='anchor',
                        clip_on=False
                    )
                else:  # left side (default)
                    lons = np.array([lon_min, lon_min + dlon])
                    lats = np.array([y, y])
                    pts = proj.transform_points(ccrs.PlateCarree(), lons, lats)
                    dx = pts[1, 0] - pts[0, 0]
                    dy = pts[1, 1] - pts[0, 1]
                    angle = np.degrees(np.arctan2(dy, dx))

                    ax.text(
                        lon_min - x_pad, y,
                        _fmt_lat(y),
                        transform=ccrs.PlateCarree(),
                        fontsize=tick_size,
                        ha='right',
                        va='center',
                        rotation=angle,
                        rotation_mode='anchor',
                        clip_on=False
                    )

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    else: 
        if isinstance(proj, ccrs.PlateCarree):
            # plain PlateCarree, normal axis ticks (not rotated)
            if show_x_ticks and xticks is not None:
                ax.set_xticks(xticks, crs=ccrs.PlateCarree())
            else:
                ax.set_xticks([])

            if show_y_ticks and yticks is not None:
                ax.set_yticks(yticks, crs=ccrs.PlateCarree())
            else:
                ax.set_yticks(yticks, crs=ccrs.PlateCarree())
                ax.tick_params(labelleft=False, labelright=False)

            if show_x_labels:
                ax.xaxis.set_major_formatter(
                    LongitudeFormatter(number_format='.0f', degree_symbol='°')
                )
            else:
                ax.tick_params(labelbottom=False)

            if show_y_labels:
                ax.yaxis.set_major_formatter(
                    LatitudeFormatter(number_format='.0f', degree_symbol='°')
                )
                if y_labels_side == 'right':
                    ax.tick_params(labelleft=False, labelright=True)
                    ax.yaxis.set_label_position('right')
                else:
                    ax.tick_params(labelleft=True, labelright=False)
            else:
                ax.tick_params(labelleft=False, labelright=False)

            ax.tick_params(labelsize=tick_size, length=10)

        else: # rotated projection, but not rotated grid 
            gl.xformatter = LongitudeFormatter(number_format='.0f', degree_symbol='°')
            gl.yformatter = LatitudeFormatter(number_format='.0f', degree_symbol='°')
            gl.xlabel_style = {'size': tick_size}
            gl.ylabel_style = {'size': tick_size}

            gl.bottom_labels = show_x_labels
            gl.top_labels = False
            if y_labels_side == 'right':
                gl.left_labels = False
                gl.right_labels = show_y_labels
            else:
                gl.left_labels = show_y_labels
                gl.right_labels = False

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        if extent is not None:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # data field
    mesh = ax.pcolormesh(
        Lon, Lat, data, transform=ccrs.PlateCarree(),
        cmap=cmap, shading=shading, rasterized=True
    )

    mesh.set_clim(crange)

    # colorbar
    cbar = None

    if add_colorbar:
        divider = make_axes_locatable(ax)
        if cbar_orientation == 'vertical':
            ax_cb = divider.append_axes('right', size='3.8%', pad=0.2, axes_class=plt.Axes)
            fig.add_axes(ax_cb)
            # ax_cb.set_in_layout(False) 
            cbar = plt.colorbar(mesh, cax=ax_cb, extend=extension, orientation='vertical')
            cbar.ax.tick_params(labelsize=ctick_size, direction='in',
                                length=8, left=True, right=True)
            if c_ticks_num:
                cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=c_ticks))
            else:
                cbar.ax.yaxis.set_major_locator(MultipleLocator(c_ticks))
        elif cbar_orientation == 'horizontal':
            ax_cb = divider.append_axes('bottom', size='6%', pad=0.1, axes_class=plt.Axes)
            fig.add_axes(ax_cb)
            ax_cb.set_in_layout(False) 
            cbar = plt.colorbar(
                mesh, cax=ax_cb, extend=extension,
                orientation='horizontal', location='bottom'
            )
            cbar.ax.tick_params(labelsize=ctick_size, direction='in',
                                length=8, top=True, bottom=True)
            if c_ticks_num:
                cbar.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=c_ticks))
            else:
                cbar.ax.xaxis.set_major_locator(MultipleLocator(c_ticks))
        
        if cbar is not None:
            cbar.set_label(label, fontsize=clabel_size, labelpad=10)

    # Selected area
    if mask_area is not None and lat_b_area is not None and lon_b_area is not None:
        polys = []
        for ii, jj in np.argwhere(mask_area):
            ring = [
                (lon_b_area[ii, jj],     lat_b_area[ii, jj]),
                (lon_b_area[ii, jj+1],   lat_b_area[ii, jj+1]),
                (lon_b_area[ii+1, jj+1], lat_b_area[ii+1, jj+1]),
                (lon_b_area[ii+1, jj],   lat_b_area[ii+1, jj]),
            ]
            poly = Polygon(ring)
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)

        if polys:
            merged = unary_union(polys)
            geoms = []
            if merged.geom_type == 'GeometryCollection':
                geoms = [g for g in merged.geoms if not g.is_empty]
            else:
                geoms = [merged]

            ax.add_geometries(
                geoms,
                crs=ccrs.PlateCarree(),
                facecolor=area_grid_facecolor,
                edgecolor=area_grid_edgecolor,
                alpha=area_grid_alpha,
                linestyle=area_grid_linestyle,
                linewidth=area_grid_linewidth,
                zorder=10
            )
        
    if isinstance(lats_area, (list, tuple)) and len(lats_area) == 2 and \
    isinstance(lons_area, (list, tuple)) and len(lons_area) == 2:
        
        if proj_area == ccrs.PlateCarree():
            y_min, y_max = sorted(lats_area)
            x_min, x_max = sorted(lons_area)
        else:
            y_min, y_max, x_min, x_max = rect_sel(lats_area, lons_area, proj_area)

        n_edge = int(5e4)

        x_bottom = np.linspace(x_min, x_max, n_edge)
        y_bottom = np.full_like(x_bottom, y_min)

        x_top = np.linspace(x_min, x_max, n_edge)
        y_top = np.full_like(x_top, y_max)

        y_left = np.linspace(y_min, y_max, n_edge)
        x_left = np.full_like(y_left, x_min)

        y_right = np.linspace(y_min, y_max, n_edge)
        x_right = np.full_like(y_right, x_max)

        x_area = np.concatenate([x_bottom, x_right, x_top[::-1], x_left[::-1]])
        y_area = np.concatenate([y_bottom, y_right, y_top[::-1], y_left[::-1]])

        ax.fill(
            x_area,
            y_area,
            transform=proj_area,
            facecolor=area_facecolor,
            edgecolor=area_edgecolor,
            linewidth=area_linewidth,
            linestyle=area_linestyle,
            alpha=area_alpha,
            zorder=15,
        )

    # Title and saving
    if title is not None:
        ax.set_title(title, fontsize=title_size, fontweight='bold')

    if save_name is not None:
        folder = '/usr/people/walj/figures/'
        save_path = os.path.join(folder, save_name + '.jpg')
        plt.savefig(save_path, 
                    format='jpg', 
                    bbox_inches='tight', 
                    dpi=800)

    if show_plot:
        plt.show()

    return mesh, cbar


def shared_colorbar(
    fig,
    axes,
    mesh,
    datasets,
    crange,
    label,
    orientation='horizontal',
    position='auto',
    c_ticks=10,
    c_ticks_num=True,
    tick_labelsize=24,
    labelsize=30,
    pad=0.02,
    thickness=0.03,
    label_pad=10,
    extendfrac=0.04,
    length_scale=1.0
):

    extension = cbar_extension(datasets, crange)

    # Calculate length adjustment to keep color range portion consistent
    # When extensions are present, we need to make the total colorbar longer
    # so that the actual color range part stays the same length
    if extendfrac != 'auto' and extendfrac is not None:
        ext_frac = float(extendfrac)
        if extension == 'both':
            # Two extensions: total length = color_range / (1 - 2*ext_frac)
            length_adjustment = 1.0 / (1.0 - 2.0 * ext_frac)
        elif extension in ('min', 'max'):
            # One extension: total length = color_range / (1 - ext_frac)
            length_adjustment = 1.0 / (1.0 - ext_frac)
        else:
            length_adjustment = 1.0
    else:
        ext_frac = 0.0
        length_adjustment = 1.0

    fig.canvas.draw()
    axes = np.atleast_1d(axes)
    positions = [ax.get_position() for ax in axes]

    if orientation == 'vertical':
        bottom = min(p.y0 for p in positions)
        top = max(p.y1 for p in positions)
        base_height = (top - bottom) * length_scale
        height = base_height * length_adjustment
        # Center the MAIN BODY (not including extensions)
        center = (top + bottom) / 2
        bottom = center - base_height / 2
        
        # Adjust position so main body is centered, extensions stick out
        if extension == 'min':
            # Extension at bottom, shift down so main body stays centered
            ext_size = height * ext_frac
            bottom = bottom - ext_size
        elif extension == 'max':
            # Extension at top, no shift needed (main body already at correct position)
            pass
        elif extension == 'both':
            # Extensions on both sides, shift down by one extension size
            ext_size = height * ext_frac
            bottom = bottom - ext_size

        # Determine left/right position
        if position == 'left':
            left = min(p.x0 for p in positions) - pad - thickness
        else:  # 'right' or 'auto'
            left = max(p.x1 for p in positions) + pad
        width = thickness

        ax_cb = fig.add_axes([left, bottom, width, height])
        ax_cb.set_in_layout(False)

        cbar = plt.colorbar(
            mesh,
            cax=ax_cb,
            orientation='vertical',
            extend=extension,
            extendfrac=extendfrac
        )

        cbar.ax.tick_params(
            labelsize=tick_labelsize,
            direction='in',
            length=8,
            left=True,
            right=True
        )
        
        # Disable minor ticks
        cbar.ax.minorticks_off()

        # Handle tick placement
        if c_ticks == 3:
            # Show only min, center, max
            cbar.set_ticks([crange[0], (crange[0] + crange[1]) / 2, crange[1]])
        elif c_ticks_num:
            cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=c_ticks))
        else:
            cbar.ax.yaxis.set_major_locator(MultipleLocator(c_ticks))
        
        # Position ticks and label on correct side for left placement
        if position == 'left':
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.set_label_position('left')

    else:
        left  = min(p.x0 for p in positions)
        right = max(p.x1 for p in positions)
        full_width = right - left
        base_width = full_width * length_scale
        width = base_width * length_adjustment
        # Center the MAIN BODY (not including extensions)
        center = (left + right) / 2
        left = center - base_width / 2
        
        # Adjust position so main body is centered, extensions stick out
        if extension == 'min':
            # Extension on left, shift left so main body stays centered
            ext_size = width * ext_frac
            left = left - ext_size
        elif extension == 'max':
            # Extension on right, no shift needed (main body already at correct position)
            pass
        elif extension == 'both':
            # Extensions on both sides, shift left by one extension size
            ext_size = width * ext_frac
            left = left - ext_size

        bottom = min(p.y0 for p in positions) - (pad + thickness)
        height = thickness

        ax_cb = fig.add_axes([left, bottom, width, height])
        ax_cb.set_in_layout(False)

        cbar = plt.colorbar(
            mesh,
            cax=ax_cb,
            orientation='horizontal',
            extend=extension,
            location='bottom',
            extendfrac=extendfrac
        )

        cbar.ax.tick_params(
            labelsize=tick_labelsize,
            direction='in',
            length=8,
            top=True,
            bottom=True
        )
        
        # Disable minor ticks
        cbar.ax.minorticks_off()

        # Handle tick placement
        if c_ticks == 3:
            # Show only min, center, max
            cbar.set_ticks([crange[0], (crange[0] + crange[1]) / 2, crange[1]])
        elif c_ticks_num:
            cbar.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=c_ticks))
        else:
            cbar.ax.xaxis.set_major_locator(MultipleLocator(c_ticks))

    cbar.set_label(label, fontsize=labelsize, labelpad=label_pad)
    return cbar


def add_shared_cbar_label(
    fig,
    cbar_axes,
    label,
    orientation='horizontal',
    fontsize=30,
    pad=0.02
):
    
    fig.canvas.draw()
    
    cbar_axes = np.atleast_1d(cbar_axes)
    
    # Get the renderer for measuring text extents
    renderer = fig.canvas.get_renderer()
    
    if orientation == 'horizontal':
        # Label goes below all colorbars
        # Need to find the bottom-most extent including tick labels
        min_bottom = 1.0  # Start at top of figure
        left_extent = 1.0
        right_extent = 0.0
        
        for ax in cbar_axes:
            # Get colorbar position
            pos = ax.get_position()
            left_extent = min(left_extent, pos.x0)
            right_extent = max(right_extent, pos.x1)
            
            # Get tick label extents
            for tick_label in ax.get_xticklabels():
                bbox = tick_label.get_window_extent(renderer=renderer)
                # Convert to figure coordinates
                bbox_fig = bbox.transformed(fig.transFigure.inverted())
                min_bottom = min(min_bottom, bbox_fig.y0)
        
        center_x = (left_extent + right_extent) / 2
        label_y = min_bottom - pad
        
        fig.text(
            center_x, label_y, label,
            ha='center', va='top',
            fontsize=fontsize,
            transform=fig.transFigure
        )
    else:
        # Vertical colorbars - label goes to the RIGHT of all colorbars, rotated 90 degrees
        # Find the rightmost extent including tick labels
        max_right = 0.0
        bottom_extent = 1.0
        top_extent = 0.0
        
        for ax in cbar_axes:
            pos = ax.get_position()
            bottom_extent = min(bottom_extent, pos.y0)
            top_extent = max(top_extent, pos.y1)
            
            # Get tick label extents (for vertical colorbars, these are on y-axis)
            for tick_label in ax.get_yticklabels():
                bbox = tick_label.get_window_extent(renderer=renderer)
                bbox_fig = bbox.transformed(fig.transFigure.inverted())
                max_right = max(max_right, bbox_fig.x1)
            
            # Also check the colorbar right edge
            max_right = max(max_right, pos.x1)
        
        # Place label to the right of the rightmost colorbar tick labels, centered vertically
        center_y = (bottom_extent + top_extent) / 2
        label_x = max_right + pad
        
        fig.text(
            label_x, center_y, label,
            ha='left', va='center',
            rotation=90,
            fontsize=fontsize,
            transform=fig.transFigure
        )