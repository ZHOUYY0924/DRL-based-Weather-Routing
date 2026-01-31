import csv
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from geopy.distance import geodesic
from shapely.geometry import LineString, box
from pyproj import Geod
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature



geod = Geod(ellps='WGS84')

# --------- 大圆路径生成 ---------
def generate_gc_path(start, end, n_points=1000):
    lons_and_lats = geod.npts(start[0], start[1], end[0], end[1], n_points)
    lons = [each[0] for each in lons_and_lats]
    lats = [each[1] for each in lons_and_lats]
    lons = [start[0]] + lons + [end[0]]
    lats = [start[1]] + lats + [end[1]]
    return list(zip(lons, lats))

def normalize_lon_360(lon):
    return lon % 360

def denormalize_lon_180(lon):
    return lon - 360 if lon >= 180 else lon

def get_crossed_grids(path, res, select_range = '-180_180'):
    normalized_path = [(normalize_lon_360(lon), lat) for lon, lat in path]
    line = LineString(normalized_path)
    min_lon, min_lat, max_lon, max_lat = line.bounds
    grids = []
    half_res = res / 2

    lon_centers = np.arange(np.floor(min_lon * 10) / 10, np.ceil(max_lon * 10) / 10 + res, res)
    lat_centers = np.arange(np.floor(min_lat * 10) / 10, np.ceil(max_lat * 10) / 10 + res, res)

    for lon_c in lon_centers:
        for lat_c in lat_centers:
            lon_min = lon_c - half_res
            lon_max = lon_c + half_res
            lat_min = lat_c - half_res
            lat_max = lat_c + half_res
            cell = box(lon_min, lat_min, lon_max, lat_max)

            if line.intersects(cell):
                intersection = line.intersection(cell)
                if intersection.is_empty:
                    continue
                segments = [intersection] if intersection.geom_type == 'LineString' else list(intersection.geoms)

                for seg in segments:
                    coords = list(seg.coords)
                    if len(coords) < 2:
                        continue
                    # 经纬度还原回 [-180, 180]
                    if select_range == '-180_180':
                        start = (denormalize_lon_180(coords[0][0]), coords[0][1])
                        end = (denormalize_lon_180(coords[-1][0]), coords[-1][1])
                        distance = segment_length([start, end])
                        azimuth = calculate_azimuth(start, end)
                        grids.append({
                            'grid_lon': round(denormalize_lon_180(lon_c), 1),
                            'grid_lat': round(lat_c, 1),
                            'distance_km': distance,
                            'heading_deg': azimuth
                        })
                    elif select_range == '0_360':
                        start = (coords[0][0], coords[0][1])
                        end = (coords[-1][0], coords[-1][1])
                        distance = segment_length([start, end])
                        azimuth = calculate_azimuth(start, end)
                        grids.append({
                            'grid_lon': round(lon_c, 1),
                            'grid_lat': round(lat_c, 1),
                            'distance_km': distance,
                            'heading_deg': azimuth
                        })

    return grids

def segment_length(coords):
    total = 0.0
    for i in range(len(coords) - 1):
        pt1 = (coords[i][1], coords[i][0])  # lat, lon
        pt2 = (coords[i+1][1], coords[i+1][0])
        total += geodesic(pt1, pt2).kilometers
    return total

def calculate_azimuth(p1, p2):
    azimuth, _, _ = geod.inv(p1[0], p1[1], p2[0], p2[1])
    return (azimuth + 360) % 360

def generate_all_grid_boxes(path, res):
    line = LineString(path)
    min_lon, min_lat, max_lon, max_lat = line.bounds

    lon_start = np.floor(min_lon / res) * res + res / 2
    lon_end   = np.ceil(max_lon / res) * res - res / 2
    lat_start = np.floor(min_lat / res) * res + res / 2
    lat_end   = np.ceil(max_lat / res) * res - res / 2

    lon_centers = np.arange(lon_start, lon_end + res, res)
    lat_centers = np.arange(lat_start, lat_end + res, res)

    grid_boxes = []
    for lon_c in lon_centers:
        for lat_c in lat_centers:
            grid_boxes.append((round(lon_c, 6), round(lat_c, 6)))
    return grid_boxes


def plot_path_and_grids(path, grids_info, res):
    half_res = res / 2
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_title('Great Circle Path Through ERA5 Grids with Coastline')
    ax.coastlines(resolution='10m', linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5)

    lons, lats = zip(*path)
    ax.plot(lons, lats, 'b-', label='Great Circle Path', linewidth=2, transform=ccrs.PlateCarree())
    ax.plot(lons[0], lats[0], 'go', label='Start', transform=ccrs.PlateCarree())
    ax.plot(lons[-1], lats[-1], 'ro', label='End', transform=ccrs.PlateCarree())

    for g in grids_info:
        lon_c = g['grid_lon']
        lat_c = g['grid_lat']
        rect = Rectangle((lon_c - half_res, lat_c - half_res), res, res,
                         linewidth=0.5, edgecolor='gray', facecolor='none', transform=ccrs.PlateCarree())
        ax.add_patch(rect)

        cx = lon_c
        cy = lat_c
        length = res * 0.4
        angle_rad = np.radians(90 - g['azimuth_deg'])
        dx = length * np.cos(angle_rad)
        dy = length * np.sin(angle_rad)
        ax.arrow(cx, cy, dx, dy, head_width=res * 0.03, color='red',
                 alpha=0.7, transform=ccrs.PlateCarree())

    all_grids = generate_all_grid_boxes(path, res)
    for lon_c, lat_c in all_grids:
        rect = Rectangle((lon_c - half_res, lat_c - half_res), res, res,
                         linewidth=0.3, edgecolor='lightgray', facecolor='none',
                         transform=ccrs.PlateCarree(), zorder=1)
        ax.add_patch(rect)

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def save_grid_info_to_csv(grids_info, filename='grid_crossing_info.csv'):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Grid_Lon', 'Grid_Lat', 'Azimuth_Deg', 'Distance_km'])
        for g in grids_info:
            writer.writerow([
                round(g['grid_lon'], 6),
                round(g['grid_lat'], 6),
                round(g['azimuth_deg'], 2),
                round(g['distance_km'], 3)
            ])
    print(f"✅ CSV 文件已保存：{filename}")

def plot_env_heatmap(ax, env_data, res, vmin=None, vmax=None, cmap='plasma'):
    half_res = res / 2
    values = [d['value'] for d in env_data]
    norm = mcolors.Normalize(vmin=vmin if vmin else min(values),
                             vmax=vmax if vmax else max(values))
    colormap = cm.get_cmap(cmap)

    for d in env_data:
        color = colormap(norm(d['value']))
        rect = Rectangle((d['lon'] - half_res, d['lat'] - half_res), res, res,
                         facecolor=color, edgecolor='none', alpha=0.8,
                         transform=ccrs.PlateCarree(), zorder=0)
        ax.add_patch(rect)

    sm = cm.ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', label='Significant Wave Height (m)')
