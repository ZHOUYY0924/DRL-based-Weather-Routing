import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
from global_land_mask import globe
import os

def get_great_circle_heading(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    angle_from_north = np.arctan2(y, x)
    heading = np.degrees(angle_from_north)
    heading = (heading + 360) % 360

    return heading

def calculate_destination_point(lon1, lat1, initial_heading, distance_degree):
    lon1_rad, lat1_rad = np.radians(lon1), np.radians(lat1)
    heading_rad = np.radians(initial_heading)
    angular_distance = np.radians(distance_degree)
    lat2_rad = np.arcsin(np.sin(lat1_rad) * np.cos(angular_distance) +
                         np.cos(lat1_rad) * np.sin(angular_distance) * np.cos(heading_rad))
    lon2_rad = lon1_rad + np.arctan2(np.sin(heading_rad) * np.sin(angular_distance) * np.cos(lat1_rad),
                                     np.cos(angular_distance) - np.sin(lat1_rad) * np.sin(lat2_rad))
    lon2, lat2 = np.degrees(lon2_rad), np.degrees(lat2_rad)

    return lon2, lat2

def main():
    lat_start, lon_start = 16, 128
    lat_end, lon_end = 31, 135

    output_csv_path = r'ship_net_combined.csv'

    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    dis_list = [round(each, 1) for each in list(np.linspace(0, 3.5, 8))]

    print(f"开始生成航路网络...")
    print(f"起点: ({lat_start}, {lon_start}) -> 终点: ({lat_end}, {lon_end})")

    g = Geod(ellps='WGS84')
    az12, az21, dist_meters = g.inv(lon_start, lat_start, lon_end, lat_end)
    dist_nm = dist_meters / 1852.0
    print(f"大圆距离: {dist_nm:.2f} nm")

    num_points = int(np.ceil(dist_nm / 60))

    intermediate_points = g.npts(lon_start, lat_start, lon_end, lat_end, num_points - 1)

    if intermediate_points:
        lon_mid, lat_mid = zip(*intermediate_points)
        lon_mid = list(lon_mid)
        lat_mid = list(lat_mid)
    else:
        lon_mid, lat_mid = [], []

    gc_lons = [lon_start] + lon_mid + [lon_end]
    gc_lats = [lat_start] + lat_mid + [lat_end]
    great_circle_data = pd.DataFrame({'lon': gc_lons, 'lat': gc_lats})

    total_steps = len(great_circle_data)
    print(f"骨干路径生成完毕，共 {total_steps} 个航点。")

    final_lats = []
    final_lons = []
    final_labels = []

    final_lats.append(lat_start)
    final_lons.append(lon_start)
    final_labels.append(0)

    for i in range(1, total_steps):
        lon1 = great_circle_data['lon'][i - 1]
        lat1 = great_circle_data['lat'][i - 1]
        lon2 = great_circle_data['lon'][i]
        lat2 = great_circle_data['lat'][i]

        heading = get_great_circle_heading(lon1, lat1, lon2, lat2)


        if i == total_steps - 1:
            current_dis_list = [0]
        else:
            steps_from_start = i
            steps_from_end = (total_steps - 1) - i

            min_steps = min(steps_from_start, steps_from_end)
            slice_idx = min_steps * 2 + 1

            if slice_idx > len(dis_list):
                current_dis_list = dis_list
            else:
                current_dis_list = dis_list[:slice_idx]

        for dist_deg in current_dis_list:
            lon3, lat3 = calculate_destination_point(lon2, lat2, heading + 90, dist_deg)

            lon3_check = lon3
            if lon3_check > 180: lon3_check -= 360
            if lon3_check < -180: lon3_check += 360

            if not globe.is_land(lat3, lon3_check):
                final_lats.append(lat3)
                final_lons.append(lon3)
                final_labels.append(i)

        for dist_deg in current_dis_list[1:]:
            lon3, lat3 = calculate_destination_point(lon2, lat2, heading - 90, dist_deg)

            lon3_check = lon3
            if lon3_check > 180: lon3_check -= 360
            if lon3_check < -180: lon3_check += 360

            if not globe.is_land(lat3, lon3_check):
                final_lats.append(lat3)
                final_lons.append(lon3)
                final_labels.append(i)

    save_lons = []
    for l in final_lons:
        if l > 180:
            l -= 360
        elif l < -180:
            l += 360
        save_lons.append(l)

    result_df = pd.DataFrame({
        'label': final_labels,
        'lat': final_lats,
        'lon': save_lons
    })

    result_df.to_csv(output_csv_path, index=False)
    print(f"网格生成完毕，已保存至: {output_csv_path}")
    print(f"共生成 {len(result_df)} 个有效航点。")

    count_start = len(result_df[result_df['label'] == 0])
    count_label_1 = len(result_df[result_df['label'] == 1])
    count_end = len(result_df[result_df['label'] == total_steps - 1])

    print("-" * 30)
    print(f"验证: Label 0 (起点) 点数: {count_start} (应为1)")
    print(f"验证: Label 1 (第2行) 点数: {count_label_1} (应 > 1)")
    print(f"验证: Label {total_steps - 1} (终点) 点数: {count_end} (应为1)")
    print("-" * 30)

    try:
        fig = plt.figure(figsize=(12, 8))
        center_lon = (lon_start + lon_end) / 2
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=center_lon))

        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

        plot_gc_lons = [l if l <= 180 else l - 360 for l in gc_lons]
        ax.plot(plot_gc_lons, gc_lats, 'r-', linewidth=2, label='Great Circle Spine', transform=ccrs.PlateCarree())

        ax.scatter(save_lons, final_lats, color='blue', s=2, marker='.', label='Navigable Network',
                   transform=ccrs.PlateCarree())

        ax.scatter([lon_start], [lat_start], color='green', s=50, marker='o', label='Start',
                   transform=ccrs.PlateCarree())
        ax.scatter([lon_end], [lat_end], color='red', s=50, marker='*', label='Goal', transform=ccrs.PlateCarree())

        ax.legend()
        plt.title("Ship Routing Network (Point-to-Point)")
        plt.savefig(os.path.join(output_dir, 'ship_network_p2p.png'))
        print("绘图已保存。")
    except Exception as e:
        print(f"绘图时发生错误: {e}")


if __name__ == '__main__':
    main()