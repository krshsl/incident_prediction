import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.colors as mcolors
from scipy.spatial import ConvexHull
import seaborn as sns
import dask.dataframe as dd
from dask import delayed    
import os

base_path = '/common/home/ks2025/rutgers/cs551/final_project/data/input/' # update your path here
output_dir = '/common/home/ks2025/rutgers/cs551/final_project/graphs' # update your path here

def setup_environment():
    data_files = {
        'Austin': os.path.join(base_path, 'Austin_2016601_2019601.csv'),
        'New York City': os.path.join(base_path, 'NewYorkCity_2016601_2019601.csv'),
        'Los Angeles': os.path.join(base_path, 'LosAngeles_2016601_2019601.csv')
    }
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    colors = sns.color_palette('viridis', 3)
    
    return base_path, data_files, output_dir, colors

def process_monthly_data(data_files):
    all_cities_monthly = []
    
    for i, (city, file_path) in enumerate(data_files.items()):
        df = pd.read_csv(file_path)
        df['StartTime'] = pd.to_datetime(df['StartTime'])
        type_6_df = df[df['Type'] == 6].copy()
        
        type_6_df['Month'] = type_6_df['StartTime'].dt.month_name()
        type_6_df['MonthNum'] = type_6_df['StartTime'].dt.month
        type_6_df['Year'] = type_6_df['StartTime'].dt.year
        
        monthly_counts = type_6_df.groupby(['Month', 'MonthNum', 'Year']).size().reset_index(name='Count')
        monthly_averages = monthly_counts.groupby(['Month', 'MonthNum'])['Count'].mean().reset_index(name='MonthlyAverage')
        monthly_averages['City'] = city
        all_cities_monthly.append(monthly_averages)
    
    combined_monthly = pd.concat(all_cities_monthly, ignore_index=True)
    month_order = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
    combined_monthly['Month'] = pd.Categorical(combined_monthly['Month'], categories=month_order, ordered=True)
    combined_monthly = combined_monthly.sort_values(['City', 'MonthNum'])
    
    return combined_monthly

def create_monthly_plot(combined_monthly, colors, output_dir):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, city in enumerate(combined_monthly['City'].unique()):
        city_data = combined_monthly[combined_monthly['City'] == city]
        ax.plot(city_data['Month'], city_data['MonthlyAverage'], 
                 marker='o', markersize=8, linewidth=2.5, 
                 color=colors[i], label=city)
    
    ax.set_title('Monthly Average of Traffic Accidents (2016-2019)', fontsize=18, pad=20)
    ax.set_xlabel('Month', fontsize=14, labelpad=10)
    ax.set_ylabel('Average Number of Accidents', fontsize=14, labelpad=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='City', title_fontsize=14, fontsize=12, loc='best')
    
    fig.tight_layout()
    output_path = os.path.join(output_dir, 'monthly_accidents_by_city.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def process_daily_data(data_files):
    all_cities_daily = []
    
    for i, (city, file_path) in enumerate(data_files.items()):
        df = pd.read_csv(file_path)
        df['StartTime'] = pd.to_datetime(df['StartTime'])
        type_6_df = df[df['Type'] == 6].copy()
        
        type_6_df['DayOfYear'] = type_6_df['StartTime'].dt.dayofyear
        type_6_df['Year'] = type_6_df['StartTime'].dt.year
        
        daily_counts = type_6_df.groupby(['DayOfYear', 'Year']).size().reset_index(name='Count')
        daily_averages = daily_counts.groupby(['DayOfYear'])['Count'].mean().reset_index(name='DailyAverage')
        daily_averages['City'] = city
        
        all_cities_daily.append(daily_averages)
    
    combined_daily = pd.concat(all_cities_daily, ignore_index=True)
    combined_daily = combined_daily.sort_values(['City', 'DayOfYear'])
    
    return combined_daily

def create_daily_plot(combined_daily, colors, output_dir):
    print(f"Creating daily plot...")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, city in enumerate(combined_daily['City'].unique()):
        city_data = combined_daily[combined_daily['City'] == city].copy()
        city_data = city_data.sort_values('DayOfYear')
        city_data['MovingAverage'] = city_data['DailyAverage'].rolling(window=7, center=True).mean()
        
        ax.plot(city_data['DayOfYear'], city_data['MovingAverage'],
                 linewidth=2.5, color=colors[i], label=f"{city}")
    
    ax.set_title('Daily Average of Traffic Accidents Throughout the Year (2016-2019)', fontsize=18, pad=20)
    ax.set_xlabel('Month', fontsize=14, labelpad=10)
    ax.set_ylabel('Average Number of Accidents', fontsize=14, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='City', title_fontsize=14, fontsize=12, loc='best')
    
    month_positions = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_names)
    
    for pos in month_positions:
        ax.axvline(x=pos, color='gray', linestyle=':', alpha=0.5)
    
    fig.tight_layout()
    output_path = os.path.join(output_dir, 'daily_accidents_by_city.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def generate_accident_hotspot_map(data_path, city_name, output_dir):
    print(f"Processing {city_name}...")
    df = pd.read_csv(data_path)
    df = df[df['Type'] == 6]
    df['StartTime'] = pd.to_datetime(df['StartTime'])
    coords = df[['LocationLat', 'LocationLng']].values
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(coords)
    cluster_counts = df['cluster'].value_counts().sort_index()
    num_days = (df['StartTime'].dt.date.max() - df['StartTime'].dt.date.min()).days + 1
    cluster_daily_avgs = cluster_counts / num_days
    cluster_centers = kmeans.cluster_centers_
    geometry = [Point(xy) for xy in zip(df['LocationLng'], df['LocationLat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#121212')
    ax.set_facecolor('#121212')
    min_count = cluster_daily_avgs.min()
    max_count = cluster_daily_avgs.max()
    norm_counts = (cluster_daily_avgs - min_count) / (max_count - min_count)
    colors = [(0, 0.3, 0.6, 0.7),
              (0, 0.6, 0.6, 0.7),
              (0.6, 0.6, 0, 0.7),
              (0.7, 0.3, 0, 0.7),
              (1.0, 0.5, 0.5, 0.7)]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("vibrant", colors)
    cluster_colors = {i: custom_cmap(norm) for i, norm in zip(cluster_daily_avgs.index, norm_counts)}
    sorted_clusters = cluster_daily_avgs.sort_values().index   
    for i in sorted_clusters:
        cluster_points = gdf[gdf['cluster'] == i]
        cluster_points.plot(ax=ax, color=cluster_colors[i], alpha=0.8, markersize=50, edgecolor='white', linewidth=0.5)
    
    ctx.add_basemap(ax, source='https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png')
    
    for i in sorted_clusters:
        cluster_points = gdf[gdf['cluster'] == i]   
        if len(cluster_points) >= 3:
            points = np.array([(p.x, p.y) for p in cluster_points.geometry])
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.append(hull_points, [hull_points[0]], axis=0)
            ax.plot(hull_points[:, 0], hull_points[:, 1], 
                    color=cluster_colors[i], linewidth=5, 
                    alpha=1.0, zorder=3)
            ax.fill(hull_points[:, 0], hull_points[:, 1], 
                    color=cluster_colors[i], alpha=0.5, zorder=2)
        
        center_point = Point(cluster_centers[i][1], cluster_centers[i][0])
        center_point = gpd.GeoSeries([center_point], crs="EPSG:4326").to_crs(epsg=3857)[0]
        plt.annotate(f"Cluster {i}\n{cluster_daily_avgs[i]:.2f} accidents/day", 
                xy=(center_point.x, center_point.y),
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", fc='#2e2e2e', ec=cluster_colors[i], lw=3, alpha=0.9),
                fontsize=12,
                fontweight='bold',
                color='white',
                zorder=6)
    
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(min_count, max_count))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6)
    cbar.set_label('Daily Average of Accidents', fontsize=12, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    plt.title(f'Daily Traffic Accident Hotspots in {city_name}', fontsize=16, color='white')
    plt.axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{city_name.lower().replace(" ", "_")}_accident_hotspots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"Accident Clusters for {city_name} (sorted by daily average):")
    for i in sorted_clusters:
        center_coords = cluster_centers[i]
        print(f"Cluster {i}: {cluster_daily_avgs[i]:.2f} accidents/day, Center: {center_coords}")
    
    return output_path

def generate_hotspot_graph(data_files, output_dir):
    for city_name, data_path in data_files.items():
        try:
            output_file = generate_accident_hotspot_map(data_path, city_name, output_dir)
            print(f"Generated hotspot map for {city_name}: {output_file}")
        except Exception as e:
            print(f"Error generating hotspot map for {city_name}: {str(e)}")

    print("All processing complete.")

def process_daylight_data(data_files):
    print("Processing daylight data using Dask...")
    
    @delayed
    def process_city(city, file_path):
        print(f"Processing daylight data for {city}...")
        ddf = dd.read_csv(file_path)
        df = ddf.compute()
        df = df[df['Type'] == 6]
        for cols in ['StartTime', 'Sunrise', 'Sunset']:
            df[cols] = pd.to_datetime(df[cols])
            df[f'{cols}_t'] = df[cols].dt.time

        df['IsDaylight'] = df.apply(lambda row: row['StartTime_t'] >= row['Sunrise_t'] and row['StartTime_t'] <= row['Sunset_t'], axis=1)
        daylight_count = np.sum(df['IsDaylight'])
        night_count = len(df) - daylight_count
        num_days = (df['StartTime'].dt.date.max() - df['StartTime'].dt.date.min()).days + 1
        daylight_avg = daylight_count / num_days
        night_avg = night_count / num_days
        city_data = pd.DataFrame({
            'City': [city, city],
            'Period': ['Daylight', 'Night'],
            'AccidentsPerDay': [daylight_avg, night_avg]
        })
        
        return city_data
    
    delayed_results = [process_city(city, file_path) for city, file_path in data_files.items()]
    results = delayed(pd.concat)(delayed_results).compute()
    return results

def create_daylight_plot(daylight_data, colors, output_dir):
    print(f"Creating daylight plot...")
    plt.figure(figsize=(12, 8))
    
    ax = sns.barplot(
        x='City',
        y='AccidentsPerDay',
        hue='Period',
        data=daylight_data,
        palette=['#FFE863', '#3A75C4'],
        edgecolor='black'
    )
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            y_offset = 5 if height > 10 else 10
            ax.annotate(
                f'{height:.2f}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                xytext=(0, y_offset),
                textcoords='offset points'
            )
    
    plt.title('Average Daily Accidents: Daylight vs Night', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Average Accidents per Day', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.legend(title='Time of Day', fontsize=12)
    
    output_path = os.path.join(output_dir, 'daylight_vs_night_accidents.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def main():
    base_path, data_files, output_dir, colors = setup_environment()
    
    combined_monthly = process_monthly_data(data_files)
    monthly_fig = create_monthly_plot(combined_monthly, colors, output_dir)
    
    combined_daily = process_daily_data(data_files)
    daily_fig = create_daily_plot(combined_daily, colors, output_dir)
    
    daylight_data = process_daylight_data(data_files)
    daylight_fig = create_daylight_plot(daylight_data, colors, output_dir)

    plt.show()

    generate_hotspot_graph(data_files, output_dir)

if __name__ == "__main__":
    main()
