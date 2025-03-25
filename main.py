import json
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from sklearn.cluster import DBSCAN
import numpy as np
from dotenv import load_dotenv
import os
import folium
from folium.plugins import HeatMap



load_dotenv()
API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")


VIRGINIA_BOUNDS = {
    "min_lat": 36.5,
    "max_lat": 39.5,
    "min_lon": -83.7,
    "max_lon": -75.2
}

# Load Google Takeout JSON
def load_takeout_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# Extract all latitude/longitude points
def extract_coordinates(data):
    coordinates = []
    virginia_coordinates = []
    for entry in data:
        if "visit" in entry:
            place = entry["visit"]["topCandidate"]
            lat, lon = map(float, place["placeLocation"].split(":")[1].split(","))
            coordinates.append((lat, lon))
            if VIRGINIA_BOUNDS["min_lat"] <= lat <= VIRGINIA_BOUNDS["max_lat"] and VIRGINIA_BOUNDS["min_lon"] <= lon <= VIRGINIA_BOUNDS["max_lon"]:
                virginia_coordinates.append((lat, lon))
    return pd.DataFrame(coordinates, columns=["latitude", "longitude"])

def export_to_csv(coordinates, output_file="coordinates.csv"):
    coordinates.to_csv(output_file, index=False)
    print(f"Coordinates exported to {output_file}")

def extract_visits(data):
    visits = []
    for entry in data:
        if 'visit' in entry:
            place = entry["visit"]["topCandidate"]
            lat, lon = map(float, place["placeLocation"].split(":")[1].split(","))
            start_time = datetime.fromisoformat(entry['startTime'].split('.')[0])
            end_time = datetime.fromisoformat(entry['endTime'].split('.')[0])
            duration = (end_time - start_time).total_seconds() / 60  # Convert to minutes
            visits.append([lat, lon, start_time, end_time, duration])
    return pd.DataFrame(visits, columns=['Latitude', 'Longitude', 'Start Time', 'End Time', 'Duration'])

def cluster_locations(df, eps=0.001, min_samples=3):
    """Clusters locations using DBSCAN to identify significant locations."""
    coords = df[['Latitude', 'Longitude']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine').fit(np.radians(coords))
    df['Cluster'] = clustering.labels_
    return df

def label_clusters(df):
    """Assigns a label to each cluster based on Google Places API."""
    cluster_labels = {}
    for cluster in df['Cluster'].unique():
        if cluster == -1:
            continue
        cluster_center = df[df['Cluster'] == cluster][['Latitude', 'Longitude']].mean()
        cluster_labels[cluster] = get_place_type(cluster_center['Latitude'], cluster_center['Longitude'])
    df['Place Type'] = df['Cluster'].map(cluster_labels)
    return df

def visualize_clusters(df, map_name='location_map.html'):
    """Creates a Folium map to visualize clustered locations."""
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
    for _, row in df.iterrows():
        color = 'blue' if row['Cluster'] == -1 else 'red'
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['Place Type']} ({row['Duration']} mins)",
            icon=folium.Icon(color=color)
        ).add_to(m)
    m.save(map_name)
    print(f"Map saved as {map_name}")

def get_place_type(lat, lon):
    """Uses Google Places API to classify a location."""
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=50&key={API_KEY}"
    response = requests.get(url).json()
    if response['status'] == 'OK' and response['results']:
        return response['results'][0]['types'][0]
    return "Unknown"

def heatmap(df):
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
    heat_data = df[['Latitude', 'Longitude', 'Duration']].values.tolist()
    HeatMap(heat_data).add_to(m)

    m.save("heatmap.html")
    print("Heatmap saved as heatmap.html")
def time_series(df):
    coords = df[['Latitude', 'Longitude']].values.tolist()
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
    folium.PolyLine(coords, color="blue", weight=2.5, opacity=1).add_to(m)

    m.save("routes.html")
    print("Route map saved as routes.html")

def cluster_map(df):
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

    for _, row in df.iterrows():
        color = 'red' if row['Cluster'] != -1 else 'blue'  # Noise points in blue
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Cluster {row['Cluster']}, {row['Place Type']}",
            icon=folium.Icon(color=color)
        ).add_to(m)

    m.save("cluster_map.html")
    print("Clustered map saved as cluster_map.html")


# Main execution
def main():
    file_path = "location-history.json"  # Update with your file path

    data = load_takeout_data(file_path)
    coordinates = extract_coordinates(data)
    df = extract_visits(data)
    df = cluster_locations(df)
    df = label_clusters(df)
    #visualize_clusters(df)
    #heatmap(df)
    #time_series(df)
    cluster_map(df)


    for cluster_id in df['Cluster'].unique():
        cluster_points = df[df['Cluster'] == cluster_id]
        print(f"Cluster {cluster_id}:")
        for _, row in cluster_points.iterrows():
            print(f"  - Location: ({row['Latitude']}, {row['Longitude']}), Duration: {row['Duration']} mins")
        print()

    # print("Extracted Coordinates:")
    # print(coordinates)



if __name__ == "__main__":
    main()
