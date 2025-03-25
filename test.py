import json
import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN
import requests
from datetime import datetime

# Google Places API Key (Replace with your own API key)
API_KEY = "YOUR_GOOGLE_PLACES_API_KEY"

def load_data(file_path):
    """Load Google Takeout JSON data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_visits(data):
    """Extracts relevant visit information from Google Takeout data."""
    visits = []
    for entry in data:
        if 'visit' in entry:
            loc = entry['visit']['topCandidate']['placeLocation']
            lat, lon = map(float, loc.replace("geo:", "").split(','))
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

def get_place_type(lat, lon):
    """Uses Google Places API to classify a location."""
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=50&key={API_KEY}"
    response = requests.get(url).json()
    if response['status'] == 'OK' and response['results']:
        return response['results'][0]['types'][0]
    return "Unknown"

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

# Main Execution
file_path = "takeout_data.json"  # Replace with actual path
data = load_data(file_path)
df = extract_visits(data)
df = cluster_locations(df)
df = label_clusters(df)
visualize_clusters(df)
