import pandas as pd
import geopandas as gpd
import requests
import time
import json
from dotenv import load_dotenv
import os
import re
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import functools

load_dotenv()

API_KEY  = os.environ.get('API_KEY')
LOC_IQ_KEY = os.environ.get('LOC_IQ_KEY')


def cache(func):
    cache = {}
    ttl = 1800  # 30 minutes in seconds

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        now = time.time()
        if key in cache:
            result, timestamp = cache[key]
            if now - timestamp < ttl:
                return result
        result = func(*args, **kwargs)
        cache[key] = (result, now)
        return result

    return wrapper


def is_postal_code(query: str):
    """
    Returns if a string matches US postal code format or not.
    The regular expression pattern matches the US ZIP code format, which can optionally include a ZIP+4 code.
    Examples: 12345, 12345-6789
    """
    pattern = r'^\d{5}(-\d{4})?$'
    match = re.match(pattern, query)
    if match:
        return True
    else:
        return False


@cache
def request_location_api(query: str, factor: int = 0):

    url = "https://us1.locationiq.com/v1/search"
    if is_postal_code(query):
        data = {
            'key': LOC_IQ_KEY,
            'postalcode': int(query),
            'format': 'json',
            'countrycodes': 'us'
        }

    else:
        data = {
            'key': LOC_IQ_KEY,
            'q': query,
            'format': 'json'
        }
    headers = {
        'Referer': 'https://www.kaggle.com/'
    }
    response = requests.get(url, params=data, headers=headers)
    data = json.loads(response.text)
    
    if data != {'error': 'Unable to geocode'}:
        bounding_box = data[0]['boundingbox']
        bbox = {
            'min_lat' : float(bounding_box[0]),
            'max_lat' : float(bounding_box[1]),
            'min_lon' : float(bounding_box[2]),
            'max_lon' : float(bounding_box[3])
        }
        
        bbox['min_lat'] -= (0.05 * 2**factor)
        bbox['max_lat'] += (0.05 * 2**factor)
        bbox['min_lon'] -= (0.05 * 2**factor)
        bbox['max_lon'] += (0.05 * 2**factor)
            
        if is_postal_code(query):
            bbox['min_lat'] -= 0.1
            bbox['max_lat'] += 0.1
            bbox['min_lon'] -= 0.1
            bbox['max_lon'] += 0.1
            
        valid_response = True
    
    else:
        data = {"message":"Please verify that you searched for a location in the United States."}
        valid_response = False
        return data, valid_response

    return bbox, valid_response

@cache
def get_sensors_bbox_response(nwlong: float, nwlat: float, selong: float, selat: float):
    
    base_url = "https://api.purpleair.com/v1/sensors/"
    fields = 'sensor_index,name,latitude,longitude,altitude,pm1.0,pm2.5,pm10.0,pm2.5_10minute,pm2.5_30minute,pm2.5_60minute'
    query = f'?fields={fields}&location_type=0'
    bbox = f'&nwlng={nwlong}&nwlat={nwlat}&selng={selong}&selat={selat}'

    url = base_url + query + bbox
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    return json.loads(response.text)



def parse_sensors_bbox_response(response_object) -> gpd.GeoDataFrame:
    data = response_object
    flat_dicts = []
    for sensor in data['data']:
        flat_dict = {
            'sensor_index': sensor[0],
            'name': sensor[1],
            'latitude': sensor[2],
            'longitude': sensor[3],
            'altitude': sensor[4],
            'pm1.0': sensor[5],
            'pm2.5': sensor[6],
            'pm10.0': sensor[7],
            'pm2.5_10minute': sensor[8],
            'pm2.5_30minute': sensor[9],
            'pm2.5_60minute': sensor[10]
        }
        flat_dicts.append(flat_dict)

    sensor_df = pd.DataFrame(flat_dicts)
    sensor_gdf = gpd.GeoDataFrame(sensor_df, geometry=gpd.points_from_xy(
        sensor_df.longitude, sensor_df.latitude))

    return sensor_gdf




def make_interpolated_polygons(sensor_gdf, expanded_search: bool = False):
    # Extract the X and Y coordinates of the sensor points
    X = sensor_gdf[["longitude", "latitude"]].values
    Z = sensor_gdf["pm2.5"].values

    # extract the bounds object from the dataframe to use for setting zoom level
    bounds = sensor_gdf.total_bounds
    bounds_obj = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
    
    # Create a grid of points to interpolate the air pollution values to
    x_min, x_max = X[:, 0].min() - 0.01, X[:, 0].max() + 0.01
    y_min, y_max = X[:, 1].min() - 0.01, X[:, 1].max() + 0.01
    grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

    # Flatten the grid_x and grid_y arrays
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    # set neighbors
    
    if len(Z) >= 5:
      neighbors = 5
    
    else:
      neighbors = len(Z)
    
    # Create a KNeighborsRegressor instance
    knn = KNeighborsRegressor(n_neighbors=neighbors, weights='distance')

    # Fit the model to the sensor data
    knn.fit(X, Z)

    # Use the predict method to interpolate values for the grid points
    interpolated_values = knn.predict(np.column_stack((grid_x, grid_y)))

    # Convert the interpolated_values array into a list of polygon features
    features = []
    for i in range(interpolated_values.shape[0]):
        polygon = [
            [grid_x[i], grid_y[i]],
            [grid_x[i] + 0.01, grid_y[i]],
            [grid_x[i] + 0.01, grid_y[i] + 0.01],
            [grid_x[i], grid_y[i] + 0.01],
            [grid_x[i], grid_y[i]]
        ]
        value = round(interpolated_values[i], 1) if np.isfinite(interpolated_values[i]) else None
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            },
            "properties": {
                "pm2.5": value
            }
        })

    # get the center point of the dataset to assist in centering the map
    center_point = [np.mean(X[:, 0]), np.mean(X[:, 1])]
    
    # also make the geojson object for just the sensor points so those can be returned back too
    
    points = json.loads(sensor_gdf.to_json())
    points_features = points['features']

    # Create a GeoJSON object that can be displayed on a React Leaflet map
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "points": points_features,
        "center_point": center_point,
        "expanded_search": expanded_search,
        "bounds": bounds_obj
        
    }
    
    return geojson
