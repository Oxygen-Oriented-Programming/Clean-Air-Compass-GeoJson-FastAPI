import os
import re
from shapely.geometry import Point, mapping
import functools
import math
import urllib
from dotenv import load_dotenv
import requests
import time
import json

load_dotenv()

API_KEY = os.environ.get("PURPLEAIR_API_KEY")
LOC_IQ_KEY = os.environ.get("LOC_IQ_KEY")


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
    pattern = r"^\d{5}(-\d{4})?$"
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
            "key": LOC_IQ_KEY,
            "postalcode": int(query),
            "format": "json",
            "countrycodes": "us",
        }

    else:
        data = {"key": LOC_IQ_KEY, "q": urllib.parse.unquote(query), "format": "json"}

    response = requests.get(url, params=data)  # headers=headers
    data = json.loads(response.text)

    if data != {"error": "Unable to geocode"}:
        bounding_box = data[0]["boundingbox"]
        bbox = {
            "min_lat": float(bounding_box[0]),
            "max_lat": float(bounding_box[1]),
            "min_lon": float(bounding_box[2]),
            "max_lon": float(bounding_box[3]),
            "size": (
                float(bounding_box[0])
                - float(bounding_box[1]) * float(bounding_box[2])
                - float(bounding_box[3])
            ),
        }

        bbox["min_lat"] -= 0.05 * 2**factor
        bbox["max_lat"] += 0.05 * 2**factor
        bbox["min_lon"] -= 0.05 * 2**factor
        bbox["max_lon"] += 0.05 * 2**factor

        if is_postal_code(query):
            bbox["min_lat"] -= 0.1
            bbox["max_lat"] += 0.1
            bbox["min_lon"] -= 0.1
            bbox["max_lon"] += 0.1

        valid_response = True

    else:
        print(response.text)
        data = {
            "message": "Please verify that you searched for a location in the United States."
        }
        valid_response = False
        return data, valid_response

    return bbox, valid_response


@cache
def get_sensors_bbox_response(nwlong: float, nwlat: float, selong: float, selat: float):
    base_url = "https://api.purpleair.com/v1/sensors/"
    fields = "sensor_index,name,latitude,longitude,altitude,pm1.0,pm2.5,pm10.0,pm2.5_10minute,pm2.5_30minute,pm2.5_60minute"
    query = f"?fields={fields}&location_type=0"
    bbox = f"&nwlng={nwlong}&nwlat={nwlat}&selng={selong}&selat={selat}"

    url = base_url + query + bbox
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    response = requests.get(url, headers=headers)
    return json.loads(response.text)


def parse_sensors_bbox_response(response_object):
    data = response_object
    geo_data = []
    for sensor in data["data"]:
        geo_dict = {
            "sensor_index": sensor[0],
            "name": sensor[1],
            "latitude": sensor[2],
            "longitude": sensor[3],
            "altitude": sensor[4],
            "pm1.0": sensor[5],
            "pm2.5": sensor[6],
            "pm10.0": sensor[7],
            "pm2.5_10minute": sensor[8],
            "pm2.5_30minute": sensor[9],
            "pm2.5_60minute": sensor[10],
            "geometry": Point(
                sensor[3], sensor[2]
            ),  # Create a Point geometry using longitude and latitude
        }
        geo_data.append(geo_dict)

    return geo_data


def calculate_total_bounds(sensor_data):
    lons = [d["longitude"] for d in sensor_data]
    lats = [d["latitude"] for d in sensor_data]
    return [min(lons), min(lats), max(lons), max(lats)]


def euclidean_distance(p1, p2):
    return math.sqrt(sum([math.pow(a - b, 2) for a, b in zip(p1, p2)]))


def knn_regression(X, Z, query_points, k=5, weights="distance"):
    predictions = []
    for query in query_points:
        distances = [euclidean_distance(query, x) for x in X]
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        neighbors = sorted_indices[:k]
        distances = [distances[i] for i in neighbors]
        values = [Z[i] for i in neighbors]

        if weights == "distance":
            weights = [1 / d if d != 0 else 1.0 for d in distances]
            prediction = sum(v * w for v, w in zip(values, weights)) / sum(weights)
        else:
            prediction = sum(values) / len(values)

        predictions.append(prediction)

    return predictions


def make_interpolated_polygons(sensor_data, expanded_search: bool = False):
    X = [[d["longitude"], d["latitude"]] for d in sensor_data]
    Z = [d["pm2.5"] for d in sensor_data]

    bounds = calculate_total_bounds(sensor_data)
    bounds_obj = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

    x_min, x_max = min(x[0] for x in X) - 0.01, max(x[0] for x in X) + 0.01
    y_min, y_max = min(x[1] for x in X) - 0.01, max(x[1] for x in X) + 0.01

    grid_x = [x_min + i * (x_max - x_min) / 99 for i in range(100)]
    grid_y = [y_min + i * (y_max - y_min) / 99 for i in range(100)]

    grid_x = [x for x in grid_x for _ in range(100)]
    grid_y = grid_y * 100

    if len(Z) >= 5:
        neighbors = 5
    else:
        neighbors = len(Z)

    query_points = [[grid_x[i], grid_y[i]] for i in range(len(grid_x))]
    interpolated_values = knn_regression(
        X, Z, query_points, k=neighbors, weights="distance"
    )

    features = []
    for i in range(len(interpolated_values)):
        polygon = [
            [grid_x[i], grid_y[i]],
            [grid_x[i] + 0.01, grid_y[i]],
            [grid_x[i] + 0.01, grid_y[i] + 0.01],
            [grid_x[i], grid_y[i] + 0.01],
            [grid_x[i], grid_y[i]],
        ]
        value = (
            round(interpolated_values[i], 1)
            if interpolated_values[i] != float("inf")
            and interpolated_values[i] != float("-inf")
            else None
        )
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
                "properties": {"pm2.5": value},
            }
        )

    center_point = [sum(x[0] for x in X) / len(X), sum(x[1] for x in X) / len(X)]

    points_features = [
        {
            "type": "Feature",
            "geometry": mapping(d["geometry"]),
            "properties": {k: v for k, v in d.items() if k != "geometry"},
        }
        for d in sensor_data
    ]

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "points": points_features,
        "center_point": center_point,
        "expanded_search": expanded_search,
        "bounds": bounds_obj,
    }
    print(geojson["center_point"])
    return geojson
