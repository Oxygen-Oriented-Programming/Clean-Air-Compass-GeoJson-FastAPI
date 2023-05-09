# Clean Air Compass - Mapping API - Serverless Version
# Clean Air Compass - Mapping API - Serverless Version

An API to handle location requests from the Clean Air Compass frontend and return an interpolated map of air pollution (PM 2.5) in the form of a list of georeferenced polygons.

## Details

The API conducts the following steps:

- Parses the location request to see if it is for a city or a US postal code
- Makes a request to the [LocationIQ API](https://locationiq.com/) and get back a bounding box of lat / lon coordinates
- Makes a request to the [Purple Air API](https://www2.purpleair.com/) to get back the list of sensors within the bounding box as well as their associated air pollution data
- Creates a spatial data frame from the sensor API response object using `shapely`
- Creates a grid of polygons based on the spatial extent of the collection of sensor points and estimates the PM 2.5 pollution at each grid cell using the K-nearest neigbors (KNN) regressor method.
- Returns the grid of polygons to the frontend as a GeoJSON object

## How to run

1. Clone the repo
2. Install all dependencies by running `pip install -r requirements.txt` in the console
3. Create your own environment file where you set the values for your own Location IQ API key (`LOC_IQ_KEY`) and Purple Air API key (`API_KEY`) respectively.
4. Run the API server locally using `uvicorn main:app --reload`
