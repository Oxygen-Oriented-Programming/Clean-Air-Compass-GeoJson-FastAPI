from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/points/{location}")
async def get_map(location: str, response: Response):
  
  # call location IQ API to get bounding box for location
  bbox, valid_response = request_location_api(location)
  
  if valid_response:
    # call the purple API to get data for sensors within the bbox
    sensors_response = get_sensors_bbox_response(nwlong = bbox['min_lon'], nwlat = bbox['max_lat'], 
                                                selong = bbox['max_lon'], selat = bbox['min_lat'])
    
    # if at first there are no sensors try to expand the bounding box
    if len(sensors_response['data']) < 1:
      expanded_search = True
      ctr = 1
      while ctr < 3:
        bbox, valid_response = request_location_api(location, factor=ctr)
        sensors_response = get_sensors_bbox_response(nwlong = bbox['min_lon'], nwlat = bbox['max_lat'], 
                                                selong = bbox['max_lon'], selat = bbox['min_lat'])     
        if len(sensors_response['data']) > 1:
          break
        ctr += 1
      
      if len(sensors_response['data']) < 1:
        bbox_polygon = {
          "type": "Feature",
          "geometry": {
            "type": "Polygon",
            "coordinates": [[[bbox['min_lon'], bbox['min_lat']],
                [bbox['min_lon'], bbox['max_lat']],
                [bbox['max_lon'], bbox['max_lat']],
                [bbox['max_lon'], bbox['min_lat']],
                [bbox['min_lon'], bbox['min_lat']]
              ]
            ]
          }
        }
        
        return json.loads(json.dumps({"message":"No sensors available in that location, please try another.", "bbox": bbox, "bbox_polygon": bbox_polygon}))
    else:
     expanded_search = False

    geo_data = parse_sensors_bbox_response(sensors_response)
    # perform interpolation and return a grid of polygons with interpolated pm2.5 values
    response = make_interpolated_polygons(geo_data, expanded_search=expanded_search)
    
    return JSONResponse(content=response, status_code=200)
  else:
    return JSONResponse(content=bbox, status_code=404)


@app.get("/average_pollution/{location}")
async def get_average_pollution(location: str):
  
  # call location IQ API to get bounding box for location
  bbox, valid_response = request_location_api(location)

  if valid_response:
    # call the purple API to get data for sensors within the bbox
    sensors_response = get_sensors_bbox_response(nwlong = bbox['min_lon'], nwlat = bbox['max_lat'], 
                                                selong = bbox['max_lon'], selat = bbox['min_lat'])

    geo_data = parse_sensors_bbox_response(sensors_response)
    pm25_sum = 0
    valid_count = 0
    for sensor_data in geo_data:
        if sensor_data['pm2.5_60minute'] is not None:
            pm25_sum += sensor_data['pm2.5_60minute']
            valid_count += 1
    response = pm25_sum / valid_count
    return JSONResponse(content=response, status_code=200)
  
  else:
    return JSONResponse(content=bbox, status_code=200)