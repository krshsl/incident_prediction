cities = {
    'Austin': {
        'coordinates': [30.079327,-97.968881, 30.596764,-97.504838],
        'weather': [(30.079, -97.969), (30.58, -97.969), (30.079, -97.718), (30.33, -97.969), (30.58, -97.718), (30.33, -97.718)],
        'sunrise': 'austin-tx',
        'timezone': 'US/Central'
    },
    'LosAngeles': {
        'coordinates': [33.700615,-118.683511,34.353627,-118.074559],
        'weather': [(34.201, -118.183), (33.7, -118.684), (33.951, -118.684), (34.201, -118.4335), (33.7, -118.183), (33.951, -118.183), (33.7, -118.4335), (33.951, -118.4335), (34.201, -118.684)],
        'sunrise': 'los-angeles-ca',
        'timezone': 'US/Pacific'
    },
    'NewYorkCity': {
        'coordinates': [40.477399,-74.259090,40.917577,-73.700272],
        'weather': [(40.477, -74.259), (40.702, -74.259), (40.477, -73.965), (40.59, -74.259), (40.702, -73.965), (40.59, -73.965)],
        'sunrise': 'new-york-ny',
        'timezone': 'US/Eastern' # also present in our dataset fyi
    }
}

city_keys = list(cities.keys())

# time interval to sample data for 
years = [2017, 2018]
months = [6, 6]
# years = [2017]
# months = [6, 9]
# years = [2018]
# months = [6] # naturally the quickest