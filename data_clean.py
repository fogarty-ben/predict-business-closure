import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import pickle
import json
from census import Census


def gdf_from_latlong(df, lat, long):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[long], df[lat]))
    return gdf


def get_acs_data(minyear, maxyear, api_key, var_list, geo_dic, output_directory):
    '''
    get acs5 data for a year range and store them in the output directory
    '''
    c = Census(api_key)
    for year in range(minyear, maxyear+1):
        print("obtaining data from {}...".format(year))
        result = c.acs5.get(var_list, geo_dic, year=year)
        # write to json file
        output_filepath = output_directory + "acs5_{}_{}.json".format(
                          year, geo_dic['for'][:-2])
        with open(output_filepath, 'w') as f:
            json.dump(test, f)
    print('finished')


def load_json(filepath):
    try:
        return json.load(open(filepath, 'r'))
    except:
        raise("ValueError", "Filepath does not exist")