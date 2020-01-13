from scipy.interpolate import NearestNDInterpolator
import numpy as np
import sys
import os
from datetime import date, timedelta
import pandas as pd
from pyproj import Proj, transform
import json
import time
import csv

import ptvsd
ptvsd.enable_attach("abc", address = ("0.0.0.0", 14000))

import capnp
if sys.platform == "win32":
    capnp.add_import_hook(additional_paths=["../vcpkg/packages/capnproto_x64-windows-static/include/", "../capnproto_schemas/"])
elif sys.platform == "linux":
    capnp.add_import_hook(additional_paths=["../capnproto_schemas/"])
import common_capnp
#import model_capnp
import climate_data_capnp

TIME_RANGES = {
    "0": {"from": 1980, "to": 2010},
    "2": {"from": 2040, "to": 2069},
    "3": {"from": 2070, "to": 2099}
}

def read_header(path_to_ascii_grid_file):
    "read metadata from esri ascii grid file"
    metadata = {}
    header_str = ""
    with open(path_to_ascii_grid_file) as _:
        for i in range(0, 6):
            line = _.readline()
            header_str += line
            sline = [x for x in line.split() if len(x) > 0]
            if len(sline) > 1:
                metadata[sline[0].strip().lower()] = float(sline[1].strip())
    return metadata, header_str

def create_ascii_grid_interpolator(arr, meta, ignore_nodata=True):
    "read an ascii grid into a map, without the no-data values"

    rows, cols = arr.shape

    cellsize = int(meta["cellsize"])
    xll = int(meta["xllcorner"])
    yll = int(meta["yllcorner"])
    nodata_value = meta["nodata_value"]

    xll_center = xll + cellsize // 2
    yll_center = yll + cellsize // 2
    yul_center = yll_center + (rows - 1)*cellsize

    points = []
    values = []

    for row in range(rows):
        for col in range(cols):
            value = arr[row, col]
            if ignore_nodata and value == nodata_value:
                continue
            r = xll_center + col * cellsize
            h = yul_center - row * cellsize
            points.append([r, h])
            values.append(value)

    return NearestNDInterpolator(np.array(points), np.array(values))

def read_file_and_create_interpolator(path_to_grid, dtype=int, skiprows=6, confirm_creation=False):
    "read file and metadata and create interpolator"

    metadata, _ = read_header(path_to_grid)
    grid = np.loadtxt(path_to_grid, dtype=dtype, skiprows=skiprows)
    interpolate = create_ascii_grid_interpolator(grid, metadata)
    if confirm_creation: 
        print("created interpolator from:", path_to_grid)
    return (interpolate, grid, metadata)

wgs84 = Proj(init="epsg:4326")
gk3 = Proj(init="epsg:3396")
gk5 = Proj(init="epsg:31469")
utm21s = Proj(init="epsg:32721")
utm32n = Proj(init="epsg:25832")

cdict = {}
def create_lat_lon_interpolator_from_csv_coords_file(path_to_csv_coords_file):
    "create interpolator from json list of lat/lon to row/col mappings"
    with open(path_to_csv_coords_file) as _:
        reader = csv.reader(_)
        next(reader)

        points = []
        values = []
        
        for line in reader:
            rowcol = float(line[0])
            row = int(rowcol / 1000)
            col = int(rowcol - row*1000)
            lat = float(line[1])
            lon = float(line[2])
            #alt = float(line[3])
            cdict[(row, col)] = (round(lat, 5), round(lon, 5))
            points.append([lat, lon])
            values.append((row, col))
            #print("row:", row, "col:", col, "clat:", clat, "clon:", clon, "h:", h, "r:", r, "val:", values[i])

        return NearestNDInterpolator(np.array(points), np.array(values))


def geo_coord_to_latlon(geo_coord):

    if not hasattr(geo_coord_to_latlon, "gk_cache"):
        geo_coord_to_latlon.gk_cache = {}
    if not hasattr(geo_coord_to_latlon, "utm_cache"):
        geo_coord_to_latlon.utm_cache = {}

    which = geo_coord.which()
    if which == "gk":
        meridian = geo_coord.gk.meridianNo
        if meridian not in geo_coord_to_latlon.gk_cache:
            geo_coord_to_latlon.gk_cache[meridian] = Proj(init="epsg:" + str(climate_data_capnp.Geo.EPSG["gk" + str(meridian)]))
        lon, lat = transform(geo_coord_to_latlon.gk_cache[meridian], wgs84, geo_coord.gk.r, geo_coord.gk.h)
    elif which == "latlon":
        lat, lon = geo_coord.latlon.lat, geo_coord.latlon.lon
    elif which == "utm":
        utm_id = str(geo_coord.utm.zone) + geo_coord.utm.latitudeBand
        if meridian not in geo_coord_to_latlon.utm_cache:
            geo_coord_to_latlon.utm_cache[utm_id] = \
                Proj(init="epsg:" + str(climate_data_capnp.Geo.EPSG["utm" + utm_id]))
        lon, lat = transform(geo_coord_to_latlon.utm_cache[utm_id], wgs84, geo_coord.utm.r, geo_coord.utm.h)

    return lat, lon


class Station(climate_data_capnp.ClimateData.Station.Server):

    def __init__(self, sim, id, geo_coord, name=None, description=None):
        self.sim = sim
        self.id = id
        self.name = name if name else id
        self.description = description if description else ""
        self.time_series_s = []
        self.geo_coord = geo_coord

    def info(self):
        return common_capnp.Common.IdInformation.new_message(id=self.id, name=self.name, description=self.description) 

    def info_context(self, context): # -> (info :IdInformation);
        context.results.info = self.info()

    def simulationInfo_context(self, context): # -> (simInfo :IdInformation);
        context.results.simInfo = self.sim.info()

    def heightNN_context(self, context): # -> (heightNN :Int32);
        context.results.heightNN = 0

    def geoCoord_context(self, context): # -> (geoCoord :Geo.Coord);
        pass
        #return {"gk": {"meridianNo": 5, "r": 1, "h": 2}}

    def allTimeSeries_context(self, context): # -> (allTimeSeries :List(TimeSeries));
        # get all time series available at this station 
        
        if len(self.time_series_s) == 0:
            for scen in self.sim.scenarios:
                for real in scen.realizations:
                    self.time_series_s.append(real.closest_time_series_at(self.geo_coord))
        
        context.results.init("allTimeSeries", len(self.time_series_s))
        for i, ts in enumerate(self.time_series_s):
            context.result.allTimeSeries[i] = ts


    def timeSeriesFor_context(self, context): # (scenarioId :Text, realizationId :Text) -> (timeSeries :TimeSeries);
        pass


def create_date(capnp_date):
    return date(capnp_date.year, capnp_date.month, capnp_date.day)

def create_capnp_date(py_date):
    return {
        "year": py_date.year if py_date else 0,
        "month": py_date.month if py_date else 0,
        "day": py_date.day if py_date else 0
    }
    
class TimeSeries(climate_data_capnp.ClimateData.TimeSeries.Server): 

    def __init__(self, realization, path_to_csv=None, dataframe=None, headers=None, start_date=None, end_date=None):
        self._path_to_csv = path_to_csv
        self._df = dataframe
        self._data = None
        self._headers = headers
        self._start_date = (date.fromisoformat(self._df.index[0]) if len(self._df) > 0 else None) if not start_date and self._df else start_date
        self._end_date = (date.fromisoformat(self._df.index[-1]) if len(self._df) > 0 else None) if not end_date and self._df else end_date
        self._real = realization

    @property
    def dataframe(self):
        "init underlying dataframe if initialized with path to csv file"
        if not self._df and self._path_to_csv:
            self._df = pd.read_csv(self._path_to_csv, skiprows=[1], index_col=0)
            #self._df = self._df.rename(columns={"windspeed": "wind"})
            if not self._start_date:
                self._start_date = date.fromisoformat(self._df.index[0]) if len(self._df) > 0 else None
            if not self._end_date:
                self._end_date = date.fromisoformat(self._df.index[-1]) if len(self._df) > 0 else None
            if self._start_date and self._end_date:
                self._df = self._df.loc[str(self._start_date):str(self._end_date)]
            if self._headers:
                self._df = self._df.loc[:, self._headers]
        return self._df


    def resolution_context(self, context): # -> (resolution :TimeResolution);
        context.results.resolution = climate_data_capnp.Climate.TimeResolution.daily

    def range_context(self, context): # -> (startDate :Date, endDate :Date);
        context.results.startDate = create_capnp_date(self._start_date)
        context.results.endDate = create_capnp_date(self._end_date)
        
    def header(self, **kwargs): # () -> (header :List(Element));
        return self._df.columns.tolist()

    def data(self, **kwargs): # () -> (data :List(List(Float32)));
        return self._df.to_numpy().tolist()

    def dataT(self, **kwargs): # () -> (data :List(List(Float32)));
        return self._df.T.to_numpy().tolist()
                
    def subrange_context(self, context): # (from :Date, to :Date) -> (timeSeries :TimeSeries);
        from_date = create_date(getattr(context.params, "from"))
        to_date = create_date(context.params.to)

        sub_df = self._df.loc[str(from_date):str(to_date)]

        context.results.timeSeries = TimeSeries(self._real, dataframe=sub_df, start_date=from_date, end_date=to_date)
        
    def subheader_context(self, context): # (elements :List(Element)) -> (timeSeries :TimeSeries);
        sub_headers = [str(e) for e in context.params.elements]
        sub_df = self._df.loc[:, sub_headers]

        context.results.timeSeries = TimeSeries(self._real, dataframe=sub_df, headers=sub_headers)


class Simulation(climate_data_capnp.ClimateData.Simulation.Server): 

    def __init__(self, id, name=None, description=None, scenario_ids=None):
        self._id = id
        self._name = name if name else self._id
        self._description = description if description else ""
        self._scens = None # []
        self._stations = None # []
        self._lat_lon_interpol = None

    @property
    def id(self):
        return self._id

    def info(self):
        return common_capnp.Common.IdInformation.new_message(id=self._id, name=self._name, description=self._description) 

    def info_context(self, context): # -> (info :IdInformation);
        context.results.info = self.info()

    @property    
    def lat_lon_interpolator(self):
        if not self._lat_lon_interpol:
            self._lat_lon_interpol = create_lat_lon_interpolator_from_csv_coords_file("macsur_european_climate_scenarios_geo_coords_and_altitude.csv")
        return self._lat_lon_interpol

    @property
    def scenarios(self):
        if not self._scens:
            self._scens = Scenario.create_scenarios(self, self._path_to_sim_dir)
        return self._scens

    @scenarios.setter
    def scenarios(self, scens):
        self._scens = scens

    def scenarios_context(self, context): # -> (scenarios :List(Scenario));
        context.results.init("scenarios", len(self.scenarios))
        for i, scen in enumerate(self.scenarios):
            context.results.scenarios[i] = scen
        

    @property
    def stations(self):
        pass

    def stations_context(self, context): # -> (stations :List(Station));
        pass
        



class Scenario(climate_data_capnp.ClimateData.Scenario.Server):

    def __init__(self, sim, id, reals=[], name=None, description=None):
        self._sim = sim
        self._id = id
        self._name = name if name else self._id
        self._description = description if description else ""
        self._reals = reals

    def info(self):
        return common_capnp.Common.IdInformation.new_message(id=self._id, name=self._name, description=self._description) 

    def info_context(self, context): # -> (info :IdInformation);
        context.results.info = self.info()
        
    @property
    def simulation(self):
        return self._sim    

    def simulation_context(self, context): # -> (simulation :Simulation);
        context.results.simulation = self._sim
        
    @property
    def realizations(self):
        return self._reals

    @realizations.setter
    def realizations(self, reals):
        self._reals = reals

    def realizations_context(self, context): # -> (realizations :List(Realization));
        context.results.init("realizations", len(self.realizations))
        for i, real in enumerate(self.realizations):
            context.results.realizations[i] = real


class Realization(climate_data_capnp.ClimateData.Realization.Server):

    def __init__(self, scen, create_paths_to_csv, id=None, name=None, description=None):
        self._scen = scen
        self._create_paths_to_csv = create_paths_to_csv
        self._id = id if id else "1"
        self._name = name if name else self._id
        self._description = description if description else ""

    def info(self):
        return common_capnp.Common.IdInformation.new_message(id=self._id, name=self._name, description=self._description) 

    def info_context(self, context): # -> (info :IdInformation);
        context.results.info = self.info()

    @property
    def scenario(self):
        return self._scen

    def scenario_context(self, context): # -> (scenario :Scenario);
        context.results.scenario = self._scen
        
    def closest_time_series_at(self, geo_coord):

        lat, lon = geo_coord_to_latlon(geo_coord)

        interpol = self.scenario.simulation.lat_lon_interpolator
        row, col = interpol(lat, lon)

        closest_time_series = [TimeSeries(self, path_to_csv) for path_to_csv in self._create_paths_to_csv(row, col)]

        return closest_time_series


    def closestTimeSeriesAt_context(self, context): # (geoCoord :Geo.Coord) -> (timeSeries :List(TimeSeries));
        # closest TimeSeries object which represents the whole time series 
        # of the climate realization at the give climate coordinate

        context.results.timeSeries = self.closest_time_series_at(context.params.geoCoord)



class Service(climate_data_capnp.ClimateData.Service.Server):

    def __init__(self, id=None, name=None, description=None):
        self._id = id if id else "macsur_european_climate_scenarios_v2"
        self._name = name if name else "MACSUR European Climate Scenarios V2"
        self._description = description if description else ""
        self._sims = create_simulations()

    def info(self):
        return common_capnp.Common.IdInformation.new_message(id=self._id, name=self._name, description=self._description) 

    def info_context(self, context): # -> (info :IdInformation);
        context.results.info = self.info()

    def getAvailableSimulations_context(self, context): # getAvailableSimulations @0 () -> (availableSimulations :List(Simulation));
        context.results.availableSimulations = self._sims
        #context.results.init("realizations", len(self._realizations))
        #for i, real in enumerate(self.realizations):
        #    context.results.realizations[i] = real
        

    def getSimulation(self, id, **kwargs): # getSimulation @1 (id :UInt64) -> (simulation :Simulation);
        for sim in self._sims:
            if sim.id == id:
                return sim




def create_simulations():
    sims_and_scens = {
        "GFDL-CM3": ["45", "85"],
        "GISS-E2-R": ["45", "85"],
        "HadGEM2-ES": ["26", "45", "85"],
        "MIROC5": ["45", "85"],
        "MPI-ESM-MR": ["26", "45", "85"]
    }

    path_template = "/beegfs/common/data/climate/macsur_european_climate_scenarios_v2/transformed/{period_id}/{sim_id}_{scen_id}/{row}_{col}_{version}.csv"

    def create_paths_to_time_series_csvs(path_template, sim_id, scen_id, version, row, col):
        return [
            path_template.format(sim_id=sim_id, scen_id=scen_id, version="v2", period_id="0", row=row, col=col),
            path_template.format(sim_id=sim_id, scen_id=scen_id, version="v2", period_id="2", row=row, col=col),
            path_template.format(sim_id=sim_id, scen_id=scen_id, version="v2", period_id="3", row=row, col=col)
        ]

    sims = []
    for sim_id, scen_ids in sims_and_scens.items():
        sim = Simulation(sim_id, name=sim_id)
        scens = []
        for scen_id in scen_ids:
            scen = Scenario(sim, scen_id, name=scen_id)
            real = Realization(scen, lambda row, col: create_paths_to_time_series_csvs(path_template, sim_id, scen_id, "v2", row, col))
            scen.realizations = [real]
            scens.append(scen)
        sim.scenarios = scens
        sims.append(sim)
    return sims


def main():
    #address = parse_args().address

    #server = capnp.TwoPartyServer("*:8000", bootstrap=DataServiceImpl("/home/berg/archive/data/"))
    server = capnp.TwoPartyServer("*:11001", bootstrap=Service())
    server.run_forever()

if __name__ == '__main__':
    main()