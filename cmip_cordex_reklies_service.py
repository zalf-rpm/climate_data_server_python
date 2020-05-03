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

#import ptvsd
#ptvsd.enable_attach(("0.0.0.0", 14000))
#ptvsd.wait_for_attach()  # blocks execution until debugger is attached

import capnp
capnp.add_import_hook(additional_paths=["../capnproto_schemas/", "../capnproto_schemas/capnp_schemas/"])
import common_capnp
#import model_capnp
import geo_coord_capnp
import climate_data2_capnp

ADAPT_TIMESERIES_TO_FUTURE_DATES = True

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
            alt = float(line[3])
            cdict[(row, col)] = {"lat": round(lat, 5), "lon": round(lon, 5), "alt": alt}
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


def lat_lon_interpolator():
    "create an interpolator for the macsur grid"
    if not hasattr(lat_lon_interpolator, "interpol"):
        lat_lon_interpolator.interpol = create_lat_lon_interpolator_from_csv_coords_file("macsur_european_climate_scenarios_geo_coords_and_altitude.csv")
    return lat_lon_interpolator.interpol


class Station(climate_data_capnp.ClimateData.Station.Server):

    def __init__(self, sim, id, geo_coord, name=None, description=None):
        self._sim = sim
        self._id = id
        self._name = name if name else id
        self._description = description if description else ""
        self._time_series = []
        self._geo_coord = geo_coord

    def info(self, **kwargs): # () -> (info :IdInformation);
        return common_capnp.Common.IdInformation.new_message(id=self._id, name=self._name, description=self._description) 

    def simulationInfo(self, **kwargs): # () -> (simInfo :IdInformation);
        return self._sim.info()

    def heightNN(self, **kwargs): # () -> (heightNN :Int32);
        return self._geo_coord["alt"]

    def geoCoord(self, **kwargs): # () -> (geoCoord :Geo.Coord);
        coord = geo_coord_capnp.Geo.Coord.new_message()
        coord.init("latlon")
        coord.latlon.lat = self._geo_coord["lat"]
        coord.latlon.lon = self._geo_coord["lon"]
        return coord
        #return {"gk": {"meridianNo": 5, "r": 1, "h": 2}}

    def allTimeSeries(self, **kwargs): # () -> (allTimeSeries :List(TimeSeries));
        # get all time series available at this station 
        
        if len(self._time_series) == 0:
            for scen in self._sim.scenarios:
                for real in scen.realizations:
                    for ts in real.closest_time_series_at(self._geo_coord["lat"], self._geo_coord["lon"]):
                        self._time_series.append(ts)
        
        return self._time_series

    def timeSeriesFor(self, scenarioId, realizationId, **kwargs): # (scenarioId :Text, realizationId :Text) -> (timeSeries :TimeSeries);
        # get all time series for a given scenario and realization at this station
        return list(filter(lambda ts: ts.scenarioInfo().id == scenarioId and ts.realizationInfo().id == realizationId, self.allTimeSeries()))


def create_date(capnp_date):
    return date(capnp_date.year, capnp_date.month, capnp_date.day)

def create_capnp_date(py_date):
    return {
        "year": py_date.year if py_date else 0,
        "month": py_date.month if py_date else 0,
        "day": py_date.day if py_date else 0
    }
    
class TimeSeries(climate_data2_capnp.Climate.TimeSeries.Server): 

    def __init__(self, metadata, path_to_csv=None, dataframe=None):
        "a supplied dataframe asumes the correct index is already set (when reading from csv then it will always be 1980 to 2010)"

        if not path_to_csv and not dataframe:
            raise Exception("Missing argument, either path_to_csv or dataframe have to be supplied!")

        self._path_to_csv = path_to_csv
        self._df = dataframe
        self._meta = metadata

    @classmethod
    def from_csv_file(cls, metadata, path_to_csv):
        return TimeSeries(metadata, path_to_csv)

    @classmethod
    def from_dataframe(cls, metadata, dataframe):
        return TimeSeries(metadata)

    @property
    def dataframe(self):
        "init underlying dataframe lazily if initialized with path to csv file"
        if self._df is None and self._path_to_csv:
            # load csv file
            self._df = pd.read_csv(self._path_to_csv, skiprows=[1], index_col=0)

            # reduce headers to the supported ones
            all_supported_headers = ["tmin", "tavg", "tmax", "precip", "globrad", "wind", "relhumid"]
            self._df = self._df.loc[:, all_supported_headers]

        return self._df

    def resolution_context(self, context): # -> (resolution :TimeResolution);
        context.results.resolution = climate_data2_capnp.Climate.TimeSeries.Resolution.daily

    def range_context(self, context): # -> (startDate :Date, endDate :Date);
        context.results.startDate = create_capnp_date(date.fromisoformat(str(self.dataframe.index[0])[:10]))
        context.results.endDate = create_capnp_date(date.fromisoformat(str(self.dataframe.index[-1])[:10]))
        
    def header(self, **kwargs): # () -> (header :List(Element));
        return self.dataframe.columns.tolist()

    def data(self, **kwargs): # () -> (data :List(List(Float32)));
        return self.dataframe.to_numpy().tolist()

    def dataT(self, **kwargs): # () -> (data :List(List(Float32)));
        return self.dataframe.T.to_numpy().tolist()
                
    def subrange(self, from, to, **kwargs): # (from :Date, to :Date) -> (timeSeries :TimeSeries);
        from_date = create_date(from)
        to_date = create_date(to)

        sub_df = self._df.loc[str(from_date):str(to_date)]

        return TimeSeries.from_dataframe(self._real, sub_df)

    def subheader(self, elements, **kwargs): # (elements :List(Element)) -> (timeSeries :TimeSeries);
        sub_headers = [str(e) for e in elements]
        sub_df = self.dataframe.loc[:, sub_headers]

        return TimeSeries.from_dataframew(self._real, sub_df)

    def metadata(self, **kwargs): # metadata @7 () -> Metadata;
        "the metadata for this time series"
        return self._metadata


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

    def info_context(self, context): # () -> (info :IdInformation);
        context.results.info = self.info()

    @property
    def scenarios(self):
        if not self._scens:
            self._scens = Scenario.create_scenarios(self, self._path_to_sim_dir)
        return self._scens

    @scenarios.setter
    def scenarios(self, scens):
        self._scens = scens

    def scenarios_context(self, context): # () -> (scenarios :List(Scenario));
        context.results.init("scenarios", len(self.scenarios))
        for i, scen in enumerate(self.scenarios):
            context.results.scenarios[i] = scen

    def stations(self, **kwargs): # () -> (stations :List(Station));
        return list([Station(self, "[r:{}/c:{}]".format(row_col[0], row_col[1]), coord) for row_col, coord in cdict.items()])


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
        
    def simulationInfo(self, **kwargs): # () -> (simulationInfo :Common.IdInformation);
        return self._sim.info()

    @property
    def simulation(self):
        return self._sim

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

    def __init__(self, scen, paths_to_csv_config, id=None, name=None, description=None, adapt_timeseries_to_future_dates=True):
        self._scen = scen
        self._paths_to_csv_config = paths_to_csv_config
        self._id = id if id else "1"
        self._name = name if name else self._id
        self._description = description if description else ""
        self._adapt_timeseries_to_future_dates = adapt_timeseries_to_future_dates

    def info(self):
        return common_capnp.Common.IdInformation.new_message(id=self._id, name=self._name, description=self._description) 

    def info_context(self, context): # -> (info :IdInformation);
        context.results.info = self.info()

    def scenarioInfo(self, **kwargs): # () -> (scenarioInfo :Common.IdInformation);
        return self._scen.info()
        
    @property
    def scenario(self):
        return self._scen

    def closest_time_series_at(self, lat, lon):

        row, col = lat_lon_interpolator()(lat, lon)

        c = self._paths_to_csv_config
        closest_time_series = []
        for time_range in c["time_ranges"]:
            formated_path = c["path_template"].format(sim_id=c["sim_id"], scen_id=c["scen_id"], version=c["version_id"], period_id=time_range, row=row, col=col)
            closest_time_series.append(TimeSeries.from_csv_file(self, formated_path, time_range, self._adapt_timeseries_to_future_dates))
            
        return closest_time_series


    def closestTimeSeriesAt(self, geoCoord, **kwargs): # (geoCoord :Geo.Coord) -> (timeSeries :List(TimeSeries));
        # closest TimeSeries object which represents the whole time series 
        # of the climate realization at the give climate coordinate
        lat, lon = geo_coord_to_latlon(geoCoord)
        return self.closest_time_series_at(lat, lon)


class Service(climate_data2_capnp.Climate.Service.Server):

    def __init__(self, path_to_data_dir, id=None, name=None, description=None):
        self._id = id if id else "cmip_cordex_reklies"
        self._name = name if name else "CMIP Cordex Reklies"
        self._description = description if description else ""
        self._metadatasets = create_metadatasets(path_to_data_dir)

    def info(self):
        return common_capnp.Common.IdInformation.new_message(id=self._id, name=self._name, description=self._description) 

    def info_context(self, context): # -> (info :IdInformation);
        context.results.info = self.info()

    def getAvailableMetadatasets(**kwargs): # getAvailableMetadatasets @0 () -> (metadatasets :List(Metadata));
        return self._metadatasets

    def getDataset(metadata, **kwargs): # getDataset @1 (metadata :Metadata) -> (dataset :Dataset);
        for mds in self._metadatasets:
            if mds == metadata:
                return mds


class GCM(climate_data2_capnp.Climate.GCM.Info.Server):

    def __init__(self, gcm_str):
        self._gcm_str = gcm_str

    def gcm(self, **kwargs): # gcm @0 () -> Common.IdInformation;
        return common_capnp.Common.IdInformation.new_message(id=self._gcm_str, name=self._gcm_str, description=self._gcm_str) 

    def gcmDict(self):
        gcm_enum = string_to_gcm(self._gcm_str)
        if gcm_enum == climate_data2_capnp.Climate.GCM.unknown:
            return {"unknown": None, "info": self}
        else: 
            return {"gcm": gcm_enum, "info": self}

    @classmethod
    def string_to_gcm(cls, gcm_str):
        return {
            "CCCma-CanESM2": climate_data2_capnp.Climate.GCM.cccmaCanEsm2,
            "ICHEC-EC-EARTH": climate_data2_capnp.Climate.GCM.ichecEcEarth,
            "IPSL-IPSL-CM5A-MR": climate_data2_capnp.Climate.GCM.ipslIpslCm5AMr,
            "MIROC-MIROC5": climate_data2_capnp.Climate.GCM.mirocMiroc5,
            "MPI-M-MPI-ESM-LR": climate_data2_capnp.Climate.GCM.mpiMMpiEsmLr
        }[gcm_str]


class RCM(climate_data2_capnp.Climate.RCM.Info.Server):

    def __init__(self, rcm_str):
        self._rcm_str = rcm_str

    def rcm(self, **kwargs): # rcm @0 () -> Common.IdInformation;
        return common_capnp.Common.IdInformation.new_message(id=self._rcm_str, name=self._rcm_str, description=self._rcm_str) 

    def rcmDict(self):
        rcm_enum = string_to_rcm(self._rcm_str)
        if rcm_enum == climate_data2_capnp.Climate.RCM.unknown:
            return {"unknown": None, "info": self}
        else: 
            return {"rcm": rcm_enum, "info": self}

    @classmethod
    def string_to_rcm(cls, rcm_str):
        return {
            "CLMcom-CCLM4-8-17": climate_data2_capnp.Climate.RCM.clmcomCclm4817,
            "GERICS-REMO2015": climate_data2_capnp.Climate.RCM.gericsRemo2015,
            "KNMI-RACMO22E": climate_data2_capnp.Climate.RCM.knmiRacmo22E,
            "SMHI-RCA4": climate_data2_capnp.Climate.RCM.smhiRca4,
            "CLMcom-BTU-CCLM4-8-17": climate_data2_capnp.Climate.RCM.clmcomBtuCclm4817,
            "MPI-CSC-REMO2009": climate_data2_capnp.Climate.RCM.mpiCscRemo2009
        }.get(rcm_str, climate_data2_capnp.Climate.RCM.unknown)


def create_metadatasets(path_to_data_dir):

     for gcm in os.listdir(path_to_data_dir):
        gcm_dir = path_to_data_dir + gcm
        if os.path.isdir(gcm_dir):
            for rcm in os.listdir(gcm_dir):
                rcm_dir = gcm_dir + "/" + rcm
                if os.path.isdir(rcm_dir):
                    for scen in os.listdir(rcm_dir):
                        scen_dir = rcm_dir + "/" + scen
                        if os.path.isdir(scen_dir):
                            for ensmem in os.listdir(scen_dir):
                                ensmem_dir = scen_dir + "/" + ensmem
                                if os.path.isdir(ensmem_dir):
                                    for version in os.listdir(ensmem_dir):
                                        version_dir = ensmem_dir + "/" + version
                                        if os.path.isdir(version_dir):
                                            metadata = {
                                                "gcm": GCM(gcm).gcmDict,
                                                "rcm": RCM(rcm).rcmDict,
        


                sims.append(cls(dirname.lower(), path_to_isimip_dir + dirname + "/", name=dirname))


    sims_and_scens = {
        "0": {"scens": ["0"], "tranges": ["0"]},
        "GFDL-CM3": {"scens": ["45", "85"], "tranges": ["2", "3"]},
        "GISS-E2-R": {"scens": ["45", "85"], "tranges": ["2", "3"]},
        "HadGEM2-ES": {"scens": ["26", "45", "85"], "tranges": ["2", "3"]},
        "MIROC5": {"scens": ["45", "85"], "tranges": ["2", "3"]},
        "MPI-ESM-MR": {"scens": ["26", "45", "85"], "tranges": ["2", "3"]}
    }

    path_template = "/beegfs/common/data/climate/macsur_european_climate_scenarios_v2/transformed/{period_id}/{sim_id}_{scen_id}/{row}_{col:03d}_{version}.csv"

    sims = []
    for sim_id, scens_and_tranges in sims_and_scens.items():
        sim = Simulation(sim_id, name=sim_id)
        scens = []
        tranges = scens_and_tranges["tranges"]
        for scen_id in scens_and_tranges["scens"]:
            scen = Scenario(sim, scen_id, name=scen_id)
            real = Realization(scen, {
                "path_template": path_template, 
                "time_ranges": tranges, 
                "sim_id": sim_id, 
                "scen_id": scen_id,
                "version_id": "v2"
            }, adapt_timeseries_to_future_dates=adapt_timeseries_to_future_dates)
            scen.realizations = [real]
            scens.append(scen)
        sim.scenarios = scens
        sims.append(sim)
    return sims


def main():
    #address = parse_args().address

    #server = capnp.TwoPartyServer("*:8000", bootstrap=DataServiceImpl("/home/berg/archive/data/"))
    server = capnp.TwoPartyServer("*:11001", bootstrap=Service("/beegfs/common/data/climate/dwd/cmip_cordex_reklies/csvs/"))
    server.run_forever()

if __name__ == '__main__':
    main()