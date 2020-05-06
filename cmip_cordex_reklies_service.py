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
def create_lat_lon_interpolator_from_json_coords_file(path_to_json_coords_file):
    "create interpolator from json list of lat/lon to row/col mappings"
    with open(path_to_json_coords_file) as _:
        data = json.load(_)
        
        points = []
        values = []
        
        for latlon, rowcol in json.load(_):
            row, col = rowcol
            lat, lon = latlon
            #alt = float(line[3])
            cdict[(row, col)] = {"lat": round(lat, 5), "lon": round(lon, 5), "alt": -1}
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

PATH_TO_LATLON_TO_ROWCOL_JSON_FILE = None
def lat_lon_interpolator():
    "create an interpolator for the macsur grid"
    if not hasattr(lat_lon_interpolator, "interpol"):
        lat_lon_interpolator.interpol = create_lat_lon_interpolator_from_json_coords_file(PATH_TO_LATLON_TO_ROWCOL_JSON_FILE)
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


class Dataset(climate_data2_capnp.Climate.Dataset.Server):

    def __init__(self, metadata, path_to_rows):
        self._meta = metadata
        self._paths_to_rows = path_to_rows

    def closest_time_series_at(self, lat, lon):
        row, col = lat_lon_interpolator()(lat, lon)
        path_to_csv = self._path_to_rows + "/row-" + str(row) + "/col-" + str(col) + ".csv"
        closest_timeseries = TimeSeries.from_csv_file(self, path_to_csv)
        return closest_time_series

    def closestTimeSeriesAt(self, geoCoord, **kwargs): # (geoCoord :Geo.Coord) -> (timeSeries :List(TimeSeries));
        # closest TimeSeries object which represents the whole time series 
        # of the climate realization at the give climate coordinate
        lat, lon = geo_coord_to_latlon(geoCoord)
        return self.closest_time_series_at(lat, lon)

    def stations(self, **kwargs): # () -> (stations :List(Station));
        return list([Station(self, "[r:{}/c:{}]".format(row_col[0], row_col[1]), coord) for row_col, coord in cdict.items()])


def string_to_gcm(gcm_str):
    return {
        "CCCma-CanESM2": climate_data2_capnp.Climate.GCM.cccmaCanEsm2,
        "ICHEC-EC-EARTH": climate_data2_capnp.Climate.GCM.ichecEcEarth,
        "IPSL-IPSL-CM5A-MR": climate_data2_capnp.Climate.GCM.ipslIpslCm5AMr,
        "MIROC-MIROC5": climate_data2_capnp.Climate.GCM.mirocMiroc5,
        "MPI-M-MPI-ESM-LR": climate_data2_capnp.Climate.GCM.mpiMMpiEsmLr
    }.get(gcm_str, None)


def gcm_to_str(gcm):
    return {
        climate_data2_capnp.Climate.GCM.cccmaCanEsm2: {"id": "CCCma-CanESM2", "name": "CCCma-CanESM2"},
        : climate_data2_capnp.Climate.GCM.ichecEcEarth: {"id": "ICHEC-EC-EARTH", "name": "ICHEC-EC-EARTH"},
        : climate_data2_capnp.Climate.GCM.ipslIpslCm5AMr: {"id": "IPSL-IPSL-CM5A-MR", "name": "IPSL-IPSL-CM5A-MR"},
        : climate_data2_capnp.Climate.GCM.mirocMiroc5: {"id": "MIROC-MIROC5", "name": "MIROC-MIROC5"},
        : climate_data2_capnp.Climate.GCM.mpiMMpiEsmLr: {"id": "MPI-M-MPI-ESM-LR", "name": "MPI-M-MPI-ESM-LR"}
    }.get(gcm, None)


def string_to_rcm(rcm_str):
    return {
        "CLMcom-CCLM4-8-17": climate_data2_capnp.Climate.RCM.clmcomCclm4817,
        "GERICS-REMO2015": climate_data2_capnp.Climate.RCM.gericsRemo2015,
        "KNMI-RACMO22E": climate_data2_capnp.Climate.RCM.knmiRacmo22E,
        "SMHI-RCA4": climate_data2_capnp.Climate.RCM.smhiRca4,
        "CLMcom-BTU-CCLM4-8-17": climate_data2_capnp.Climate.RCM.clmcomBtuCclm4817,
        "MPI-CSC-REMO2009": climate_data2_capnp.Climate.RCM.mpiCscRemo2009
    }.get(rcm_str, None)


def string_to_ensmem(ensmem_str):
    # split r<N>i<M>p<L>

    sr, sip = ensmem_str[1:].split("i")
    si, sp = sip.split("p")
    return {"real": int(sr), "init": int(si), "pert": int(sp)}


class Metadata_Info(climate_data2_capnp.Climate.Metadata.Info.Server):

    def __init__(self, metadata):
        self._meta = metadata

    def for(self, entry, **kwargs): # for @0 (entry :Entry) -> Common.IdInformation;
        pass
        #return common_capnp.Common.IdInformation.new_message(id=self._rcm_str, name=self._rcm_str, description=self._rcm_str) 

    def all(self, **kwargs): # all @0 () -> :List(Common.IdInformation);
        pass
        #return common_capnp.Common.IdInformation.new_message(id=self._rcm_str, name=self._rcm_str, description=self._rcm_str) 


def create_metadatasets(path_to_data_dir):

    metadatasets = []
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
                                                "entries": [
                                                    {"gcm": string_to_gcm(gcm)},
                                                    {"rcm": string_to_rcm(rcm)},
                                                    {"historical": None} if scen == "historical" else {"rcp": scen},
                                                    {"ensMem": string_to_ensmem(ensmem)},
                                                    {"version": version}
                                                ],
                                                "dataset": None
                                            }
                                            metadata["info"] = Metadata_Info(metadata)
                                            metadata["dataset"] = Dataset(metadata, version_dir)
                                            metadatasets.append(metadata)
    return metadatasets


class Service(climate_data2_capnp.Climate.Service.Server):

    def __init__(self, path_to_data_dir, id=None, name=None, description=None):
        self._id = id if id else "cmip_cordex_reklies"
        self._name = name if name else "CMIP Cordex Reklies"
        self._description = description if description else ""
        self._metadatasets = create_metadatasets(path_to_data_dir)

    def info(self):
        return {"id": self._id, "name": self._name, "description": self._description} #common_capnp.Common.IdInformation.new_message(id=self._id, name=self._name, description=self._description) 

    def info_context(self, context): # -> (info :IdInformation);
        context.results.info = self.info()

    def getAvailableMetadatasets(self, **kwargs): # getAvailableMetadatasets @0 () -> (metadatasets :List(Metadata));
        return self._metadatasets

    def getDataset(self, metadata, **kwargs): # getDataset @1 (metadata :Metadata) -> (dataset :Dataset);
        access_entries = {
            "gcm": lambda e: e.gcm,
            "rcm": lambda e: e.rcm,
            "historical": lambda e: true,
            "rcp": lambda e: e.rcp,
            "ssp": lambda e: e.ssp,
            "ensMem": lambda e: e.ensMem,
            "version": lambda e: e.version,
            "start": lambda e: create_date(e.start),
            "end": lambda e: create_date(e.end)
        }

        search_entry_to_value = {}
        for e in metadata.entries:
            which = e.which()
            entry_dict[which] = access_entries[which](e)

        def contains_search_entries(mds):
            for e in mds.entries:
                which = e.which()
                if search_entry_to_value[which] != access_entries[which](e):
                    return False
            return True

        metadatasets = filter(contains_search_entries, self._metadatasets)
        return metadatasets




def main():
    #address = parse_args().address

    #server = capnp.TwoPartyServer("*:8000", bootstrap=DataServiceImpl("/home/berg/archive/data/"))
    path_to_data = "/beegfs/common/data/climate/dwd/cmip_cordex_reklies/"
    PATH_TO_LATLON_TO_ROWCOL_JSON_FILE = path_to_data + "latlon-to-rowcol.json"
    server = capnp.TwoPartyServer("*:11001", bootstrap=Service(path_to_data + "csvs/"))
    server.run_forever()

if __name__ == '__main__':
    main()