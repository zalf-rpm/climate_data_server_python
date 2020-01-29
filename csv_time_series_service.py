import sys
import os
from datetime import date, timedelta
import numpy
import pandas as pd
import json
import time

#import argparse
#import common
import capnp
capnp.add_import_hook(additional_paths=["../capnproto_schemas/", "../capnproto_schemas/capnp_schemas/"])
import common_capnp
import climate_data_capnp


def create_date(capnp_date):
    return date(capnp_date.year, capnp_date.month, capnp_date.day)


def create_capnp_date(py_date):
    return {
        "year": py_date.year if py_date else 0,
        "month": py_date.month if py_date else 0,
        "day": py_date.day if py_date else 0
    }


class CSV_TimeSeries(climate_data_capnp.ClimateData.TimeSeries.Server):

    def __init__(self, dataframe=None, path_to_csv_file=None, headers=None, start_date=None, end_date=None):
        if path_to_csv_file:
            dataframe = pd.read_csv(path_to_csv_file, skiprows=[1], index_col=0, delimiter=";")
        elif not dataframe:
            return
        self._df = dataframe.rename(columns={"windspeed": "wind"})
        self._data = None
        self._headers = headers
        self._start_date = start_date if start_date else (date.fromisoformat(dataframe.index[0]) if len(dataframe) > 0 else None)
        self._end_date = end_date if end_date else (date.fromisoformat(dataframe.index[-1]) if len(dataframe) > 0 else None)
        #self._real = realization

    def resolution_context(self, context): # -> (resolution :TimeResolution);
        context.results.resolution = climate_data_capnp.ClimateData.TimeResolution.daily

    def range_context(self, context): # -> (startDate :Date, endDate :Date);
        context.results.startDate = create_capnp_date(self._start_date)
        context.results.endDate = create_capnp_date(self._end_date)

    def header(self, **kwargs): # () -> (header :List(Element));
        return self._df.columns.tolist()

    def data(self, **kwargs): # () -> (data :List(List(Float32)));
        print("data requested")
        return [[x for x in range(1,8)] for _ in range(1,330+1)]
        return self._df.to_numpy().tolist()

    def dataT(self, **kwargs): # () -> (data :List(List(Float32)));
        print("dataT requested")
        return self._df.T.to_numpy().tolist()

    # (from :Date, to :Date) -> (timeSeries :TimeSeries);
    def subrange_context(self, context):
        from_date = create_date(getattr(context.params, "from"))
        to_date = create_date(context.params.to)

        sub_df = self._df.loc[str(from_date):str(to_date)]

        context.results.timeSeries = CSV_TimeSeries(dataframe=sub_df, headers=self._headers,
                                                    start_date=from_date, end_date=to_date)

    # (elements :List(Element)) -> (timeSeries :TimeSeries);
    def subheader_context(self, context):
        sub_headers = [str(e) for e in context.params.elements]
        sub_df = self._df.loc[:, sub_headers]

        context.results.timeSeries = CSV_TimeSeries(dataframe=sub_df, headers=sub_headers,
                                                    start_date=self._start_date, end_date=self._end_date)

class List_Tests(climate_data_capnp.ClimateData.ListTests.Server):

    def __init__(self):
        self.lf32 = list([1 for x in range(1000)])
        print("lf32 done")
        self.llf32 = list([[x for x in range(1,8)] for _ in range(1000)])
        print("llf32 done")
        self.li32 = list([1 for x in range(2895)])
        print("li32 done")
        self.lli32 = list([[x for x in range(1,8)] for _ in range(1000)])
        print("lli32 done")
        self.i = 0

    def testLF32(self, **kwargs): # () -> (test1 :List(Float32));
        return self.lf32

    def testLLF32(self, **kwargs): # () -> (test1 :List(List(Float32)));
        return self.llf32

    def testLI32(self, **kwargs): # () -> (test1 :List(Int32));
        self.i = self.i + 1
        return [1 for x in range(self.i)]
        #return self.li32

    def testLLI32(self, **kwargs): # () -> (test1 :List(List(Int32)));
        return self.lli32



#class CSV_TimeSeries_CH(climate_data_capnp.ClimateData.Test.Server):

#    def __init__(self, dataframe=None, path_to_csv_file=None, headers=None, start_date=None, end_date=None):
#        self._ts = CSV_TimeSeries(dataframe=dataframe, path_to_csv_file=path_to_csv_file,
#                                  headers=headers, start_date=start_date, end_date=end_date)

#    def timeSeries(self, **kwargs):
#        print("dataT requested")
#        return common.CapHolderImpl(self._ts, "sturdy ref", lambda: print("cleanup called"))


def main():

    config = {
        "port": "11002",
        "server": "*",
        "path_to_csv_file": "climate-iso.csv"
    }
    # read commandline args only if script is invoked directly from commandline
    if len(sys.argv) > 1 and __name__ == "__main__":
        for arg in sys.argv[1:]:
            k, v = arg.split("=")
            if k in config:
                config[k] = v

    server = capnp.TwoPartyServer(config["server"] + ":" + config["port"],
                                  bootstrap=List_Tests())
                                  #bootstrap=CSV_TimeSeries_CH(path_to_csv_file=config["path_to_csv_file"]))
                                  #bootstrap=CSV_TimeSeries(path_to_csv_file=config["path_to_csv_file"]))
    server.run_forever()


if __name__ == '__main__':
    main()
