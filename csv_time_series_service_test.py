from multiprocessing import Process
import time
from datetime import date, timedelta
import csv_time_series_service as ts_service
import pytest
import capnp
capnp.add_import_hook(additional_paths=["../capnproto_schemas"])
import climate_data_capnp


@pytest.fixture(scope="module")
def start_time_series_service():
    p = Process(target = ts_service.main, kwargs={"port": 6666})
    p.start()
    time.sleep(0.1)
    yield
    p.terminate()

@pytest.fixture(scope="module")
def time_series_cap(start_time_series_service):
    csv_time_series = capnp.TwoPartyClient("localhost:6666").bootstrap().cast_as(climate_data_capnp.Climate.TimeSeries)
    return csv_time_series


def test_header(time_series_cap):
    header = time_series_cap.header().wait().header
    assert len(header) == 7
    assert list(header) == ["tavg", "tmin", "tmax", "wind", "globrad", "precip", "relhumid"]

def test_subheader(time_series_cap):
    ts = time_series_cap.subheader(["tavg", "precip", "wind"]).wait().timeSeries
    header = ts.header().wait().header
    assert len(header) == 3
    assert list(header) == ["tavg", "precip", "wind"]

def test_range(time_series_cap):
    r = time_series_cap.range().wait()
    assert ts_service.create_date(r.startDate) == date(1991, 1, 1)
    assert ts_service.create_date(r.endDate) == date(1997, 12, 31)

def test_subrange(time_series_cap):
    ts = time_series_cap.subrange({"year": 1992, "month": 2, "day": 3}, {"year": 1993, "month": 11, "day": 25}).wait().timeSeries
    r = ts.range().wait()
    assert ts_service.create_date(r.startDate) == date(1992, 2, 3)
    assert ts_service.create_date(r.endDate) == date(1993, 11, 25)

    ts2 = time_series_cap.subrange({"year": 1991, "month": 2, "day": 1}, {"year": 1991, "month": 2, "day": 20}).wait().timeSeries
    data2 = ts2.data().wait().data
    assert len(data2) == 20
    
    data0 = time_series_cap.data().wait().data
    assert list(data0[31]) == list(data2[0])
   
def test_resolution(time_series_cap):
    r = time_series_cap.resolution().wait().resolution
    assert str(r) == "daily"

#def test_location(time_series_cap):
#    l = time_series_cap.location().wait()
#    assert l.timeSeries == time_series_cap

def test_data(time_series_cap):
    r = time_series_cap.range().wait()
    timedelta = ts_service.create_date(r.endDate) - ts_service.create_date(r.startDate)
    d = time_series_cap.data().wait().data
    assert len(d) == timedelta.days + 1

def test_dataT(time_series_cap):
    d = time_series_cap.data().wait().data
    dt = time_series_cap.dataT().wait().data
    d0 = list(d[0])
    dt0 = list(dt[i][0] for i in range(7))
    assert d0 == dt0

    dx = list(d[-1])
    dtx = list(dt[i][-1] for i in range(7))
    assert dx == dtx


