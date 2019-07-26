#from scipy.interpolate import NearestNDInterpolator
#import numpy as np
import sys
import os
from datetime import date, timedelta
#import pandas as pd
#from pyproj import Proj, transform
import json
import time
import uuid

import capnp
#import climate_data_capnp
capnp.remove_import_hook()
#climate_data_capnp = capnp.load('capnproto_schemas/climate_data.capnp')
#model_capnp = capnp.load('capnproto_schemas/model.capnp')
cluster_admin_service_capnp = capnp.load("capnproto_schemas/cluster_admin_service.capnp")
#common_capnp = capnp.load('capnproto_schemas/common.capnp')


class SlurmRuntime(cluster_admin_service_capnp.Cluster.Runtime.Server):

    def __init__(self, cores, admin_master):
        self._admin_master = admin_master
        self._cores = cores
        self._used_cores = 0
        self._uuid4 = uuid.uuid4()
        self._factories = {}

    def info_context(self, context): # info @0 () -> (info :IdInformation);
        # interface to retrieve id information from an object
        return {"id": str(self._uuid4), "name": "SlurmRuntime(" + str(self._uuid4) + ")", "description": ""}

    def registerModelInstanceFactory_context(self, context): # registerModelInstanceFactory @0 (aModelId :Text, aFactory :ModelInstanceFactory);
        "register a model instance factory for the given model id"
        self._factories[context.params.aModelId] = context.params.aFactory

    def availableModels_context(self, context): # availableModels @1 () -> (factories :List(ModelInstanceFactory));
        "# the model instance factories this runtime has access to"
        pass

    def numberOfCores_context(self, context): # numberOfCores @2 () -> (cores :Int16);
        "# how many cores does the runtime offer"
        return self._cores 

    def freeNumberOfCores_context(self, context): # freeNumberOfCores @3 () -> (cores :Int16);
        "# how many cores are still unused"
        return self._cores - self._used_cores

    def reserveNumberOfCores_context(self, context): # reserveNumberOfCores @4 (reserveCores :Int16, aModelId :Text) -> (reservedCores :Int16);
        "# try to reserve number of reserveCores for model aModelId and return the number of actually reservedCores"
        pass

def main():
    #address = parse_args().address

    admin_master = capnp.TwoPartyClient("localhost:8000").bootstrap().cast_as(cluster_admin_service_capnp.Cluster.AdminMaster)

    #server = capnp.TwoPartyServer("*:8000", bootstrap=DataServiceImpl("/home/berg/archive/data/"))
    server = capnp.TwoPartyServer("*:8000", bootstrap=SlurmRuntime(cores=4, admin_master=admin_master))
    server.run_forever()

if __name__ == '__main__':
    main()