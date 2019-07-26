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
model_capnp = capnp.load('capnproto_schemas/model.capnp')
cluster_admin_service_capnp = capnp.load("capnproto_schemas/cluster_admin_service.capnp")
common_capnp = capnp.load('capnproto_schemas/common.capnp')


class SlurmMonicaInstanceFactory(cluster_admin_service_capnp.Cluster.ModelInstanceFactory.Server):

    def __init__(self):
        self._uuid4 = uuid.uuid4()

    def modelId_context(self, context): # modelId @4 () -> (id :Text);
        "# return the id of the model this factory creates instances of"
        return "monica_v2.1"

    def info_context(self, context): # info @0 () -> (info :IdInformation);
        # interface to retrieve id information from an object
        return {"id": str(self._uuid4), "name": "SlurmMonicaInstanceFactory(" + str(self._uuid4) + ")", "description": ""}

    def newInstance_context(self, context): # newInstance @0 () -> (instance :AnyPointer);
        "# return a new instance of the model"
        pass

    def newInstances_context(self, context): # newInstances @1 (numberOfInstances :Int16) -> (instances :AnyList);
        "# return the requested number of model instances"
        pass

    def newCloudViaZmqPipelineProxies_context(self, context): # newCloudViaZmqPipelineProxies @2 (numberOfInstances :Int16) -> (zmqInputAddress :Text, zmqOutputAddress :Text);
        "# return the TCP addresses of two ZeroMQ proxies to access the given number of instances of the model"
        pass

    def newCloudViaProxy_context(self, context): # newCloudViaProxy @3 (numberOfInstances :Int16) -> (proxy :AnyPointer);
        "# return a model proxy acting as a gateway to the requested number of model instances"
        pass

def main():
    #address = parse_args().address

    #server = capnp.TwoPartyServer("*:8000", bootstrap=DataServiceImpl("/home/berg/archive/data/"))
    server = capnp.TwoPartyServer("*:8000", bootstrap=SlurmMonicaInstanceFactory())
    server.run_forever()

if __name__ == '__main__':
    main()