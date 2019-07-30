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
import subprocess

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
        self._registry = {}

    # registerModelInstance @5 [ModelInstance] (instance :ModelInstance, registrationToken :Text = "") -> (unregister :Common.Callback);
    def registerModelInstance(self, instance, registrationToken, _context, **kwargs):
        if registrationToken in self._registry:
            self._registry[registrationToken] = instance
            

        

        pass

    def modelId(self, _context, **kwargs): # modelId @4 () -> (id :Text);
        "# return the id of the model this factory creates instances of"
        return "monica_v2.1"
        
    def info(self, _context, **kwargs): # info @0 () -> (info :IdInformation);
        # interface to retrieve id information from an object
        return {"id": str(self._uuid4), "name": "SlurmMonicaInstanceFactory(" + str(self._uuid4) + ")", "description": ""}

    def newInstance(self, _context, **kwargs): # newInstance @0 () -> (instance :AnyPointer);
        "# return a new instance of the model"

        uid = uuid.uuid4()
        subprocess.Popen(["monica-capnp-server", ""])



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

    runtime_available = False
    while not runtime_available:
        try:
            runtime = capnp.TwoPartyClient("localhost:9000").bootstrap().cast_as(cluster_admin_service_capnp.Cluster.Runtime)
            runtime_available = True
        except:
            #time.sleep(1)
            pass

    monicaFactory = SlurmMonicaInstanceFactory()
    registered_factory = False
    while not registered_factory:
        try:
            unreg = runtime.registerModelInstanceFactory("monica_v2.1", monicaFactory).wait().unregister
            registered_factory = True
        except capnp.KjException as e:
            print(e)
            time.sleep(1)

    #server = capnp.TwoPartyServer("*:8000", bootstrap=DataServiceImpl("/home/berg/archive/data/"))
    server = capnp.TwoPartyServer("*:10000", bootstrap=monicaFactory)
    server.run_forever()

if __name__ == '__main__':
    main()