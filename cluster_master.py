#from scipy.interpolate import NearestNDInterpolator
#import numpy as np
import sys
import os
from datetime import date, timedelta
#import pandas as pd
#from pyproj import Proj, transform
import json
import time
from collections import defaultdict
import uuid

import capnp
#import climate_data_capnp
capnp.remove_import_hook()
#climate_data_capnp = capnp.load('capnproto_schemas/climate_data.capnp')
#model_capnp = capnp.load('capnproto_schemas/model.capnp')
cluster_admin_service_capnp = capnp.load("capnproto_schemas/cluster_admin_service.capnp")
common_capnp = capnp.load('capnproto_schemas/common.capnp')

class AdminMasterImpl(cluster_admin_service_capnp.Cluster.AdminMaster.Server):
    "Implementation of the Cluster.AdminMaster Cap'n Proto server interface."

    def __init__(self):
        self._uuid4 = uuid.uuid4()
        self._factories = defaultdict(list)

    def info_context(self, context): # info @0 () -> (info :IdInformation);
        "# interface to retrieve id information from an object"
        return {"id": str(self._uuid4), "name": "AdminMaster(" + str(self._uuid4) + ")", "description": ""}

    def registerModelInstanceFactory_context(self, context): # registerModelInstanceFactory @0 (aModelId :Text, aFactory :ModelInstanceFactory););
        "# register a model instance factory for the given model id"
        self._factories[context.params.aModelId].append(context.params.aFactory)

    def availableModels_context(self, context): # availableModels @1 () -> (factories :List(ModelInstanceFactory));
        "# get instance factories to all the available models on registered runtimes"
        context.results.init("modelInfos", len(self._nodes))
        fs = []
        for model_id, factories in self._factories.items():
            if len(factories) == 1:
                fs.append(factories[0])
            elif len(factories > 1):
                fs.append(MultiRuntimeModelInstanceFactory(model_id, factories))

        context.results.factories = fs


class UserMasterImpl(cluster_admin_service_capnp.Cluster.UserMaster.Server):

    def __init__(self, admin_master):
        self._uuid4 = uuid.uuid4()
        self._admin_master = admin_master

    def info_context(self, context): # info @0 () -> (info :IdInformation);
        # interface to retrieve id information from an object
        return {"id": str(self._uuid4), "name": "UserMaster(" + str(self._uuid4) + ")", "description": ""}

    def availableModels(self, context): # availableModels @0 () -> (factories :List(ModelInstanceFactory));
        "# get instance factories to all the available models to the user"
        return self._admin_master.availableModels()


class MultiRuntimeModelInstanceFactory(cluster_admin_service_capnp.Cluster.ModelInstanceFactory.Server):

    def __init__(self, model_id, factories):
        self._model_id = model_id
        self._factories = factories
        self._uuid4 = uuid.uuid4()

    def info_context(self, context): # info @0 () -> (info :IdInformation);
        # interface to retrieve id information from an object
        return {"id": str(self._uuid4), "name": "MultiRuntimeModelInstanceFactory(" + self._model_id + ")(" + str(self._uuid4) + ")", "description": ""}

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
    server = capnp.TwoPartyServer("*:8000", bootstrap=UserMasterImpl(AdminMasterImpl()))
    server.run_forever()

if __name__ == '__main__':
    main()