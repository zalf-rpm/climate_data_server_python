import sys
import os
from datetime import date, timedelta
import json
import time

import capnp
capnp.add_import_hook(additional_paths=["../capnproto_schemas/", "../capnproto_schemas/capnp_schemas/"])
import fbp_capnp

class Process(fbp_capnp.FBP.Input.Server):

    def __init__(self, out):
        self._out = out

    def input_context(self, context): # (data :Text);
        data = context.params.data
        return self._out.input(data + " -> output")

class Consumer(fbp_capnp.FBP.Input.Server):
    def __init__(self):
        pass

    def input(self, data, **kwargs): # (data :Text);
        print(data)

def produce(config):
    process = capnp.TwoPartyClient("localhost:" + config["process_port"]).bootstrap().cast_as(fbp_capnp.FBP.Input)

    for i in range(1000):
        process.input("data " + str(i)).wait()


def main():

    # start = [Process, Consumer, Producer]
    config = {
        "process_port": "10001",
        "consumer_port": "10002",
        "server": "*",
        "start": "Process"
    }
    # read commandline args only if script is invoked directly from commandline
    if len(sys.argv) > 1 and __name__ == "__main__":
        for arg in sys.argv[1:]:
            k, v = arg.split("=")
            if k in config:
                config[k] = v

    cs = config["start"]
    if cs == "Process":
        consumer = capnp.TwoPartyClient("localhost:" + config["consumer_port"]).bootstrap().cast_as(fbp_capnp.FBP.Input)
        server = capnp.TwoPartyServer(config["server"] + ":" + config["process_port"], bootstrap=Process(consumer))
        server.run_forever()
    elif cs == "Consumer":
        server = capnp.TwoPartyServer(config["server"] + ":" + config["consumer_port"], bootstrap=Consumer())
        server.run_forever()
    elif cs == "Producer":
        produce(config)

if __name__ == '__main__':
    main()
