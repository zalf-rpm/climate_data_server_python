import sys
import os
from datetime import date, timedelta
import json
import time
from collections import defaultdict
import uuid

import capnp
capnp.remove_import_hook()
common_capnp = capnp.load('capnproto_schemas/common.capnp')

class CallbackImpl(common_capnp.Common.Callback.Server):

    def __init__(self, callback, *args, exec_callback_on_del=False, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._callback = callback
        self._already_called = False

    def __del__(self):
        if exec_callback_on_del and not self._already_called:
            self._callback(*self._args, **self._kwargs)

    def call(self, _context, **kwargs): # call @0 ();
        self._callback(*self._args, **self._kwargs)
        self._already_called = True


def main():
    pass
    #server = capnp.TwoPartyServer("*:8000", bootstrap=AdminMasterImpl()) #UserMasterImpl(AdminMasterImpl()))
    #server.run_forever()

if __name__ == '__main__':
    main()