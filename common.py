import sys
import os
from datetime import date, timedelta
import json
import time
from collections import defaultdict
import uuid

import capnp
capnp.add_import_hook(additional_paths=["../capnproto_schemas/", "../capnproto_schemas/capnp_schemas/"])
import common_capnp

# interface Callback
class CallbackImpl(common_capnp.Common.Callback.Server):

    def __init__(self, callback, *args, exec_callback_on_del=False, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._callback = callback
        self._already_called = False
        self._exec_callback_on_del = exec_callback_on_del

    def __del__(self):
        if self._exec_callback_on_del and not self._already_called:
            self._callback(*self._args, **self._kwargs)

    def call(self, _context, **kwargs): # call @0 ();
        self._callback(*self._args, **self._kwargs)
        self._already_called = True


# interface CapHolder(CapType) extends(Persistent.Persistent(Text, Text))
class CapHolderImpl(common_capnp.Common.CapHolder.Server):

    def __init__(self, cap, sturdy_ref, cleanup_func, cleanup_on_del=False):
        self._cap = cap
        self._sturdy_ref = sturdy_ref
        self._cleanup_func = cleanup_func
        self._already_cleaned_up = False
        self._cleanup_on_del = cleanup_on_del

    def __del__(self):
        if self._cleanup_on_del and not self._already_cleaned_up:
            self.cleanup_func()

    def cap_context(self, context): # cap @0 () -> (cap :CapType);
        context.results.cap = self._cap

    def free_context(self, context): # free @1 ();
        self._cleanup_func()
        self._cleanup_on_del = True

    def save_context(self, context): # save @0 SaveParams -> SaveResults;
        context.results.sturdyRef = self._sturdy_ref


def main():
    pass
    #server = capnp.TwoPartyServer("*:8000", bootstrap=AdminMasterImpl()) #UserMasterImpl(AdminMasterImpl()))
    #server.run_forever()

if __name__ == '__main__':
    main()