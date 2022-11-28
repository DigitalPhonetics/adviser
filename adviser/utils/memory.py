from typing import Any, Callable
from datetime import datetime, timedelta
from threading import Thread, RLock
import time


GC_INTERVAL = 600 # garbabe collection interval (in seconds): 600 ~ run GC every 10 minutes
KEEP_DATA_FOR_N_SECONDS = timedelta(seconds=600) # duration to store user data in memory


class _GC:
    __managed = set()
    _gc_thread = None

    @staticmethod
    def register(obj):
        """
        Registers the object with the garbage collector.
        Creates a GC thread on first call.
        """
        if _GC._gc_thread == None:
            # init GC (make GC daemon so it ends on program exit)
            _GC._gc_thread = Thread(target=_GC.gc, daemon=True)
            _GC._gc_thread.start()
        _GC.__managed.add(obj)
    
    @staticmethod
    def gc():
        while True:
            time.sleep(GC_INTERVAL)    # wait for next GC event
            # sweep & clean memory
            for mem in _GC.__managed:
                mem._gc()
                # print("GC:", mem.mem)



class UserState:
    def __init__(self, value_factory: Callable = None) -> None:
        self.value_factory = value_factory
        self.lock = RLock()
        self.mem = {}
        self.timestamps = {}
        _GC.register(self)

    def __getitem__(self, user_id: int) -> Any:
        self.lock.acquire()
        self.timestamps[user_id] = datetime.now()
        if not user_id in self.mem and self.value_factory:
            self.mem[user_id] = self.value_factory()
        res = self.mem[user_id] 
        self.lock.release()
        return res

    def __setitem__(self, user_id: int, value: Any):
        self.lock.acquire()
        self.mem[user_id] = value
        self.timestamps[user_id] = datetime.now()
        self.lock.release()

    def _gc(self):
        self.lock.acquire()
        # collect expired user data
        now = datetime.now()
        expired_users = [user_id for user_id in self.timestamps if now - self.timestamps[user_id] > KEEP_DATA_FOR_N_SECONDS]
        # delete expired user data
        for user_id in expired_users:
            del self.timestamps[user_id]
            del self.mem[user_id]
        self.lock.release()
    