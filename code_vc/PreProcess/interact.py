import threading
import ctypes
import inspect


class interact(threading.Thread):
    def __init__(self, valid_cmd, lead_key='-'):
        super().__init__()
        self.daemon = True
        self.valid_cmd = set(valid_cmd)
        assert len(lead_key) == 1
        self.lead_key = lead_key
        self.cmd = ''
        self.wait = threading.Event()
        self.wait.set()

    def __enter__(self):
        return self.cmd

    def __exit__(self, type, value, trace):
        self.cmd = ''
        self.wait.set()

    def run(self):
        while True:
            self.wait.wait()
            try:
                x = input('The input is being monitored\n')
            except UnicodeDecodeError:
                print('Invalid cmd.')
                continue
            if len(x) <= 1:
                continue
            if x[0] != self.lead_key or not set(x[1:]) <= self.valid_cmd:
                print('Invalid cmd: {}.'.format(x))
                continue
            self.cmd = x[1:]
            self.wait.clear()
            print('Cmd has been accepted, monitor has been locked')

    def stop_thread(self, exctype=SystemExit):
        """raises the exception, performs cleanup if needed"""
        tid = self.ident
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")
