import logging, sys
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue

FMT = "%(asctime)s %(processName)s %(levelname)s %(name)s: %(message)s"

def start_parent_listener(level=logging.INFO):
    q = Queue(-1)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FMT))
    listener = QueueListener(q, handler)
    listener.start()
    return q, listener

def init_worker_logging(q, level=logging.INFO):
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(QueueHandler(q))
