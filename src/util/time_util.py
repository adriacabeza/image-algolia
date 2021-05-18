from functools import wraps
from time import time

import src.util.log_util as log


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        log.info("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result
    return wrap
