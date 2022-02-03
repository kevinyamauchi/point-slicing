import functools
from timeit import default_timer

from utils.logger import logger


def timer(func):
    """
    Decorator to time a function
    """

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """time_wrapper's doc string"""
        start = default_timer()
        result = func(*args, **kwargs)
        end = default_timer()
        logger.info(f"Function: {func.__name__}, Time: {end-start:e} s")
        return result

    return time_closure
