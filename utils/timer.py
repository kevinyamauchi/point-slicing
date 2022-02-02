from timeit import default_timer

from utils.logger import logger


def timer(func):
    def wrapper(*args, **kwargs):
        start = default_timer()
        func(*args, **kwargs)
        end = default_timer()
        logger.info(f"Time taken for {func.__name__} was {end - start:e}")

    return wrapper
