from timeit import default_timer as timer
from time import sleep


def time_it(fn):
    def wrapper():
        print('asd')
        start = timer()
        rv = fn()
        end = timer()
        print('it took:', end - start)
        return rv

    return wrapper

