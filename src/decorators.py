from timeit import default_timer as timer


def time_it(fn):
    def wrapper():
        start = timer()
        rv = fn()
        end = timer()
        print('it took:', end - start)
        return rv

    return wrapper


def print_decorator(fn):
    def wrapper():

        print('fn name:', fn.__name__)
        rv = fn()
        print('2')
        return rv

    return wrapper


@print_decorator
@time_it
def test():
    print('test print')


test()