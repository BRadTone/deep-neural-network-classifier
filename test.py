def gen():
    print('first')
    yield
    print('second')
    yield
    print('last')
    yield


gen = gen()

next(gen)
next(gen)
next(gen)