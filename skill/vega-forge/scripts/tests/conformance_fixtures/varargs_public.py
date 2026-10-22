"""C12: *args and **kwargs in a public signature."""

def spread(*args, **kwargs):
    return args, kwargs


class Holder:
    def __init__(self, *args):
        self.args = args
