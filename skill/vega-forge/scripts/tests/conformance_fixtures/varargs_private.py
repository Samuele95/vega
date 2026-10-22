"""C12 boundary: *args and **kwargs in _prefixed helpers."""

def _spread(*args, **kwargs):
    return args, kwargs


class _Holder:
    def __init__(self, *args):
        self.args = args

    def call(self, **kwargs):
        return kwargs
