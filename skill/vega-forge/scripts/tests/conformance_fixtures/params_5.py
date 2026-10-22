"""C12 boundary: five parameters beyond self/cls."""

def five(a, b, c, d, e):
    return a


class Holder:
    def method(self, a, b, c, d, e):
        return a

    @classmethod
    def factory(cls, a, b, c, d, e):
        return cls

    def keywords(self, a, b, *, c, d, e):
        return a
