"""C12: six parameters; a staticmethod discounts no first parameter, whatever its name."""

def six(a, b, c, d, e, f):
    return a


class Holder:
    @staticmethod
    def static(a, b, c, d, e, f):
        return a

    def mixed(self, a, b, c, d, *rest):
        return a

    @staticmethod
    def named_cls(cls, a, b, c, d, e):
        return a
