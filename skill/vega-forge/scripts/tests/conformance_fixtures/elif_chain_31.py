"""C11: an if/elif/else chain of sixteen arms is thirty-one statements; the AST, not the eye, counts elif."""

def chain(v):
    if v == 0:
        return 0
    elif v == 1:
        return 1
    elif v == 2:
        return 2
    elif v == 3:
        return 3
    elif v == 4:
        return 4
    elif v == 5:
        return 5
    elif v == 6:
        return 6
    elif v == 7:
        return 7
    elif v == 8:
        return 8
    elif v == 9:
        return 9
    elif v == 10:
        return 10
    elif v == 11:
        return 11
    elif v == 12:
        return 12
    elif v == 13:
        return 13
    elif v == 14:
        return 14
    else:
        return -1
