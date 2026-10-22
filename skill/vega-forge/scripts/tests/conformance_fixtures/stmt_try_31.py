"""C11: what try/except/else/finally, except*, match/case, for-else, while-else and with hold
is counted like any body; the comments number the statements (`guarded` has 31)."""


def guarded(value, rows):
    try:                                        # 1
        number = int(value)                     # 2
        scaled = number * 2                     # 3
    except ValueError:
        number = 0                              # 4
        scaled = 0                              # 5
    except TypeError as exc:
        number = -1                             # 6
        scaled = str(exc)                       # 7
    else:
        scaled += 1                             # 8
    finally:
        done = True                             # 9
        rows = list(rows)                       # 10
    match number:                               # 11
        case 0:
            label = "zero"                      # 12
            weight = 0                          # 13
        case 1 | 2:
            label = "small"                     # 14
            weight = 1                          # 15
        case _:
            label = "large"                     # 16
            weight = 2                          # 17
    for row in rows:                            # 18
        if row is None:                         # 19
            break                               # 20
    else:
        rows.append(None)                       # 21
    while weight > 0:                           # 22
        weight -= 1                             # 23
    else:
        weight = 0                              # 24
    with open(__file__, encoding="utf-8") as handle:   # 25
        first = handle.readline()               # 26
    later = lambda: first                       # 27 (a lambda is an expression, not a def)
    del later                                   # 28
    assert done                                 # 29
    result = (label, scaled, first)             # 30
    return result                               # 31


def grouped(tasks):
    try:                                        # 1
        results = [task() for task in tasks]    # 2
    except* ValueError:
        results = []                            # 3
    except* KeyError as group:
        results = list(group.exceptions)        # 4
    return results                              # 5
