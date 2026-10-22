"""
expected.py — the known measures of the measurement modules beside this file, which
`test_conformance_sizes.py` asserts exactly (D6's `CONFORMANCE_MEASURES`, code-contract §5.1).

A library of the test, not a measurement fixture.
"""
from __future__ import annotations

# (file, symbol): (statements, parameters, variadic, helper)
FUNCTIONS = {
    ("stmt_30.py", "thirty"): (30, 0, (), False),
    ("stmt_31.py", "thirty_one"): (31, 0, (), False),
    ("nested_31.py", "outer"): (3, 0, (), False),
    ("nested_31.py", "inner"): (31, 0, (), False),
    ("elif_chain_31.py", "chain"): (31, 1, (), False),
    ("stmt_try_31.py", "guarded"): (31, 2, (), False),
    ("stmt_try_31.py", "grouped"): (5, 1, (), False),
    ("params_5.py", "five"): (1, 5, (), False),
    ("params_5.py", "Holder.method"): (1, 5, (), False),
    ("params_5.py", "Holder.factory"): (1, 5, (), False),
    ("params_5.py", "Holder.keywords"): (1, 5, (), False),
    ("params_6.py", "six"): (1, 6, (), False),
    ("params_6.py", "Holder.static"): (1, 6, (), False),
    ("params_6.py", "Holder.mixed"): (1, 5, ("*rest",), False),
    ("params_6.py", "Holder.named_cls"): (1, 6, (), False),
    ("varargs_public.py", "spread"): (1, 2, ("*args", "**kwargs"), False),
    ("varargs_public.py", "Holder.__init__"): (1, 1, ("*args",), False),
    ("varargs_private.py", "_spread"): (1, 2, ("*args", "**kwargs"), True),
    ("varargs_private.py", "_Holder.__init__"): (1, 1, ("*args",), True),
    ("varargs_private.py", "_Holder.call"): (1, 1, ("**kwargs",), True),
}

# (file, class): lines, from the first decorator to the last statement
CLASSES = {("class_200.py", "Holder"): 200, ("class_201.py", "Holder"): 201}

# file: lines, from line 1 to the last statement
MODULE_LINES = {"module_400.py": 400, "module_401.py": 401}

# file: the (code, line, symbol, measured) of every finding size_findings returns, exactly
FINDINGS = {
    "stmt_30.py": [],
    "stmt_31.py": [("C11", 3, "thirty_one", 31)],
    "nested_31.py": [("C11", 4, "inner", 31)],
    "elif_chain_31.py": [("C11", 3, "chain", 31)],
    "stmt_try_31.py": [("C11", 5, "guarded", 31)],
    "params_5.py": [],
    "params_6.py": [("C12", 3, "six", 6), ("C12", 9, "Holder.static", 6),
                    ("C12", 12, "Holder.mixed", None), ("C12", 16, "Holder.named_cls", 6)],
    "varargs_public.py": [("C12", 3, "spread", None), ("C12", 8, "Holder.__init__", None)],
    "varargs_private.py": [],
    "class_200.py": [],
    "class_201.py": [("C13", 8, "class Holder", 201)],
    "module_400.py": [],
    "module_401.py": [("C13", 1, "module", 401)],
}
