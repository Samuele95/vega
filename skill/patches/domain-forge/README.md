# domain-forge patch: the validator's browser-profile cleanup race

`validate_model-tempdir.patch` changes three lines of `domain-forge/scripts/validate_model.py`.
Each of the three `tempfile.TemporaryDirectory()` blocks that host a headless Chromium
`--user-data-dir` becomes `tempfile.TemporaryDirectory(ignore_cleanup_errors=True)`. Nothing else
changes.

| pre-image line | function | invariant | what the temp dir holds |
|---|---|---|---|
| 760 | `check_projector_round_trip` | 12 (RDF projectors round-trip) | `instrumented.html` and the browser `profile/` |
| 946 | `check_kg_render` | 18–19 (knowledge-graph render) | `instr.html` and the browser `profile/` |
| 1366 | `check_layers_headless` (through `_l16_run`) | 16 (per-layer isolation render) | `l16-<tag>.html` and one `profile-<tag>/` per render |

The patch was made against the installed domain-forge `scripts/validate_model.py` (1468 lines):

| image | sha256 |
|---|---|
| before the patch (pre-image) | `86e6db2b3b606685cfba52d1e30338607a11943e58de44d6944641002c2b02d3` |
| after the patch (post-image) | `6e46edbef1ced45f085faee50b18ce6af6469a99822ee6260ea32d6f947cb18d` |

It needs Python 3.10 or later, because `ignore_cleanup_errors` first appeared in 3.10.

## Why

The chain's acceptance scripts went red at random on this machine (design D6 §1.2 and §18 G17).
`validate_model.py` died after its checks had already run, with this traceback:

```
  File ".../domain-forge/scripts/validate_model.py", line 760, in check_projector_round_trip
    with tempfile.TemporaryDirectory() as td:
  File "/usr/lib/python3.12/tempfile.py", line 1085, in __exit__
    self.cleanup()
  ...
  File "/usr/lib/python3.12/shutil.py", line 695, in _rmtree_safe_fd
    os.rmdir(entry.name, dir_fd=topfd)
OSError: [Errno 39] Directory not empty: 'Default'
```

`subprocess.run(... '--dump-dom' ...)` returns when the main Chromium process exits. A child
process can still be writing `profile/Default` at that moment, so the `rmtree` of the temp
directory races with it and loses. It happened in 3 of 5 calls made inside the acceptance flow,
right after heavy steps. It happened in 0 of 30 isolated calls (12 sequential, 18 at six-way
concurrency).

The failure carries no information about the model. The DOM was already dumped and parsed
before `__exit__` runs, so the verdict is complete. The failed cleanup only turns that verdict
into a traceback and exit status 1. `ignore_cleanup_errors=True` keeps the verdict and ignores
the cleanup error.

The one other `TemporaryDirectory` in the file (line 652, invariant 8) is left alone, because
no browser runs inside it. It holds only the files of two kinds of child process, and
`subprocess.run` waits for each of them to exit. `extract_component.py` starts no browser and no
subprocess. The nested `validate_model.py <component> --no-recurse` runs skip every headless
check: `main` calls `check_projector_round_trip`, `check_kg_render` and `check_layers_headless`
only when `--no-recurse` is absent. So nothing is still writing in that directory when it is
removed.

**Trade-off.** After a lost race, the Chromium profile (a few MB) can stay behind in `$TMPDIR`.
The other upstream fix would remove that leftover: wait for the browser's process group to exit
before the cleanup. Either fix closes G17. This one is the smallest change that makes the
verdict deterministic.

## Apply

The installed skill is replaced, never edited in place by this package. To install the fix:

```
patch -p1 -d <installed domain-forge skill dir> < patches/domain-forge/validate_model-tempdir.patch
# or: git apply --directory=<path of the skill inside the git work tree> validate_model-tempdir.patch
```

`python3 vega/scripts/tests/test_prereqs.py` reads the installed `validate_model.py` and tells
three states apart:

| state | how it is recognised | the test |
|---|---|---|
| unpatched | the pre-image sha256, or the patch applies forward with no fuzz | applies the patch to a copy and checks that it applies with no offset and no fuzz (else: regenerate the patch), that it only adds `ignore_cleanup_errors=True`, and that the result compiles |
| patched here | the post-image sha256, or the patch applies in reverse with no fuzz | passes, and prints `NOTE: … the patch is applied locally` |
| changed upstream | neither | fails. When every browser-profile temp dir of the new file already ignores cleanup errors, the fix has landed upstream: delete this patch and the `df-profile-race` known-flake entry. Otherwise, regenerate the patch against the new file |

Applying the patch as above therefore keeps the test green. Only a changed upstream file that
already carries the fix is the signal to delete the patch and the flake entry.

## Until it is applied: the known flake

While the installed domain-forge is unpatched, the package's acceptance scripts (forge, train,
eval, pack) call `validate_model.py` through `vega/scripts/tests/known_flake.py`. That wrapper
retries a command **once**, and only when the command and its output match a signature in
`vega/scripts/tests/lanekit/flakes.json` (`vega/known-flakes@1`). The one entry is
`df-profile-race`: command `validate_model\.py`, output `OSError: [Errno 39] Directory not
empty: 'Default'` with `tempfile` in the traceback, owner `domain-forge`, upstream this file. Every
retry is reported (`WARN:` on stderr, and a JSON line when `--record` or `VEGA_FLAKE_RECORD` is
set). A second failure, or any failure outside a known signature, is never retried.

The list of known flakes may only shrink. Once domain-forge itself ships the fix (the "changed
upstream" state above, with every browser-profile temp dir already tolerant), the entry is
deleted, and so is this patch. Applying the patch on one machine deletes neither: other
installations still carry the race.
