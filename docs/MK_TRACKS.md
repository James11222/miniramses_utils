# `mk_tracks.py` — Particle evolutionary tracks for MINI-RAMSES

`mk_tracks` reads the `tree` particle outputs and `clump` catalogs of a
MINI-RAMSES run and stitches them into **time-series histories** for every
tracked object. For each object you get its position, velocity, mass, and halo
properties (`r200`, `c200`, `vmax`, …) at every snapshot, all packed into tidy
NumPy arrays.

It's the front end of the analysis pipeline: `mk_track` does the heavy I/O once,
and other tools (e.g. [`mergertrees.py`](MERGERTREES.md)) just consume the
dictionary it returns.

> **Requirement:** the simulation must have been run with `merger_tree=.true.`
> in the namelist, so that the `tree` particle outputs exist.

## Quick start

```python
import miniramses as ram
import mk_tracks as mk

# Follow every tracked object across snapshots 1..100, using 8 cores
tracks = mk.mk_track(100, n_cores=8)

# tracks is a dict of arrays. Most are shaped (ngal, nsnapshots):
mass = tracks["mass"]     # halo mass history of every object
x, y, z = tracks["x"], tracks["y"], tracks["z"]

# Mass history of the first object:
import matplotlib.pyplot as plt
plt.plot(tracks["aexp"][0], tracks["mass"][0])
plt.xlabel("expansion factor a"); plt.ylabel("mass [code units]")
```

`mk_track(100)` processes outputs `1` through `100` inclusive. The number of
tracked objects (`ngal`) is determined from the final snapshot you ask for.

## The main function

```python
tracks = mk.mk_track(snapshot, **kwargs)
```

- `snapshot` (int) — the **final** output number to process. Outputs `1` through
  `snapshot` are read.

### Common options

| kwarg | default | meaning |
|-------|---------|---------|
| `n_cores` | `cpu_count() - 1` | CPU cores for parallel snapshot reading. Use `1` to run serially. |
| `bins` | `10` | Smoothing window for the mass histories. Set `bins=1` to disable smoothing (do this for merger trees — see note below). |
| `sigma` | `0.0` | Lognormal scatter applied to `vmax` (`0.0` = none). |
| `window_size` | `3` | Window for the maximum-mass calculation. |
| `mass_max` | `True` | Also compute the peak mass reached by each track. |
| `verbose` | `True` | Print progress and diagnostics. |

Any extra keyword arguments are forwarded to the underlying `miniramses`
readers (e.g. a `path=` to point at the simulation directory).

## What you get back

`mk_track` returns a dictionary. Unless noted, every value is a 2-D array of
shape `(ngal, nsnapshots)` — one row per tracked object, one column per snapshot.

| key | description |
|-----|-------------|
| `aexp`, `texp` | expansion factor and proper time at each snapshot |
| `x`, `y`, `z` | Cartesian positions |
| `vx`, `vy`, `vz` | Cartesian velocities |
| `mass` | clump mass (smoothed unless `bins=1`) |
| `mpatch` | patch mass |
| `m200` | M200c, the NFW mass within `r200` |
| `r200`, `c200`, `rmax`, `vmax` | virial radius, concentration, radius of max circular velocity, and max circular velocity |
| `peak_id` | associated density-peak ID |
| `pop` | population type: `1` central, `2` most-massive central, `3` satellite, `4` orphan |
| `merge_arr` | merger flag (`1` in the snapshot where the object merges) |
| `birth_id`, `tracking_id`, `merging_id` | bookkeeping IDs from the tree output |
| `mass_max` | **1-D** array `(ngal,)` — peak mass reached by each track |

All quantities are in **code units**. Use a `miniramses.Info` object
(`ram.rd_info(snapshot)`) for conversions to physical units.

## A couple of handy extras

**Trace a merger branch** — `collect` walks the tree and returns the `birth_id`s
of every object that eventually merges into a target halo:

```python
tree = ram.rd_part(100, prefix="tree")
branch = mk.collect(tree, itrack=42)
print(f"{len(branch)} objects merge into halo 42")
```

## Notes & gotchas

- **Indexing convention:** row `i` of the returned arrays corresponds to
  `birth_id = i + 1`.
- **Turn off mass smoothing for merger trees.** Smoothing (`bins`, default 10)
  bleeds mass into post-merger snapshots and distorts merger times. Build tracks
  with `bins=1` when feeding them to [`mergertrees.py`](MERGERTREES.md).
  Smoothing is fine for other analyses.
- Snapshots with no peak/clump data are skipped automatically; their columns
  stay zero.
- Reading is parallelised over snapshots, so more cores mainly speeds up the I/O.
  A progress bar is shown while snapshots are processed.
