# `miniramses.py` — Reading and visualizing MINI-RAMSES data

`miniramses` is the workhorse of this toolkit. It reads the raw (mostly
unformatted Fortran binary) outputs of a MINI-RAMSES run — particles, AMR/hydro
cells, clump catalogs, info/units, maps, initial conditions, lightcones — into
convenient NumPy-backed Python objects, and provides a few plotting helpers on
top.

The convention throughout is to import it as `ram`:

```python
import miniramses as ram
```

Snapshots are referred to by their integer number `nout`: the directory
`output_000012/` corresponds to `nout=12`.

## Quick start

```python
import miniramses as ram
import matplotlib.pyplot as plt
import numpy as np

# Simulation metadata, cosmology and unit conversions for snapshot 12
info = ram.rd_info(12)
print(info.aexp, info.redshift, info.boxlen)

# Dark-matter / star particles inside a sphere
p = ram.rd_part(12, center=[0.5, 0.5, 0.5], radius=0.1)
print(p.npart, np.max(p.pos[0]))

# Hydro leaf cells (AMR), then a quick density map
c = ram.rd_cell(12, center=[0.5, 0.5, 0.5], radius=0.1)
ram.visu(c.x[0], c.x[1], c.dx, c.u[0], log=1)   # gas density (u[0]) in log
plt.show()
```

## Common options

Most of the snapshot readers (`rd_info`, `rd_part`, `rd_cell`, `rd_amr`,
`rd_hydro`, `rd_clump`) accept the same keyword arguments:

| kwarg | default | meaning |
|-------|---------|---------|
| `path` | `"./"` | directory containing the `output_00*` folders |
| `center` | `None` | `[x, y, z]` center of a sphere to restrict the region read in |
| `radius` | `None` | radius of that sphere (set together with `center`) |
| `backup` | `False` | read `backup_00*` restart files instead of `output_00*` |
| `silent` | `False` | suppress the "Found N particles/clumps" prints |

`center`/`radius` filtering is periodic-aware (it wraps around the box).

## Reading snapshots

### `rd_info(nout, **kwargs)` → `Info`

Reads `info.txt`: grid sizes, cosmology, and unit conversions. Key attributes:

- `ncpu`, `ndim`, `levelmin`, `nlevelmax`, `boxlen`
- `time`, `texp`, `aexp`, `redshift`
- `H0`, `h`, `omega_m`, `omega_l`, `omega_k`, `omega_b`, `gamma`
- `unit_l`, `unit_d`, `unit_t`, `unit_v`, `unit_m`, `unit_T2` — multiply code
  units by these to reach CGS (e.g. `mass_cgs = p.mass * info.unit_m`)

Pass `units=True` to attach `astropy` units to the unit/box/time attributes.
Pass `rt=True` to also load `rt_info.txt` (radiative-transfer groups).

### `rd_part(nout, **kwargs)` → `Part`

Reads particle files. Choose the type with `prefix`:

| `prefix` | contents | extra fields |
|----------|----------|--------------|
| `"part"` (default) | dark matter / generic particles | — |
| `"star"` | star particles | `metallicity`, `birth_date` |
| `"tree"` | merger-tree tracer particles | `birth_date`, `merging_date`, `merging_id`, `tracking_id` |
| `"sink"` | sink particles | `angmom`, `accel`, `birth_date` |

Always available: `npart`, `ndim`, `pos` (`pos[0]`=x, …), `vel`, `mass`,
`level`, `birth_id`. Add `peak=True` to also read the clump-finder peak
association (`halo_id`, `peak_id`).

```python
t = ram.rd_part(100, prefix="tree", peak=True)
```

### `rd_cell(nout, **kwargs)` → `Cell`

Reads the AMR grid and hydro variables and returns the **leaf cells** (the cells
actually used in the simulation), which is usually what you want. Attributes:

- `ncell`, `ndim`, `nvar`
- `x` — cell centers (`x[0]`=x, `x[1]`=y, …)
- `u` — hydro variables (`u[0]`=density, `u[1..ndim]`=velocity, …)
- `dx` — individual cell sizes; `level` — refinement levels

Add `geom="square"` to use a cubic instead of spherical `center`/`radius` cut.
The lower-level `rd_amr` and `rd_hydro` are available if you need the raw
grid hierarchy.

### `rd_clump(nout, **kwargs)` → `ClumpCat`

Reads the clump-finder catalog (`clump.0000*`). Per-clump arrays include
`index`, `parent`, `halo`, `mass`, `mpatch`, position (`x`,`y`,`z`), velocity
(`u`,`v`,`w`), densities (`dmin`/`dmax`/`dave`/`dsad`), and halo properties
`r200`, `rmax`, `c200`.

## Reading other files

| function | reads | returns |
|----------|-------|---------|
| `rd_map(filename)` | a 2-D map from `amr2map`/`part2map` | `Map` (`.data`, `.time`, `.nx`, `.ny`) |
| `rd_histo(filename)` | a 2-D histogram from the `histo` utility | `Histo` (`.data` plus axis bounds) |
| `rd_cool(filename)` | a cooling table | `Cool` (`.cool`, `.heat`, `.spec`, `.xion`, …) |
| `rd_grafic(filename)` | a GRAFIC initial-conditions field (MUSIC) | `GraficFile` (`.data`, grid + cosmology) |
| `wr_grafic(dat, header1, header2, fileout)` | — | writes a GRAFIC field |
| `rd_log(filename)` | a 1-D run log | list of `Snap1d` (`.x`, `.d`, `.u`, …) |

Lightcone shells are handled by the `LightconeReader` class (static methods
`rd_part`, `rd_cell`, `rd_metadata`, `get_shells`, `rd_positions_as_healpix`).

## Visualization & analysis helpers

### `visu(x, y, dx, v, **kwargs)`

Scatter-plots AMR cells as colored squares sized by `dx` — a quick way to look
at 2-D (or projected 3-D) cell data. Useful kwargs: `log`, `vmin`, `vmax`,
`cmap`, `colorbar`, `grid`, and `sort` (draw order, helpful for 3-D).

```python
c = ram.rd_cell(2)
ram.visu(c.x[0], c.x[1], c.dx, c.u[0], sort=c.u[0], log=1, vmin=-3, vmax=1)
```

### `mk_image(x, y, dx, var)` / `mk_cube(x, y, z, dx, var)`

Resample AMR leaf cells onto a uniform 2-D image or 3-D cube (handy for FFTs,
`imshow`, or volume rendering).

### `rotate_view(c, center=..., velocity=...)`

Rotate cells into a frame whose z-axis is aligned with the gas angular-momentum
vector (i.e. face-on disk). Returns rotated `x, y, z`.

### `mk_movie(**kwargs)`

Turn a numbered sequence of `.map` files into PNG frames and stitch them into an
MP4. Requires `ffmpeg` and ImageMagick's `convert`. See the docstring for the
full set of parameters (`start`, `stop`, `path`, `prefix`, `cmap`, `cbar`, …).

### `plot_tree(nout, pid)`

A minimal merger-tree sketch for the tree particles of peak `pid`. For
publication-quality trees, use [`mergertrees.py`](MERGERTREES.md) on top of
[`mk_tracks.py`](MK_TRACKS.md) instead.

## Notes

- All quantities are in **code units** unless you convert them with the
  `unit_*` factors from `rd_info` (or pass `units=True`).
- `pos`, `vel`, `x`, and `u` are stored "dimension-first": `pos[0]` is the x
  array over all particles, `u[0]` is density over all cells, etc.
- A handful of functions (`hilbert2d`, `hilbert3d`, `get_cpu_list`) are internal
  helpers for spatial indexing and aren't meant to be called directly.
