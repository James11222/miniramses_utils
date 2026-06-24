# `mergertrees.py` — Merger-tree diagrams for MINI-RAMSES

An object-oriented re-implementation of the mergertree plotting scripts in RAMSES (found at `ramses/utils/py/mergertreeplot.py`), for the MINI-RAMSES data format. It turns the tree particle tracks built by `mk_tracks.py` into a publication-style tree: vertical
dot-chains for each halo, diagonal connectors at mergers, snapshot on the left axis and redshift on the right (optional).

## Quick start

```python
import miniramses as ram
import mk_tracks as mk
import mergertrees as mt

tracks = mk.mk_track(100, n_cores=8, bins=1)   # follow every object, snaps 1..100 (no smoothing)
info   = ram.rd_info(100)                      # cosmology / unit conversions

# Simplest call — auto-pick the most massive surviving halo as the root:
fig, ax = mt.plot_merger_tree(tracks, info=info)
fig.savefig("merger_tree.png", dpi=150, bbox_inches="tight")
```

That's the whole pipeline: `mk_track` does the heavy I/O (reading `tree` particles
and `clump` catalogs for every snapshot), and `mergertrees` only consumes the
returned dictionary.

## Command line

```bash
# Build tracks for snapshots 1..100 and plot the tree of halo 14, coloured by mass
python mergertrees.py 100 14 --color-by mass -o tree_14.png

# Auto-pick the root, plain black esthetic, label every node with its birth_id
python mergertrees.py 100 --color-by black --labels
```

Arguments: `python mergertrees.py <snapshot> [root_birth_id]`
with `--path`, `--ncores`, `--color-by`, `--labels`, `-o/--output`.

## The API within the script

### High level

```python
fig, ax = mt.plot_merger_tree(tracks, root=None, info=None, ax=None, **plot_kwargs)
```

- `tracks` — the dict from `mk_tracks.mk_track`.
- `root` — `birth_id` of the root halo. Omit it to auto-select the most massive
  halo that survives (never merges) to the final snapshot.
- `info` — a `miniramses.Info`, needed only for physical-unit colourbars
  (`mass` → M⊙, `r200` → kpc).
- `**plot_kwargs` — forwarded to `MergerTree.plot` (see below).

### Object oriented

```python
tree = mt.MergerTree(tracks, root=14, info=info)
print(tree.summary())                 # text description of the tree
fig, ax = tree.plot(color_by="mass")  # draw it
```

Useful attributes: `tree.root_id`, `tree.nodes` (list of `Branch`),
`tree.ncols`, `tree.snap_min`, `tree.snap_max`.

## `plot` options (`color_by` and friends)

| kwarg | default | meaning |
|-------|---------|---------|
| `color_by` | `'column'` | `'column'` = one colour per branch (structure view); `'black'` = plain black tree (the RAMSES esthetic figure); or **any per-snapshot field in `tracks`** (`'mass'`, `'vmax'`, `'r200'`, `'c200'`, …) to colour the node dots by that field with a colourbar. |
| `labels` | `False` | label the bottom of each branch with its `birth_id` in the same color as the branch. |
| `marker_size` | `6` | dot size (points) when `labels=False`. |
| `linewidth` | `1.6` | width of branch spines and merger connectors. |
| `cmap` | `'viridis'` | colormap for field colouring. |
| `norm` | `'log'` | `'log'` or `'linear'` colour normalisation. |
| `vmin`, `vmax` | auto | colour limits in **code units**. |
| `show_redshift` | `True` | add the right-hand redshift axis (uses `aexp` from `tracks`). |
| `title` | auto | plot title, can be set to `None` |
| `figsize` | auto | override the automatic figure size. |


## Example Output

```python
import mk_tracks as mk
import mergertrees as mt

# generate tracks from simulation
tracks = mk.mk_track(100, n_cores=8, bins=1)     # follow every tracked object, snaps 1..100

N = 3
fig, axes = plt.subplots(1,N, figsize=(6*N, 12))

ax = axes[0] 
mt.plot_merger_tree(tracks, ax=ax, info=info, color_by='k', title=None, show_redshift=False)

ax = axes[1]
mt.plot_merger_tree(tracks, ax=ax, info=info, color_by='column', labels=True, title=None, show_redshift=False)

ax = axes[2]
_,_,cbar = mt.plot_merger_tree(tracks, ax=ax, info=info, color_by='r200', norm='linear', title=None, show_redshift=True)
```

<p align="center">
  <img width="1595" height="1064" alt="image" src="https://github.com/user-attachments/assets/90db4b11-e493-4998-8969-603370ad870b" />
</p>


<!--
## How it works (and how it maps to the RAMSES script)

MINI-RAMSES already follows each object continuously through time, so **one
tracked object is one branch** — a vertical chain of the snapshots in which it is
a resolved clump (`mass > 0`). Mergers are encoded by a single field per object,
`merging_id` (the `birth_id` of the halo it merges into, `0` if it survives).
That makes the tree a simplified version of the RAMSES one, drawn with the same
five-phase layout from `ramses_mergertrees/mergertreeplot_guide.md`:

1. **Build** — wrap each object in a `Branch`, build the `children` map from
   `merging_id`, then BFS from the root to gather the tree (`MergerTree._build`).
2. **Size** — recursively count sub-branches per branch (`branches_tot`).
3. **Sort** — order sibling branches by merge time, ties broken by sub-tree size.
4. **Lay out** — the heart of it: x-positions come from *ordered column
   insertion* (`Column`, the port of RAMSES `_branch_x`). Each merging branch is
   inserted as a new column right next to its parent's column, alternating sides;
   the final integer x is just the position in the ordered list, which keeps
   branches near their trunk and guarantees no overlaps.
5. **Render** — a vertical spine per branch, a diagonal connector per merger, a
   dot/label per node, and dual snapshot/redshift axes.


| RAMSES (`mergertreeplot.py`) | here (`mergertrees.py`) |
|------------------------------|-------------------------|
| `_branch_x` | `Column` |
| `Node` + main-progenitor "branch" | `Branch` (one tracked object) |
| `make_tree` (progenitor/descendant lists) | `MergerTree._build` (from `merging_id`) |
| `_walk_tree` / `_sum_branches` | `MergerTree._size` |
| `_sort_branch` | `MergerTree._sorted_children` |
| `_get_x` (ordered column insertion) | `MergerTree._layout` / `_assign_columns` |
| `_draw_straight_lines` / `_draw_tree` | `MergerTree._draw_branch` / `_draw_connector` |
| `_get_plotcolors` | `_column_color` |
| `_tweak_treeplot` | `MergerTree._style_axes` |
-->
