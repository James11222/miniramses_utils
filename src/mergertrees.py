"""
mergertrees.py - Object-oriented merger-tree construction & plotting for MINI-RAMSES
====================================================================================

A faithful re-implementation, for the MINI-RAMSES data format, of the merger-tree
diagram produced by the official RAMSES script ``utils/py/mergertreeplot.py``.

Why a re-implementation was needed
----------------------------------
The RAMSES script consumes per-snapshot *progenitor / descendant* lists and treats
a tree node as a *(clump, snapshot)* pair, collapsing chains of "main progenitors"
into branches.  MINI-RAMSES exposes its merger history completely differently: the
``mk_tracks.mk_track`` pipeline already follows every tracked object continuously
through time, so **one tracked object is already one branch** (a vertical chain of
the snapshots in which it is a resolved clump).  Mergers are encoded by a single
field per object, ``merging_id`` -- the ``birth_id`` of the halo it eventually
merges into (0 if it survives to z=0).

That makes the MINI-RAMSES tree a *simplified* RAMSES tree, and lets us port the
same elegant 5-phase layout algorithm described in ``mergertreeplot_guide.md``:

    1. Build   : turn the ``mk_track`` dict into ``Branch`` objects + a children map,
                 then BFS from a chosen root to gather the tree.
    2. Size    : recursively count sub-branches (``branches_tot``) per branch.
    3. Sort    : order sibling branches by merge time (earliest first), ties broken
                 by sub-tree size (bushier branches pushed further out).
    4. Lay out : assign x-positions with the RAMSES *ordered-column-insertion* trick
                 (the ``_branch_x`` idea, here :class:`Column`): each merging branch
                 is inserted as a new column immediately next to its parent's column,
                 alternating sides; final x = position in the ordered list.
    5. Render  : one vertical line per branch, a diagonal connector per merger, a
                 dual y-axis (snapshot on the left, redshift on the right), and a
                 dot (or labelled box) per node.  Nodes can be coloured per-column
                 (structure view) or by a physical field -- mass, vmax, ... -- read
                 straight from the ``tracks`` dict (the MINI-RAMSES extension).

Typical usage
-------------
    >>> import miniramses as ram, mk_tracks as mk, mergertrees as mt
    >>> tracks = mk.mk_track(100, n_cores=8)         # build all tracks
    >>> info   = ram.rd_info(100)                    # for unit conversions
    >>> fig, ax = mt.plot_merger_tree(tracks, info=info)        # auto-pick main halo
    >>> fig, ax = mt.plot_merger_tree(tracks, root=14)          # explicit root
    >>> fig, ax = mt.plot_merger_tree(tracks, color_by='mass', info=info)

Or the object-oriented form, mirroring the RAMSES design:

    >>> tree = mt.MergerTree(tracks, root=14, info=info)
    >>> print(tree.summary())
    >>> fig, ax = tree.plot(labels=True)

Author: ported for MINI-RAMSES from Mladen Ivkovic's RAMSES ``mergertreeplot.py``.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
import astropy.units as u


# =============================================================================
# LAYOUT PRIMITIVE  (port of RAMSES ``_branch_x``)
# =============================================================================

class Column:
    """
    A mutable x-coordinate shared by a single branch, plus that branch's
    vertical extent.

    Ported from the RAMSES ``_branch_x`` class.  The whole point is that the
    objects live in an *ordered list*: branches hold a *reference* to their
    Column, and new branches are inserted next to existing ones, so the final
    left-to-right ordering -- and hence every x value -- simply falls out of the
    list order once we enumerate it.  Storing a reference (not a copy) is what
    lets an insertion shift every column to its right for free.
    """

    __slots__ = ("x", "ymin", "ymax")

    def __init__(self, ymin, ymax):
        self.x = 0
        self.ymin = ymin
        self.ymax = ymax

    def extend_y(self, ymin, ymax):
        self.ymin = min(self.ymin, ymin)
        self.ymax = max(self.ymax, ymax)

    def set_x(self, x):
        self.x = x


# =============================================================================
# TREE NODE  (one tracked object = one branch)
# =============================================================================

class Branch:
    """
    One MINI-RAMSES tracked object across its whole lifetime -- i.e. one vertical
    track in the diagram.

    This is the MINI-RAMSES analogue of a RAMSES *branch* (a main-progenitor
    spine).  Because ``mk_track`` already follows each object continuously, a
    branch needs no list of per-snapshot nodes; it just remembers the snapshots
    in which it is a resolved clump.

    Attributes
    ----------
    bid : int
        The object's ``birth_id`` (1-based; row ``bid - 1`` in the tracks arrays).
    snaps : ndarray
        1-based snapshot numbers in which the object exists as a resolved clump.
    ymin, ymax : int
        First / last such snapshot (bottom / top of the vertical track).
    merge_snap : int or None
        Snapshot at which this object merges into its target (top of the diagonal
        connector), or ``None`` if it survives to the final snapshot.
    target : int
        ``birth_id`` of the object this one merges into (0 if it survives).
    children : list[Branch]
        Branches that merge *into* this one (its progenitors-by-merger).
    branches_tot : int
        Total number of sub-branches in this branch's sub-tree (Phase 2).
    column : Column
        The x-layout column assigned in Phase 4.
    """

    __slots__ = ("bid", "snaps", "ymin", "ymax", "merge_snap", "target",
                 "children", "branches_tot", "column")

    def __init__(self, bid, snaps, merge_snap, target):
        self.bid = bid
        self.snaps = snaps
        self.ymin = int(snaps.min())
        self.ymax = int(snaps.max())
        self.merge_snap = merge_snap
        self.target = target
        self.children = []
        self.branches_tot = 0
        self.column = None


# =============================================================================
# THE MERGER TREE
# =============================================================================

class MergerTree:
    """
    Build and draw a merger tree from a ``mk_tracks.mk_track`` dictionary.

    Parameters
    ----------
    tracks : dict
        Output of :func:`mk_tracks.mk_track`.  Uses the ``mass``, ``merging_id``,
        ``merge_arr`` and ``aexp`` arrays, all shaped ``(ngal, nsnap)`` and indexed
        by ``birth_id - 1``.
    root : int, optional
        ``birth_id`` of the halo to use as the tree root.  If omitted, the most
        massive halo that *survives* to the final snapshot is chosen automatically.
    info : miniramses.Info, optional
        Only needed when colouring nodes by a physical field that requires unit
        conversion (e.g. ``mass`` -> Msun).

    Notes
    -----
    After construction the tree is ready to draw; all five layout phases have run.
    """

    def __init__(self, tracks, root=None, info=None):
        self.tracks = tracks
        self.info = info
        self.mass = tracks["mass"]
        self.merging_id = tracks["merging_id"]
        self.merge_arr = tracks["merge_arr"]
        self.ngal, self.nsnap = self.mass.shape

        # aexp per snapshot column (identical for every halo where it exists, so
        # the column-wise max recovers it while ignoring the zero-padded rows).
        self.aexp_col = tracks["aexp"].max(axis=0) if "aexp" in tracks else None

        self._build(root)        # Phase 0/1: branches + children + BFS from root
        self._size(self.root)    # Phase 2: branches_tot
        self.columns = []        # Phase 4 scratch (ordered list of Columns)
        self._layout()           # Phases 3+4: sort siblings, assign x

    # ------------------------------------------------------------------ build
    def _merge_target(self, row):
        """birth_id this object merges into (0 if it never merges)."""
        nz = self.merging_id[row][self.merging_id[row] > 0]
        return int(nz[0]) if len(nz) else 0

    def _merge_snap(self, row):
        """1-based snapshot at which this object merges, or None."""
        ma = np.where(self.merge_arr[row] == 1)[0]
        return int(ma[0]) + 1 if len(ma) else None

    def _build(self, root):
        """
        Phase 0/1 -- turn the tracks dict into Branch objects, wire up the
        children map (who merges into whom), pick a root, and gather its tree.
        """
        # One Branch per object that is ever a resolved clump.
        self._branch = {}                  # birth_id -> Branch
        children_of = defaultdict(list)    # target birth_id -> [child birth_id]

        for row in range(self.ngal):
            cols = np.where(self.mass[row] > 0)[0]
            if len(cols) == 0:
                continue                   # never a resolved clump -> no track
            bid = row + 1
            target = self._merge_target(row)
            self._branch[bid] = Branch(bid, cols + 1, self._merge_snap(row), target)
            if target > 0:
                children_of[target].append(bid)

        if not self._branch:
            raise ValueError("No resolved halos found in tracks - nothing to plot.")

        # Choose the root.
        if root is None:
            root = self._auto_root()
        if root not in self._branch:
            raise ValueError(
                f"Root birth_id {root} is not a resolved halo in these tracks.")
        self.root_id = root

        # Attach children, restricting to objects that actually exist, and gather
        # the sub-tree reachable from the root by breadth-first search over the
        # merger links (mirrors mk_tracks.collect, but on Branch objects).
        order = [root]
        for bid in order:
            br = self._branch[bid]
            for cid in children_of.get(bid, []):
                if cid in self._branch:
                    br.children.append(self._branch[cid])
                    order.append(cid)

        self.root = self._branch[root]
        self.nodes = [self._branch[b] for b in order]
        self._warn_if_mass_smoothed()

    def _warn_if_mass_smoothed(self):
        """
        Warn if any branch's lifetime extends past its merger snapshot.

        Without smoothing, the snapshots where an object is a resolved clump
        (``mass > 0``) and the snapshots where it is flagged as merged are
        disjoint, with the merger strictly *after* the last clump snapshot, so a
        merger connector always points forward in time.  ``mk_track`` optionally
        *smooths* the mass history (its ``bins`` kwarg, default 10), which bleeds
        mass into the post-merger snapshots and inflates the apparent lifetime --
        making a connector point backward in time and, more importantly,
        producing incorrect merger/lifetime values if those are quoted.  Mass
        smoothing is not needed for the merger tree, so we flag it here.
        """
        bad = [n.bid for n in self.nodes
               if n.target > 0 and n.merge_snap is not None
               and n.merge_snap <= n.ymax]
        if bad:
            import warnings
            preview = bad if len(bad) <= 10 else bad[:10] + ["..."]
            warnings.warn(
                f"Branch(es) {preview} have a clump lifetime that extends past "
                "their merger snapshot. This is a sign that the mass history was "
                "smoothed when the tracks were built (mk_track's `bins` kwarg), "
                "which inflates lifetimes and would report incorrect merger times. "
                "Rebuild the tracks without smoothing for a correct merger tree, "
                "e.g. mk_tracks.mk_track(snapshot, bins=1).",
                stacklevel=3,
            )

    def _auto_root(self):
        """Most massive halo that survives (never merges) to the final snapshot."""
        final = self.mass[:, -1].copy()
        for row in range(self.ngal):
            if self._merge_target(row) > 0:
                final[row] = 0.0           # keep survivors only
        if not np.any(final > 0):
            raise ValueError("No surviving halo with mass at the final snapshot.")
        return int(np.argmax(final)) + 1

    # ------------------------------------------------------------------- size
    def _size(self, branch):
        """Phase 2 -- recursively count sub-branches in this branch's sub-tree."""
        branch.branches_tot = 0
        for child in branch.children:
            self._size(child)
            branch.branches_tot += child.branches_tot + 1

    # ----------------------------------------------------------------- layout
    def _sorted_children(self, branch):
        """
        Phase 3 -- order the branches merging into ``branch`` by time of
        appearance (earliest merger first), ties broken by sub-tree size so that
        bushier branches are pushed further from the trunk.
        """
        def key(child):
            t = child.merge_snap if child.merge_snap is not None else child.ymax
            return (t, child.branches_tot)
        return sorted(branch.children, key=key)

    def _layout(self):
        """
        Phase 4 -- assign integer x-positions via ordered column insertion.

        The root holds the first column.  For every branch we insert each child's
        column immediately to the right or left of the parent's column
        (alternating by the child's sorted index, even->right / odd->left), so
        merging branches always nestle against their trunk.  We place all of a
        node's direct children first, then recurse -- matching the two-pass
        structure of RAMSES ``_get_x``.  Final x = index in the ordered list.
        """
        self.root.column = Column(self.root.ymin, self.root.ymax)
        self.columns = [self.root.column]
        self._assign_columns(self.root)
        for i, col in enumerate(self.columns):
            col.set_x(i)

    def _assign_columns(self, branch):
        kids = self._sorted_children(branch)
        for idx, child in enumerate(kids):
            x = self.columns.index(branch.column)
            col = Column(child.ymin, child.ymax)
            if idx % 2 == 0:                       # go right
                self.columns.insert(x + 1, col)
            else:                                  # go left
                self.columns.insert(x, col)
            child.column = col
        for child in kids:
            self._assign_columns(child)

    # ---------------------------------------------------------------- queries
    @property
    def ncols(self):
        return len(self.columns)

    @property
    def snap_min(self):
        return min(n.ymin for n in self.nodes)

    @property
    def snap_max(self):
        return max(n.ymax for n in self.nodes)

    def summary(self):
        """Return a short human-readable description of the tree."""
        nmerge = sum(1 for n in self.nodes if n is not self.root)
        lines = [
            f"Merger tree rooted at birth_id {self.root_id}",
            f"  branches (tracked objects) : {len(self.nodes)}",
            f"  mergers into the tree      : {nmerge}",
            f"  snapshot span              : {self.snap_min} - {self.snap_max}",
            f"  layout columns             : {self.ncols}",
        ]
        if self.aexp_col is not None:
            a_hi = self.aexp_col[self.snap_max - 1]
            a_lo = self.aexp_col[self.snap_min - 1]
            if a_hi > 0 and a_lo > 0:
                z_hi = max(0.0, 1 / a_hi - 1)
                z_lo = max(0.0, 1 / a_lo - 1)
                lines.append(
                    f"  redshift span              : {z_hi:.2f} - {z_lo:.2f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------- draw
    def plot(self, ax=None, **kwargs):
        """
        Draw the merger tree.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created if omitted.

        Other Parameters
        ----------------
        color_by : {'column', 'black', <field>}, default 'column'
            'column'  - each branch gets its own colour (structure view);
            'black'   - plain black tree, like the RAMSES esthetic figure;
            <field>   - any per-snapshot field in ``tracks`` ('mass', 'vmax',
                        'r200', 'c200', 'm200', ...): node dots are coloured by that
                        field's value with a colorbar (the MINI-RAMSES extension).
        labels : bool, default False
            Draw the ``birth_id`` in a coloured box at each node instead of a dot.
        marker_size : float, default 6
            Node marker size (points) when ``labels`` is False.
        linewidth : float, default 1.6
            Width of the branch spines and merger connectors.
        cmap : str, default 'viridis'
            Colormap used when ``color_by`` is a physical field.
        norm : {'log', 'linear'}, default 'log'
            Colour normalisation for field colouring.
        vmin, vmax : float, optional
            Colour limits (in code units) for field colouring.
        show_redshift : bool, default True
            Add a redshift axis on the right (needs ``aexp`` in tracks).
        title : str, optional
            Plot title.  Defaults to "Merger tree for halo <root> at snapshot <n>".
        figsize : tuple, optional
            Override the automatic figure size.

        Returns
        -------
        fig, ax
        """
        color_by = kwargs.get("color_by", "column")
        labels = kwargs.get("labels", False)
        marker_size = kwargs.get("marker_size", 6)
        lw = kwargs.get("linewidth", 1.6)
        cmap_name = kwargs.get("cmap", "viridis")
        norm_kind = kwargs.get("norm", "log")
        show_redshift = kwargs.get("show_redshift", True)

        field_mode = color_by not in ("column", "black", "k")
        if field_mode and color_by not in self.tracks:
            raise ValueError(f"color_by='{color_by}' is not a field in tracks.")

        # ---- figure ---------------------------------------------------------
        if ax is None:
            figsize = kwargs.get("figsize", self._auto_figsize())
            fig, ax = plt.subplots(figsize=figsize)
            owns_fig = True
        else:
            fig = ax.figure
            owns_fig = False

        # ---- field colour setup (shared scalar mappable + colorbar) ---------
        smap = None
        if field_mode:
            conv, clabel = self._field_conversion(color_by)
            vmin, vmax = self._field_limits(color_by, kwargs.get("vmin"),
                                            kwargs.get("vmax"))
            Norm = mpl.colors.LogNorm if norm_kind == "log" else mpl.colors.Normalize
            smap = mpl.cm.ScalarMappable(norm=Norm(vmin * conv, vmax * conv),
                                         cmap=cmap_name)
            self._field_norm = Norm(vmin, vmax)      # in code units, for scatter
            self._field_cmap = cmap_name

        # ---- draw every branch ---------------------------------------------
        for node in self.nodes:
            self._draw_branch(ax, node, color_by, labels, marker_size, lw, smap)

        # ---- merger connectors ---------------------------------------------
        for node in self.nodes:
            if node.target > 0 and node.target in self._branch \
                    and self._branch[node.target] in self.nodes:
                self._draw_connector(ax, node, color_by, lw)

        ax2 = self._style_axes(ax, kwargs, show_redshift)

        # Only lay out the figure when we created it. When the caller supplies an
        # axes (e.g. one panel of a multi-subplot figure) the layout is theirs to
        # manage -- calling tight_layout here would fight their arrangement and
        # resize the sibling panels. tight_layout can warn harmlessly when a twin
        # axis is present, so silence that one warning.
        if owns_fig:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout()

        if smap is not None:
            cb = self._add_field_colorbar(fig, ax, ax2, smap, clabel, owns_fig)
            return fig, ax, cb

        return fig, ax, None

    def _add_field_colorbar(self, fig, ax, ax2, smap, clabel, owns_fig):
        """Pin a horizontal colorbar beneath the plot, robust to figure size.

        Rather than letting matplotlib steal space (whose thickness/aspect drift
        as the figure is resized), the colorbar axes is given a *locator* that
        positions it from the main axes' live geometry on every draw:

        * its **width** matches the main axes exactly, so the bar's left/right
          edges stay flush with the left (snapshot) and right (redshift) y-axes;
        * its **thickness** and the gap above it are defined in *inches*, so they
          stay constant whatever the figure size or aspect ratio;
        * it hangs just under the x-axis with a small, nearly-flush pad, and -
          because the locator re-reads the axes position each draw - it keeps
          tracking the plot through later ``tight_layout`` calls or resizes.

        When we own the figure (single axes) the axes is shrunk upward first so
        the bar and its labels are guaranteed to fit inside the figure. When the
        caller supplies the axes (a panel in a multi-subplot figure) the axes is
        left untouched - it stays the same size as its siblings and the bar simply
        hangs below it, into the figure's bottom margin.
        """
        fs = self._fontsizes(fig)
        _, fig_h = fig.get_size_inches()

        # Geometry in inches (constant across figure sizes); the locator converts
        # to figure fraction at draw time using the then-current figure height.
        bar_in = float(np.clip(fs["label"] / 72.0 * 1.1, 0.25, 0.6))  # thickness
        pad_in = 0.12                                                 # axes->bar gap
        # Room beneath the bar for its tick labels + axis label, scaled to the
        # (figure-dependent) font sizes so big, tall trees still leave enough.
        label_in = 0.16 + (fs["tick"] + fs["label"]) / 72.0 * 1.7

        if owns_fig:
            # Shrink the (single) plot upward, top fixed, to free a bottom strip.
            pos = ax.get_position()
            reserve = (pad_in + bar_in + label_in) / fig_h
            new_box = [pos.x0, pos.y0 + reserve, pos.width, pos.height - reserve]
            ax.set_position(new_box)
            if ax2 is not None:
                ax2.set_position(new_box)

        cax = fig.add_axes([0, 0, 1, 1])  # placeholder; the locator drives it
        cax.set_axes_locator(self._cbar_locator(ax, bar_in, pad_in))
        cb = fig.colorbar(smap, cax=cax, orientation="horizontal")
        cb.set_label(clabel, fontsize=fs["label"])
        cb.ax.tick_params(labelsize=fs["tick"])
        return cb

    @staticmethod
    def _cbar_locator(parent, bar_in, pad_in):
        """Locator that hangs a colorbar axes below ``parent``, flush and fixed.

        Returns a callable ``(cax, renderer) -> Bbox`` (figure-fraction) that
        matplotlib invokes each draw, so the bar follows the parent axes if the
        figure is resized or re-laid-out, always a constant ``bar_in`` inches
        thick and ``pad_in`` inches below the axes, spanning its full width.
        """
        from matplotlib.transforms import Bbox

        def _locate(cax, renderer):
            _, fig_h = parent.figure.get_size_inches()
            p = parent.get_position()
            bar_h = bar_in / fig_h
            pad_h = pad_in / fig_h
            return Bbox.from_bounds(p.x0, p.y0 - pad_h - bar_h, p.width, bar_h)

        return _locate

    # --------------------------------------------------------- drawing helpers
    def _draw_branch(self, ax, node, color_by, labels, marker_size, lw, smap):
        """
        Vertical spine + per-snapshot markers for one branch.

        With ``labels`` the markers are drawn as squares and the branch's
        ``birth_id`` is written once, in large type, just below its earliest
        (lowest) snapshot -- rather than repeating it at every snapshot.
        """
        inner, edge = self._branch_colors(node, color_by)
        x = node.column.x
        marker = "s" if labels else "o"

        # The spine: a single line spanning the branch's vertical extent.
        ax.plot([x, x], [node.ymin, node.ymax], "-", color=inner, lw=lw, zorder=1)

        ys = node.snaps
        if color_by not in ("column", "black", "k") and smap is not None:
            # Colour each marker by the physical field value at that snapshot.
            vals = self.tracks[color_by][node.bid - 1][ys - 1]
            ax.scatter(np.full(len(ys), x), ys, c=vals, cmap=self._field_cmap,
                       norm=self._field_norm, s=(marker_size + 2) ** 2,
                       marker=marker, edgecolor="k", linewidth=0.4, zorder=3)
        else:
            ax.plot(np.full(len(ys), x), ys, marker, color=inner,
                    markeredgecolor=edge, markeredgewidth=0.6,
                    ms=marker_size, zorder=2)

        if labels:
            self._draw_branch_id(ax, node, edge, marker_size)

    def _draw_branch_id(self, ax, node, color, marker_size):
        """Write the branch's birth_id once, below its earliest snapshot."""
        id_fs = 0.8 * self._fontsizes(ax.figure)["label"]
        ax.annotate(str(node.bid), (node.column.x, node.ymin),
                    textcoords="offset points", xytext=(0, -(marker_size + 6)),
                    ha="center", va="top", fontsize=id_fs, fontweight="bold",
                    color=color, zorder=5)

    def _draw_connector(self, ax, node, color_by, lw):
        """Diagonal link from a branch's tip to its descendant's trunk."""
        parent = self._branch[node.target]
        inner, _ = self._branch_colors(node, color_by)
        y_top = node.ymax
        y_join = node.merge_snap if node.merge_snap is not None else node.ymax + 1
        ax.plot([node.column.x, parent.column.x], [y_top, y_join],
                "-", color=inner, lw=lw, zorder=1)

    # ----------------------------------------------------------- colour logic
    def _branch_colors(self, node, color_by):
        """Return (inner, edge) colours for a branch's spine / markers."""
        if color_by == "black" or color_by == "k":
            return "0.15", "0.15"
        if color_by == "column":
            return _column_color(node.column.x)
        # Field mode: spine/edges stay neutral; the dots carry the field colour.
        return "0.35", "0.25"

    def _field_limits(self, field, vmin, vmax):
        vals = []
        for node in self.nodes:
            v = self.tracks[field][node.bid - 1][node.snaps - 1]
            vals.append(v[v > 0])
        vals = np.concatenate(vals) if vals else np.array([1.0])
        lo = vmin if vmin is not None else float(vals.min())
        hi = vmax if vmax is not None else float(vals.max())
        return lo, hi

    def _field_conversion(self, field):
        """Code-unit -> physical conversion factor and a colorbar label."""

        if self.info is not None:
            try:
                # Check to see if units were used in info object, if not warn the user and use code units instead.
                if field == "mass":
                    conv = self.info.unit_m.to(u.Msun).value * self.info.h # cgs to Msun/h
                    return conv, r"$M_{\rm clump}\;[\mathrm{M}_\odot/h]$"
                if field == "m200":
                    conv = self.info.unit_m.to(u.Msun).value * self.info.h # cgs to Msun/h
                    return conv, r"$M_{\rm 200c}\;[\mathrm{M}_\odot/h]$"
                if field == "r200":
                    conv = self.info.unit_l.to(u.kpc).value * self.info.h # cgs to kpc/h
                    return conv, r"$r_{\rm 200c}\;[\mathrm{kpc}/h]$"
                    
            except AttributeError:
                import warnings
                warnings.warn(
                    "Please set units=True when loading info with miniramses.py to get physical units"
                    "using code units for now.",
                    stacklevel=3,
                )
            
        labels = {"vmax": r"$V_{\max}$ [code units]",
                  "r200": r"$r_{\rm 200c}$ [code units]",
                  "c200": r"$c_{\rm 200c}$", 
                  "mass": r"$M_{\rm clump}$ [code units]",
                  "m200": r"$M_{\rm 200c}$ [code units]"
                 }
                  
        return 1.0, labels.get(field, field)

    # ------------------------------------------------------------- axis style
    def _auto_figsize(self):
        span = self.snap_max - self.snap_min + 1
        height = float(np.clip(0.18 * span, 6, 24))
        width = float(np.clip(0.7 * self.ncols + 2.5, 5, 30))
        return (width, height)

    @staticmethod
    def _fontsizes(fig):
        """Text sizes scaled to the figure so labels stay legible on tall plots.

        Rather than hard-coding point sizes (which look tiny on the large,
        auto-sized tree figures), derive them from the figure height.
        """
        _, fh = fig.get_size_inches()
        label = float(np.clip(2.0 * fh, 14, 34))
        return {"label": label, "tick": 0.8 * label, "title": 1.15 * label}

    def _style_axes(self, ax, kwargs, show_redshift):
        fs = self._fontsizes(ax.figure)
        has_rax = show_redshift and self.aexp_col is not None
        # snapshot increases upward (latest at top), x is layout-only.
        ax.set_xticks([])
        # Never show the top, and only show the right when a redshift axis lives
        # there. Without a redshift axis, keep a bottom baseline so the plot is
        # still framed (left + bottom), but no right/top.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(not has_rax)
        ax.tick_params(top=False, right=False, which="both")
        ax.set_ylabel("Snapshot", fontsize=fs["label"])
        ax.tick_params(axis="y", labelsize=fs["tick"])

        from matplotlib.ticker import MaxNLocator

        lo, hi = self.snap_min, self.snap_max
        # With labels=True the birth_id is printed below each branch's lowest
        # snapshot, so leave extra room beneath the deepest branch for it.
        labels = kwargs.get("labels", False)
        bottom_extra = 1.0 + (0.07 * (hi - lo + 1) if labels else 0.0)
        ax.set_ylim(lo - bottom_extra, hi + 1)
        # Centre the frame on the surviving root's trunk: use a symmetric x-range
        # about the root column so the trunk sits in the middle of the plot, with
        # progenitor branches splaying out to either side.
        pad = max(1.0, 0.05 * self.ncols)
        xs = [n.column.x for n in self.nodes]
        root_x = self.root.column.x
        half = max(root_x - min(xs), max(xs) - root_x) + pad
        ax.set_xlim(root_x - half, root_x + half)
        grid = kwargs.get("grid", False)

        if grid:
            ax.grid(axis="y", which="major", color="0.85", lw=0.5, zorder=0)

        # Shared major-tick snapshots, so the snapshot (left) and redshift (right)
        # axes line up exactly: every left tick has a matching right tick at the
        # same y-position. No minor ticks on either axis.
        locator = MaxNLocator(nbins=8, integer=True, steps=[1, 2, 5, 10])
        snaps = [int(t) for t in locator.tick_values(lo, hi)
                 if lo <= t <= hi
                 and self.aexp_col is not None and self.aexp_col[int(t) - 1] > 0]
        ax.set_yticks(snaps)
        ax.set_yticklabels([str(s) for s in snaps])
        ax.minorticks_off()

        title = kwargs.get("title",
                           f"Merger tree for halo {self.root_id} "
                           f"at snapshot {self.snap_max}")
        ax.set_title(title, fontsize=fs["title"])

        # Redshift axis on the right: ticks at the *same* snapshots, each labelled
        # by the redshift at that snapshot, so the left snapshot tick and the right
        # redshift tick correspond exactly.
        if show_redshift and self.aexp_col is not None:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.spines["top"].set_visible(False)
            ax2.set_yticks(snaps)
            ax2.set_yticklabels(
                [f"{max(0.0, 1.0 / self.aexp_col[s - 1] - 1.0):.2f}"
                 for s in snaps])
            ax2.set_ylabel("Redshift", fontsize=fs["label"], rotation=270, labelpad=30)
            ax2.tick_params(axis="y", labelsize=fs["tick"])
            ax2.minorticks_off()
            return ax2
        return None


# =============================================================================
# PER-COLUMN COLOURS  (port of RAMSES ``_get_plotcolors`` idea)
# =============================================================================

# A qualitative palette; column 0 (the trunk) gets the first entry.
_PALETTE = (list(plt.get_cmap("tab10").colors)
            + list(plt.get_cmap("tab20b").colors))


def _column_color(x):
    """Return (inner, edge) colours for layout column index ``x``."""
    inner = _PALETTE[x % len(_PALETTE)]
    edge = tuple(0.55 * c for c in inner[:3])      # darker shade for the outline
    return inner, edge


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def plot_merger_tree(tracks, root=None, info=None, ax=None, **kwargs):
    """
    One-call helper: build a :class:`MergerTree` and draw it.

    Parameters
    ----------
    tracks : dict
        Output of :func:`mk_tracks.mk_track`.
    root : int, optional
        ``birth_id`` of the root halo (auto-selected if omitted).
    info : miniramses.Info, optional
        Needed only for physical-unit colourbars.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    **kwargs
        Passed through to :meth:`MergerTree.plot`.

    Returns
    -------
    fig, ax

    Examples
    --------
    >>> fig, ax = plot_merger_tree(tracks, info=info)
    >>> fig, ax = plot_merger_tree(tracks, root=14, color_by='mass', info=info)
    """
    tree = MergerTree(tracks, root=root, info=info)
    return tree.plot(ax=ax, **kwargs)


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def _main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Build and plot a MINI-RAMSES merger tree.")
    parser.add_argument("snapshot", type=int,
                        help="final snapshot number to analyse (e.g. 100)")
    parser.add_argument("root", type=int, nargs="?", default=None,
                        help="birth_id of the root halo (auto if omitted)")
    parser.add_argument("--path", default="./", help="simulation directory")
    parser.add_argument("--ncores", type=int, default=None,
                        help="cores for mk_track (default: all but one)")
    parser.add_argument("--color-by", default="column",
                        help="'column', 'black', 'k', or a field name (mass, vmax, ...)")
    parser.add_argument("--labels", action="store_true",
                        help="label nodes with birth_id instead of drawing dots")
    parser.add_argument("-o", "--output", default=None,
                        help="output image filename (default: merger_tree_<root>.png)")
    args = parser.parse_args(argv)

    import mk_tracks as mk
    import miniramses as ram

    mk_kwargs = {"path": args.path}
    if args.ncores is not None:
        mk_kwargs["n_cores"] = args.ncores
    tracks = mk.mk_track(args.snapshot, **mk_kwargs)
    info = ram.rd_info(args.snapshot, path=args.path)

    tree = MergerTree(tracks, root=args.root, info=info)
    print(tree.summary())

    fig, ax = tree.plot(color_by=args.color_by, labels=args.labels)
    out = args.output or f"merger_tree_{tree.root_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    _main()
