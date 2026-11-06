"""
RAMSES Particle Track Analysis - Optimized Parallel Version
=============================================================

This module provides utilities for reading and analyzing particle/tracer outputs
from MINI-RAMSES simulations. It builds time-series histories for tracked objects
including positions, velocities, masses, and halo properties.

Main Functions:
- mk_track: Build particle tracking histories across simulation outputs
- collect: Recursively collect merger tree branches
- plot_mass: Visualize mass evolution along merger branches
- plot_traj: Visualize spatial trajectories
- plot_tree: Visualize complete merger tree structure
"""

import numpy as np
import miniramses as ram
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# TRACK FIELDS
# =============================================================================

# Tracks returned by mk_track when `fields` is not given
DEFAULT_FIELDS = ('aexp', 'vmax', 'mass', 'm200c', 'x', 'y', 'z', 'peak_id',
                  'mpeak', 'vpeak', 'pop', 'birth_id', 'tracking_id', 'merging_id')

# Per-snapshot scalars read from Info, the same for every tracked object
INFO_FIELDS = ('aexp', 'texp')

# Read from the tree particles, so defined for every tracked object
PART_FIELDS = {
    'x': lambda t: t.pos[0],
    'y': lambda t: t.pos[1],
    'z': lambda t: t.pos[2],
    'vx': lambda t: t.vel[0],
    'vy': lambda t: t.vel[1],
    'vz': lambda t: t.vel[2],
    'birth_id': lambda t: t.birth_id,
    'tracking_id': lambda t: t.tracking_id,
    'merging_id': lambda t: t.merging_id,
}

# Read from the clump catalog, only for matched, non-orphan, non-merged objects
CLUMP_FIELDS = ('vmax', 'mass', 'm200c', 'r200c', 'c200c', 'rmax', 'mpatch')

# Built in process_snapshot from the clump match
SPECIAL_FIELDS = ('peak_id', 'pop')

# Running peaks computed in mk_track: {output field: field it is the peak of}
DERIVED_FIELDS = {'mpeak': 'm200c', 'vpeak': 'vmax'}

# Tracks flag_orphans and flag_merged need to build pop
POP_INPUTS = ('tracking_id', 'merging_id', 'aexp')

# ALL FIELDS: this represents every available field constructed from the clump catalogues
ALL_FIELDS = (INFO_FIELDS + tuple(PART_FIELDS) + CLUMP_FIELDS + SPECIAL_FIELDS
              + tuple(DERIVED_FIELDS))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# def f(x):
#     """
#     NFW profile helper function.

#     Parameters
#     ----------
#     x : float or array-like
#         Dimensionless radius (r/rs)

#     Returns
#     -------
#     float or array-like
#         Value of ln(1+x) - x/(1+x)
#     """
#     return np.log(1 + x) - x / (1 + x)

# def compute_M200(r200, c200, info):
#     """
#     Compute M200c for an NFW halo.

#     Parameters
#     ----------
#     r200 : float or array-like
#         r200c (where density = 200 * critical density) [code units]
#     c200 : float or array-like
#         Concentration parameter for c200c (r200c / rs)
#     info : miniramses.Info
#         Info object containing cosmological parameters

#     Returns
#     -------
#     float or array-like
#         M200c mass [code units]
#     """

#     rho_m = 1.0 # Mean matter density in code units 
#     omega_mz = info.omega_m * info.aexp**(-3) / (info.omega_m * info.aexp**(-3) + info.omega_l)
#     d200c = 200 / omega_mz  # 200 times the critical density
#     # d200c = 200  # 200 times the background density (HACK FOR OLD SIMS)

#     M200 = (4 * np.pi / 3) * r200**3 * d200c * rho_m
#     return M200

# def compute_vmax(r200, c200, rmax, info):
#     """
#     Compute maximum circular velocity (Vmax) for an NFW halo.

#     Uses the formalism from Mo, van den Bosch & White (2010),
#     Galaxy Formation and Evolution, pages 352-353.

#     Parameters
#     ----------
#     r200 : float or array-like
#         Virial radius (where density = 200 * critical density) [code units]
#     c200 : float or array-like
#         Concentration parameter (r200 / rs)
#     rmax : float or array-like
#         Radius of maximum circular velocity [code units]
#     info : miniramses.Info
#         Info object containing cosmological parameters

#     Returns
#     -------
#     float or array-like
#         Maximum circular velocity [code units]
#     """
#     rho_m = 1.0  # Mean matter density in code units
#     G = 3 / (8 * np.pi) * info.omega_m * info.aexp  # Gravitational constant (H0=1)

#     rs = r200 / c200  # Scale radius
#     xmax = rmax / rs  # Dimensionless radius at Vmax

#     # Characteristic overdensity
#     delta_char = (200 / 3) * (c200**3 / f(c200))

#     # Mass enclosed at rmax
#     M_max = 4 * np.pi * rho_m * delta_char * rs**3 * f(xmax)

#     # Maximum circular velocity
#     V_max = np.sqrt(G * M_max / rmax)

#     return V_max


# def compute_max_running_average(x, y, window_size=3):
#     """
#     Compute the maximum running average of y over a sliding window.

#     Parameters
#     ----------
#     x : array-like
#         Independent variable (e.g., time or scale factor)
#     y : array-like
#         Dependent variable to average (e.g., mass)
#     window_size : int, default=3
#         Number of points in the sliding window

#     Returns
#     -------
#     x_ave_max : float
#         Value of x corresponding to maximum running average
#     y_ave_max : float
#         Maximum running average value
#     """

#     i = 0
#     y_ave_max, x_ave_max = 0, 0
#     y_window, x_window = [], []

#     if len(x) <= window_size: # not enough points for running average
#         return max(x), max(y)

#     while i < len(x):
#         if len(y_window) < window_size - 1:
#             y_window.append(y[i]); x_window.append(x[i])
#         else:
#             y_window.append(y[i]); x_window.append(x[i])
#             y_ = sum(y_window) / window_size; x_ = sum(x_window) / window_size
#             y_ave_max = np.maximum(y_ave_max, y_)
#             if y_ave_max == y_: x_ave_max = x_
#             y_window.pop(0); x_window.pop(0)
#         i += 1

#     return x_ave_max, y_ave_max


def compute_running_peak(y, window_size=3):
    """
    Compute the running maximum of the trailing average of y.

    Parameters
    ----------
    y : array-like
        Dependent variable to average (e.g., mass)
    window_size : int, default=3
        Number of points in the trailing window

    Returns
    -------
    y_peak : ndarray
        Same shape as y. y_peak[k] is the largest trailing average up to
        point k. The first window_size - 1 points average what is available.
    """
    y = np.asarray(y, dtype=float)
    n = np.arange(1, y.size + 1)
    y_ave = np.convolve(y, np.ones(window_size))[:y.size] / np.minimum(n, window_size)
    return np.maximum.accumulate(y_ave)


def flag_orphans(pop, tracking_id, aexp):
    """Set pop to -1 from the snapshot where tracking_id shuts off permanently."""
    tracked = tracking_id != 0
    tracked_before = np.logical_or.accumulate(tracked, axis=1)
    tracked_later = np.logical_or.accumulate(tracked[:, ::-1], axis=1)[:, ::-1]
    pop[tracked_before & ~tracked_later & (aexp > 0)] = -1


def flag_merged(pop, merging_id, aexp):
    """Set pop to nan from the snapshot where merging_id turns on."""
    merged = merging_id != 0
    merged_before = np.logical_or.accumulate(merged, axis=1)
    merged_later = np.logical_or.accumulate(merged[:, ::-1], axis=1)[:, ::-1]
    pop[merged_before & merged_later & (aexp > 0)] = np.nan


def resolve_fields(fields):
    """
    Work out which per-snapshot fields must be read to build `fields`.

    Parameters
    ----------
    fields : iterable of str
        Field names requested from mk_track

    Returns
    -------
    tuple of str
        Requested fields with derived fields replaced by their inputs, plus
        the inputs of pop, plus `mass`, which the min_part mask always needs

    Raises
    ------
    ValueError
        If a name is not in ALL_FIELDS
    """
    fields = tuple(fields)
    unknown = [f for f in fields if f not in ALL_FIELDS]
    if unknown:
        raise ValueError(f"Unknown track fields {unknown}. Valid fields: {ALL_FIELDS}")

    raw_fields = ['mass']
    for f in fields:
        if f in DERIVED_FIELDS:
            # Running peaks are only defined where m200c and aexp are positive
            needed = (DERIVED_FIELDS[f], 'm200c', 'aexp')
        elif f == 'pop':
            needed = ('pop',) + POP_INPUTS
        else:
            needed = (f,)
        raw_fields.extend(n for n in needed if n not in raw_fields)

    return tuple(raw_fields)


# =============================================================================
# CORE PARALLELIZATION FUNCTIONS
# =============================================================================

def process_snapshot(iout, ngal, ind, sigma, verbose, fields, kwargs):
    """
    Process a single simulation snapshot in parallel.

    Reads tree particle data and clump catalogs for one output, extracts
    the requested fields for all tracked particles.

    Parameters
    ----------
    iout : int
        Output number to process
    ngal : int
        Total number of tracked galaxies/halos
    ind : array-like
        Sorted indices of birth_ids
    sigma : float
        Scatter to apply to vmax (lognormal sigma)
    verbose : bool
        Print diagnostic information
    fields : tuple of str
        Per-snapshot fields to extract, as returned by resolve_fields
    kwargs : dict
        Additional arguments for miniramses readers

    Returns
    -------
    dict or None
        Dictionary with `iout` and one (ngal,) array per name in `fields`,
        or None if failed/empty
    """

    # -------------------------------------------------------------------------
    # Read snapshot data
    # -------------------------------------------------------------------------
    try:
        tt = ram.rd_part(iout, prefix="tree", peak=True, silent=True, **kwargs)
        info = ram.rd_info(iout, **kwargs)
    except FileNotFoundError:
        return None  # Peak data doesn't exist
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to read snapshot {iout}: {e}")
        return None

    # Skip empty snapshots
    if tt.npart == 0:
        return None

    # if verbose: # the number of tree particles at each snapshot
    #     print(iout, tt.npart)

    # -------------------------------------------------------------------------
    # Read clump catalog
    # -------------------------------------------------------------------------
    try:
        cc = ram.rd_clump(iout, silent=True, **kwargs)
    except FileNotFoundError:
        if verbose:
            print(f"Warning: No clump data for snapshot {iout}, skipping")
        return None
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to read clump data for snapshot {iout}: {e}")
        return None

    # -------------------------------------------------------------------------
    # Match tracked particles to clumps
    # -------------------------------------------------------------------------
    ii = np.argsort(tt.birth_id)  # Sort by birth_id for consistent ordering
    ntrack = ii.size

    peak_id = tt.peak_id[ii]
    order = np.argsort(cc.index)
    pos = np.searchsorted(cc.index, peak_id, sorter=order)
    ipeak = order[np.minimum(pos, cc.index.size - 1)]

    matched = (peak_id > 0) & (cc.index[ipeak] == peak_id)
    orphan = tt.tracking_id[ii] == 0
    merged = tt.merging_id[ii] > 0
    live = matched & ~orphan & ~merged  # Halo properties only for these

    # -------------------------------------------------------------------------
    # Fill output arrays
    # -------------------------------------------------------------------------
    snap = {f: np.zeros(ngal) for f in fields}

    for f in fields:
        if f in INFO_FIELDS:
            snap[f][:ntrack] = getattr(info, f)
        elif f in PART_FIELDS:
            snap[f][:ntrack] = PART_FIELDS[f](tt)[ii]
        elif f in CLUMP_FIELDS:
            snap[f][:ntrack][live] = getattr(cc, f)[ipeak[live]]

    if 'peak_id' in snap:
        snap['peak_id'][:ntrack][live] = peak_id[live]

    # Classify Population Type (simplified)
    if 'pop' in snap:
        snap['pop'][:ntrack][matched & cc.is_central[ipeak]] = 1  # Central
        snap['pop'][:ntrack][matched & cc.is_satellite[ipeak]] = 2  # Satellite

    snap['iout'] = iout
    return snap


# =============================================================================
# MAIN TRACKING FUNCTION
# =============================================================================

def mk_track(snapshot, fields=DEFAULT_FIELDS, **kwargs):
    """
    Build time-series tracks for particles across simulation outputs.

    Reads tree/tracer particle outputs and clump catalogs for outputs 1 through
    `snapshot`, constructing complete histories for each tracked object. Supports
    parallel processing across multiple CPU cores.

    Parameters
    ----------
    snapshot : int
        Final output number to process (processes outputs 1 to snapshot inclusive)
    fields : iterable of str, optional
        Tracks to return (default: DEFAULT_FIELDS). Any name in ALL_FIELDS is valid.

    Other Parameters
    ----------------
    n_cores : int, optional
        Number of CPU cores for parallel processing (default: cpu_count() - 1)
    bins : int, optional
        Smoothing window size for mass histories (default: 1)
    sigma : float, optional
        Lognormal scatter to apply to Vmax (default: 0.0)
    window_size : int, optional
        Trailing window size for mpeak and vpeak (default: 3)
    min_part : int, optional
        Drop tracks whose mass never exceeds min_part particle masses in any
        snapshot (default: 200). Use min_part=0 to keep every ever-resolved track.
    verbose : bool, optional
        Print diagnostic information (default: True)

    Returns
    -------
    dict
        One (nkept, snapshot) array per name in `fields`, where nkept is the
        number of tracks that pass the min_part mask. Rows are not indexed by
        birth_id - 1 after masking, so use `birth_id` to identify a row. It is
        0 before an object exists, so tracks['birth_id'].max(axis=1) gives the
        ID of each row. Available fields:
        - aexp : Expansion factor at each output
        - texp : Proper time at each output
        - r200c : Radius enclosing 200 times the critical density
        - c200c : Concentration parameter
        - rmax : Radius of maximum circular velocity
        - vmax : Maximum circular velocity
        - mass : Halo mass (smoothed if bins > 2)
        - mpatch : Patch mass
        - m200c : Mass enclosed by r200c
        - x, y, z : Cartesian positions
        - vx, vy, vz : Cartesian velocities
        - peak_id : Associated peak ID
        - pop : Population type (1=central, 2=satellite, -1=permanent orphan,
          0=no matched clump, nan=merged)
        - birth_id : ID of the tracked object
        - tracking_id : Tracking ID (0 if orphan)
        - merging_id : birth_id of the merge target (0 if not merged)
        - mpeak : Peak smoothed m200c reached up to each output
        - vpeak : Peak smoothed vmax reached up to each output

    Examples
    --------
    >>> # Process 100 outputs using 8 cores
    >>> tracks = mk_track(100, n_cores=8)

    >>> # Sequential processing with custom smoothing
    >>> tracks = mk_track(50, n_cores=1, bins=20)

    >>> # Only positions and masses, keeping every ever-resolved track
    >>> tracks = mk_track(100, fields=('x', 'y', 'z', 'mass'), min_part=0)
    """
    # -------------------------------------------------------------------------
    # Parse arguments and initialize
    # -------------------------------------------------------------------------
    fields = tuple(fields)
    raw_fields = resolve_fields(fields)

    verbose = kwargs.get("verbose", True)
    sigma = kwargs.get("sigma", 0.0)
    window_size = kwargs.get("window_size", 3)
    n_cores = kwargs.get("n_cores", cpu_count() - 1)
    nsmooth_in = kwargs.get("bins", 1)
    min_part = kwargs.get("min_part", 200)

    # Read initial snapshot to determine number of tracks
    info = ram.rd_info(snapshot, **kwargs)
    t = ram.rd_part(snapshot, prefix="tree", silent=True, **kwargs)
    ind = np.argsort(t.birth_id)
    ngal = len(ind)

    # Print configuration
    if verbose:
        print(f"Found {ngal} tracks")
        print(f"Using {n_cores} cores for parallel processing")
        print(f"Applying mass smoothing with window size: {nsmooth_in}")
        print(f"Applying scatter to vmax with sigma: {sigma}")
        # print(f"Mass max flag is set to: {mpeak_flag}")
        print(f"Finding maximum m200c and vmax history with window size: {window_size}")

    # Print diagnostics
    if verbose:
        n_orphan = np.sum(t.tracking_id == 0)
        n_clump = np.sum(t.tracking_id > 0)
        n_merged = np.sum(t.merging_id > 0)
        print(f"Found {n_orphan} orphans")
        print(f"Found {n_clump} clumps")
        print(f"Found {n_merged} merged")

    # -------------------------------------------------------------------------
    # Pre-allocate output arrays
    # -------------------------------------------------------------------------
    tracks = {f: np.zeros((ngal, snapshot)) for f in raw_fields}

    # -------------------------------------------------------------------------
    # Process snapshots in parallel
    # -------------------------------------------------------------------------
    process_func = partial(
        process_snapshot,
        ngal=ngal,
        ind=ind,
        sigma=sigma,
        verbose=verbose,
        fields=raw_fields,
        kwargs=kwargs,
    )

    snapshot_range = range(1, snapshot + 1)

    if n_cores > 1:
        with Pool(processes=n_cores) as pool:
            results = list(tqdm(
                pool.imap(process_func, snapshot_range),
                total=snapshot,
                desc="Processing snapshots",
                unit="snapshot",
            ))
    else:
        results = [
            process_func(iout)
            for iout in tqdm(
                snapshot_range,
                desc="Processing snapshots",
                unit="snapshot",
            )
        ]

    # -------------------------------------------------------------------------
    # Combine results from parallel workers
    # -------------------------------------------------------------------------
    if verbose: print("\nCombining results...")

    for result in results:
        if result is not None:
            iloop = result['iout'] - 1
            for f in raw_fields:
                tracks[f][:, iloop] = result[f]

    # -------------------------------------------------------------------------
    # Remove clumps that never have more than min_part particles in any snapshot
    # -------------------------------------------------------------------------
    mask = np.max(tracks['mass'], axis=1) > (min_part / info.npart)
    tracks = {f: track[mask] for f, track in tracks.items()}
    ngal = int(np.sum(mask))

    if verbose:
        print(f"Kept {ngal} of {mask.size} tracks with min_part: {min_part}")

    # -------------------------------------------------------------------------
    # Set pop to -1 once a track has permanently orphaned and to nan once it has
    # merged. flag_merged runs last so an orphan that later merges ends as nan.
    # -------------------------------------------------------------------------
    if 'pop' in tracks:
        flag_orphans(tracks['pop'], tracks['tracking_id'], tracks['aexp'])
        flag_merged(tracks['pop'], tracks['merging_id'], tracks['aexp'])

    # -------------------------------------------------------------------------
    # Apply smoothing to mass histories
    # -------------------------------------------------------------------------
    if verbose: print("\nApplying mass smoothing...")

    mass = tracks['mass']

    for igal in range(ngal):
        msave = mass[igal].copy()  # Save original
        mm = mass[igal]

        ind_nozero = np.where(mm > 0)[0]
        nsmooth = np.min([nsmooth_in, len(ind_nozero)])

        if nsmooth > 2:
            kernel = np.ones(nsmooth) / nsmooth
            mass[igal] = np.convolve(mm, kernel, mode='same')
            # Restore last 10 points to avoid edge artifacts
            mass[igal, -10:] = msave[-10:]

    # -------------------------------------------------------------------------
    # Compute maximum mass histories
    # -------------------------------------------------------------------------
    peak_fields = [f for f in fields if f in DERIVED_FIELDS]

    if peak_fields:
        if verbose: print("\nComputing maximum m200c and vmax histories...")

        for f in peak_fields:
            tracks[f] = np.zeros((ngal, snapshot))

        for i in range(ngal):
            ind = np.where((tracks["m200c"][i] > 0) & (tracks["aexp"][i] > 0))[0]

            if len(ind) > 0:
                # Map each snapshot from first resolved onward to latest resolved point
                fill = np.searchsorted(ind, np.arange(ind[0], snapshot), side="right") - 1
                for f in peak_fields:
                    source = tracks[DERIVED_FIELDS[f]][i]
                    tracks[f][i, ind[0]:] = compute_running_peak(source[ind], window_size)[fill]

    # -------------------------------------------------------------------------
    # Package and return results
    # -------------------------------------------------------------------------
    if verbose:
        print("\n✓ Track processing complete!")

    return {f: tracks[f] for f in fields}


# =============================================================================
# MERGER TREE FUNCTIONS
# =============================================================================

def collect(tree, itrack):
    """
    Recursively collect all particles in a merger branch.

    Traces the merger tree to find all particles that eventually merge into
    the target halo identified by `itrack`.

    Parameters
    ----------
    tree : miniramses.Part
        Tree particle data with merging_id and birth_id arrays
    itrack : int
        Birth ID of the root halo to trace

    Returns
    -------
    list of int
        Birth IDs of all particles in the merger tree branch

    Examples
    --------
    >>> tree = ram.rd_part(100, prefix="tree")
    >>> merger_tree = collect(tree, itrack=42)
    >>> print(f"Found {len(merger_tree)} halos in merger branch")
    """
    res = [itrack]

    # Breadth-first search through merger tree
    for i in res:
        ind = np.where(tree.merging_id == i)[0]
        if len(ind) > 0:
            children = tree.birth_id[ind].astype(int)
            res.extend(children.tolist())

    return res


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

# def plot_tree(itrack, peak_tree, tracks=None, info=None, **kwargs):
#     """
#     Visualize merger tree with colored nodes showing halo evolution.
# 
#     Creates a tree diagram where horizontal position represents birth_id and
#     vertical position represents time. Nodes can be colored by any field
#     (mass, r200, c200, etc.).
# 
#     Parameters
#     ----------
#     itrack : int
#         Birth ID of the root halo
#     peak_tree : miniramses.Part
#         Peak tree data with birth_id, merging_id, birth_date, merging_date
#     tracks : dict, optional
#         Track data from mk_track() (required for colored nodes)
#     info : miniramses.Info, optional
#         Info object for unit conversions (required for colored nodes)
# 
#     Other Parameters
#     ----------------
#     field : str, optional
#         Field to color nodes by: 'mass', 'r200', 'c200', etc. (default: 'mass')
#     marker_size : float, optional
#         Fixed marker size, or -1 for size scaling with field (default: -1)
#     norm : {'log', 'linear'}, optional
#         Color normalization (default: 'log')
#     cmap : str, optional
#         Matplotlib colormap name (default: 'viridis')
#     vmin : float, optional
#         Minimum value for color scaling in code units
#     vmax : float, optional
#         Maximum value for color scaling in code units
# 
#     Returns
#     -------
#     fig : matplotlib.figure.Figure
#         Figure object
#     ax : matplotlib.axes.Axes
#         Axes object
# 
#     Examples
#     --------
#     >>> # Basic tree structure
#     >>> fig, ax = plot_tree(42, peak_tree)
# 
#     >>> # Colored by mass with custom colormap
#     >>> fig, ax = plot_tree(42, peak_tree, tracks=tracks, info=info,
#     ...                     field='mass', cmap='plasma')
# 
#     >>> # Colored by concentration with linear scaling
#     >>> fig, ax = plot_tree(42, peak_tree, tracks=tracks, info=info,
#     ...                     field='c200', norm='linear')
#     """
#     fig, ax = plt.subplots(1, 1, figsize=(8, 10))
# 
#     # Parse kwargs
#     field = kwargs.get("field", 'mass')
#     marker_size = kwargs.get("marker_size", -1)
#     norm = kwargs.get("norm", 'log')
#     norm_func = plt.cm.colors.LogNorm if norm == 'log' else plt.cm.colors.Normalize
#     cmap = kwargs.get("cmap", 'viridis')
#     _vmin = kwargs.get("vmin", None)
#     _vmax = kwargs.get("vmax", None)
# 
#     # Collect merger tree
#     merger_tree = collect(peak_tree, itrack)
# 
#     # Plot tree structure
#     for idb in merger_tree:
#         ind = np.where(peak_tree.birth_id == idb)
#         idm = peak_tree.merging_id[ind][0].astype(int)
#         tm = peak_tree.merging_date[ind][0]
#         tb = peak_tree.birth_date[ind][0]
#         tend = 0 if tm == -1000.0 else tm
# 
#         # Horizontal segment (merger connection)
#         ax.plot([idm, idb], [tend, tend], color='k', linewidth=1.5, zorder=0)
# 
#         # Vertical segment (lifetime)
#         ax.plot([idb, idb], [tb, tend], color='k', linewidth=1.5, zorder=0)
# 
#         # Add colored scatter points if tracks provided
#         if tracks is not None and info is not None:
#             valid_mask = (
#                 (tracks[field][idb - 1] > 0) &
#                 (tracks['aexp'][idb - 1] > 0) &
#                 (tracks['texp'][idb - 1] <= tend)
#             )
# 
#             # Setup colorbar once for root halo
#             if idb == merger_tree[0]:
#                 vmin = _vmin if _vmin is not None else np.min(tracks[field][idb - 1][valid_mask])
#                 vmax = _vmax if _vmax is not None else np.max(tracks[field][idb - 1][valid_mask])
# 
#                 # Unit conversions
#                 conversion_factor = {
#                     'mass': (info.unit_d * info.unit_l**3) / 2e33,  # to Msun
#                     'r200': info.unit_l * 3.24078e-25  # to kpc
#                 }.get(field, 1.0)
# 
#                 label = {
#                     'mass': r'$M_{200} \; [\mathrm{M}_\odot]$',
#                     'r200': r'$r_{200} \; [\mathrm{kpc}]$',
#                     'c200': r'$c_{200}$'
#                 }.get(field, field)
# 
#                 ax.set_xlabel('Birth ID')
#                 ax.set_ylabel('Time [code units]')
# 
#                 fig.colorbar(
#                     plt.cm.ScalarMappable(
#                         norm=norm_func(vmin * conversion_factor, vmax * conversion_factor),
#                         cmap=cmap
#                     ),
#                     ax=ax,
#                     orientation='vertical',
#                     label=label
#                 )
# 
#             # Scale marker size
#             s = marker_size if marker_size > 0 else 1e3 * (tracks[field][idb - 1][valid_mask] / vmax)
# 
#             # Plot colored points
#             ax.scatter(
#                 np.full(np.sum(valid_mask), idb),
#                 tracks['texp'][idb - 1][valid_mask],
#                 c=tracks[field][idb - 1][valid_mask],
#                 s=s,
#                 norm=norm_func(vmin, vmax),
#                 cmap=cmap,
#                 edgecolor='k',
#                 linewidth=0.5,
#                 zorder=10
#             )
# 
#     ax.grid(alpha=0.3)
#     return fig, ax
