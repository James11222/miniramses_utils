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
# HELPER FUNCTIONS
# =============================================================================

def f(x):
    """
    NFW profile helper function.

    Parameters
    ----------
    x : float or array-like
        Dimensionless radius (r/rs)

    Returns
    -------
    float or array-like
        Value of ln(1+x) - x/(1+x)
    """
    return np.log(1 + x) - x / (1 + x)

def compute_M200(r200, c200, info):
    """
    Compute M200c for an NFW halo.

    Parameters
    ----------
    r200 : float or array-like
        r200c (where density = 200 * critical density) [code units]
    c200 : float or array-like
        Concentration parameter for c200c (r200c / rs)
    info : miniramses.Info
        Info object containing cosmological parameters

    Returns
    -------
    float or array-like
        M200c mass [code units]
    """

    rho_m = 1.0 # Mean matter density in code units 
    omega_mz = info.omega_m * info.aexp**(-3) / (info.omega_m * info.aexp**(-3) + info.omega_l)
    d200c = 200 / omega_mz  # 200 times the critical density
    
    M200 = (4 * np.pi / 3) * r200**3 * d200c * rho_m
    return M200

def compute_vmax(r200, c200, rmax, info):
    """
    Compute maximum circular velocity (Vmax) for an NFW halo.

    Uses the formalism from Mo, van den Bosch & White (2010),
    Galaxy Formation and Evolution, pages 352-353.

    Parameters
    ----------
    r200 : float or array-like
        Virial radius (where density = 200 * critical density) [code units]
    c200 : float or array-like
        Concentration parameter (r200 / rs)
    rmax : float or array-like
        Radius of maximum circular velocity [code units]
    info : miniramses.Info
        Info object containing cosmological parameters

    Returns
    -------
    float or array-like
        Maximum circular velocity [code units]
    """
    rho_m = 1.0  # Mean matter density in code units
    G = 3 / (8 * np.pi) * info.omega_m * info.aexp  # Gravitational constant (H0=1)

    rs = r200 / c200  # Scale radius
    xmax = rmax / rs  # Dimensionless radius at Vmax

    # Characteristic overdensity
    delta_char = (200 / 3) * (c200**3 / f(c200))

    # Mass enclosed at rmax
    M_max = 4 * np.pi * rho_m * delta_char * rs**3 * f(xmax)

    # Maximum circular velocity
    V_max = np.sqrt(G * M_max / rmax)

    return V_max


def compute_max_running_average(x, y, window_size=3):
    """
    Compute the maximum running average of y over a sliding window.

    Parameters
    ----------
    x : array-like
        Independent variable (e.g., time or scale factor)
    y : array-like
        Dependent variable to average (e.g., mass)
    window_size : int, default=3
        Number of points in the sliding window

    Returns
    -------
    x_ave_max : float
        Value of x corresponding to maximum running average
    y_ave_max : float
        Maximum running average value
    """

    i = 0
    y_ave_max, x_ave_max = 0, 0
    y_window, x_window = [], []

    if len(x) <= window_size: # not enough points for running average
        return max(x), max(y)

    while i < len(x):
        if len(y_window) < window_size - 1:
            y_window.append(y[i]); x_window.append(x[i])
        else:
            y_window.append(y[i]); x_window.append(x[i])
            y_ = sum(y_window) / window_size; x_ = sum(x_window) / window_size
            y_ave_max = np.maximum(y_ave_max, y_)
            if y_ave_max == y_: x_ave_max = x_
            y_window.pop(0); x_window.pop(0)
        i += 1

    return x_ave_max, y_ave_max


# =============================================================================
# CORE PARALLELIZATION FUNCTIONS
# =============================================================================

def process_snapshot(iout, ngal, ind, sigma, verbose, kwargs):
    """
    Process a single simulation snapshot in parallel.

    Reads tree particle data and clump catalogs for one output, extracts
    positions, velocities, and halo properties for all tracked particles.

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
    kwargs : dict
        Additional arguments for miniramses readers

    Returns
    -------
    dict or None
        Dictionary containing arrays for this snapshot, or None if failed/empty
        Keys: iout, aexp, texp, r200, c200, rmax, vmax, mass, mpatch,
              x, y, z, vx, vy, vz, peak, pop, merge
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
    # Initialize output arrays
    # -------------------------------------------------------------------------
    aexp_snap = np.zeros(ngal)
    texp_snap = np.zeros(ngal)
    r200_snap = np.zeros(ngal)
    c200_snap = np.zeros(ngal)
    rmax_snap = np.zeros(ngal)
    vmax_snap = np.zeros(ngal)
    mass_snap = np.zeros(ngal)
    mpatch_snap = np.zeros(ngal)
    m200_snap = np.zeros(ngal)
    x_snap = np.zeros(ngal)
    y_snap = np.zeros(ngal)
    z_snap = np.zeros(ngal)
    vx_snap = np.zeros(ngal)
    vy_snap = np.zeros(ngal)
    vz_snap = np.zeros(ngal)
    peak_snap = np.zeros(ngal)
    pop_snap = np.zeros(ngal)
    merge_snap = np.zeros(ngal)
    birth_id_snap = np.zeros(ngal)
    tracking_id_snap = np.zeros(ngal)
    merging_id_snap = np.zeros(ngal)

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
    # Process each tracked particle
    # -------------------------------------------------------------------------
    ii = np.argsort(tt.birth_id)  # Sort by birth_id for consistent ordering

    # Build lookup dictionary for O(1) clump matching (optimization)
    clump_dict = {cc.index[i]: i for i in range(len(cc.index))}

    for igal in range(len(ii)):
        ind_gal = ii[igal]

        # Extract particle metadata
        peak_id = tt.peak_id[ind_gal]
        orphan = tt.tracking_id[ind_gal] == 0
        merged = tt.merging_id[ind_gal] > 0

        # Record time information
        aexp_snap[igal] = info.aexp
        texp_snap[igal] = info.texp

        # Record kinematics
        x_snap[igal] = tt.pos[0, ind_gal]
        y_snap[igal] = tt.pos[1, ind_gal]
        z_snap[igal] = tt.pos[2, ind_gal]
        vx_snap[igal] = tt.vel[0, ind_gal]
        vy_snap[igal] = tt.vel[1, ind_gal]
        vz_snap[igal] = tt.vel[2, ind_gal]
        birth_id_snap[igal] = tt.birth_id[ind_gal]
        tracking_id_snap[igal] = tt.tracking_id[ind_gal]
        merging_id_snap[igal] = tt.merging_id[ind_gal]

        # # Match to clump catalog
        # ipeak_arr = np.where(cc.index == peak_id)[0]

        # if len(ipeak_arr) > 0:
        #     ipeak = ipeak_arr[0]

        # Use dictionary lookup instead of np.where
        if peak_id > 0 and peak_id in clump_dict:
            ipeak = clump_dict[peak_id]

            # Extract halo properties (only for non-orphan, non-merged)
            if (not orphan) and (not merged):
                r200_snap[igal] = cc.r200[ipeak]
                rmax_snap[igal] = cc.rmax[ipeak]
                c200_snap[igal] = cc.c200[ipeak]
                mass_snap[igal] = cc.mass[ipeak]
                mpatch_snap[igal] = cc.mpatch[ipeak]
                peak_snap[igal] = peak_id

                # Compute Vmax with optional scatter
                vmax_snap[igal] = compute_vmax(
                    cc.r200[ipeak],
                    cc.c200[ipeak],
                    cc.rmax[ipeak],
                    info
                ) * (10 ** np.random.normal(0, sigma))

                m200_snap[igal] = compute_M200(
                    cc.r200[ipeak],
                    cc.c200[ipeak],
                    info
                )

            # Classify population type
            is_most_massive_central = (cc.index[ipeak] == cc.parent[ipeak]) and \
                                     (cc.index[ipeak] == cc.halo[ipeak])
            is_central = cc.index[ipeak] == cc.parent[ipeak]

            if is_most_massive_central:
                pop_snap[igal] = 2  # Most massive central
            elif is_central:
                pop_snap[igal] = 1  # Central
            else:
                pop_snap[igal] = 3  # Satellite

            if merged:
                merge_snap[igal] = 1

    # -------------------------------------------------------------------------
    # Return packaged results
    # -------------------------------------------------------------------------
    return {
        'iout': iout,
        'aexp': aexp_snap, 'texp': texp_snap,
        'r200': r200_snap, 'c200': c200_snap, 'rmax': rmax_snap, 'vmax': vmax_snap,
        'mass': mass_snap, 'mpatch': mpatch_snap, 'm200': m200_snap,
        'x': x_snap, 'y': y_snap, 'z': z_snap,
        'vx': vx_snap, 'vy': vy_snap, 'vz': vz_snap,
        'peak': peak_snap, 'pop': pop_snap, 'merge': merge_snap,
        'birth_id':birth_id_snap,
        'tracking_id': tracking_id_snap, 'merging_id': merging_id_snap
    }


# =============================================================================
# MAIN TRACKING FUNCTION
# =============================================================================

def mk_track(snapshot, **kwargs):
    """
    Build time-series tracks for particles across simulation outputs.

    Reads tree/tracer particle outputs and clump catalogs for outputs 1 through
    `snapshot`, constructing complete histories for each tracked object. Supports
    parallel processing across multiple CPU cores.

    Parameters
    ----------
    snapshot : int
        Final output number to process (processes outputs 1 to snapshot inclusive)

    Other Parameters
    ----------------
    n_cores : int, optional
        Number of CPU cores for parallel processing (default: cpu_count() - 1)
    bins : int, optional
        Smoothing window size for mass histories (default: 10)
    sigma : float, optional
        Lognormal scatter to apply to Vmax (default: 0.0)
    window_size : int, optional
        Window size for maximum mass computation (default: 3)
    mass_max : bool, optional
        Compute maximum mass history per track (default: True)
    verbose : bool, optional
        Print diagnostic information (default: True)

    Returns
    -------
    dict
        Dictionary with keys:
        - aexp : (ngal, snapshot) - Expansion factor at each output
        - texp : (ngal, snapshot) - Proper time at each output
        - r200 : (ngal, snapshot) - Virial radius
        - c200 : (ngal, snapshot) - Concentration parameter
        - rmax : (ngal, snapshot) - Radius of maximum circular velocity
        - vmax : (ngal, snapshot) - Maximum circular velocity
        - mass : (ngal, snapshot) - Halo mass (smoothed)
        - mpatch : (ngal, snapshot) - Patch mass
        - x, y, z : (ngal, snapshot) - Cartesian positions
        - vx, vy, vz : (ngal, snapshot) - Cartesian velocities
        - peak_id : (ngal, snapshot) - Associated peak ID
        - pop : (ngal, snapshot) - Population type (1=central, 2=massive central,
                                                     3=satellite, 4=orphan)
        - merge_arr : (ngal, snapshot) - Merger flag (1 if merged)
        - mass_max : (ngal,) - Maximum mass achieved by each track

    Examples
    --------
    >>> # Process 100 outputs using 8 cores
    >>> tracks = mk_track(100, n_cores=8)

    >>> # Sequential processing with custom smoothing
    >>> tracks = mk_track(50, n_cores=1, bins=20)
    """
    # -------------------------------------------------------------------------
    # Parse arguments and initialize
    # -------------------------------------------------------------------------
    verbose = kwargs.get("verbose", True)
    sigma = kwargs.get("sigma", 0.0)
    window_size = kwargs.get("window_size", 3)
    mass_max_flag = kwargs.get("mass_max", True)
    n_cores = kwargs.get("n_cores", cpu_count() - 1)
    nsmooth_in = kwargs.get("bins", 10)

    # Read initial snapshot to determine number of tracks
    t = ram.rd_part(snapshot, prefix="tree", silent=True, **kwargs)
    ind = np.argsort(t.birth_id)
    ngal = len(ind)

    # Print configuration
    if verbose:
        print(f"Found {ngal} tracks")
        print(f"Using {n_cores} cores for parallel processing")
        print(f"Applying mass smoothing with window size: {nsmooth_in}")
        print(f"Applying scatter to vmax with sigma: {sigma}")
        print(f"Mass max flag is set to: {mass_max_flag}")
        print(f"Finding maximum mass history with window size: {window_size}")

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
    aexp = np.zeros((ngal, snapshot))
    texp = np.zeros((ngal, snapshot))
    r200 = np.zeros((ngal, snapshot))
    c200 = np.zeros((ngal, snapshot))
    rmax = np.zeros((ngal, snapshot))
    vmax = np.zeros((ngal, snapshot))
    mass = np.zeros((ngal, snapshot))
    mpatch = np.zeros((ngal, snapshot))
    m200 = np.zeros((ngal, snapshot))
    x = np.zeros((ngal, snapshot))
    y = np.zeros((ngal, snapshot))
    z = np.zeros((ngal, snapshot))
    vx = np.zeros((ngal, snapshot))
    vy = np.zeros((ngal, snapshot))
    vz = np.zeros((ngal, snapshot))
    peak = np.zeros((ngal, snapshot))
    mass_max = np.zeros(ngal)
    pop = np.zeros((ngal, snapshot))
    merge_arr = np.zeros((ngal, snapshot))
    birth_id = np.zeros((ngal, snapshot))
    tracking_id = np.zeros((ngal, snapshot))
    merging_id = np.zeros((ngal, snapshot))
    # -------------------------------------------------------------------------
    # Process snapshots in parallel
    # -------------------------------------------------------------------------
    process_func = partial(
        process_snapshot,
        ngal=ngal,
        ind=ind,
        sigma=sigma,
        verbose=verbose,
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
            aexp[:, iloop] = result['aexp']
            texp[:, iloop] = result['texp']
            r200[:, iloop] = result['r200']
            c200[:, iloop] = result['c200']
            rmax[:, iloop] = result['rmax']
            vmax[:, iloop] = result['vmax']
            mass[:, iloop] = result['mass']
            mpatch[:, iloop] = result['mpatch']
            m200[:, iloop] = result['m200']
            x[:, iloop] = result['x']
            y[:, iloop] = result['y']
            z[:, iloop] = result['z']
            vx[:, iloop] = result['vx']
            vy[:, iloop] = result['vy']
            vz[:, iloop] = result['vz']
            peak[:, iloop] = result['peak']
            pop[:, iloop] = result['pop']
            merge_arr[:, iloop] = result['merge']
            birth_id[:, iloop] = result['birth_id']
            tracking_id[:, iloop] = result['tracking_id']
            merging_id[:, iloop] = result['merging_id']

    # -------------------------------------------------------------------------
    # Apply smoothing to mass histories
    # -------------------------------------------------------------------------
    if verbose: print("\nApplying mass smoothing...")

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

    out = {
        "aexp": aexp,
        "texp": texp,
        "r200": r200,
        "rmax": rmax,
        "vmax": vmax,
        "c200": c200,
        "mass": mass,
        "mpatch": mpatch,
        "m200": m200,
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "peak_id": peak,
        "mass_max": mass_max,
        "pop": pop,
        "merge_arr": merge_arr,
        "birth_id": birth_id,
        "tracking_id": tracking_id,
        "merging_id": merging_id
    }

    # -------------------------------------------------------------------------
    # Compute maximum mass histories
    # -------------------------------------------------------------------------
    if mass_max_flag:
        if verbose: print("\nComputing maximum mass histories...")

        for i in range(ngal):
            mmm = out["mass"][i]
            aaa = out["aexp"][i]
            ind = np.where((mmm > 0) & (aaa > 0))[0]

            if len(ind) > 0:
                _, y_ave_max = compute_max_running_average(
                    aaa[ind],
                    mmm[ind],
                    window_size=window_size
                )
                out["mass_max"][i] = y_ave_max

    # -------------------------------------------------------------------------
    # Package and return results
    # -------------------------------------------------------------------------
    if verbose:
        print("\n✓ Track processing complete!")

    return out


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
