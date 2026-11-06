import numpy as np
import miniramses as ram 
import treecorr
import astropy.units as u  
import astropy.constants as const
import mk_tracks as mkt
from scipy.interpolate import interp1d
import sham
import pyccl as ccl
import os 
import hashlib

class Simulation:
    """
    Simulation object class to handle halo track data and analysis.

    Parameters:
        path (str): Path to the simulation data.
        snapshot (int): Snapshot number to analyze.
        npart (int): Number of particles in the simulation.
        kwargs: Additional keyword arguments for track creation and analysis.
    
    Attributes:
        path (str): Path to the simulation data.
        snapshot (int): Snapshot number to analyze.
        npart (int): Number of particles in the simulation.
        info (object): Simulation information object from miniramses.
        L_box (float): Box size of the simulation in Mpc/h.
        sim_name (str): Name of the simulation.
        cosmo (object): Cosmology object from pyccl.
        min_halo_mass (float): Minimum halo mass in code units.
        max_halo_mass (float): Maximum halo mass in code units.
        min_halo_mass_Msun (float): Minimum halo mass in Msun/h.
        max_halo_mass_Msun (float): Maximum halo mass in Msun/h.
        tracks (dict): Dictionary containing halo tracks.

        Methods:
        make_tracks(verbose, smooth, sigma, window_size, min_part, fields): Create or load halo tracks.
        get_aexp_index(aexp): Get index of scale factor closest to specified value.
        create_mask(population, verbose, aexp): Create mask for selecting halo population.
        grab(property_name, aexp, population): Grab specified property from tracks.
        sort_halos(population, sort_by, verbose, aexp): Sort halos based on specified property.
        compute_number_densities(population, sort_by, verbose, aexp): Compute number densities for different thresholds.
        compute_2pcf(n_thresh, aexp, **kwargs): Compute 2-point correlation function for halos above threshold.
        compute_HMF(aexp, **kwargs): Compute Halo Mass Function for specified halo population.
    """
    def __init__(self, path, snapshot, npart, **kwargs):
        self.path = path
        self.snapshot = snapshot
        self.npart = npart
        self.info = ram.rd_info(snapshot, path=path, units=True)
        self.info.unit_v = self.info.unit_l / self.info.unit_t
        self.info.unit_m = self.info.unit_d * self.info.unit_l**3
        self.L_box = int(self.info.boxlen.to(u.Mpc).value * self.info.h) #Msun/h
        self.sim_name = self.path.split('/')[-1]

        # determine the min and max halo mass of the simulation
        self.cosmo = ccl.Cosmology(Omega_b=0.0492, Omega_c=(self.info.omega_m - 0.0492), 
                      h=self.info.h, A_s=2.2065e-09, n_s=0.9645)
        self.min_halo_mass = sham.compute_halo_Mmin(self.L_box, N_part=self.npart, cosmo=self.cosmo, code_units=True)
        self.max_halo_mass = sham.compute_halo_Mmax(self.L_box, cosmo=self.cosmo, code_units=True)
        self.min_halo_mass_Msun = self.min_halo_mass * (self.info.unit_m * self.info.h).to(u.Msun).value #Msun/h
        self.max_halo_mass_Msun = self.max_halo_mass * (self.info.unit_m * self.info.h).to(u.Msun).value #Msun/h

        self.make_tracks(verbose=kwargs.get('verbose', False), 
                         smooth=kwargs.get('smooth', False), 
                         sigma=kwargs.get('sigma', 0.0),
                         window_size=kwargs.get('window_size', 3),
                         min_part=kwargs.get('min_part', 200),
                         fields=kwargs.get('fields', mkt.DEFAULT_FIELDS))
        
        # experimental 
        self.delta_aout = kwargs.get('delta_aout', 0.01)

    def make_tracks(self, verbose, smooth, sigma, window_size, min_part, fields):
        """
        This function creates or loads the halo tracks for the simulation.

        Parameters:
        verbose (bool): Whether to print progress messages.
        sigma (float): Scatter to apply to vmax.
        min_part (int): Drop tracks that never exceed this many particles.
        fields (iterable of str): Track fields to build.
        Returns:
        tracks (dict): A dictionary containing the halo tracks.
        """

        file = f'{self.path}/{self.sim_name}_S{str(sigma).replace(".", "p")}_W{window_size}_P{min_part}'
        if set(fields) != set(mkt.DEFAULT_FIELDS):
            file += '_F' + hashlib.md5(','.join(sorted(fields)).encode()).hexdigest()[:8]
        file += '_tracks_smoothed.npy' if smooth else '_tracks.npy'

        if os.path.exists(file):
            if verbose: print(f'Loading tracks from {file}')
            self.tracks = np.load(file, allow_pickle=True).item()
        else:
            if verbose: print(f'Creating tracks and saving to {file}')
            self.tracks = mkt.mk_track(self.snapshot, fields=fields, path=self.path, 
                                  verbose=verbose, 
                                  bins=1,
                                  sigma=sigma,
                                  window_size=window_size,
                                  min_part=min_part)
            
            np.save(file, self.tracks, allow_pickle=True)
        
        return self.tracks

    def get_aexp_index(self, aexp):
        """
        This function returns the index of the scale factor closest to the specified value.

        Parameters:
        aexp (float): The scale factor to find the index for.

        Returns:
        index (int): The index of the scale factor closest to the specified value.
        """
        # round the scale factor to 2 decimal places and create an array of scale factors from 0 to the max scale factor in the tracks
        aexp_array = np.arange(0.0, np.round(self.info.aexp, 2) + self.delta_aout, self.delta_aout) 

        # find the index of the scale factor closest to the specified value
        if np.min(np.abs(aexp_array - aexp)) > self.delta_aout:
            raise ValueError(f"""Specified scale factor {aexp} is not within {self.delta_aout} of any scale factor in the tracks. 
                             The max scale factor in the tracks is {self.info.aexp:0.2f}.""")

        index = np.argmin(np.abs(aexp_array - aexp))
        return index
        
    def create_mask(self, population, verbose, aexp):
        """
        This function creates a mask for selecting a subset of the halos
        from the tracks based on the specified population type.

        Parameters:
        population (str): The type of halo population to select. 
                          Options are 'cen' for centrals, 'sat' for satellites, 
                          and 'all' for all halos. Additionally 'cen+' can be used 
                          to select only the most massive centrals in each halo. 
        
        Returns:
        mask (np.ndarray): A boolean mask array indicating the selected halos.
        """

        

        # select only valid halos
        valid_mask = self.tracks['mass'][:, self.get_aexp_index(aexp)] >= self.min_halo_mass 
        # valid_mask = self.tracks['mass_max'] >= self.min_halo_mass

        if verbose: print(f"Fraction of halos that have a max mass above min halo mass {np.count_nonzero(valid_mask) / len(valid_mask):0.2f}")

        # orphans have their tracking_id set to 0, so we can use that to select them, otherwise we can use pop
        if population == 'orph':
            mask = valid_mask & (self.tracks['tracking_id'][:, self.get_aexp_index(aexp)] == 0)
        elif population == 'all':
            mask = valid_mask
        else:
            if population == 'cen':    # all centrals, not including orphans
                mask = valid_mask & ((self.tracks['pop'][:, self.get_aexp_index(aexp)] == 1) 
                                 | (self.tracks['pop'][:, self.get_aexp_index(aexp)] == 2))
            elif population == 'cen+': # most massive centrals only
                mask = valid_mask & (self.tracks['pop'][:, self.get_aexp_index(aexp)] == 2)
            elif population == 'sat':  # all satellites, not including orphans
                mask = valid_mask & (self.tracks['pop'][:, self.get_aexp_index(aexp)] == 3)
            elif population == 'sat+': # most massive satellites only
                mask = valid_mask & ((self.tracks['pop'][:, self.get_aexp_index(aexp)] == 1) 
                                 | (self.tracks['pop'][:, self.get_aexp_index(aexp)] == 3))
           
        if verbose: print(f'Made mask for {population} halos: {np.count_nonzero(mask) / np.count_nonzero(valid_mask):0.2f} of all valid halos were selected.')
    
        self.mask = mask

        return mask

    def grab(self, property_name, aexp, population):
        """
        This function grabs the specified property from the tracks at the specified scale factor.

        Parameters:
        property_name (str): The name of the property to grab.
        aexp (float): The scale factor to grab the property at.
        population (str): The type of halo population to select. 
                          Options are 'cen' for centrals, 'sat' for satellites, 
                          and 'all' for all halos.

        Returns:
        property_array (np.ndarray): An array containing the values of the specified property at the specified scale factor.
        """
        
        index = self.get_aexp_index(aexp)
        return self.tracks[property_name][:, index][self.create_mask(population, verbose=False, aexp=aexp)]

    def sort_halos(self, population, sort_by, verbose, aexp):
        """
        This function sorts the halos in the clump data based on the specified property.

        Parameters:
        population (str): The type of halo population to select. 
                          Options are 'cen' for centrals, 'sat' for satellites, 
                          and 'all' for all halos.
        sort_by (str): The property to sort the halos by. Options are 'Mmax' or 'Vmax'.
        
        Returns:
        sorted_indices (np.ndarray): An array of indices that sorts the halos.
        """

        # make sure mask exists
        self.create_mask(population, verbose, aexp)

        if sort_by == 'Mmax': #to be properly implemented later, for now just sort by mass
            sorted_indices = np.argsort(self.tracks['mass_max'][self.mask])
            sorted_values  = self.tracks['mass_max'][self.mask][sorted_indices]
        elif sort_by == 'mass':
            sorted_indices = np.argsort(self.tracks['mass'][:, self.get_aexp_index(aexp)][self.mask])
            sorted_values  = self.tracks['mass'][:, self.get_aexp_index(aexp)][self.mask][sorted_indices]
        elif sort_by == 'm200':
            sorted_indices = np.argsort(self.tracks['m200'][:, self.get_aexp_index(aexp)][self.mask])
            sorted_values  = self.tracks['m200'][:, self.get_aexp_index(aexp)][self.mask][sorted_indices]
        elif sort_by == 'Vmax':
            sorted_indices = np.argsort(self.tracks['vmax'][:, self.get_aexp_index(aexp)][self.mask])
            sorted_values  = self.tracks['vmax'][:, self.get_aexp_index(aexp)][self.mask][sorted_indices]
        
        self.sorted_indices = sorted_indices
        self.sorted_values = sorted_values

        if verbose: print(f'Sorted {population} halos by {sort_by}.')

        return sorted_indices, sorted_values

    def compute_number_densities(self, population, sort_by, verbose, aexp):
        """
        This function computes the number density of halos for different Vmax thresholds.

        Parameters:
        population (str): The type of halo population to select. 
                          Options are 'cen' for centrals, 'sat' for satellites, 
                          and 'all' for all halos.
        sort_by (str): The property to sort the halos by. Options are 'Mmax' or 'Vmax'.
        
        Returns:
        n_inv (interp1d): An interpolation function that gives the threshold Vmax 
                          as a function of number density.
        """

        # make sure halos are sorted
        self.sort_halos(population, sort_by, verbose, aexp)

        box_volume = (self.info.boxlen**3).to(u.Mpc**3).value # in Mpc^3
        
        # Since sorted_values is already sorted, we can compute number densities directly
        # Number of halos above threshold i is simply (total_halos - i)
        N_halos = len(self.sorted_values)
        counts_above = np.arange(N_halos, 0, -1)  # [N, N-1, N-2, ..., 1]
        number_densities = counts_above / box_volume
        thresholds = self.sorted_values

        n_inv = interp1d(number_densities[::-1], 
                        thresholds[::-1], 
                        kind='linear', 
                        fill_value='extrapolate')

        self.n_inv = n_inv
        self.number_densities = number_densities
        self.thresholds = thresholds

        if verbose: print(f'Computed number densities for {population} halos sorted by {sort_by}.')

        return n_inv, thresholds, number_densities

    def compute_2pcf(self, n_thresh, aexp, **kwargs):
        """
        This function computes the 2-point correlation function (2PCF) for halos above a specified number density threshold.
        Parameters:
        n_thresh (float): The desired number density threshold in units of (Mpc/h)^-3.
        kwargs: Additional keyword arguments for mask creation, halo sorting, and 2PCF computation.
            nbins (int): Number of bins for the 2PCF computation. Default is 50.
            rmax (float): Maximum separation for the 2PCF computation in Mpc/h. Default is 50.0 Mpc/h.
            rmin (float): Minimum separation for the 2PCF computation in Mpc/h. Default is 0.1 Mpc/h.
            population (str): The type of halo population to select. Options are 'cen' for centrals, 'sat' for satellites, and 'all' for all halos. Default is 'cen'.
            sort_by (str): The property to sort the halos by. Options are 'Mmax' or 'Vmax'. Default is 'Vmax'.
            verbose (bool): Whether to print progress messages. Default is True.

        Returns:
        r (np.ndarray): The separation bins in Mpc/h.
        xi (np.ndarray): The 2-point correlation function values corresponding to the separation bins.
        """
        nbins      = kwargs.get('nbins', 50)
        rmax       = kwargs.get('rmax', 50.0) #Mpc/h
        rmin       = kwargs.get('rmin', 0.1)  #Mpc/h 
        population = kwargs.get('population', 'cen')
        sort_by    = kwargs.get('sort_by', 'Vmax')
        verbose    = kwargs.get('verbose', True)

        # make sure the halos have been sorted and number densities computed
        self.compute_number_densities(population, sort_by, verbose, aexp)

        # select halos above the value threshold corresponding to the desired number density
        self.threshold_needed = self.n_inv(n_thresh * self.info.h**3)  # threshold value needed for desired number density
        if sort_by == 'Vmax':
            if verbose: print(f"Threshold Vmax needed for n={n_thresh:0.2e} h^3 Mpc^-3: {self.threshold_needed * (self.info.unit_v).to(u.km/u.s).value :.2f} km/s")
            mask = (self.tracks['vmax'][:, self.get_aexp_index(aexp)] > self.threshold_needed) & self.mask
        elif sort_by == 'mass':
            if verbose: print(f"Threshold mass needed for n={n_thresh:0.2e} h^3 Mpc^-3: {self.threshold_needed * (self.info.unit_m * self.info.h).to(u.Msun).value :.2e} Msun/h")
            mask = (self.tracks['mass'][:, self.get_aexp_index(aexp)] > self.threshold_needed) & self.mask
        elif sort_by == 'm200':
            if verbose: print(f"Threshold m200 needed for n={n_thresh:0.2e} h^3 Mpc^-3: {self.threshold_needed * (self.info.unit_m * self.info.h).to(u.Msun).value :.2e} Msun/h")
            mask = (self.tracks['m200'][:, self.get_aexp_index(aexp)] > self.threshold_needed) & self.mask
        elif sort_by == 'Mmax':
            if verbose: print(f"Threshold Mmax needed for n={n_thresh:0.2e} h^3 Mpc^-3: {self.threshold_needed * (self.info.unit_m * self.info.h).to(u.Msun).value :.2e} Msun/h")
            mask = (self.tracks['mass_max'] > self.threshold_needed) & self.mask
        if verbose: print(f"Number of halos selected for 2PCF: {np.count_nonzero(mask)}")

        # grab positions of remaining halos 
        x, y, z = self.tracks['x'][:, self.get_aexp_index(aexp)][mask], self.tracks['y'][:, self.get_aexp_index(aexp)][mask], self.tracks['z'][:, self.get_aexp_index(aexp)][mask]

        # compute the 2PCF
        r, xi = sham.compute_xi(x, y, z, self.info,
                                nbins=nbins,
                                min_sep = rmin / (self.info.unit_l * self.info.h).to(u.Mpc).value, 
                                max_sep = rmax/ (self.info.unit_l * self.info.h).to(u.Mpc).value,
        )
    
        return r * (self.info.unit_l * self.info.h).to(u.Mpc).value, xi  #Mpc/h, unitless
    
    def compute_HMF(self, aexp, **kwargs):
        """
        This function computes the Halo Mass Function (HMF) for the specified halo population.
        The HMF is calculated as the differential number density of halos per unit volume per dlog(M).
        Parameters:
        kwargs: Additional keyword arguments for mask creation and HMF computation.
            nbins (int): Number of bins for the HMF computation. Default is 50.
            population (str): The type of halo population to select. Options are 'cen' for centrals, 'sat' for satellites, and 'all' for all halos. Default is 'cen'.
            verbose (bool): Whether to print progress messages. Default is True.

        Returns:
        bin_centers (np.ndarray): The centers of the mass bins in Msun/h.
        dndlogM (np.ndarray): The differential number density of halos per unit volume per
                                dlog(M) in units of (Mpc/h)^-3.
        """

        nbins = kwargs.get('nbins', 50)
        population = kwargs.get('population', 'cen')
        verbose = kwargs.get('verbose', True)
        weighted = kwargs.get('weighted', False)
        
        # make sure clump data is loaded
        self.create_mask(population, verbose, aexp)

        log_bins = np.logspace(np.log10(self.min_halo_mass), np.log10(self.max_halo_mass), nbins+1)
        counts, bin_edges = np.histogram(self.tracks['mass'][:, self.get_aexp_index(aexp)][self.mask], bins=log_bins)

        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean
        volume = (self.info.boxlen.to(u.Mpc).value) ** 3

        # Calculating the differential mass function in terms of counts per unit volume per dlog(M)
        dlogM = np.diff(np.log10(bin_edges))
        dndlogM = counts / (volume * dlogM)
        errors = np.sqrt(counts) / (volume * dlogM)
         
        bin_centers = bin_centers * (self.info.unit_m * self.info.h).to(u.Msun).value #Msun/h
        dndlogM = dndlogM / (self.info.h**3)  # (Mpc/h)^-3
        errors = errors / (self.info.h**3)  # (Mpc/h)^-3

        if weighted:
            return bin_centers, bin_centers * dndlogM, bin_centers * errors,  #Msun/h, h^2 Msun/Mpc^3 
        else:
            return bin_centers, dndlogM, errors  #Msun/h, (Mpc/h)^-3
        
    def compute_binned_statistics(self, x, y, n_bins=10):
        """
        Compute binned averages and standard deviations of y based on x values.
        
        Parameters:
        -----------
        x : array-like
            Independent variable values
        y : array-like
            Dependent variable values
        n_bins : int, optional
            Number of bins to create (default: 10)
        
        Returns:
        --------
        bin_centers : numpy array
            Center point of each bin
        bin_means : numpy array
            Mean of y values in each bin
        bin_stds : numpy array
            Standard deviation of y values in each bin
        """
        
        # Create bins based on x values
        bins = np.geomspace(x.min(), x.max(), n_bins + 1)
        
        # Assign each point to a bin
        bin_indices = np.digitize(x, bins)
        
        # Initialize arrays for results
        bin_centers = np.zeros(n_bins)
        bin_means = np.zeros(n_bins)
        bin_stds = np.zeros(n_bins)
        
        # Compute statistics for each bin
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_centers[i-1] = (bins[i-1] + bins[i]) / 2
                bin_means[i-1] = np.mean(y[mask])
                bin_stds[i-1] = np.std(y[mask], ddof=1)  # Sample std
            else:
                bin_centers[i-1] = (bins[i-1] + bins[i]) / 2
                bin_means[i-1] = np.nan
                bin_stds[i-1] = np.nan
        
        return bin_centers, bin_means, bin_stds

