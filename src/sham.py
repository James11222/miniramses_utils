import numpy as np
import treecorr
from astropy import units as u
import miniramses as ram
from scipy.interpolate import interp1d
import pyccl as ccl

def xi_powerlaw(info, **kwargs):

    rmin = kwargs.get("rmin", 0.1) # h^-1 Mpc
    rmax = kwargs.get("rmax", 10.0) # h^-1 Mpc
    nbins = kwargs.get("nbins", 100)

    rmin = rmin / info.h  # convert to Mpc
    rmax = rmax / info.h  # convert to Mpc

    # Power-law fit parameters from the SDSS data
    r0_h_inv_mpc = 6.1   # r0 in h^-1 Mpc
    gamma = 1.84

    # Convert r0 to Mpc using h
    r0_mpc = r0_h_inv_mpc / info.h

    # Define projected separation rp in Mpc (converted from 10 h^-1 kpc to 10 h^-1 Mpc)
    rp_mpc = np.logspace(np.log10(rmin), np.log10(rmax), nbins)  # in Mpc

    # Compute the projected 3D correlation function ξ*(r)
    xi_r = (rp_mpc / r0_mpc) ** (-gamma)

    return rp_mpc, xi_r

def compute_xi(x,y,z,info,**kwargs):
    """
    This function computes the 3D two-point correlation function using the Landy-Szalay estimator via the TreeCorr package.

    Parameters:
    x, y, z : array-like
        The coordinates of the points in 3D space, assuming a cubic box
        with periodic boundary conditions in code units.
    **kwargs : dict
        Additional keyword arguments to customize the TreeCorr NNCorrelation object.
    Returns:
    r : array-like
        The separation distances at which the correlation function is computed.
    xi : array-like
        The computed two-point correlation function values at the corresponding separations.
    """

    # default parameters
    min_sep  = kwargs.get('min_sep', 0.1 / (info.unit_l * info.h).to(u.Mpc).value)
    max_sep  = kwargs.get('max_sep', 10 / (info.unit_l * info.h).to(u.Mpc).value)
    nbins    = kwargs.get('nbins', 50)
    bin_type = kwargs.get('bin_type', 'Log')
    metric   = kwargs.get('metric', 'Euclidean')
    nr       = kwargs.get('nr', 50)

    config = {
        'min_sep': min_sep,
        'max_sep': max_sep,
        'nbins': nbins,
        'bin_type': bin_type,
        'metric': metric,
    }

    N = x.size

    data_cat = treecorr.Catalog(x = x, y = y, z = z)
    randoms_cat = treecorr.Catalog(x = np.random.rand(nr*N),
                                   y = np.random.rand(nr*N),
                                   z = np.random.rand(nr*N))

    nn = treecorr.NNCorrelation(config)
    rr = treecorr.NNCorrelation(config)
    dr = treecorr.NNCorrelation(config)

    nn.process(data_cat)
    rr.process(randoms_cat)
    dr.process(data_cat, randoms_cat)

    xi, varxi = nn.calculateXi(rr=rr, dr=dr) #Landy-Szalay estimator : xi = (DD - 2DR + RR) / RR
    r = nn.meanr

    return r, xi #variance is so small we ignore it here

def compute_halo_Mmax(L_box, **kwargs):
    """
    This function computes what the maximum mass should be for a simulation 
    given the size of the box in units of Mpc/h. 

    Args:
        L_box     - float - size of the periodic box on any given side in units of Mpc/h 
        kwargs:
            cosmo     - ccl.Cosmology - optional - cosmology object 
            a         - float - optional - scale factor 
            N_halos   - integer - optional - the minimum number of halos desired at the maximum mass 
            mass_def  - ccl.halos.MassDef - optional - mass definition of halos 
            mass_func - ccl.halos.MassFunc - optional - halo mass function

    Return:
        Mmax - float - maximum halo mass of the simulation given the box size. Units of Msun / h.
    """

    cosmo     = kwargs.get("cosmo", ccl.Cosmology(Omega_b=0.0492, Omega_c=0.2650, h=0.6724, A_s=2.2065e-09, n_s=0.9645))
    a         = kwargs.get("a", 1.0)
    N_halos   = kwargs.get("N_halos", 10)
    mass_def  = kwargs.get("mass_def", ccl.halos.MassDef200m)
    mass_func = kwargs.get("mass_func", ccl.halos.MassFuncTinker08(mass_def=mass_def))
    cu        = kwargs.get("code_units", False)
                           
    m_arr     = np.geomspace(1E10,1E16,256)
    inverse   = interp1d(x=mass_func(cosmo, m_arr, a) / cosmo['h']**3, y=m_arr * cosmo['h']) 
    m_max     = float(inverse(N_halos / L_box**3)) # Return in Msun/h units

    if cu:
        rho_mean = ccl.rho_x(cosmo, a, 'matter') # in Msun / Mpc^3
        return m_max / (rho_mean * (L_box / cosmo['h'])**3)  # Convert to code units
    else:
        return m_max


def compute_halo_Mmin(L_box, N_part, **kwargs):
    """
    This function computes what the minimum halo mass should be for a simulation
    given the size of the box in units of Mpc/h and the number of particles on a side.

    Args:
        L_box   - float - size of the periodic box on any given side in units of Mpc/h 
        N_part  - integer - number of particles on a side of the box    
        kwargs:
            cosmo   - ccl.Cosmology - optional - cosmology object 
            a       - float - optional - scale factor
    
    Return:
        Mmin - float - minimum halo mass of the simulation given the box size and number of particles. Units of Msun / h.
    """

    cosmo     = kwargs.get("cosmo", ccl.Cosmology(Omega_b=0.0492, Omega_c=0.2650, h=0.6724, A_s=2.2065e-09, n_s=0.9645))
    a         = kwargs.get("a", 1.0)
    cu        = kwargs.get("code_units", False)

    m_part    = 1/int(N_part**3)

    if cu:
        return 200 * m_part  # Return in code units
    else:
        rho_mean  = ccl.rho_x(cosmo, a, 'matter') * (u.Msun/u.Mpc**3)

        L_box = L_box * u.Mpc / cosmo['h'] # Convert from Mpc/h to Mpc
        m_part = m_part * (rho_mean * L_box**3) # mass per particle in Msun

        return float((200 * m_part).value * cosmo['h'])  # Return in Msun/h units

