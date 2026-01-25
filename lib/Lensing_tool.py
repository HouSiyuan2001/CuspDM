import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from jax import jit
import jax

import jax.scipy.ndimage as jndimage
from functools import partial

import numpy as np
import sys
from tqdm import tqdm
from scipy.optimize import root_scalar
from lenstronomy.Cosmo.nfw_param import NFWParam

from sklearn.cluster import DBSCAN
from collections import Counter
from itertools import combinations

import itertools
sys.path.append("./libs") 

from astropy.table import Table
from astropy import units as u 
import astropy.constants as const 
from astropy.cosmology import FlatwCDM,FlatLambdaCDM

vc = const.c.to(u.km/u.s).value 
G =  const.G.to(u.Mpc/u.solMass *(u.km/u.second)*(u.km/u.second)).value 
apr =  1.0/np.pi*180.0*3600  # arcsec per rad

H0 = 70.0  # Hubble constant in km/s/Mpc
Omega_b = 0.05  # Baryon density parameter
Omega_c = 0.25  # Cold dark matter density parameter
Omega_m = Omega_b+Omega_c  # Total matter density parameter
Omega_Lambda = 0.7  # Dark energy density parameter
Omega_k = 0.0  # Curvature density parameter (set to 0 for a flat universe)
w0 = -1.0  # Equation of state parameter for dark energy

cosmo = FlatwCDM(H0=H0, Om0=Omega_m, Ob0=Omega_b, w0=w0)
# cosmo=FlatLambdaCDM(H0=72,Om0=0.26,Tcmb0=2.725)


# Virial overdensity
def dv(z): 
    ov = 1.0/cosmo.Om(z)-1.0
    res = 18.8*np.pi*np.pi*(1.0+0.4093*ov**0.9052)
    return res

# Calculate r200
def rvir_mvir(m, z, stype="vir"):
    if stype == "vir":
        res = (3.0 * m / 4.0 / np.pi / rho_crit(z) / dv(z))**(1.0 / 3.0)
    elif stype == "200":
        res = (3.0 * m / 4.0 / np.pi / rho_crit(z) / 200.0)**(1.0 / 3.0)
    else:
        print("wrong stype!!!")
    return res

def SigmaCrit(z1, z2):
    '''
        Critical surface density for the case of lens plane at z1 and source plane at z2.
    '''
    res = (vc * vc / 4.0 / jnp.pi / G * Dc(z2) / (Dc(z1) / (1.0 + z1)) / Dc2(z1, z2))
    return res

def rho_crit(z, densType="crit"): 
    if densType == "matter":
        # Matter density 
        res = cosmo.Om(z) * cosmo.critical_density(z).to(u.solMass / u.Mpc / u.Mpc / u.Mpc).value / cosmo.h / cosmo.h # M_sun Mpc^-3 *h^2
    elif densType == "crit":
        # Critical density
        res = cosmo.critical_density(z).to(u.solMass / u.Mpc / u.Mpc / u.Mpc).value / cosmo.h / cosmo.h # M_sun Mpc^-3 *h^2
    else:
        print("error!!!")
    return res

# Precompute cosmological distances using Astropy
# Calculate comoving distance
def Dc0(z):
    res = cosmo.comoving_distance(z).value * cosmo.h  #Mpc/h
    return res

# Comoving distance between two points
def Dc20(z1, z2):
    Dcz1 = (cosmo.comoving_distance(z1).value * cosmo.h)
    Dcz2 = (cosmo.comoving_distance(z2).value * cosmo.h)
    res = Dcz2 - Dcz1 + 1e-8
    return res

# Angular diameter distance
def Da0(z):
    res = cosmo.angular_diameter_distance(z).value * cosmo.h
    return res #Mpc/h

def Da20(z1, z2):
    res = cosmo.angular_diameter_distance_z1z2(z1, z2).value * cosmo.h
    return res #Mpc/h

# Function to calculate comoving distance using JAX and trapezoid integration
@jit
def E_func(z):
    return jnp.sqrt(Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_Lambda)

@jit
def Dc(z):
    # Create an array of redshift values to integrate over
    z_values = jnp.linspace(0.0, z, 1000000)  # Create a fine grid of z values for integration
    
    # Calculate E(z) values for each z
    E_values = E_func(z_values)
    
    # Perform numerical integration using trapezoid rule
    integral = trapezoid(1.0 / E_values, z_values)
    
    # Calculate the comoving distance
    distance = (vc / H0) * integral  # Return the comoving distance in Mpc
    return distance*cosmo.h #Mpc/h
@jit
def Dc2(z1,z2):
    Dcz1 = Dc(z1)
    Dcz2 = Dc(z2)
    res = Dcz2-Dcz1+1e-8
    return res
# Function to calculate angular diameter distance from comoving distance
@jit
def Da(z):
    # Calculate the comoving distance first
    D_C = Dc(z)
    
    # Apply the formula D_A(z) = D_C(z) / (1+z)
    D_A = D_C / (1 + z)
    
    return D_A #Mpc/h
# Function to calculate angular diameter distance between two redshifts
@jit
def Da2(z1, z2):
    # Calculate the comoving distance first
    D_C = Dc(z2) - Dc(z1)
    
    # Apply the formula D_A(z1, z2) = D_C(z1, z2) / (1+z2)
    D_A = D_C / (1 + z2)
    
    return D_A #Mpc/h

def potential_to_alphas(potential, dsx_arc):
    alpha1 = jnp.gradient(potential, dsx_arc, axis=0)
    alpha2 = jnp.gradient(potential, dsx_arc, axis=1)
    return alpha1, alpha2

def get_eff_kappa(alpha1_in, alpha2_in, dsx_arc):
    al11_tmp, al12_tmp = jnp.gradient(alpha1_in, dsx_arc)
    al21_tmp, al22_tmp = jnp.gradient(alpha2_in, dsx_arc)
    kappa = 0.5*(al11_tmp + al22_tmp)
    return kappa
def adding_external_to_alpha(alpha1_in, alpha2_in,xi1,xi2, external_kappa):
    alpha1_out = alpha1_in+ xi1*external_kappa
    alpha2_out = alpha2_in+ xi2*external_kappa
    return alpha1_out, alpha2_out

def SigmaCrit0(z1, z2):
    '''
        Critical surface density for the case of lens plane at z1 and source plane at z2.
    '''
    res = (vc * vc / 4.0 / np.pi / G * Dc0(z2) / (Dc0(z1) / (1.0 + z1)) / Dc20(z1, z2))
    return res

def alphas_to_mu(alpha1_in, alpha2_in, dsx_arc, xi1, xi2, external_kappa=None):
    al11_tmp, al12_tmp = jnp.gradient(alpha1_in, dsx_arc)
    al21_tmp, al22_tmp = jnp.gradient(alpha2_in, dsx_arc)

    if external_kappa is not None:
        kappa = 0.5*(al11_tmp + al22_tmp)+external_kappa
    else:
        kappa = 0.5*(al11_tmp + al22_tmp)
    gamma1 = 0.5*(al22_tmp - al11_tmp)
    gamma2 = al12_tmp
    gamma_sq = gamma1**2.0 + gamma2**2.0 
    # mu_out = 1.0/(1.0 - (al11_tmp + al22_tmp) + al11_tmp*al22_tmp - al12_tmp*al21_tmp)
    mu_out = 1.0/((1.0 - kappa)**2 - gamma_sq)
    y1_out = xi1-alpha1_in
    y2_out = xi2-alpha2_in
    return y1_out, y2_out, mu_out, kappa, gamma1, gamma2

def timedelay(potential, alpha1, alpha2, zlens, zsource, angle_unit='arcsec'):
    """
    Geometric + potential time delay: Δt = ((1+zl)/c) * (Dl*Ds/Dls) * φ.
    Parameters:
        potential : float
            Lens potential ψ(θ). If angle_unit='arcsec', ψ is normalized so ∇ψ is in arcsec;
            the function multiplies by (arcsec→rad)^2 to convert to radians.
        alpha1, alpha2 : float
            Components of (θ-β) (note: not the deflection α), in the units set by angle_unit.
        zlens, zsource : float
            Lens and source redshifts.
        angle_unit : {'arcsec','rad'}
            Units of the angular inputs.
    Returns:
        float : Time delay in days.
    """
    # Angle to radians
    if angle_unit == 'arcsec':
        a1 = (alpha1 * u.arcsec).to_value(u.rad)
        a2 = (alpha2 * u.arcsec).to_value(u.rad)
        psi = potential * (u.arcsec.to(u.rad))**2
    elif angle_unit == 'rad':
        a1, a2 = float(alpha1), float(alpha2)
        psi = float(potential)
    else:
        raise ValueError("angle_unit must be 'arcsec' or 'rad'")

    # Dimensionless Fermat potential
    phi = 0.5 * (a1*a1 + a2*a2) - psi

    # Distances and speed of light with astropy units
    Dl  = cosmo.angular_diameter_distance(zlens)     # Mpc
    Ds  = cosmo.angular_diameter_distance(zsource)   # Mpc
    Dls = cosmo.angular_diameter_distance_z1z2(zlens, zsource)  # Mpc
    T0 = ((1.0 + zlens) * (Dl * Ds / Dls) / const.c).to_value(u.s)

    # Convert to days
    return (T0 * phi) / 86400.0


def Mass_c_to_rhos_rs(Mvir, cvir,zlens):
    Rvir = rvir_mvir(Mvir,zlens)
    rs = Rvir / cvir  # Mpc
    rhos = rho_crit(zlens)*dv(zlens)/3.0*cvir**3.0/(jnp.log(1.0+cvir)-cvir/(1+cvir))
    rhos = rhos.item()  # Convert JAX Array to float
    return rhos, rs

from scipy.optimize import brentq

def find_radius_at_density(rho_func, rho_ref, r_min=1e-3, r_max=5.0):
    """
    Find r such that rho_func(r) = rho_ref.

    Parameters:
        rho_func: Callable returning density at radius r.
        rho_ref: Target density (e.g., 200 * rho_crit).
        r_min: Lower bound of search interval (default 1e-3).
        r_max: Upper bound of search interval (default 5.0).

    Returns:
        Radius satisfying the condition.
    """
    # print("r_min =", r_min, "rho_func(r_min) =", rho_func(r_min))
    def f(r):
        return rho_func(r) - rho_ref
    
    return brentq(f, r_min, r_max)

from scipy.integrate import simpson
def Sigma_g_to_Mvir(thetaE_MPC, sigma_g, zlens):
    """
    Convert velocity dispersion to NFW virial mass M_vir.
    thetaE_MPC : mpc/h
    Returns:
    M_200 : float
        Estimated virial mass (units: M_sun/h)
    """
    This_rho_crit = rho_crit(zlens)
    r = np.logspace(np.log10(0.0001*thetaE_MPC), np.log10(thetaE_MPC*1000), 2000)  # Units: Mpc/h
    rho_SIE = SIEDensity3D(sigma_g,r)
    rho_ref =  200*This_rho_crit
    rho_func = lambda r: SIEDensity3D(sigma_g, r)
    r200_SIE = find_radius_at_density(rho_func, rho_ref)
    r = np.logspace(np.log10(0.0001*r200_SIE), np.log10(r200_SIE), 2000)
    # Compute rho(r)
    rho_SIE = np.array([rho_func(ri) for ri in r])

    # Integrate mass
    mass_SIE = simpson(y=4 * np.pi * r**2 * rho_SIE, x=r)
    return mass_SIE 

def compute_sigma_host(r_ein, rs, rhos,arcsec_1):
    """
    Compute the projected mass density Σ(R) for an NFW halo at a characteristic
    projected radius (typically the Einstein radius), used as Σ_host for lensing
    fluctuation normalization (cf. pyhalo-ULDM).

    This uses the analytic NFW projected density at R = r_ein * arcsec_1.

    Parameters
    ----------
    r_ein : float
        Characteristic angular scale in arcsec (typically the Einstein angle).
    rs : float
        NFW scale radius (units consistent with arcsec_1, e.g., Mpc/h).
    rhos : float
        NFW characteristic density (Msun / (Mpc/h)^3).
    arcsec_1 : float
        Physical scale per arcsec (Mpc/h per arcsec) from cosmology at z_lens.

    Returns
    -------
    sigma_host : float
        Projected density Σ(R) at r_ein (Msun / (Mpc/h)^2) for normalization.

    Notes
    -----
    Assumes a spherical NFW profile; r_ein is the most lensing-sensitive region.
    """
    r_perp = r_ein * arcsec_1 #Mpc/h
    x = r_perp / rs

    if x < 1:
        Fx = np.arctanh(np.sqrt(1 - x ** 2)) / np.sqrt(1 - x ** 2)
    else:
        Fx = np.arctan(np.sqrt(-1 + x ** 2)) / np.sqrt(-1 + x ** 2)

    sigma_host = 2 * rhos * rs * (1 - Fx) / (x ** 2 - 1)
    return sigma_host


# Ensure that make_c_coor and other utility functions are also using jax.numpy
def make_c_coor(bs,nc):
    '''
        Draw the mesh grids for a bs*bs box with nc*nc pixels
    '''
    ds = bs/nc
    xx01 = np.linspace(-bs/2.0,bs/2.0-ds,nc)+0.5*ds
    xx02 = np.linspace(-bs/2.0,bs/2.0-ds,nc)+0.5*ds
    xg1,xg2 = np.meshgrid(xx01,xx02)
    return xg1,xg2

def Einstein_angle(M,D_L,D_S,D_LS):
    """
    unit: rad
    """
    return np.sqrt((4 * G * M * D_LS) / (vc**2 * D_L * D_S))

def Get_M_from_Einstein_angle(theta_E, D_L, D_S, D_LS):
    """
    Calculate the mass M from the Einstein radius theta_E.
    """
    return (theta_E**2 * vc**2 * D_L * D_S) / (4 * G * D_LS)

apr =  1.0/np.pi*180.0*3600  # arcsec per rad
def Rad_to_arcsec(rad):
    return rad * apr
def arcsec_to_Rad(arcsec):
    return arcsec / apr



# SIE
def SIEDensity3D(sigma0, r):
    # r = np.maximum(r, 1e-4)
    rho = sigma0**2 / (2 * jnp.pi * G * r**2)
    return rho

# SIE
def SIELensingPot(x1, x2, xc1, xc2, sigma0, ql, pa, z1, z2):
    resFact = 4.0 * jnp.pi * (sigma0 / vc) ** 2.0 * Da2(z1, z2) / Da(z2)
    # print(f"resFact:{resFact}")
    rrr = jnp.sqrt((x1 - xc1) ** 2 + (x2 - xc2) ** 2)

    if jnp.isclose(ql, 1.0):
        # Fall back to the SIS potential form
        res = resFact * rrr * apr
    else:
        rpa = jnp.arctan2(x2 - xc2, x1 - xc1)
        ellm = (1.0 - ql) / (1.0 + ql)

        ell = (1. - jnp.sqrt(1 - ellm * ellm)) / ellm
        res = resFact * rrr * jnp.sqrt(1.0 - ell * jnp.cos(2.0 * (rpa - pa))) * apr

    return res


class SIE_Model(object):
    """
    Verified against Lenstronomy.
    """
    def __init__(self, theta_E, xc1=0.0, xc2=0.0, ql=1.0, pa=0.0):
        """
        Initialize the SIE model.
        
        Parameters
        ----------
        theta_E : float
            SIS Einstein radius
        lambda_f : float
            Dynamical normalization factor lambda(f)
        xc1, xc2 : float
            Lens center coordinates
        ql : float
            Axis ratio (minor/major), 0<ql<=1
        pa : float
            Ellipse position angle in radians, counterclockwise from x-axis
        """
        self.theta_E = theta_E
        self.xc1 = xc1
        self.xc2 = xc2
        self.ql = ql
        self.pa = pa

    def _rotate(self, xi1, xi2):
        """Translate and rotate coordinates to the ellipse principal axis."""
        xt = xi1 - self.xc1
        yt = xi2 - self.xc2
        x_rot = xt * np.cos(self.pa) + yt * np.sin(self.pa)
        y_rot = -xt * np.sin(self.pa) + yt * np.cos(self.pa)
        return x_rot, y_rot

    def kappa(self, xi1, xi2):
        """Compute convergence κ(x, y)."""
        x_rot, y_rot = self._rotate(xi1, xi2)
        # Non-dimensionalize
        x_nd = x_rot / (self.theta_E)
        y_nd = y_rot / (self.theta_E)
        r = np.sqrt(x_nd**2 + (self.ql**2) * y_nd**2)
        kappa_val = (np.sqrt(self.ql) / 2.0) / r
        return kappa_val
    def potential(self, xi1, xi2):
        """
        Compute lensing potential Psi(x, y) (Eq. 5.84).
        """
        # Rotate coordinates
        x_rot, y_rot = self._rotate(xi1, xi2)

        # Non-dimensionalize
        x_nd = x_rot / (self.theta_E)
        y_nd = y_rot / (self.theta_E)

        # Polar coordinates
        x = np.sqrt(x_nd**2 + y_nd**2)
        phi = np.arctan2(y_nd, x_nd)

        f = self.ql
        f_prime = np.sqrt(1 - f**2)

        term1 = np.sin(phi) * np.arcsin(f_prime * np.sin(phi))
        term2 = np.cos(phi) * np.arcsinh((f_prime / f) * np.cos(phi))
        psi = x * (np.sqrt(f) / f_prime) * (term1 + term2)

        # Restore scale
        psi *=  self.theta_E**2
        return psi
    def deflection_angle(self, xi1, xi2):
        """
        Compute SIE deflection components (α1, α2) (Eq. 5.87).
        """
        # Rotate coordinates
        x_rot, y_rot = self._rotate(xi1, xi2)

        # Non-dimensionalize
        x_nd = x_rot / (self.theta_E)
        y_nd = y_rot / (self.theta_E)

        # Polar coordinates
        phi = np.arctan2(y_nd, x_nd)

        f = self.ql
        f_prime = np.sqrt(1 - f**2)

        # Dimensionless deflection components
        alpha1_nd = (np.sqrt(f) / f_prime) * np.arcsinh((f_prime / f) * np.cos(phi))
        alpha2_nd = (np.sqrt(f) / f_prime) * np.arcsin(f_prime * np.sin(phi))

        # Restore to dimensional angles
        alpha1_rot = alpha1_nd * self.theta_E
        alpha2_rot = alpha2_nd * self.theta_E

        # Rotate back to observed frame
        alpha_x = alpha1_rot * np.cos(self.pa) - alpha2_rot * np.sin(self.pa)
        alpha_y = alpha1_rot * np.sin(self.pa) + alpha2_rot * np.cos(self.pa)

        return alpha_x, alpha_y



#SIS
def SIS_with_shear_potential(x1, x2, xc1, xc2, b, gamma, theta_gamma, z1, z2):
    """
    Implement ψ = b * r + γ * b * r * cos(2 * (θ - θγ)).
    Parameters:
        x1, x2       : Observed coordinates
        xc1, xc2     : Lens center
        b            : Einstein radius
        gamma        : Shear strength
        theta_gamma  : Shear angle (radians)
    Returns:
        Lensing potential ψ
    """
    dx = x1 - xc1
    dy = x2 - xc2
    r = jnp.sqrt(dx**2 + dy**2)
    theta = jnp.arctan2(dy, dx)

    psi = b * r + gamma * b * r * jnp.cos(2.0 * (theta - theta_gamma/180*jnp.pi))
    return psi

def calculate_velocity_dispersion(theta_E, D_S, D_ls):
    sigma = np.sqrt((vc**2 / (4 * np.pi)) * (D_S / D_ls) * theta_E)
    return sigma
def calculate_theta_E(velocity_dispersion, D_S, D_ls):
    theta_E = (4 * np.pi / vc**2) * (D_ls / D_S) * (velocity_dispersion**2)
    return theta_E

def rhoSIS(r,sigma0): 
    return sigma0**2/(2.0*np.pi*G*r**2)

# NFW

def NFWLensingPot(x1, x2, xc1, xc2, c, m, ql, pa, z1, z2):
    # ref: https://www.aanda.org/articles/aa/pdf/2002/30/aah3555.pdf, Eq. 8

    pa = jnp.deg2rad(pa) 
    r = jnp.sqrt((x1-xc1)**2.0+(x2-xc2)**2.0)
    rvir = rvir_mvir(m,z1) 
    print(f"rvir:{rvir}")
    rs = rvir/c
    rs_arc = rs/Da(z1)*apr 
    print(f"rs:{rs}, rs_arc:{rs_arc}")

    ell = (1.0-ql)/(1.0+ql) 
    rpa = jnp.arctan2(x2-xc2, x1-xc1)
    xx = r/rs_arc*jnp.sqrt(1.0-ell*jnp.cos(2.0*(rpa - pa)))
    rhos = rho_crit(z1)*dv(z1)/3.0*c**3.0/(jnp.log(1.0+c)-c/(1+c))
    kappas = rs*rhos/SigmaCrit(z1,z2)

    # Compute h(x)
    ss = xx*0.0
    idx1_1 = xx>0.0
    idx1_2 = xx<1.0
    idx1 = idx1_1&idx1_2
    ss = ss.at[idx1].set(jnp.log(xx[idx1]*0.5)**2.0 - jnp.arccosh(1.0/xx[idx1])**2.0)
    idx2 = xx>=1.0
    ss = ss.at[idx2].set(jnp.log(xx[idx2]*0.5)**2.0 + jnp.arccos(1.0/xx[idx2])**2.0)

    # Compute potential
    res = 2.0*kappas*rs_arc**2.0*ss 
    return res

def rhoNFW(r,rhos,rs):
    return rhos/(r/rs)/pow(1+r/rs,2)

# Hernquist

def rhoHern(r,rhoH,rH):
    return rhoH/((r/rH)*pow(1+r/rH,3))


# Enternal shear
class ext_shear(object):
    """
    Verified against Lenstronomy.
    """
    def __init__(self, g, phi_g):
        """
        Initialize an external shear using the amplitude g and the angle phi_g (in rad)
        """
        self.g = g
        self.phi_g = phi_g

    def psi(self, x, phi):
        """
        Returns the lensing potential at polar coordinates x, phi
        """
        return 0.5 * self.g * x**2 * np.cos(2 * (phi - self.phi_g))

    def alpha(self, x, phi):
        """
        Returns the components of the deflection angle at polar coordinates x, phi
        """
        a1 = self.g * x * np.cos(2 * self.phi_g - phi)
        a2 = self.g * x * np.sin(2 * self.phi_g - phi)
        return a1, a2

    def gamma(self):
        """
        Returns the components of the shear at polar coordinates x, phi
        """
        g1 = self.g * np.cos(2 * self.phi_g)
        g2 = self.g * np.sin(2 * self.phi_g)
        return g1, g2



# fft

def zero_padding(in_arr, Nx, Ny):
    out = np.zeros((2*Nx, 2*Ny), dtype=in_arr.dtype)
    out[:Nx, :Ny] = in_arr
    return out

def corner_matrix(in_arr, Nx, Ny):
    return in_arr[:Nx, :Ny]

def roll_a_matrix(in_arr, roll_nx1, roll_nx2):
    # Use numpy.roll to shift the array
    return np.roll(np.roll(in_arr, roll_nx1, axis=0), roll_nx2, axis=1)

def kernel_alphas_iso_I(Ncc, dsx):
    # Kernel with I (original) boundary conditions
    alpha1_iso = np.zeros((Ncc, Ncc))
    alpha2_iso = np.zeros((Ncc, Ncc))
    half = Ncc // 2
    for i in range(Ncc):
        for j in range(Ncc):
            if i <= half and j <= half:
                x = (i)*dsx + 0.5*dsx
                y = (j)*dsx + 0.5*dsx
                r = np.sqrt(x*x + y*y)
                if r > dsx*(Ncc/2.0):
                    alpha1_iso[i, j] = 0.0
                    alpha2_iso[i, j] = 0.0
                else:
                    val = 1.0/(np.pi*r*r)
                    alpha1_iso[i, j] = x*val
                    alpha2_iso[i, j] = y*val
            else:
                # Fill using symmetry
                if i <= half and j > half:
                    alpha1_iso[i, j] =  alpha1_iso[i, Ncc-j]
                    alpha2_iso[i, j] = -alpha2_iso[i, Ncc-j]
                if i > half and j <= half:
                    alpha1_iso[i, j] = -alpha1_iso[Ncc-i, j]
                    alpha2_iso[i, j] =  alpha2_iso[Ncc-i, j]
                if i > half and j > half:
                    alpha1_iso[i, j] = -alpha1_iso[Ncc-i, Ncc-j]
                    alpha2_iso[i, j] = -alpha2_iso[Ncc-i, Ncc-j]
    return alpha1_iso, alpha2_iso

def kernel_alphas_iso_P(Ncc, dsx):
    # Kernel with P (periodic) boundary conditions
    alpha1_iso = np.zeros((Ncc, Ncc))
    alpha2_iso = np.zeros((Ncc, Ncc))
    for i in range(Ncc):
        for j in range(Ncc):
            x = ((i - Ncc//2) + 0.5)*dsx
            y = ((j - Ncc//2) + 0.5)*dsx
            r = np.sqrt(x*x + y*y)
            if r > dsx*(Ncc/2.0):
                alpha1_iso[i, j] = 0.0
                alpha2_iso[i, j] = 0.0
            else:
                alpha1_iso[i, j] = x/(np.pi*r*r)
                alpha2_iso[i, j] = y/(np.pi*r*r)
    return alpha1_iso, alpha2_iso

def convolve_fft(in1, in2, dx, dy):
    f1 = np.fft.rfft2(in1)
    f2 = np.fft.rfft2(in2)
    out_fft = f1 * f2
    out = np.fft.irfft2(out_fft, s=in1.shape)
    out = out * dx * dy
    return out

def call_kappa_to_alphas(Kappa, Bsz, Ncc, boundary_type='I'):
    Kappa = np.array(Kappa, dtype=np.float64)
    dsx = Bsz / Ncc

    if boundary_type == 'I':
        # Use I-type kernel
        alpha1_iso, alpha2_iso = kernel_alphas_iso_I(2*Ncc, dsx)
        # Zero padding
        kappa = zero_padding(Kappa, Ncc, Ncc)
        # Convolution
        alpha1_tmp = convolve_fft(kappa, alpha1_iso, dsx, dsx)
        alpha2_tmp = convolve_fft(kappa, alpha2_iso, dsx, dsx)
        alpha1 = corner_matrix(alpha1_tmp, Ncc, Ncc)
        alpha2 = corner_matrix(alpha2_tmp, Ncc, Ncc)
    
    elif boundary_type == 'P':
        # Use P-type kernel (no zero padding needed, consistent with C code)
        alpha1_iso, alpha2_iso = kernel_alphas_iso_P(Ncc, dsx)
        # Directly convolve with Kappa
        alpha1_tmp = np.zeros_like(Kappa)
        alpha2_tmp = np.zeros_like(Kappa)

        alpha1_tmp = convolve_fft(Kappa, alpha1_iso, dsx, dsx)
        alpha2_tmp = convolve_fft(Kappa, alpha2_iso, dsx, dsx)

        # Roll processing
        alpha1 = roll_a_matrix(alpha1_tmp, Ncc//2, Ncc//2)
        alpha2 = roll_a_matrix(alpha2_tmp, Ncc//2, Ncc//2)
    
    else:
        raise ValueError("boundary_type must be 'I' or 'P'.")

    return alpha1, alpha2

# Get Sorce position when adding point source

def mapping_triangles_vec(ys, xgrids1, xgrids2, lgrids1, lgrids2, nc):
    ntris = nc - 1

    # Generate all triangles (two types of divisions) at once
    def generate_all_triangles(grids1, grids2):
        p0_a = np.stack([grids1[:-1, :-1], grids2[:-1, :-1]], axis=-1).reshape(-1, 2)
        p1_a = np.stack([grids1[1:, :-1], grids2[1:, :-1]], axis=-1).reshape(-1, 2)
        p2_a = np.stack([grids1[1:, 1:], grids2[1:, 1:]], axis=-1).reshape(-1, 2)

        p0_b = p0_a
        p1_b = p2_a
        p2_b = np.stack([grids1[:-1, 1:], grids2[:-1, 1:]], axis=-1).reshape(-1, 2)

        # Combine all triangles
        p0 = np.vstack([p0_a, p0_b])
        p1 = np.vstack([p1_a, p1_b])
        p2 = np.vstack([p2_a, p2_b])

        return p0, p1, p2

    # Vertices of all triangles
    lp0, lp1, lp2 = generate_all_triangles(lgrids1, lgrids2)
    xp0, xp1, xp2 = generate_all_triangles(xgrids1, xgrids2)

    # Batch check if point ys is inside all lens triangles
    def signP_batch(pt, v1, v2):
        return (pt[0] - v2[:, 0]) * (v1[:, 1] - v2[:, 1]) - (v1[:, 0] - v2[:, 0]) * (pt[1] - v2[:, 1])

    b0 = signP_batch(ys, lp0, lp1) < 0.0
    b1 = signP_batch(ys, lp1, lp2) < 0.0
    b2 = signP_batch(ys, lp2, lp0) < 0.0

    inside_mask = (b0 == b1) & (b1 == b2)

    if not np.any(inside_mask):
        return np.empty((0, 2))

    lp0_in, lp1_in, lp2_in = lp0[inside_mask], lp1[inside_mask], lp2[inside_mask]
    xp0_in, xp1_in, xp2_in = xp0[inside_mask], xp1[inside_mask], xp2[inside_mask]

    # Batch area calculation
    def tri_area_batch(p0, p1, p2):
        return 0.5 * np.abs(p0[:, 0]*(p1[:, 1]-p2[:, 1]) +
                            p1[:, 0]*(p2[:, 1]-p0[:, 1]) +
                            p2[:, 0]*(p0[:, 1]-p1[:, 1]))

    area_total = tri_area_batch(lp0_in, lp1_in, lp2_in)
    area0 = tri_area_batch(np.tile(ys, (len(lp0_in), 1)), lp1_in, lp2_in)
    area1 = tri_area_batch(np.tile(ys, (len(lp0_in), 1)), lp2_in, lp0_in)
    area2 = tri_area_batch(np.tile(ys, (len(lp0_in), 1)), lp0_in, lp1_in)

    bary = np.stack([area0, area1, area2], axis=-1) / area_total[:, None]

    # Convert back to Cartesian coordinates
    xroots = xp0_in * bary[:, [0]] + xp1_in * bary[:, [1]] + xp2_in * bary[:, [2]]

    return xroots

@jax.jit
def mapping_triangles_vec_jax(ys, xgrids1, xgrids2, lgrids1, lgrids2):
    """
    ys:   (2,)         Single point [y, x]
    xgrids1,xgrids2: (N, M) grid coordinates
    lgrids1,lgrids2: (N, M) grid coordinates for triangle membership
    Returns:
      xroots_all:    (2*(N-1)*(M-1), 2)   Interpolation results for all triangles
      inside_mask:  (2*(N-1)*(M-1),)     Boolean mask indicating triangles containing ys
    """
    def gen_triangles(g1, g2):
        # Split each cell into two triangles
        p0a = jnp.stack([g1[:-1, :-1], g2[:-1, :-1]], axis=-1).reshape(-1, 2)
        p1a = jnp.stack([g1[1: , :-1], g2[1: , :-1]], axis=-1).reshape(-1, 2)
        p2a = jnp.stack([g1[1: ,  1: ], g2[1: ,  1: ]], axis=-1).reshape(-1, 2)
        p0b = p0a
        p1b = p2a
        p2b = jnp.stack([g1[:-1, 1:], g2[:-1, 1:]], axis=-1).reshape(-1, 2)
        p0 = jnp.concatenate([p0a, p0b], axis=0)
        p1 = jnp.concatenate([p1a, p1b], axis=0)
        p2 = jnp.concatenate([p2a, p2b], axis=0)
        return p0, p1, p2

    # Build triangle vertices for lens grid and output grid
    lp0, lp1, lp2 = gen_triangles(lgrids1, lgrids2)
    xp0, xp1, xp2 = gen_triangles(xgrids1, xgrids2)

    # Test whether ys lies inside each triangle (same as original signP_batch)
    def sign(pt, v1, v2):
        return (pt[0] - v2[:,0]) * (v1[:,1] - v2[:,1]) - (v1[:,0] - v2[:,0]) * (pt[1] - v2[:,1])

    b0 = sign(ys, lp0, lp1) < 0
    b1 = sign(ys, lp1, lp2) < 0
    b2 = sign(ys, lp2, lp0) < 0
    inside_mask = (b0 == b1) & (b1 == b2)  # (n_triangles,)

    # Compute barycentric interpolation for all triangles
    def tri_area(a, b, c):
        return 0.5 * jnp.abs(
            a[:,0] * (b[:,1] - c[:,1]) +
            b[:,0] * (c[:,1] - a[:,1]) +
            c[:,0] * (a[:,1] - b[:,1])
        )

    # Broadcast ys to (n_triangles, 2)
    ys_tile = jnp.broadcast_to(ys, lp0.shape)

    area_tot = tri_area(lp0, lp1, lp2)       # (n_triangles,)
    a0 = tri_area(ys_tile, lp1, lp2)         # (n_triangles,)
    a1 = tri_area(ys_tile, lp2, lp0)
    a2 = tri_area(ys_tile, lp0, lp1)

    bary = jnp.stack([a0, a1, a2], axis=1) / area_tot[:, None]  # (n_triangles,3)
    xroots_all = (
        bary[:,0:1] * xp0 +
        bary[:,1:2] * xp1 +
        bary[:,2:3] * xp2
    )  # (n_triangles,2)

    return xroots_all, inside_mask
# Calculate R_cusp
def cal_Rcusp_circle(alpha1_global, alpha2_global,xi1,xi2,dsx_arc,sharpest_point, data_file=None ):
    if data_file is not None:
        # Read npz file
        data = np.load(data_file)

        # Extract alpha1 and alpha2
        alpha1_sub = data['alpha1_sub']
        alpha2_sub = data['alpha2_sub']
    else:
        # Use default zeros when no data file is provided
        alpha1_sub = np.zeros_like(alpha1_global)
        alpha2_sub = np.zeros_like(alpha2_global)
    # Rescale
    yi1, yi2, mu_global, kappa_main, gamma1_gobal, gamma2_gobal = alphas_to_mu(alpha1_global, alpha2_global, dsx_arc, xi1, xi2)

    alpha1_global = alpha1_global + alpha1_sub
    alpha2_global = alpha2_global + alpha2_sub

    yi1, yi2, mu_global, kappa_global, gamma1_gobal, gamma2_gobal = alphas_to_mu(alpha1_global, alpha2_global, dsx_arc, xi1, xi2)

    # Stack kappa_subs onto kappa_main and renormalization
    rescale_factor = kappa_main.sum()/kappa_global.sum()
    alpha1_global = alpha1_global*rescale_factor
    alpha2_global = alpha2_global*rescale_factor
    yi1, yi2, mu_global, kappa_global, gamma1_gobal, gamma2_gobal = alphas_to_mu(alpha1_global, alpha2_global, dsx_arc, xi1, xi2)

    ys1 = sharpest_point[1]
    ys2 = sharpest_point[0]
    ys = np.array([ys1, ys2])
    xroots = mapping_triangles_vec(ys, xi1, xi2,yi1, yi2, len(xi1))

    # Input data (replace with actual data)
    interpolation_points = jnp.array(xroots)
    jarray = jnp.array(mu_global)
    dx = dsx_arc
    dy = dsx_arc

    # Convert coordinates to index coordinates
    interpolation_points_ = (interpolation_points - jnp.array([xi1[0, 0], xi2[0, 0]])) / jnp.array([dx, dy])
    interpolation_points_x = interpolation_points_[:, 0]
    interpolation_points_y = interpolation_points_[:, 1]

    # Interpolate using map_coordinates
    mu_values_jax = jndimage.map_coordinates(
        jarray, [interpolation_points_x, interpolation_points_y],
        order=1, mode='nearest'
    )

    # Find indices of the three largest absolute values
    top_indices = jnp.argsort(jnp.abs(mu_values_jax))[-3:]

    # Retrieve those three elements
    mu1, mu2, mu3 = mu_values_jax[top_indices]

    # Compute R_cusp
    R_cusp = jnp.abs((mu1 + mu2 + mu3)) / (jnp.abs(mu1) + jnp.abs(mu2) + jnp.abs(mu3))
    return R_cusp

def cal_Rcusp(xroots, mu_global, xi1, xi2):
    import jax.numpy as jnp
    from jax.scipy import ndimage as jndimage

    dx = xi1[1, 0] - xi1[0, 0]
    dy = xi2[0, 1] - xi2[0, 0]
    jarray = jnp.array(mu_global)
    interpolation_points = jnp.array(xroots)

    # Convert coordinates to index coordinates
    interpolation_points_ = (interpolation_points - jnp.array([xi1[0, 0], xi2[0, 0]])) / jnp.array([dx, dy])
    interpolation_points_x = interpolation_points_[:, 0]
    interpolation_points_y = interpolation_points_[:, 1]

    # Interpolate using map_coordinates
    mu_values_jax = jndimage.map_coordinates(
        jarray, [interpolation_points_x, interpolation_points_y],
        order=1, mode='nearest'
    )

    # Find indices of the three largest absolute values
    top_indices = jnp.argsort(jnp.abs(mu_values_jax))[-3:]

    # Retrieve those three elements and coordinates
    mu_top3 = mu_values_jax[top_indices]
    points_top3 = interpolation_points[top_indices]

    # Compute R_cusp
    R_cusp = jnp.abs(jnp.sum(mu_top3)) / jnp.sum(jnp.abs(mu_top3))

    return R_cusp, points_top3

def max_angle_from_center(center_point, points_top3):

    # Three vectors from center to the three points
    vectors = points_top3 - center_point

    angles = []
    for i in range(3):
        for j in range(i + 1, 3):
            v1 = vectors[i]
            v2 = vectors[j]
            cos_theta = jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2))
            cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
            angle = jnp.arccos(cos_theta)
            angles.append(angle)

    return jnp.max(jnp.stack(angles))

def get_mu_of_three_points(xroots, mu_global, xi1, xi2):
    dx = xi1[1, 0] - xi1[0, 0]
    dy = xi2[0, 1] - xi2[0, 0]
    jarray = jnp.array(mu_global)
    interpolation_points = jnp.array(xroots)

    # Convert coordinates to index coordinates
    interpolation_points_ = (interpolation_points - jnp.array([xi1[0, 0], xi2[0, 0]])) / jnp.array([dx, dy])
    interpolation_points_x = interpolation_points_[:, 0]
    interpolation_points_y = interpolation_points_[:, 1]

    # Interpolate using map_coordinates
    mu_values_jax = jndimage.map_coordinates(
        jarray, [interpolation_points_x, interpolation_points_y],
        order=1, mode='nearest'
    )
    
    # Build list of positions and magnifications
    result = []
    for idx in range(len(interpolation_points)):
        point = interpolation_points[idx]
        mu = mu_values_jax[idx]
        result.append({
            "position": [float(point[0]), float(point[1])],
            "mu": float(mu)
        })

    return result
def interpolate_maps_at_points(xroots, maps_dict, xi1, xi2):
    dx = xi1[1, 0] - xi1[0, 0]
    dy = xi2[0, 1] - xi2[0, 0]
    interpolation_points = jnp.array(xroots)

    # Convert coordinates to index coordinates
    interpolation_points_ = (interpolation_points - jnp.array([xi1[0, 0], xi2[0, 0]])) / jnp.array([dx, dy])
    interpolation_points_x = interpolation_points_[:, 0]
    interpolation_points_y = interpolation_points_[:, 1]

    # Interpolate each map
    interpolated_results = {}
    for map_name, map_array in maps_dict.items():
        jarray = jnp.array(map_array)
        values = jndimage.map_coordinates(
            jarray, [interpolation_points_x, interpolation_points_y],
            order=1, mode='nearest'
        )
        interpolated_results[map_name] = values

    # Build list with all positions and interpolated map values
    result = []
    for idx in range(len(interpolation_points)):
        point = interpolation_points[idx]
        entry = {"position": [float(point[0]), float(point[1])]}
        for map_name, values in interpolated_results.items():
            entry[map_name] = float(values[idx])
        result.append(entry)

    return result


def classify_lens_configuration(mu_info, class_way = "position"):
    # Applicable to smooth cases only
    # Extract all (abs(mu), info) tuples
    abs_mu_with_info = [(abs(info["mu"]), info) for info in mu_info]

    # Sort by absolute value (ascending)
    abs_mu_with_info_sorted = sorted(abs_mu_with_info, key=lambda x: x[0])
    
    # Take the largest three (from the tail)
    top3_infos = [entry[1] for entry in abs_mu_with_info_sorted[-3:]]
    top4_infos = [entry[1] for entry in abs_mu_with_info_sorted[-4:]]

    
    # Extract μ values sorted by absolute magnitude
    mu_values = sorted([abs(info["mu"]) for info in top3_infos])
    mu1, mu2, mu3 = mu_values
    R_cusp = jnp.abs((mu1 + mu2 + mu3)) / (jnp.abs(mu1) + jnp.abs(mu2) + jnp.abs(mu3))
    
    if class_way == "position":
        # get position
        positions = np.array([info["position"] for info in top4_infos])
        classification_code = label_quad(positions)
    elif class_way == "mu":
        # get mu
        classification_code = label_quad_using_mu(mu1, mu2, mu3)
    return {
        "mu_info": top3_infos,
        "classification": classification_code,  # 0=fold, 1=cusp, 2=Einstein Cross
        "R_cusp": R_cusp
    }


def label_quad_using_mu(mu1,mu2,mu3):
    # Compute two ratios
    if mu1 == 0:
        ratio_all = float('inf')
    else:
        ratio_all = mu3 / mu1

    if mu2 == 0:
        ratio_top2 = float('inf')
    else:
        ratio_top2 = mu3 / mu2

    # Classification logic
    if ratio_all < 1.5:
        classification_code = 2  # Einstein Cross
    elif ratio_top2 < 1.2:
        classification_code = 0  # Fold
    else:
        classification_code = 1  # Rcusp
    
    return classification_code


def label_quad(points, eps_factor=0.7):
    """
    Cluster and label 4 points:
      1 -> three points form a cluster (cusp)
      2 -> two pairs (fold)
      0 -> no clear cluster (cross or other)
    points: np.ndarray, shape (4,2)
    eps_factor: clustering distance threshold = mean_pairwise_dist * eps_factor
    """
    # 1. Pairwise distances
    dists = np.linalg.norm(points[:,None,:] - points[None,:,:], axis=2)
    mean_d = np.mean(dists[np.triu_indices(4, k=1)])
    eps = mean_d * eps_factor

    # 2. Clustering
    db = DBSCAN(eps=eps, min_samples=2).fit(points)
    labels = db.labels_  # -1 denotes noise

    # 3. Cluster size statistics
    cnt = Counter(l for l in labels if l >= 0)
    sizes = sorted(cnt.values(), reverse=True)

    # 4. Assign labels
    if 3 in sizes:
        return 1
    if sizes == [2, 2]:
        return 2
    return 0
def label_quad_up(points, eps_factor = 0.7):
    """
    Cluster and label 4 points:
      1 -> three points form a cluster (cusp)
      2 -> one 2-point cluster with others as noise (fold: 2+1+1), also try forming a triplet
      3 -> all points are noise (cross: 1+1+1+1), also try forming a triplet via rotating line
      0 -> otherwise (e.g., 2+2 or all-in-one cluster)
    """
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    mean_d = np.mean(dists[np.triu_indices(4, k=1)])
    eps = mean_d * eps_factor

    db = DBSCAN(eps=eps, min_samples=2).fit(points)
    labels = db.labels_

    cnt = Counter(l for l in labels if l >= 0)
    sizes = sorted(cnt.values(), reverse=True)
    cluster_groups = {}
    cusp_negative_point = None
    label = 0

    for i, lbl in enumerate(labels):
        if lbl >= 0:
            cluster_groups.setdefault(lbl, []).append(i)

    if 3 in sizes:
        label = 1
        largest_label = cnt.most_common(1)[0][0]
        idxs = cluster_groups[largest_label]
        cluster_points = points[idxs]
        center_idx = np.argmin([
            np.sum(np.linalg.norm(cluster_points[i] - np.delete(cluster_points, i, axis=0), axis=1))
            for i in range(3)
        ])
        cusp_negative_point = cluster_points[center_idx]

    elif sizes == [2]:
        label = 2
        cluster_label = cnt.most_common(1)[0][0]
        two_idxs = cluster_groups[cluster_label]
        remaining_idxs = list(set(range(4)) - set(two_idxs))

        distances = []
        for idx in remaining_idxs:
            avg_dist = np.mean([np.linalg.norm(points[idx] - points[tid]) for tid in two_idxs])
            distances.append((idx, avg_dist))
        third_idx = sorted(distances, key=lambda x: x[1])[0][0]

        triplet_idxs = two_idxs + [third_idx]
        cluster_points = points[triplet_idxs]
        center_idx = np.argmin([
            np.sum(np.linalg.norm(cluster_points[i] - np.delete(cluster_points, i, axis=0), axis=1))
            for i in range(3)
        ])
        cusp_negative_point = cluster_points[center_idx]

        cluster_groups = {0: triplet_idxs}
        other = list(set(range(4)) - set(triplet_idxs))
        if other:
            cluster_groups[-1] = other

    elif len(cnt) == 0:
        label = 3
        origin = np.array([0.0, 0.0])
        best_score = np.inf
        best_triplet = None

        for angle_deg in np.arange(0, 180, 1.0):  # rotate every 1°
            theta = np.radians(angle_deg)
            u = np.array([np.cos(theta), np.sin(theta)])
            dists_to_line = np.abs(np.cross(points - origin, u)) / np.linalg.norm(u)
            idx_sorted = np.argsort(dists_to_line)
            idx1, idx2 = idx_sorted[:2]

            # Tolerance: the two points on the line must be close
            if dists_to_line[idx1] > 0.15 or dists_to_line[idx2] > 0.15:
                continue

            rest = list(set(range(4)) - {idx1, idx2})
            d0 = np.mean([np.linalg.norm(points[rest[0]] - points[i]) for i in [idx1, idx2]])
            d1 = np.mean([np.linalg.norm(points[rest[1]] - points[i]) for i in [idx1, idx2]])
            third_idx = rest[0] if d0 < d1 else rest[1]

            triplet = [idx1, idx2, third_idx]
            subp = points[triplet]
            internal_score = np.mean([
                np.linalg.norm(subp[i] - subp[j])
                for i in range(3) for j in range(i+1, 3)
            ])

            if internal_score < best_score:
                best_score = internal_score
                best_triplet = triplet

        if best_triplet is not None:
            cluster_points = points[best_triplet]
            center_idx = np.argmin([
                np.sum(np.linalg.norm(cluster_points[i] - np.delete(cluster_points, i, axis=0), axis=1))
                for i in range(3)
            ])
            cusp_negative_point = cluster_points[center_idx]
            cluster_groups = {0: best_triplet}
            other = list(set(range(4)) - set(best_triplet))
            if other:
                cluster_groups[-1] = other

    return {
        "label": label,
        "cluster_labels": labels.tolist(),
        "cusp_negative_point": cusp_negative_point,
        "cluster_groups": cluster_groups,
        "eps": eps
    }

def compute_Rcusp_distribution(mu_global,x_center, y_center, xi1, xi2, yi1, yi2, desired_points=100, alpha=10):
    collected = 0
    results = {}

    # Grid step size (used for local sampling sigma)
    dx = xi1[1, 0] - xi1[0, 0]
    dy = xi2[0, 1] - xi2[0, 0]
    sigma_x = alpha * dx
    sigma_y = alpha * dy


    # Track last accepted point
    last_hit = None
    local_count = 0
    with tqdm(total=desired_points, desc="Collecting valid points") as pbar:
        while collected < desired_points:
            if last_hit is not None:
                # Locally uniform sampling near last_hit
                x0, y0 = last_hit
                x_rand = x0 + np.random.uniform(low=-sigma_x, high=sigma_x)
                y_rand = y0 + np.random.uniform(low=-sigma_y, high=sigma_y)

            else:
                dx_uniform = alpha * dx * 2  # Controls uniform sampling range
                # Uniform sampling near center
                x_rand = np.random.uniform(low=x_center - dx_uniform, high=x_center + dx_uniform)
                y_rand = np.random.uniform(low=y_center - dx_uniform, high=y_center + dx_uniform)


            point = np.array([x_rand, y_rand])
            ys = np.array([point[1], point[0]])

            # Triangle mapping
            xroots_all, mask = mapping_triangles_vec_jax(ys, xi1, xi2, yi1, yi2)
            xroots = xroots_all[mask]
            if len(xroots) != 5:
                continue

            mu_info = get_mu_of_three_points(xroots, mu_global, xi1, xi2)
            Rcusp_info = classify_lens_configuration_up(mu_info)

            code = Rcusp_info["classification"]

            if code != 1:
                continue
            edge_points = np.array([p["position"] for p in Rcusp_info["edge_points"]])
            negative_point = Rcusp_info["negative_point"]
            center_point = np.array(Rcusp_info["center_point"]["position"])

            angles = compute_theoretical_angles(edge_points, negative_point, center_point)
            print()

            # Hit: record and start the next round of local sampling
            results[tuple(point)] = {
                'point': point,
                'xroots': xroots,
                'Rcusp_info': Rcusp_info,
                'angles': angles
            }
            collected += 1
            local_count += 1
            last_hit = point.copy()
            pbar.update(1)

    return results

def compute_opening_angle_with_uncertainty(x_imageB,y_imageB,x_imageC,y_imageC, wavelength_arcsec, origin=(0, 0), n_samples=1000, seed=42):
    rng = np.random.default_rng(seed)


    theo_sigma = wavelength_arcsec / 2

    angles = []
    for _ in range(n_samples):
        # Perturb the two points
        p1 = rng.normal(loc=[x_imageB,y_imageB], scale=[theo_sigma, theo_sigma])
        p2 = rng.normal(loc=[x_imageC,y_imageC], scale=[theo_sigma, theo_sigma])

        v1 = p1 - origin
        v2 = p2 - origin
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_theta))
        angles.append(theta)

    angle_mean = np.mean(angles)
    angle_std = np.std(angles)
    return angle_mean, angle_std

def compute_AOB_AOC_with_uncertainty(x_imageA, y_imageA,
                                     x_imageB, y_imageB,
                                     x_imageC, y_imageC,
                                     wavelength_arcsec,
                                     n_samples=1000,
                                     seed=42):
    rng = np.random.default_rng(seed)
    theo_sigma = wavelength_arcsec / 2

    angles_AOB = []
    angles_AOC = []

    for _ in range(n_samples):
        ptA = rng.normal(loc=[x_imageA, y_imageA], scale=[theo_sigma, theo_sigma])
        ptB = rng.normal(loc=[x_imageB, y_imageB], scale=[theo_sigma, theo_sigma])
        ptC = rng.normal(loc=[x_imageC, y_imageC], scale=[theo_sigma, theo_sigma])

        # AOB: angle between OA and OB
        vOA = ptA
        vOB = ptB
        cos_AOB = np.dot(vOA, vOB) / (np.linalg.norm(vOA) * np.linalg.norm(vOB))
        angle_AOB = np.degrees(np.arccos(np.clip(cos_AOB, -1.0, 1.0)))
        angles_AOB.append(angle_AOB)

        # AOC: angle between OA and OC
        vOC = ptC
        cos_AOC = np.dot(vOA, vOC) / (np.linalg.norm(vOA) * np.linalg.norm(vOC))
        angle_AOC = np.degrees(np.arccos(np.clip(cos_AOC, -1.0, 1.0)))
        angles_AOC.append(angle_AOC)

    mean_AOB, std_AOB = np.mean(angles_AOB), np.std(angles_AOB)
    mean_AOC, std_AOC = np.mean(angles_AOC), np.std(angles_AOC)

    return (mean_AOB, std_AOB), (mean_AOC, std_AOC)

# Position anomaly
# Compute Euclidean distance between two points
def calculate_distance(coord1, coord2):
    return np.sqrt(np.sum((coord1 - coord2) ** 2))

def cal_position_ano(xroots_main, xroots):
    #input: arcsec
    # Track differences to the nearest pair and their grouping
    differences = []
    pairs_info = []

    # For each point in marked_coords, find the nearest point in marked_coords_nfw and compute differences
    for i, coord in enumerate(xroots):
        # Find the nearest point in marked_coords_nfw
        min_distance = float('inf')
        closest_coord = None
        closest_idx = None
        # Convert coord and closest_coord to numpy arrays before computing differences
        coord = np.array(coord)
        closest_coord = np.array(closest_coord)
        for j, nfw_coord in enumerate(xroots_main):
            nfw_coord = np.array(nfw_coord)
            distance = calculate_distance(coord, nfw_coord)
            if distance < min_distance:
                min_distance = distance
                closest_coord = nfw_coord
                closest_idx = j


        # Difference to the nearest pair
        difference = (coord - closest_coord)*1000
        differences.append(difference)
        
        # Sum of squares for x and y
        square_sum = np.sum(np.square(difference))
        
        # Record grouping and paired points with squared sum
        pairs_info.append({
            'marked_coords_idx': i,
            'marked_coords': coord,
            'marked_coords_nfw_idx': closest_idx,
            'marked_coords_nfw': closest_coord,
            'difference': difference,
            'square_sum': square_sum
        })
    return pairs_info #output: mas

# adding source--Sersic

zpoints = {'u': 30.0, 'g': 30.0, 'r': 30.0, 'i': 30.0, 'z': 30.0} 
@jit
def call_sersic_2d_hist_phosim_c(xi1, xi2, xc1, xc2, a_major, b_minor, pha, ndex=1.0):
    '''
        Sersic Profile from the document of PhoSim
    '''
    phirad = jnp.pi * (pha / 180.0) + jnp.pi / 2.0
    bn = 2.0*ndex-1/3.0+0.009876/ndex
    xi1new = (xi1-xc1)*jnp.cos(phirad)+(xi2-xc2)*jnp.sin(phirad)
    xi2new = (xi2-xc2)*jnp.cos(phirad)-(xi1-xc1)*jnp.sin(phirad)
    R_scale = jnp.sqrt((xi1new/a_major)**2+(xi2new/b_minor)**2)
    R_scale_th = 0.02
    R_scale = jnp.maximum(R_scale, R_scale_th)
    img = jnp.exp(-bn*((R_scale)**(1.0/ndex)-1.0))
    res = img/jnp.exp(-bn*((R_scale_th)**(1.0/ndex)-1.0))
    return res
@jit
def sumfunction(arr):
    return jnp.sum(arr)
# Scaling factor
def sfactor(feq_name,mag,sfc,sum_ss):
    return (10.0**((zpoints[feq_name] - mag)*0.4))*sfc/sum_ss

def add_g_structure(xi1,xi2,var_dict, mag, yi1, yi2, images,disk,bulge,namefeq): 
    ys1 = var_dict["ys1"] 
    ys2 = var_dict["ys2"] 
    sda = var_dict["sda"] 
    sdb = var_dict["sdb"]
    spa = var_dict["spa"] 
    sdn = var_dict["sdn"] 
    sba = var_dict["sba"] 
    sbb = var_dict["sbb"] 
    sbn = var_dict["sbn"] 
    sfc = var_dict["sfc"]
  
    # lensed disks
    sum_ss_disk  = sumfunction(call_sersic_2d_hist_phosim_c(xi1, xi2, ys1, ys2, sda, sdb, spa, sdn))
    sfactor_disk = [0.0] * len(namefeq)

    # lensed bulges
    sum_ss_bulge = sumfunction(call_sersic_2d_hist_phosim_c(xi1, xi2, ys1, ys2, sba, sbb, spa, sbn))
   
    sfactor_bulge = [0.0] * len(namefeq)
    ss_tmp_disk  = call_sersic_2d_hist_phosim_c(yi1, yi2, ys1, ys2, sda, sdb, spa, sdn)
    ss_tmp_bulge = call_sersic_2d_hist_phosim_c(yi1, yi2, ys1, ys2, sba, sbb, spa, sbn)
   
    if sum_ss_disk > 1e-8:
        for pix in range(len(namefeq)):
            sfactor_disk[pix] = sfactor(namefeq[pix],mag[pix],(1.0-sfc),sum_ss_disk)
            sfactor_bulge[pix] = sfactor(namefeq[pix],mag[pix],sfc,sum_ss_bulge)
   
    # stack all images together 
    
    for pix in range(len(namefeq)):
        images[pix] = images[pix] + ss_tmp_disk*sfactor_disk[pix] + ss_tmp_bulge*sfactor_bulge[pix]
        disk[pix] = disk[pix] + ss_tmp_disk*sfactor_disk[pix]
        bulge[pix] = bulge[pix] + ss_tmp_bulge*sfactor_bulge[pix]
   
    print(np.mean(images[0]),np.mean(disk[0]),np.mean(bulge[0]))
    return images


# Gaussian

def gaussian_2d(x, y, x0, y0, sigma, I0, cutoff=0.002):
    gauss = I0 * jnp.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    gauss = gauss.at[gauss < cutoff].set(0)
    return gauss



def mark_brightest_points(map, xi1, xi2, n_points=3, min_distance=0.2):
    marked_coords = []  # Track marked coordinates
    remaining_points = np.unravel_index(np.argsort(map, axis=None)[::-1], map.shape)  # Points sorted by brightness (desc)

    # Find n bright spots
    while len(marked_coords) < n_points and len(remaining_points[0]) > 0:
        # Get the brightest remaining point
        idx = (remaining_points[0][0], remaining_points[1][0])  # Index of brightest point
        
        # Use xi1 and xi2 to convert to coordinates
        x_coord = xi2[idx[1], :]  # Actual X coordinate
        y_coord = xi1[:, idx[0]]  # Actual Y coordinate
        # Enforce x < 0 condition
        if x_coord[0] <= 0:
            # Skip if x does not meet the condition
            remaining_points = (remaining_points[0][1:], remaining_points[1][1:])
            continue
        too_close = False
        # Check proximity to already marked points
        for (y_m, x_m) in marked_coords:
            dist = np.sqrt((x_coord - x_m)**2 + (y_coord - y_m)**2)
            if (dist < min_distance).any():
                too_close = True
                break
        
        # Mark the point if it is not too close to existing ones
        if not too_close:
            marked_coords.append((y_coord[0], x_coord[0]))
            # ax.scatter(x_coord, y_coord, color='red', label='Max Brightness', marker='o', s=20, linewidths=3, edgecolors='black')  # Larger marker with outline

        # Remove the processed point from remaining list
        remaining_points = (remaining_points[0][1:], remaining_points[1][1:])  # Remove the first element

    return marked_coords
