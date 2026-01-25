
import numpy as np
from scipy.special import gamma
from scipy.special import gammaincc, gammaln
from scipy.special import exp1
from Lensing_tool import make_c_coor, SigmaCrit,Da,Da2
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
from jax import jit, vmap,lax
from jax.ops import segment_sum
from tqdm import tqdm
def de_broglie_wavelength(m_eV: float, v_kms: float) -> float:
    """
    Calculate the de Broglie wavelength of a dark matter particle in kpc.

    Parameters
    ----------
    m_eV : float
        Particle mass in eV (e.g., 1e-22).
    v_kms : float
        Velocity in km/s (e.g., 250).

    Returns
    -------
    float
        The de Broglie wavelength in kpc.
    """
    lambda_dB = 0.48 * (1e-22 / m_eV) * (250 / v_kms)
    return lambda_dB

def compute_delta_kappa_rms(kappa_host, A_fluc = 10**-1.3, f=0.48, m_psi=0.8*10**(-22)):
    """
    Compute sqrt(<delta_kappa^2>) using the given parameters.
    arxiv: 2206.11269 eq(19)
    Parameters
    ----------
    A_fluc : float
        Amplitude of fluctuations.
    f : float
        Fractional contribution, e.g. 0.48.+-0.15
    m_psi : Quantity
        Particle mass in units of energy (e.g. 1e-22 * u.eV).
    kappa_host: Kappa value at the NFW reference radius.
 
    """
    term_m = np.sqrt((10**(-22)/m_psi))
    term_sigma_ratio = 2.875
    
    delta_kappa_rms = A_fluc * f  * term_m * kappa_host * term_sigma_ratio
    return delta_kappa_rms


def compute_eps2_dtheta2(delta_kappa_rms,theta_E):
    """
    Compute ε² ⟨δθₓ²⟩ = (3/2)⟨δκ²⟩
    arxiv: 2311.18211 eq(16)
    Parameters
    ----------
    delta_kappa_rms : float
        sqrt(<delta_kappa^2>)

    theta_E: arcsec

    Returns
    -------
    eps2_dtheta2 : float
        ⟨δθₓ²⟩
    """
    epsilon = 1/theta_E
    return np.sqrt(1.5) * delta_kappa_rms/ epsilon
def gaussian_2D_density(r, amp, sigma):
    """
    Compute a 2D Gaussian surface density:
        Sigma(r) = amp * exp(-r^2 / (2 σ^2))

    :param r: 2D radius (scalar or array)
    :param amp: 2D Gaussian amplitude (corresponding to ρ(0))
    :param sigma: Standard deviation
    """
    return amp * np.exp(-r**2 / (2 * sigma**2)) 

def gaussian_lens_potential(R, amp, sigma):
    """
    Axisymmetric lensing potential for a Gaussian surface density:
    Φ(R) = amp * σ² * (γ + exp1(R² / (2σ²)) + log(R² / (2σ²)))
    where γ is the Euler–Mascheroni constant (EULER_GAMMA).

    Parameters
    ----------
    R : array-like
        Radius.
    amp : float
        Peak surface density of the 2D Gaussian.
    sigma : float
        Gaussian width.

    Returns
    -------
    array-like
        Φ(R) evaluated at R.
    """
    EULER_GAMMA = 0.57721566490153286060
    x = R**2 / (2 * sigma**2)
    phi = amp * sigma**2 * (EULER_GAMMA + exp1(x) + np.log(x))
    return phi


def gaussian_deflection(R, amp, sigma):
    """
    Axisymmetric deflection angle α(R) for a Gaussian surface density:
    α(R) = (2 * amp * sigma^2 / R) * (1 - exp(-R^2 / (2 sigma^2)))

    Parameters
    ----------
    R : array-like
        Radius.
    amp : float
        Peak surface density of the 2D Gaussian.
    sigma : float
        Gaussian width.

    Returns
    -------
    array-like
        Magnitude of the deflection angle α(R).
    """
    R = np.asarray(R)
    alpha = (2 * amp * sigma**2 / R) * (1 - np.exp(-R**2 / (2 * sigma**2)))

    return alpha

@jit
def gaussian_deflection_jax(R, amp, sigma):
    eps = 1e-8
    return (2 * amp * sigma**2 / (R + eps)) * (1 - jnp.exp(-R**2 / (2 * sigma**2)))


class GAUSSIAN_parametric_Multiplane(object):
    def __init__(self, Gaussian_params, bsz_arc=200, dsx_arc=0.2, nnn=2000, xi1=None, xi2=None, zsource=1.0):
        self.zsource = zsource
        # Grid setup
        if xi1 is None or xi2 is None:
            self.bsz_arc, self.dsx_arc = bsz_arc, dsx_arc
            self.nnn = nnn
            self.xi2, self.xi1 = make_c_coor(self.bsz_arc, self.nnn)
        else:
            self.xi1, self.xi2 = xi1, xi2
            self.dsx_arc = xi1[1,0] - xi1[0,0]
            self.nnn    = xi1.shape[0]

        # Extract parameters and convert to JAX arrays
        amps   = [p['amp']       for p in Gaussian_params]
        sigmas = [p['sigma']     for p in Gaussian_params]
        zs     = [p['redshift']  for p in Gaussian_params]
        xs     = [p['position_x'] for p in Gaussian_params]
        ys     = [p['position_y'] for p in Gaussian_params]

        amps   = [a / (2 * np.pi * s**2) for a, s in zip(amps, sigmas)]  # Convert to kappa0


        self.amp_array      = jnp.array(amps)
        self.sigma_array    = jnp.array(sigmas)
        self.redshift_array = jnp.array(zs)
        self.pos_x_array    = jnp.array(xs)
        self.pos_y_array    = jnp.array(ys)

        # Sort by redshift and build grouping indices
        idx = jnp.argsort(self.redshift_array)
        self.amp_array      = self.amp_array[idx]
        self.sigma_array    = self.sigma_array[idx]
        self.redshift_array = self.redshift_array[idx]
        self.pos_x_array    = self.pos_x_array[idx]
        self.pos_y_array    = self.pos_y_array[idx]

        self.grouped_redshift_array, self._inv_z_idx = jnp.unique(
            self.redshift_array, return_inverse=True
        )
        self._sigma_cr_array = None
        
    def _prepare_common_arrays(self):
        if self._sigma_cr_array is None:
            # Vectorized Σ_crit calculation
            self._sigma_cr_array = vmap(lambda z: SigmaCrit(z, self.zsource))(
                self.redshift_array
            )


    def _compute_grouped_alpha_maps(self, batch_size=30):
        """
        Compute combined raw α1, α2 maps for each redshift bin.
        Batch processing controls memory usage.
        Returns grouped_alpha1, grouped_alpha2.
        """
        # 1) Prepare required parameters
        G     = self.grouped_redshift_array.shape[0]
        nx, ny = self.xi1.shape


        # 2) Initialize accumulators
        grouped_alpha1 = jnp.zeros((G, nx, ny), dtype=jnp.float32)
        grouped_alpha2 = jnp.zeros((G, nx, ny), dtype=jnp.float32)

        # 3) Effective amplitudes
        amps_eff = self.amp_array

        n_halo = amps_eff.shape[0]
        for start in tqdm(range(0, n_halo, batch_size), desc="Computing α maps"):
            end = min(start + batch_size, n_halo)
            idx = slice(start, end)

            # 4.1) Compute offsets lazily to avoid large allocations
            arc_dx = self.xi1[None] - self.pos_x_array[idx, None, None]
            arc_dy = self.xi2[None] - self.pos_y_array[idx, None, None]
            R_ang  = jnp.sqrt(arc_dx**2 + arc_dy**2)

            # 4.2) Batch parameters
            amp_batch   = amps_eff[idx][:, None, None]
            sigma_batch = self.sigma_array[idx][:, None, None]

            # 4.3) Parallel compute α(R) for this batch
            alpha_batch = vmap(gaussian_deflection_jax, in_axes=(0,0,0))(
                R_ang, amp_batch, sigma_batch
            )  # shape = (batch, nx, ny)

            # 4.4) Split into α1, α2
            alpha1_batch = alpha_batch * (arc_dx / (R_ang + 1e-8))
            alpha2_batch = alpha_batch * (arc_dy / (R_ang + 1e-8))

            # 4.5) Flatten and accumulate by redshift group
            flat1 = alpha1_batch.reshape((end - start, -1)).astype(jnp.float32)
            flat2 = alpha2_batch.reshape((end - start, -1)).astype(jnp.float32)
            inv_idx = self._inv_z_idx[idx]  # length = end-start

            sum1 = segment_sum(flat1, inv_idx, G)  # (G, nx*ny)
            sum2 = segment_sum(flat2, inv_idx, G)

            # 4.6) Release temporary arrays to reduce peak memory
            del flat1, flat2, alpha1_batch, alpha2_batch
            del arc_dx, arc_dy, R_ang, amp_batch, sigma_batch, alpha_batch
            grouped_alpha1 += sum1.reshape((G, nx, ny))
            grouped_alpha2 += sum2.reshape((G, nx, ny))



        return grouped_alpha1, grouped_alpha2




def uldm_density(r, kappa_0, theta_c, slope=8):
    r"""
    ULDM soliton core 3D density:
        ρ(r) = ρ̃₀ [1 + a (r/θ_c)^2]^{-β}

    where:
      - β = slope (default 8);
      - a = 0.5^{-1/β} - 1;
      - ρ̃₀ = κ₀ Γ(β) / Γ(β - 1/2) * sqrt(0.5^{-1/β} - 1) / (sqrt(π) θ_c).

    :param r: 3D radius (scalar or NumPy array, same units as θ_c)
    :param kappa_0: Central convergence κ₀
    :param theta_c: Core angular radius θ_c
    :param slope: Power-law index β (default 8)
    :return: ρ(r) with units consistent with the angular inputs
    """
    # 1. Compute a = 0.5^{-1/β} - 1
    a = 0.5 ** (-1.0 / slope) - 1.0
    # 2. Compute ρ̃₀

    rho_tilde = (
        kappa_0
        * gamma(slope) / gamma(slope - 0.5)
        * np.sqrt(a / np.pi) / theta_c
    )
    # 3. Return ρ(r)
    return rho_tilde / (1.0 + a * (r / theta_c) ** 2) ** slope


def uldm_deflection(x, y, kappa_0, theta_c, slope=8, center_x=0, center_y=0, eps=1e-8):
    """
    Deflection angle (α_x, α_y) for a ULDM soliton core.

    Parameters
    ----------
    x, y : array-like
        Angular coordinates of the target point (scalar or NumPy array).
    kappa_0 : float
        Central convergence κ₀.
    theta_c : float
        Core angular radius θ_c.
    slope : float
        Power-law index β (default 8).
    center_x, center_y : float
        Core center coordinates (default 0, 0).
    eps : float
        Small number to avoid division by zero when r=0.

    Returns
    -------
    tuple of array-like
        (alpha_x, alpha_y) with the same shape as x, y.
    """
    # 1. Compute radial distance r
    x_rel = np.array(x) - center_x
    y_rel = np.array(y) - center_y
    r = np.sqrt(x_rel**2 + y_rel**2)
    # Avoid division by zero at r=0
    r_safe = np.maximum(r, eps)

    # 2. Compute a = 0.5^{-1/β} - 1
    a = 0.5 ** (-1.0 / slope) - 1.0

    # 3. Prefactor for the radial deflection
    #    α(r) = [2 κ₀ θ_c² / ((2β - 3) a)] * [1 - (1 + a (r/θ_c)²)^{3/2 - β}] / r
    pref = 2.0 * kappa_0 * theta_c**2 / ((2 * slope - 3) * a)
    denom = (1.0 + a * (r_safe / theta_c)**2) ** (slope - 1.5)
    alpha_r = pref * (1.0 - 1.0 / denom) / r_safe

    # 4. Vector components
    alpha_x = alpha_r * (x_rel / r_safe)
    alpha_y = alpha_r * (y_rel / r_safe)

    return alpha_x, alpha_y

@jit
def uldm_deflection_jax(R, kappa0, theta_c, slope=8.0):
    """
    JAX version of the ULDM soliton core deflection magnitude α(r).

    Parameters
    ----------
    R : array-like
        Radial distance (scalar or array).
    kappa0 : float
        Central convergence κ₀.
    theta_c : float
        Core angular radius θ_c.
    slope : float
        Power-law index β (default 8).

    Returns
    -------
    array-like
        α(r) with the same shape as R.
    """
    eps = 1e-8
    r_safe = jnp.maximum(R, eps)
    a = 0.5 ** (-1.0 / slope) - 1.0
    pref = 2.0 * kappa0 * theta_c**2 / ((2 * slope - 3) * a)
    denom = (1.0 + a * (r_safe / theta_c)**2) ** (slope - 1.5)
    return pref * (1.0 - 1.0 / denom) / r_safe

class ULDM_parametric_Multiplane(object):
    def __init__(self, ULDM_params,
                 bsz_arc=200, dsx_arc=0.2, nnn=2000,
                 xi1=None, xi2=None,
                 zsource=1.0, slope=8.0):
        self.zsource = zsource
        self.slope   = slope

        # Grid setup
        if xi1 is None or xi2 is None:
            self.bsz_arc, self.dsx_arc = bsz_arc, dsx_arc
            self.nnn = nnn
            self.xi2, self.xi1 = make_c_coor(self.bsz_arc, self.nnn)
        else:
            self.xi1, self.xi2 = xi1, xi2
            self.dsx_arc = xi1[1,0] - xi1[0,0]
            self.nnn    = xi1.shape[0]

        # Extract parameters and convert to JAX arrays
        kappas  = [p['kappa_0']    for p in ULDM_params]
        thetacs = [p['theta_c']   for p in ULDM_params]
        zs      = [p['redshift']  for p in ULDM_params]
        xs      = [p['position_x'] for p in ULDM_params]
        ys      = [p['position_y'] for p in ULDM_params]

        self.kappa0_array   = jnp.array(kappas)
        self.theta_c_array  = jnp.array(thetacs)
        self.redshift_array = jnp.array(zs)
        self.pos_x_array    = jnp.array(xs)
        self.pos_y_array    = jnp.array(ys)

        # Sort by redshift and build grouping indices
        idx = jnp.argsort(self.redshift_array)
        self.kappa0_array   = self.kappa0_array[idx]
        self.theta_c_array  = self.theta_c_array[idx]
        self.redshift_array = self.redshift_array[idx]
        self.pos_x_array    = self.pos_x_array[idx]
        self.pos_y_array    = self.pos_y_array[idx]

        self.grouped_redshift_array, self._inv_z_idx = jnp.unique(
            self.redshift_array, return_inverse=True
        )
        self._sigma_cr_array = None

    def _prepare_common_arrays(self):
        if self._sigma_cr_array is None:
            # Precompute Σ_crit for potential normalization
            self._sigma_cr_array = vmap(lambda z: SigmaCrit(z, self.zsource))(
                self.redshift_array
            )

    def _compute_grouped_alpha_maps(self, batch_size=30):
        """
        Compute combined raw α1, α2 maps for each redshift bin.
        Batch processing controls memory usage.
        Returns grouped_alpha1, grouped_alpha2.
        """
        # 1) Prepare
        G     = self.grouped_redshift_array.shape[0]
        nx, ny = self.xi1.shape

        # 2) Initialize output
        grouped_alpha1 = jnp.zeros((G, nx, ny), dtype=jnp.float32)
        grouped_alpha2 = jnp.zeros((G, nx, ny), dtype=jnp.float32)

        # 3) Effective kappa (can switch to kappa0/σ_cr if needed)
        kappa_eff = self.kappa0_array
        n_halo    = kappa_eff.shape[0]

        # 4) Iterate by batch
        for start in tqdm(range(0, n_halo, batch_size), desc="Computing α maps"):
            end = min(start + batch_size, n_halo)
            idx = slice(start, end)

            # 4.1) Offsets and radial distance for this batch
            arc_dx = self.xi1[None] - self.pos_x_array[idx, None, None]
            arc_dy = self.xi2[None] - self.pos_y_array[idx, None, None]
            R      = jnp.sqrt(arc_dx**2 + arc_dy**2)

            # 4.2) Batch parameters
            k0b = kappa_eff[idx][:, None, None]
            tcb = self.theta_c_array[idx][:, None, None]

            # 4.3) Parallel deflection calculation
            alpha_batch = vmap(
                uldm_deflection_jax,
                in_axes=(0, 0, 0, None),
                out_axes=0
            )(R, k0b, tcb, self.slope)  # (batch, nx, ny)

            # 4.4) Split into α1, α2
            alpha1_b = alpha_batch * (arc_dx / (R + 1e-8))
            alpha2_b = alpha_batch * (arc_dy / (R + 1e-8))

            # 4.5) Flatten and accumulate by redshift group
            flat1   = alpha1_b.reshape((end-start, -1)).astype(jnp.float32)
            flat2   = alpha2_b.reshape((end-start, -1)).astype(jnp.float32)
            inv_idx = self._inv_z_idx[idx]

            sum1 = segment_sum(flat1, inv_idx, G)  # (G, nx*ny)
            sum2 = segment_sum(flat2, inv_idx, G)

            del flat1, flat2, alpha1_b, alpha2_b
            del arc_dx, arc_dy, R, k0b, tcb, alpha_batch

            grouped_alpha1 = grouped_alpha1 + sum1.reshape((G, nx, ny))
            grouped_alpha2 = grouped_alpha2 + sum2.reshape((G, nx, ny))


        return grouped_alpha1, grouped_alpha2
    
@jit
def _nfw_kernel_jax(x):
    """
    Core NFW function G_nfw(x):
    - x < 1: (1−x²)^(-1/2) arctanh(√(1−x²))
    - x > 1: (x²−1)^(-1/2) arctan(√(x²−1))
    """
    eps = 1e-6
    x_safe = jnp.where(x == 1.0, 1.0 - eps, x)
    return jnp.where(
        x_safe < 1.0,
        (1.0 - x_safe**2)**(-0.5) * jnp.arctanh(jnp.sqrt(1.0 - x_safe**2)),
        (x_safe**2 - 1.0)**(-0.5) * jnp.arctan(jnp.sqrt(x_safe**2 - 1.0))
    )

@jit
def _G_jax(X, b):
    """
    G(x, b) function for CNFW.
    """
    c = 1e-6
    b_safe = jnp.where(b == 1.0, 1.0 + c, b)
    x2 = X**2
    b2 = b_safe**2
    fac  = (1.0 - b_safe)**2
    pref = 1.0 / fac

    # Two branches: use term1 when |X - b| > c, otherwise term2
    term1 = pref * (
        fac * jnp.log(0.25 * x2)
        - b2 * jnp.log(b2)
        + 2.0 * (b2 - x2) * _nfw_kernel_jax(X / b_safe)
        + 2.0 * (1.0 + b_safe * (x2 - 2.0)) * _nfw_kernel_jax(X)
    )
    term2 = pref * (
        2.0 * (1.0 - 2.0 * b_safe + b_safe**3) * _nfw_kernel_jax(b_safe)
        + fac * (-1.38692 + jnp.log(b2))
        - b2 * jnp.log(b2)
    )

    return 0.5 * jnp.where(jnp.abs(X - b_safe) > c, term1, term2)

@jit
def _alpha2rho0(alpha_Rs, Rs, r_core):
    """
    Convert alpha_Rs to ρ0:
      b   = r_core / Rs
      gx  = _G_jax(1.0, b)
      ρ0  = alpha_Rs / (4 * Rs^2 * gx)
    """
    b = r_core / Rs
    gx = _G_jax(1.0, b)
    rho0 = alpha_Rs / (4 * Rs**2 * gx)
    return rho0


@jit
def cnfw_deflection_jax(R, Rs, rho0, r_core):
    """
    JAX implementation of the CNFW radial deflection α_r(R).

    Parameters
    ----------
    R : array-like
        Radial distance (scalar or array).
    Rs : float
        NFW scale radius.
    rho0 : float
        Density normalization (derived from alpha_Rs).
    r_core : float
        Core radius.

    Returns
    -------
    array-like
        α_r with the same shape as R.
    """
    eps = 1e-8
    R_safe = jnp.maximum(R, eps)
    x = R_safe / Rs
    b = r_core / Rs
    gx = _G_jax(x, b)
    return 4.0 * rho0 * Rs**2 * gx / x
class CNFW_parametric_Multiplane(object):
    def __init__(self, CNFW_params, bsz_arc=200, dsx_arc=0.2, nnn=2000,
                 xi1=None, xi2=None, zsource=1.0):
        self.zsource = zsource

        # Grid setup
        if xi1 is None or xi2 is None:
            self.bsz_arc, self.dsx_arc = bsz_arc, dsx_arc
            self.nnn = nnn
            self.xi2, self.xi1 = make_c_coor(self.bsz_arc, self.nnn)
        else:
            self.xi1, self.xi2 = xi1, xi2
            self.dsx_arc = xi1[1,0] - xi1[0,0]
            self.nnn    = xi1.shape[0]

        # Extract parameters and convert to JAX arrays 
        Rs      = jnp.array([p['Rs']       for p in CNFW_params])
        aRs     = jnp.array([p['alpha_Rs'] for p in CNFW_params])
        rcore   = jnp.array([p['r_core']   for p in CNFW_params])
        zs      = jnp.array([p['redshift'] for p in CNFW_params])
        xs      = jnp.array([p['position_x'] for p in CNFW_params])
        ys      = jnp.array([p['position_y'] for p in CNFW_params])

        # Sort by redshift and build grouping indices 
        idx = jnp.argsort(zs)
        self.Rs_array       = Rs[idx]
        self.alphaRs_array  = aRs[idx]
        self.r_core_array   = rcore[idx]
        self.redshift_array = zs[idx]
        self.pos_x_array    = xs[idx]
        self.pos_y_array    = ys[idx]

        self.grouped_redshift_array, self._inv_z_idx = jnp.unique(
            self.redshift_array, return_inverse=True
        )
        self._sigma_cr_array = None

    def _prepare_common_arrays(self):
        if self._sigma_cr_array is None:
            # Vectorized Σ_crit calculation
            self._sigma_cr_array = vmap(lambda z: SigmaCrit(z, self.zsource))(
                self.redshift_array
            )

    def _compute_grouped_alpha_maps(self, batch_size=32):
        """
        Compute combined raw α1, α2 maps for each redshift bin.
        Batch processing controls memory usage.
        Returns grouped_alpha1, grouped_alpha2.
        """
        # 1) Prepare
        G      = self.grouped_redshift_array.shape[0]
        nx, ny = self.xi1.shape

        # 2) Compute effective rho0
        aRsb_eff = self.alphaRs_array
        rho0_eff = vmap(_alpha2rho0)(aRsb_eff, self.Rs_array, self.r_core_array)

        # 3) Initialize output
        grouped_alpha1 = jnp.zeros((G, nx, ny), dtype=jnp.float32)
        grouped_alpha2 = jnp.zeros((G, nx, ny), dtype=jnp.float32)

        n_halo = self.Rs_array.shape[0]
        for start in tqdm(range(0, n_halo, batch_size), desc="Computing α maps"):
            end = min(start + batch_size, n_halo)
            idx = slice(start, end)

            # 4.1) Compute offsets and radial distance for this batch
            arc_dx = self.xi1[None] - self.pos_x_array[idx, None, None]
            arc_dy = self.xi2[None] - self.pos_y_array[idx, None, None]
            R_ang  = jnp.sqrt(arc_dx**2 + arc_dy**2)

            # 4.2) Batch parameters
            Rsb  = self.Rs_array[idx][:, None, None]
            rcb  = self.r_core_array[idx][:, None, None]
            rho0b = rho0_eff[idx][:, None, None]

            # 4.3) Parallel computation of α_r for this batch
            alpha_r_batch = vmap(
                cnfw_deflection_jax,
                in_axes=(0,0,0,0),
                out_axes=0
            )(R_ang, Rsb, rho0b, rcb)  # (batch, nx, ny)

            # 4.4) Split into α1, α2
            a1 = alpha_r_batch * (arc_dx / (R_ang + 1e-8))
            a2 = alpha_r_batch * (arc_dy / (R_ang + 1e-8))

            # 4.5) Flatten and accumulate by redshift group
            flat1   = a1.reshape((end-start, -1)).astype(jnp.float32)
            flat2   = a2.reshape((end-start, -1)).astype(jnp.float32)
            inv_idx = self._inv_z_idx[idx]

            sum1 = segment_sum(flat1, inv_idx, G)  # (G, nx*ny)
            sum2 = segment_sum(flat2, inv_idx, G)

            # 4.6) Release temporary arrays to reduce peak memory
            del arc_dx, arc_dy, R_ang, Rsb, rcb, rho0b, alpha_r_batch , a1, a2, flat1, flat2
            grouped_alpha1 += sum1.reshape((G, nx, ny))
            grouped_alpha2 += sum2.reshape((G, nx, ny))



        return grouped_alpha1, grouped_alpha2


import numpy as np

from pyHalo.PresetModels.wdm import WDM
from pyHalo.concentration_models import preset_concentration_models
from pyHalo.pyhalo import pyHalo
from pyHalo.realization_extensions import RealizationExtensions
from pyHalo.utilities import MinHaloMassULDM, de_broglie_wavelength


def ULDM_r(z_lens, z_source, log10_m_uldm, log10_fluc_amplitude=-0.8, fluctuation_size_scale=0.05,
          fluctuation_size_dispersion=0.2, n_fluc_scale=1.0, velocity_scale=200, sigma_sub=0.025,
         log10_sigma_sub=None, log_mlow=6., log_mhigh=10.,
        mass_function_model_subhalos='SHMF_SCHIVE2016', kwargs_mass_function_subhalos={},
        mass_function_model_fieldhalos='SCHIVE2016', kwargs_mass_function_fieldhalos={},
        concentration_model_subhalos='LAROCHE2022', kwargs_concentration_model_subhalos={},
        concentration_model_fieldhalos='LAROCHE2022', kwargs_concentration_model_fieldhalos={},
        truncation_model_subhalos='TRUNCATION_GALACTICUS', kwargs_truncation_model_subhalos={},
        truncation_model_fieldhalos='TRUNCATION_RN', kwargs_truncation_model_fieldhalos={},
        shmf_log_slope=-1.9, cone_opening_angle_arcsec=6., log_m_host=13.3, r_tidal=0.25,
        LOS_normalization=1.0, geometry_type='DOUBLE_CONE', kwargs_cosmo=None,
         uldm_plaw=1 / 3, flucs=True, flucs_shape='aperture', flucs_args={}, n_cut=50000, r_ein=1.0):

    # constants
    m22 = 10**(log10_m_uldm + 22)
    log_m0 = np.log10(1.6e10 * m22**(-4/3))

    # FIRST WE CREATE AN INSTANCE OF PYHALO, WHICH SETS THE COSMOLOGY
    pyhalo = pyHalo(z_lens, z_source, kwargs_cosmo)

    # compute M_min as described in documentation
    log_m_min = MinHaloMassULDM(log10_m_uldm, pyhalo.astropy_cosmo, log_mlow)
    kwargs_density_profile = {}
    kwargs_density_profile['log10_m_uldm'] = log10_m_uldm
    kwargs_density_profile['scale_nfw'] = False
    kwargs_density_profile['uldm_plaw'] = uldm_plaw
    kwargs_wdm = {'z_lens': z_lens, 'z_source': z_source, 'log_mc': log_m0, 'sigma_sub': sigma_sub,
                  'log10_sigma_sub': log10_sigma_sub,
                  'log_mlow': log_m_min, 'log_mhigh': log_mhigh,
                  'mass_function_model_subhalos': mass_function_model_subhalos,
                  'kwargs_mass_function_subhalos': kwargs_mass_function_subhalos,
                  'mass_function_model_fieldhalos': mass_function_model_fieldhalos,
                  'kwargs_mass_function_fieldhalos': kwargs_mass_function_fieldhalos,
                  'concentration_model_subhalos': concentration_model_subhalos,
                  'kwargs_concentration_model_subhalos': kwargs_concentration_model_subhalos,
                  'concentration_model_fieldhalos': concentration_model_fieldhalos,
                  'kwargs_concentration_model_fieldhalos': kwargs_concentration_model_fieldhalos,
                  'truncation_model_subhalos': truncation_model_subhalos,
                  'kwargs_truncation_model_subhalos': kwargs_truncation_model_subhalos,
                  'truncation_model_fieldhalos': truncation_model_fieldhalos,
                  'kwargs_truncation_model_fieldhalos': kwargs_truncation_model_fieldhalos,
                  'shmf_log_slope': shmf_log_slope, 'cone_opening_angle_arcsec': cone_opening_angle_arcsec,
                  'log_m_host': log_m_host, 'r_tidal': r_tidal, 'LOS_normalization': LOS_normalization,
                  'geometry_type': geometry_type, 'kwargs_cosmo': kwargs_cosmo,
                  'mdef_subhalos': 'ULDM', 'mdef_field_halos': 'ULDM',
                  'kwargs_density_profile': kwargs_density_profile
                  }

    uldm_no_fluctuations = WDM(**kwargs_wdm)
    if flucs:  # add fluctuations to realization
        ext = RealizationExtensions(uldm_no_fluctuations)
        lambda_dB = de_broglie_wavelength(log10_m_uldm, velocity_scale)  # de Broglie wavelength in kpc

        if flucs_args == {}:
            raise Exception('Must specify fluctuation arguments, see realization_extensions.add_ULDM_fluctuations')

        a_fluc = 10 ** log10_fluc_amplitude
        m_psi = 10 ** log10_m_uldm

        zlens_ref, zsource_ref = 0.5, 2.0
        mhost_ref = 10**13.3
        rein_ref = 1.0
        r_perp_ref = rein_ref * uldm_no_fluctuations.lens_cosmo.cosmo.kpc_proper_per_asec(zlens_ref)

        model, _ = preset_concentration_models('DIEMERJOYCE19')
        concentration_model_for_host = model(pyhalo.astropy_cosmo)
        sigma_crit_ref = uldm_no_fluctuations.lens_cosmo.get_sigma_crit_lensing(zlens_ref, zsource_ref)
        c_host_ref = concentration_model_for_host.nfw_concentration(mhost_ref, z_lens)
        rhos_ref, rs_ref, _ = uldm_no_fluctuations.lens_cosmo.NFW_params_physical(mhost_ref, c_host_ref, zlens_ref)
        xref = r_perp_ref / rs_ref
        if xref < 1:
            Fxref = np.arctanh(np.sqrt(1 - xref ** 2)) / np.sqrt(1 - xref ** 2)
        else:
            Fxref = np.arctan(np.sqrt(-1 + xref ** 2)) / np.sqrt(-1 + xref ** 2)
        sigma_host_ref = 2 * rhos_ref * rs_ref * (1 - Fxref) / (xref**2 - 1)

        # host at actual lens
        r_perp = r_ein * uldm_no_fluctuations.lens_cosmo.cosmo.kpc_proper_per_asec(z_lens)
        sigma_crit = uldm_no_fluctuations.lens_cosmo.get_sigma_crit_lensing(z_lens, z_source)
        c_host = concentration_model_for_host.nfw_concentration(10**log_m_host, z_lens)
        rhos, rs, _ = uldm_no_fluctuations.lens_cosmo.NFW_params_physical(10**log_m_host, c_host, z_lens)

        # Helper to compute the local Sigma_host(r)
        def _sigma_host_at_rperp(r_perp_kpc, rhos_, rs_):
            x = r_perp_kpc / rs_
            if x < 1:
                Fx = np.arctanh(np.sqrt(1 - x ** 2)) / np.sqrt(1 - x ** 2)
            else:
                Fx = np.arctan(np.sqrt(-1 + x ** 2)) / np.sqrt(-1 + x ** 2)
            return 2 * rhos_ * rs_ * (1 - Fx) / (x ** 2 - 1)

        # Scale fluctuation amplitude by radius per annulus
        prefactor = a_fluc * (m_psi / 1e-22) ** (-0.5) * (sigma_crit_ref / sigma_crit) / sigma_host_ref
        kpc_per_arcsec = uldm_no_fluctuations.lens_cosmo.cosmo.kpc_proper_per_asec(z_lens)

        N_RING = 100                         # Tunable: number of rings
        r_in_min = 0.01 * r_ein             # Tunable: inner radius (arcsec)
        r_out_max = 3 * r_ein            # Tunable: outer radius (arcsec)
        r_edges = np.linspace(r_in_min, r_out_max, N_RING + 1)

        # Weight by ring area to avoid rounding small rings to zero
        areas = []
        for i in range(N_RING):
            rin_arc = r_edges[i]
            rout_arc = r_edges[i + 1]
            # Area ∝ r_out^2 - r_in^2 (arcsec^2 units keep the same scaling)
            areas.append(max(1e-12, rout_arc**2 - rin_arc**2))
        area_sum = sum(areas)

        uldm_realization = None
        for i in range(N_RING):
            rin_arc = r_edges[i]
            rout_arc = r_edges[i + 1]
            rmid_arc = 0.5 * (rin_arc + rout_arc)
            rmid_kpc = rmid_arc * kpc_per_arcsec

            sigma_local = _sigma_host_at_rperp(rmid_kpc, rhos, rs)
            A_ring = prefactor * sigma_local


            
            # Key steps: specify rmin/rmax and accumulate instead of overwrite
            uldm_realization = ext.add_ULDM_fluctuations(
                de_Broglie_wavelength=lambda_dB,
                fluctuation_amplitude=A_ring,
                fluctuation_size=lambda_dB * fluctuation_size_scale,
                fluctuation_size_variance=lambda_dB * fluctuation_size_scale * fluctuation_size_dispersion,
                n_fluc_scale=n_fluc_scale,
                shape='ring',
                args={'rmin': rin_arc, 'rmax': rout_arc},   # pyHalo expects rmin/rmax in arcsec
                n_cut=n_cut
            )

            # Allow the next ring to build on the updated realization
            ext = RealizationExtensions(uldm_realization)

        return uldm_realization

    else:
        return uldm_no_fluctuations
