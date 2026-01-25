import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
from jax import jit, vmap,lax
from jax.ops import segment_sum
import numpy as np
from Lensing_tool import make_c_coor, SigmaCrit,Da,Da2
import pickle
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u 
import astropy.constants as const 
from SIDM_density_fluid import Numerical_2d_density
from tqdm import tqdm


Omegam0 = 0.3157
Omegab0 = 0.04936
sigma8 = 0.8116
Omegar0 = 5.3815e-5
Omegak0 = 0
OmegaL0 = 1 - Omegam0
c = const.c.to(u.km/u.s).value 
G = const.G.to(u.Mpc/u.solMass *(u.km/u.second)*(u.km/u.second)).value 
apr =  1.0/np.pi*180.0*3600  # arcsec per rad
cosmo = FlatLambdaCDM(H0=70, Om0=Omegam0)
h = cosmo.h

class SIDM_parametric_simple(object):
    # PSIDM-25 model
    def get_density(self, r, current_tr, rhoss=1.0, rss=1.0):
        '''Calculate the density at a given radius.'''
        current_rhos = self.fitted_rhos(current_tr)
        current_rs = self.fitted_rs(current_tr)
        current_rc = self.fitted_rc(current_tr)
        current_a = self.fitted_beta(current_tr)
        current_c = self.fitted_gamma(current_tr)
        Density = self.funrho(r/rss, current_rhos, current_rs, current_rc, current_a, current_c) * rhoss
        return Density

    def get_density_baryon(self, r, tr0, trb, rhoss=1.0, rss=1.0):
        '''Calculate the baryon-corrected density at a given radius.'''
        Ft = tr0 / trb
        current_rhos = self.fitted_rhos(trb)
        current_rs = self.fitted_rs(trb)
        current_rc = self.fitted_rc(trb) * Ft**2
        current_a = self.fitted_beta(trb)
        current_c = self.fitted_gamma(trb)
        Density = self.funrho(r/rss, current_rhos, current_rs, current_rc, current_a, current_c) * rhoss
        return Density

    def fitted_gamma(self, x):
        '''Calculate the fitted gamma coefficient.'''
        popt_c = [2.54741454, -2.75585114, 6.11802594, -7.28902641, 3.41868933, 1.33017751]
        a, b, c, d, g, h = popt_c
        A = 0.1
        B = 3
        C = -0.5
        return a + b * x + c * x**2 + d * x**3 + g * x**4 + 1 / (np.log(0.001)) * (1 - a) * np.log(x**h + 0.001) + A * (np.arctan(B * (x + C)) - np.arctan(B * C))

    def fitted_rhos(self, x):
        '''Calculate the fitted rho coefficient.'''
        a, b, c, d, g, h, i = -1.54891697e+01, 7.22099942e+00, 1.77835229e+01, -7.21790903e+02, 5.64041228e+02, -4.52714226e-01, -1.24201628e+00
        return np.exp(a * pow(x, 0.3) + b * x**0.2 + c * x**0.7 + d * x*0.8 + g * x + h * x**6 + i * x**12)

    def fitted_rs(self, x):
        '''Calculate the fitted rs coefficient.'''
        a, b, c, d, g, h, i = 2.79650252, -0.02224725, -2.68712924, 3.42099585, 0.06969549, 2.02747497, 0.03140837
        return (1 + a * pow(x, 0.6) + c * x**1 + d * x**3 + g * x**10 + h * x**17 + i * x**80 + b * x**83)

    def fitted_beta(self, x):
        '''Calculate the fitted alpha coefficient.'''
        a, b, c, d, g, h, i = -0.41711945, 0.29619283, 0.20237198, 0.93450782, -2.10957393, 2.00795074, -0.57618027
        return (1 + a * pow(x, 0.6) + c * x**2 + d * x**13 + g * x**26 + h * x**39 + i * x**66 + b * x**71)

    def fitted_rc(self, x):
        '''Calculate the fitted rc coefficient.'''
        a, b, c, d, g, h, i = 6.04642694e+00, -6.95669182e+00, 1.48570521e+00, -6.23443139e-01, 2.09573098e-01, -3.37958160e-02, 5.02603829e-05
        return a * pow(x, 0.6) + b * pow(x, 0.8) + c * x**2 + d * x**5 + g * x**12 + h * x**27 + i * x**80

    def funrho(self, r, rhos, rs, rc, a, c):
        '''Calculate the density using the given formula.'''
        k = 4  # Define the slope parameter
        core_term = (r / rs)**k + (rc / rs)**k
        core_component = core_term**(c / k)
        outer_term = 1 + (r / rs)**a
        outer_component = outer_term**((3 - c) / a)
        return rhos / (core_component * outer_component)

    # Beta4 model
    def get_density_beta4(self, r, current_tr, rhoss=1.0, rss=1.0):
        '''Calculate the density using the beta4 model.'''
        current_rhos = self.rhost(current_tr, rhoss, rss)
        current_rs = self.rst(current_tr, rhoss, rss)
        current_rc = self.rct(current_tr, rhoss, rss)
        return self.frho(r, current_rhos, current_rs, current_rc)

    def get_density_beta4_baryon(self, r, tr0, trb, rhoss=1.0, rss=1.0):
        '''Calculate the baryon-corrected density using the beta4 model.'''
        Ft = tr0 / trb
        current_rhos = self.rhost(trb, rhoss, rss)
        current_rs = self.rst(trb, rhoss, rss)
        current_rc = self.rct(trb, rhoss, rss) * Ft**2
        return self.frho(r, current_rhos, current_rs, current_rc)

    def frho(self, r, rhos, rs, rc):
        '''Compute the rho function for density.'''
        self.k = 4
        return rhos * rs / pow(pow(r, self.k) + pow(rc, self.k), 1. / self.k) / pow(1 + r / rs, 2)

    def rhost(self, tr, rhoss, rss):
        '''Calculate the rho density for a given temperature.'''
        val = 2.03305816 + 0.73806287 * tr + 7.26368767 * pow(tr, 5) - 12.72976657 * pow(tr, 7) + 9.91487857 * pow(tr, 9) - 0.1448 * (1 - 2.03305816) * np.log(tr + 0.001)
        return val * rhoss

    def rst(self, tr, rhoss, rss):
        '''Calculate the rs parameter for a given temperature.'''
        val = 0.71779858 - 0.10257242 * tr + 0.24743911 * pow(tr, 2) - 0.40794176 * pow(tr, 3) - 0.1448 * (1 - 0.71779858) * np.log(tr + 0.001)
        return val * rss

    def rct(self, tr, rhoss, rss):
        '''Calculate the rc parameter for a given temperature.'''
        val = 2.55497727 * np.sqrt(tr) - 3.63221179 * tr + 2.13141953 * pow(tr, 2) - 1.41516784 * pow(tr, 3) + 0.46832269 * pow(tr, 4)
        return val * rss


# Some functions for PSIDM-25 parametric model

@jit
def fit_potential(r, a, b, c, d, e):
    result = a * jnp.log(1 + b * r + c * r**2)**c - jnp.log(d * r + 1)**e
    return result


@jit
def fit_alpha(r, a, b, c, d, e):
    term1 = (a * c * (b + 2 * c * r) * jnp.log(b * r + c * r**2 + 1)**(c - 1)) / (b * r + c * r**2 + 1)
    term2 = (d * e * jnp.log(d * r + 1)**(e - 1)) / (d * r + 1)
    derivative = term1 - term2
    return derivative


@jit
def fit_Sigma(r, a, b, c, d, e):
    term1 = (2 * a * c**2 * jnp.log(b * r + c * r**2 + 1)**(c - 1)) / (b * r + c * r**2 + 1)
    term2 = (a * (c - 1) * c * (b + 2 * c * r)**2 * jnp.log(b * r + c * r**2 + 1)**(c - 2)) / (b * r + c * r**2 + 1)**2
    term3 = (a * c * (b + 2 * c * r)**2 * jnp.log(b * r + c * r**2 + 1)**(c - 1)) / (b * r + c * r**2 + 1)**2
    term4 = (d**2 * e * jnp.log(d * r + 1)**(e - 1)) / (d * r + 1)**2
    term5 = (d**2 * (e - 1) * e * jnp.log(d * r + 1)**(e - 2)) / (d * r + 1)**2
    term6 = (a * c * (b + 2 * c * r) * jnp.log(b * r + c * r**2 + 1)**(c - 1)) / (b * r + c * r**2 + 1)
    term7 = (d * e * jnp.log(d * r + 1)**(e - 1)) / (d * r + 1)
    
    numerator = r * (term1 + term2 - term3 + term4 - term5) + term6 - term7
    
    derivative = numerator / (2 * r)
    return derivative


@jit
def psi(rs, rho_s, Sigma_cr, psi_hat,zlens):
    rs_alpha=rs*apr/Da(zlens)
    return psi_hat * (rho_s * rs *rs_alpha**2) / Sigma_cr
@jit
def func_alpha(rs,rho_s, Sigma_cr, alpha_hat,zlens):
    rs_alpha=rs*apr/Da(zlens)
    return alpha_hat * (rho_s * rs *rs_alpha) / Sigma_cr
@jit
def func_Sigma(rs,rho_s, Sigma_hat):
    return Sigma_hat* rho_s* rs

def fited_a(x):
    a, b, c, d, g, h, i, j, k = 1.11323951e+00, -1.26183924e+00,  2.87611831e+00, -3.61577534e+00,2.77768241e+00, -4.15935892e-01,  1.35746091e-02, -3.86240111e-02,-2.63332737e-04
    return a + b*x**0.1 + c*x**0.2 + d*x**0.5 + g*x**0.9+h*x**2+i*x**12+j*x**20+k*x**87

def fited_b(x):
    a, b, c, d, g, h, i, j,k,l = 6.59669818e+00,  2.21289181e+00, -9.56284544e+00,  9.10385757e+00, -4.01289646e+00,  3.26789716e+00,  2.07641520e+00,  1.18377949e-01, -2.10519668e-04,  4.21396052e-07
    return a + b*x**0.01 + c*x**0.2 + d*x**0.6 + g*x**0.9+h*x**2+i*x**12+j*x**51+k*x**109+l*x**203

def fited_c(x):
    a, b, c, d, g, h, i, j = 1.79309631e+00,  5.10780441e-01, -1.43007683e+00,  1.39244251e+00, -7.85787233e-01,  1.60384916e-01,  3.14558126e-02,  7.14758319e-04
    return a + b*x**0.1 + c*x**0.2 + d*x**0.4 + g*x**0.9+h*x**2+i*x**12+j*x**67

def fited_p(x):
    a, b, c, d, g, h, i, j = 6.94266549, 4.96669252, -12.52779021, 7.54562512, 3.21238241, 2.86297753, -0.32646638, 0.30926922
    return a + b*x**0.2 + c*x**0.3 + d*x**0.8 + g*x**2+h*x**11+i*x**22+j*x**37

def fited_s(x):
    a, b, c, d, g, h, i, j = 1.82544721e+00, -5.99111491e-01, 8.44850940e-01, -4.34593883e-01, 8.17550753e-02, 1.67462383e-02, -1.75109111e-03, 3.82847738e-03
    return a + b*x**0.2 + c*x**0.3 + d*x**0.8 + g*x**2+h*x**11+i*x**22+j*x**37

@jax.jit
def get_para_scalar(trx):
    # Use jax.numpy for conditional logic
    a, b, c, p, s = jnp.where(trx == 0, 
                                jnp.array([0.758150, 3.585676, 1.941042, 2.573823, 1.993227]), 
                                jnp.stack([fited_a(trx), fited_b(trx), fited_c(trx), fited_p(trx), fited_s(trx)]))
    return a, b, c, p, s

get_para = vmap(get_para_scalar)

class SIDM_parametric(object):
    def __init__(self, tnfw_params=None, filename=None, bsz_arc=200, dsx_arc=0.2, xi1=None, xi2=None, zsource=1.0, halonum=None):
        self.zsource = zsource
        # Create a grid
        if xi1 is None or xi2 is None:
            self.bsz_arc = bsz_arc
            self.dsx_arc = dsx_arc
            self.nnn = int(np.ceil(self.bsz_arc / self.dsx_arc))
            self.xi2, self.xi1 = make_c_coor(self.bsz_arc, self.nnn)  # Create coordinate grid
        else:
            self.xi1 = xi1
            self.xi2 = xi2
            self.bsz_arc = xi1.max() - xi1.min()
            self.dsx_arc = xi1[1, 0] - xi1[0, 0]
            self.nnn = xi1.shape[0]

        # Read parameters from file if provided
        if filename is not None:
            self.tnfw_params = self.read_tnfw_params(filename, halonum)
        else:
            if tnfw_params is None:
                raise ValueError("Either filename or tnfw_params must be provided.")
            else:
                if halonum is not None:
                    self.tnfw_params = tnfw_params[:halonum]  # Slice the parameters if halonum is provided
                else:
                    self.tnfw_params = tnfw_params

        self.tau = []

        # Extract necessary parameters from the input parameters
        self.rs_list, self.rhos_list, self.redshift_list, self.halo_age_list = [], [], [], []
        self.pos_x_list, self.pos_y_list = [], []
        self.mass = []

        # Loop through the parameters to extract values
        for params in self.tnfw_params:
            self.mass.append(params['mass'])
            self.rs_list.append(params['tnfw_params']['rs'])
            self.rhos_list.append(params['tnfw_params']['rhos'])
            self.redshift_list.append(params['redshift'])

            if 'halo_age' in params:
                self.halo_age_list.append(params['halo_age'])
                self.state = 'Transfer'  # Set state if 'halo_age' is found
            elif 'tau' in params:
                self.tau.append(params['tau'])
                self.state = 'Known_tau'  # Set state if 'tau' is found
            else:
                raise ValueError("Either 'halo_age' or 'tau' must be provided.")  # Error if neither is found

            self.pos_x_list.append(params['position_x'])
            self.pos_y_list.append(params['position_y'])

        # Convert lists to numpy arrays for faster operations
        self.rs_array = jnp.array(self.rs_list)
        self.rhos_array = jnp.array(self.rhos_list)
        self.redshift_array = jnp.array(self.redshift_list)
        self.pos_x_array = jnp.array(self.pos_x_list)
        self.pos_y_array = jnp.array(self.pos_y_list)

        # Handle halo_age or tau arrays based on the state
        if self.state == 'Transfer':
            self.halo_age_array = jnp.array(self.halo_age_list)
        elif self.state == 'Known_tau':
            self.tau_array = jnp.array(self.tau)

        # Cache arrays to avoid redundant calculations
        self._tau_array_computed = False
        self._sigma_cr_array = None
        self._Sigma_s_array = None
        self._R_s_array = None
        self._R_c_array = None

        self._kappa_map = None
        self._potential_map = None
        self._alpha1_map = None
        self._alpha2_map = None

    # Read parameters from the file (pickle format)
    def read_tnfw_params(self, filename, num_items=None):
        with open(filename, 'rb') as file:
            tnfw_params = pickle.load(file)  # Load the parameters
            if num_items is not None:
                tnfw_params = tnfw_params[:num_items]  # Limit the number of items if provided
        return tnfw_params

    # Calculate tau if needed (based on state 'Transfer')
    def _compute_tau_if_needed(self):
        if self._tau_array_computed:
            return
        if self.state == 'Transfer':
            tau_list = []
            for i in range(len(self.rs_array)):
                rs = self.rs_array[i]
                rhos = self.rhos_array[i]
                tl = self.halo_age_array[i]
                tr = get_tau(rhos, rs, tl)  # Call external function to get tau
                tau_list.append(tr)
            self.tau_array = jnp.array(tau_list)
        self._tau_array_computed = True

    # Prepare common arrays for further calculations
    def _prepare_common_arrays(self, batch_size=30):
        if self._Sigma_s_array is not None:
            return  # Arrays already prepared

        self._compute_tau_if_needed()  # Ensure tau is calculated

        num_lenses = len(self.rs_array)
        num_batches = (num_lenses + batch_size - 1) // batch_size  # Calculate number of batches
        sigma_cr_list = []
        a_values_list = []
        b_values_list = []
        c_values_list = []
        d_values_list = []
        e_values_list = []

        sigma_crit_vectorized = vmap(lambda z_lens: SigmaCrit(z_lens, self.zsource))  # Vectorized SigmaCrit calculation

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_lenses)
            redshift_batch = self.redshift_array[start_idx:end_idx]
            tau_batch = self.tau_array[start_idx:end_idx]
            sigma_cr_batch = sigma_crit_vectorized(redshift_batch)  # Calculate sigma_crit for the batch
            sigma_cr_list.append(sigma_cr_batch)

            # Get parameters a, b, c, d, e
            a_values_batch, b_values_batch, c_values_batch, d_values_batch, e_values_batch = get_para(tau_batch)
            a_values_list.append(a_values_batch)
            b_values_list.append(b_values_batch)
            c_values_list.append(c_values_batch)
            d_values_list.append(d_values_batch)
            e_values_list.append(e_values_batch)

        # Combine all the computed values into arrays
        self._sigma_cr_array = jnp.concatenate(sigma_cr_list)
        self.a_values, self.b_values, self.c_values, self.d_values, self.e_values = jnp.concatenate(a_values_list), jnp.concatenate(b_values_list), jnp.concatenate(c_values_list), jnp.concatenate(d_values_list), jnp.concatenate(e_values_list)

    # Compute kappa map
    def compute_kappa_map(self, batch_size=30):
        if self._kappa_map is not None:
            return self._kappa_map  # Return the cached kappa map

        self._prepare_common_arrays(batch_size=batch_size)  # Prepare common arrays if not already done

        num_lenses = len(self.rs_array)
        num_batches = (num_lenses + batch_size - 1) // batch_size
        total_kappa = jnp.zeros_like(self.xi1)  # Initialize kappa map

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_lenses)

            pos_x_batch = self.pos_x_array[start_idx:end_idx]
            pos_y_batch = self.pos_y_array[start_idx:end_idx]
            redshift_batch = self.redshift_array[start_idx:end_idx]
            a_batch = self.a_values[start_idx:end_idx]
            b_batch = self.b_values[start_idx:end_idx]
            c_batch = self.c_values[start_idx:end_idx]
            d_batch = self.d_values[start_idx:end_idx]
            e_batch = self.e_values[start_idx:end_idx]
            r_s0_batch = self.rs_array[start_idx:end_idx]
            rho_s0_batch = self.rhos_array[start_idx:end_idx]
            sigma_cr_batch = self._sigma_cr_array[start_idx:end_idx]

            def compute_surface_density_single(pos_x, pos_y, rs, rhos, zlens, a, b, c, d, e):
                delta_x = (self.xi1 - pos_x)/apr * Da(zlens)  # Convert x to Mpc
                delta_y = (self.xi2 - pos_y)/apr * Da(zlens)  # Convert y to Mpc
                R = jnp.sqrt(delta_x**2 + delta_y**2) / rs  # Compute normalized radius
                Sigmahat = fit_Sigma(R, a, b, c, d, e)  # Fit surface density model
                result = func_Sigma(rs, rhos, Sigmahat)  # Compute surface density
                return result

            compute_surface_density_single_vmapped = vmap(
                compute_surface_density_single,
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                out_axes=0
            )

            # Compute surface density for each batch and calculate kappa
            Sigma_batch = compute_surface_density_single_vmapped(
                pos_x_batch, pos_y_batch, r_s0_batch, rho_s0_batch, redshift_batch, a_batch, b_batch, c_batch, d_batch, e_batch
            )
            kappa_batch = Sigma_batch / sigma_cr_batch[:, None, None]

            kappa_batch_sum = jnp.sum(kappa_batch, axis=0)
            total_kappa += kappa_batch_sum

        self._kappa_map = total_kappa  # Store the final result
        return self._kappa_map

    def compute_potential_map(self, batch_size=30):
        # If potential map is already computed, return it directly
        if self._potential_map is not None:
            return self._potential_map

        # Prepare common arrays if sigma_cr_array is not yet computed
        if self._sigma_cr_array is None:
            self._prepare_common_arrays(batch_size=batch_size)

        num_lenses = len(self.rs_array)
        num_batches = (num_lenses + batch_size - 1) // batch_size
        total_potential = jnp.zeros_like(self.xi1)  # Initialize total potential map to zero

        # Loop over each batch to compute potential for all lenses
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_lenses)

            # Extract relevant data for the current batch
            r_s0_batch = self.rs_array[start_idx:end_idx]
            rho_s0_batch = self.rhos_array[start_idx:end_idx]
            pos_x_batch = self.pos_x_array[start_idx:end_idx]
            pos_y_batch = self.pos_y_array[start_idx:end_idx]
            redshift_batch = self.redshift_array[start_idx:end_idx]
            sigma_cr_batch = self._sigma_cr_array[start_idx:end_idx]
            a_batch = self.a_values[start_idx:end_idx]
            b_batch = self.b_values[start_idx:end_idx]
            c_batch = self.c_values[start_idx:end_idx]
            d_batch = self.d_values[start_idx:end_idx]
            e_batch = self.e_values[start_idx:end_idx]

            # Function to compute potential for a single lens
            def compute_potential_single(pos_x, pos_y, rs, rhos, sigma_cr, zlens, a, b, c, d, e):
                delta_x = (self.xi1 - pos_x) / apr * Da(zlens)  # Mpc
                delta_y = (self.xi2 - pos_y) / apr * Da(zlens)  # Mpc
                R = jnp.sqrt(delta_x**2 + delta_y**2) / rs  # Normalized radial distance
                phi_value = fit_potential(R, a, b, c, d, e)  # Compute the potential
                psi_value = psi(rs, rhos, sigma_cr, phi_value, zlens)  # Compute the deflection potential
                return psi_value

            # Vectorizing the potential computation for the batch
            compute_potential_single_vmapped = vmap(
                compute_potential_single,
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                out_axes=0
            )

            # Compute potential for the batch
            batch_potentials = compute_potential_single_vmapped(
                pos_x_batch, pos_y_batch, r_s0_batch, rho_s0_batch, sigma_cr_batch,
                redshift_batch, a_batch, b_batch, c_batch, d_batch, e_batch
            )

            # Add the batch potential sum to the total potential map
            total_potential += jnp.sum(batch_potentials, axis=0)

        # Store and return the computed potential map
        self._potential_map = total_potential
        return self._potential_map

    def compute_alpha_map(self, batch_size=30):
        # Prepare common arrays if sigma_cr_array is not yet computed
        if self._sigma_cr_array is None:
            self._prepare_common_arrays(batch_size=batch_size)

        # Return precomputed alpha maps if available
        if self._alpha1_map is not None and self._alpha2_map is not None:
            return self._alpha1_map, self._alpha2_map

        num_lenses = len(self.rs_array)
        num_batches = (num_lenses + batch_size - 1) // batch_size
        total_alpha1 = jnp.zeros_like(self.xi1)  # Initialize total alpha1 map to zero
        total_alpha2 = jnp.zeros_like(self.xi1)  # Initialize total alpha2 map to zero

        # Loop over each batch to compute alpha for all lenses
        with tqdm(total=num_batches, desc="Computing alpha") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_lenses)

                # Extract relevant data for the current batch
                r_s0_batch = self.rs_array[start_idx:end_idx]
                rho_s0_batch = self.rhos_array[start_idx:end_idx]
                pos_x_batch = self.pos_x_array[start_idx:end_idx]
                pos_y_batch = self.pos_y_array[start_idx:end_idx]
                redshift_batch = self.redshift_array[start_idx:end_idx]
                sigma_cr_batch = self._sigma_cr_array[start_idx:end_idx]
                a_batch = self.a_values[start_idx:end_idx]
                b_batch = self.b_values[start_idx:end_idx]
                c_batch = self.c_values[start_idx:end_idx]
                d_batch = self.d_values[start_idx:end_idx]
                e_batch = self.e_values[start_idx:end_idx]

                # Compute the angular position deltas
                arc_delta_x_batch = self.xi1[None, :, :] - pos_x_batch[:, None, None]
                arc_delta_y_batch = self.xi2[None, :, :] - pos_y_batch[:, None, None]
                arc_R_batch = jnp.sqrt(arc_delta_x_batch**2 + arc_delta_y_batch**2)  # Angular radius

                vmap_Da = vmap(Da, in_axes=(0))
                # Convert to physical distances
                delta_x_batch = arc_delta_x_batch / apr * vmap_Da(redshift_batch)[:, None, None]
                delta_y_batch = arc_delta_y_batch / apr * vmap_Da(redshift_batch)[:, None, None]
                R_batch = jnp.sqrt(delta_x_batch**2 + delta_y_batch**2)  # Physical distance

                # Function to compute alpha for a single lens
                def compute_alpha_single(R, rho_s0, r_s0, sigma_cr, a_val, b_val, c_val, d_val, e_val, zlens):
                    alpha_hat = fit_alpha(R / r_s0, a_val, b_val, c_val, d_val, e_val)  # Compute alpha hat
                    alpha = func_alpha(r_s0, rho_s0, sigma_cr, alpha_hat, zlens)  # Compute deflection angle
                    return alpha

                # Vectorizing the alpha computation for the batch
                compute_alpha_single_vmapped = vmap(
                    compute_alpha_single,
                    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                    out_axes=0
                )

                # Compute alpha for the batch
                batch_alpha = compute_alpha_single_vmapped(
                    R_batch, rho_s0_batch, r_s0_batch, sigma_cr_batch,
                    a_batch,b_batch,c_batch,d_batch,e_batch, redshift_batch
                )
                batch_alpha2 = batch_alpha * (arc_delta_x_batch/arc_R_batch)
                batch_alpha1 = batch_alpha * (arc_delta_y_batch/arc_R_batch)

                # Add the batch alpha to the total alpha maps
                total_alpha1 += jnp.sum(batch_alpha1, axis=0)
                total_alpha2 += jnp.sum(batch_alpha2, axis=0)
                pbar.update(1)
        

        # Store and return the computed alpha maps
        self._alpha1_map = total_alpha1
        self._alpha2_map = total_alpha2
        return self._alpha1_map, self._alpha2_map
    


class SIDM_parametric_split(object):
    def __init__(self, tnfw_params=None, filename=None, bsz_arc=200, dsx_arc=0.2, xi1=None, xi2=None, zsource=1.0, halonum=None, sort_redshift=True, descending=False):
        self.zsource = zsource
        # Create a grid
        if xi1 is None or xi2 is None:
            self.bsz_arc = bsz_arc
            self.dsx_arc = dsx_arc
            self.nnn = int(np.ceil(self.bsz_arc / self.dsx_arc))
            self.xi2, self.xi1 = make_c_coor(self.bsz_arc, self.nnn)  # Create coordinate grid
        else:
            self.xi1 = xi1
            self.xi2 = xi2
            self.bsz_arc = xi1.max() - xi1.min()
            self.dsx_arc = xi1[1, 0] - xi1[0, 0]
            self.nnn = xi1.shape[0]

        # Read parameters from file if provided
        if filename is not None:
            self.tnfw_params = self.read_tnfw_params(filename, halonum)
        else:
            if tnfw_params is None:
                raise ValueError("Either filename or tnfw_params must be provided.")
            else:
                if halonum is not None:
                    self.tnfw_params = tnfw_params[:halonum]  # Slice the parameters if halonum is provided
                else:
                    self.tnfw_params = tnfw_params

        self.tau = []

        # Extract necessary parameters from the input parameters
        self.rs_list, self.rhos_list, self.redshift_list, self.halo_age_list = [], [], [], []
        self.pos_x_list, self.pos_y_list = [], []
        self.mass = []

        # Loop through the parameters to extract values
        for params in self.tnfw_params:
            self.mass.append(params['mass'])
            self.rs_list.append(params['tnfw_params']['rs'])
            self.rhos_list.append(params['tnfw_params']['rhos'])
            self.redshift_list.append(params['redshift'])

            if 'halo_age' in params:
                self.halo_age_list.append(params['halo_age'])
                self.state = 'Transfer'  # Set state if 'halo_age' is found
            elif 'tau' in params:
                self.tau.append(params['tau'])
                self.state = 'Known_tau'  # Set state if 'tau' is found
            else:
                raise ValueError("Either 'halo_age' or 'tau' must be provided.")  # Error if neither is found

            self.pos_x_list.append(params['position_x'])
            self.pos_y_list.append(params['position_y'])

        # Convert lists to numpy arrays for faster operations
        self.rs_array = jnp.array(self.rs_list)
        self.rhos_array = jnp.array(self.rhos_list)
        self.redshift_array = jnp.array(self.redshift_list)
        self.pos_x_array = jnp.array(self.pos_x_list)
        self.pos_y_array = jnp.array(self.pos_y_list)

        # Handle halo_age or tau arrays based on the state
        if self.state == 'Transfer':
            self.halo_age_array = jnp.array(self.halo_age_list)
        elif self.state == 'Known_tau':
            self.tau_array = jnp.array(self.tau)

        # Cache arrays to avoid redundant calculations
        self._tau_array_computed = False
        self._sigma_cr_array = None
        self._Sigma_s_array = None
        self._R_s_array = None
        self._R_c_array = None

        self._kappa_maps = None
        self._potential_maps = None
        self._alpha1_maps = None
        self._alpha2_maps = None

        # Optionally sort at initialization
        if sort_redshift:
            self.sort_by_redshift(descending=descending)

    def sort_by_redshift(self, descending=False):
        """Sort internal data by redshift and reset cached maps."""
        idx = jnp.argsort(self.redshift_array)
        if descending:
            idx = idx[::-1]
        order = idx.tolist()
        # Reorder list-based attributes
        self.tnfw_params = [self.tnfw_params[i] for i in order]
        self.rs_list = [self.rs_list[i] for i in order]
        self.rhos_list = [self.rhos_list[i] for i in order]
        self.redshift_list = [self.redshift_list[i] for i in order]
        self.pos_x_list = [self.pos_x_list[i] for i in order]
        self.pos_y_list = [self.pos_y_list[i] for i in order]
        if self.state=='Transfer':
            self.halo_age_list = [self.halo_age_list[i] for i in order]
        else:
            self.tau = [self.tau[i] for i in order]
        # Reorder arrays
        self.rs_array = self.rs_array[idx]
        self.rhos_array = self.rhos_array[idx]
        self.redshift_array = self.redshift_array[idx]
        self.pos_x_array = self.pos_x_array[idx]
        self.pos_y_array = self.pos_y_array[idx]
        if self.state=='Transfer':
            self.halo_age_array = self.halo_age_array[idx]
        else:
            self.tau_array = self.tau_array[idx]


    # Read parameters from the file (pickle format)
    def read_tnfw_params(self, filename, num_items=None):
        with open(filename, 'rb') as file:
            tnfw_params = pickle.load(file)  # Load the parameters
            if num_items is not None:
                tnfw_params = tnfw_params[:num_items]  # Limit the number of items if provided
        return tnfw_params

    # Calculate tau if needed (based on state 'Transfer')
    def _compute_tau_if_needed(self):
        if self._tau_array_computed:
            return
        if self.state == 'Transfer':
            tau_list = []
            for i in range(len(self.rs_array)):
                rs = self.rs_array[i]
                rhos = self.rhos_array[i]
                tl = self.halo_age_array[i]
                tr = get_tau(rhos, rs, tl)  # Call external function to get tau
                tau_list.append(tr)
            self.tau_array = jnp.array(tau_list)
        self._tau_array_computed = True

    # Prepare common arrays for further calculations
    def _prepare_common_arrays(self, batch_size=30):
        if self._Sigma_s_array is not None:
            return  # Arrays already prepared

        self._compute_tau_if_needed()  # Ensure tau is calculated

        num_lenses = len(self.rs_array)
        num_batches = (num_lenses + batch_size - 1) // batch_size  # Calculate number of batches
        sigma_cr_list = []
        a_values_list = []
        b_values_list = []
        c_values_list = []
        d_values_list = []
        e_values_list = []

        sigma_crit_vectorized = vmap(lambda z_lens: SigmaCrit(z_lens, self.zsource))  # Vectorized SigmaCrit calculation

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_lenses)
            redshift_batch = self.redshift_array[start_idx:end_idx]
            tau_batch = self.tau_array[start_idx:end_idx]
            sigma_cr_batch = sigma_crit_vectorized(redshift_batch)  # Calculate sigma_crit for the batch
            sigma_cr_list.append(sigma_cr_batch)

            # Get parameters a, b, c, d, e
            a_values_batch, b_values_batch, c_values_batch, d_values_batch, e_values_batch = get_para(tau_batch)
            a_values_list.append(a_values_batch)
            b_values_list.append(b_values_batch)
            c_values_list.append(c_values_batch)
            d_values_list.append(d_values_batch)
            e_values_list.append(e_values_batch)

        # Combine all the computed values into arrays
        self._sigma_cr_array = jnp.concatenate(sigma_cr_list)
        self.a_values, self.b_values, self.c_values, self.d_values, self.e_values = jnp.concatenate(a_values_list), jnp.concatenate(b_values_list), jnp.concatenate(c_values_list), jnp.concatenate(d_values_list), jnp.concatenate(e_values_list)

    def compute_kappa_map(self, batch_size=30):

        if self._kappa_maps is not None:
            return self._kappa_maps
        self._prepare_common_arrays(batch_size=batch_size)
        kappa_list = []
        num_lenses = len(self.rs_array)
        num_batches = (num_lenses + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_lenses)
            # Batch parameter extraction
            pos_x = self.pos_x_array[start:end]
            pos_y = self.pos_y_array[start:end]
            sigma_cr = self._sigma_cr_array[start:end]
            r_s0 = self.rs_array[start:end]
            rho_s0 = self.rhos_array[start:end]
            a_batche, b_batch, c_batch, d_batch, e_batch = (
                self.a_values[start:end], self.b_values[start:end],
                self.c_values[start:end], self.d_values[start:end], self.e_values[start:end]
            )
            # Define and vmap the single-halo Sigma calculation
            def _sigma_single(px, py, rs, rhos, zlens, a, b, c, d, e):
                dx = (self.xi1 - px)/apr * Da(zlens)
                dy = (self.xi2 - py)/apr * Da(zlens)
                R = jnp.sqrt(dx**2 + dy**2)/rs
                Sigmahat = fit_Sigma(R, a, b, c, d, e)
                return func_Sigma(rs, rhos, Sigmahat)
            sigma_vm = vmap(_sigma_single, in_axes=(0,0,0,0,0,0,0,0,0,0), out_axes=0)
            # Compute and store each halo's kappa
            Sigma_batch = sigma_vm(pos_x, pos_y, r_s0, rho_s0, self.redshift_array[start:end],
                                    a_batche, b_batch, c_batch, d_batch, e_batch)
            kappa_batch = Sigma_batch / sigma_cr[:, None, None]
            kappa_list.append(kappa_batch)
        self._kappa_maps = jnp.concatenate(kappa_list, axis=0)
        return self._kappa_maps

    def compute_potential_map(self, batch_size=30):

        if self._potential_maps is not None:
            return self._potential_maps
        if self._sigma_cr_array is None:
            self._prepare_common_arrays(batch_size=batch_size)
        pot_list = []
        num_lenses = len(self.rs_array)
        num_batches = (num_lenses + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_lenses)
            args = {
                'pos_x': self.pos_x_array[start:end], 'pos_y': self.pos_y_array[start:end],
                'rs': self.rs_array[start:end], 'rhos': self.rhos_array[start:end],
                'sigma_cr': self._sigma_cr_array[start:end], 'zlens': self.redshift_array[start:end],
                'a': self.a_values[start:end], 'b': self.b_values[start:end],
                'c': self.c_values[start:end], 'd': self.d_values[start:end], 'e': self.e_values[start:end]
            }
            def _psi_single(px, py, rs, rhos, sigma_cr, zlens, a, b, c, d, e):
                dx = (self.xi1 - px)/apr * Da(zlens)
                dy = (self.xi2 - py)/apr * Da(zlens)
                R = jnp.sqrt(dx**2 + dy**2)/rs
                phi_hat = fit_potential(R, a, b, c, d, e)
                return psi(rs, rhos, sigma_cr, phi_hat, zlens)
            psi_vm = vmap(_psi_single, in_axes=(0,0,0,0,0,0,0,0,0,0,0), out_axes=0)
            batch_pot = psi_vm(**args)
            pot_list.append(batch_pot)
        self._potential_maps = jnp.concatenate(pot_list, axis=0)
        return self._potential_maps

    def compute_alpha_map(self, batch_size=30):
        """Compute deflection angles per halo; returns two arrays of shape (N_halo, nx, ny)."""
        if self._alpha1_maps is not None and self._alpha2_maps is not None:
            return self._alpha1_maps, self._alpha2_maps
        if self._sigma_cr_array is None:
            self._prepare_common_arrays(batch_size=batch_size)
        a1_list, a2_list = [], []
        num_lenses = len(self.rs_array)
        num_batches = (num_lenses + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_lenses)
            arc_dx = self.xi1[None] - self.pos_x_array[start:end, None, None]
            arc_dy = self.xi2[None] - self.pos_y_array[start:end, None, None]
            R_ang = jnp.sqrt(arc_dx**2 + arc_dy**2)
            phys_dx = arc_dx/apr * vmap(Da)(self.redshift_array[start:end])[:,None,None]
            phys_dy = arc_dy/apr * vmap(Da)(self.redshift_array[start:end])[:,None,None]
            R_phys = jnp.sqrt(phys_dx**2 + phys_dy**2)
            def _alpha_single(R, rhos, rs, sigma_cr, a, b, c, d, e, zlens):
                a_hat = fit_alpha(R/rs, a, b, c, d, e)
                return func_alpha(rs, rhos, sigma_cr, a_hat, zlens)
            alpha_vm = vmap(_alpha_single,
                            in_axes=(0,0,0,0,0,0,0,0,0,0), out_axes=0)
            alpha_batch = alpha_vm(R_phys,
                                self.rhos_array[start:end], self.rs_array[start:end],
                                self._sigma_cr_array[start:end],
                                self.a_values[start:end], self.b_values[start:end],
                                self.c_values[start:end], self.d_values[start:end],
                                self.e_values[start:end], self.redshift_array[start:end])
            a1_list.append(alpha_batch * (arc_dy/R_ang))
            a2_list.append(alpha_batch * (arc_dx/R_ang))
        self._alpha1_maps = jnp.concatenate(a1_list, axis=0)
        self._alpha2_maps = jnp.concatenate(a2_list, axis=0)
        return self._alpha1_maps, self._alpha2_maps


#-------------------------------------------------------
# function for the multiple lens plane ray-tracing
#

# --- JAX-accelerated version ---
@jit
def inverse_cic_JAX(source_map, posy1, posy2, ysc1, ysc2, dsi, nsx, nsy):
    sm = source_map.ravel()
    p1 = posy1.ravel()
    p2 = posy2.ravel()

    xb1 = (p1 - ysc1) / dsi + nsx/2.0 - 0.5
    xb2 = (p2 - ysc2) / dsi + nsy/2.0 - 0.5

    i1 = jnp.floor(xb1).astype(jnp.int32)
    j1 = jnp.floor(xb2).astype(jnp.int32)

    valid = (i1 >= 0) & (i1 < nsx-1) & (j1 >= 0) & (j1 < nsy-1)

    wx = 1.0 - (xb1 - i1)
    wy = 1.0 - (xb2 - j1)
    ww1 = wx * wy
    ww2 = wx * (1.0 - wy)
    ww3 = (1.0 - wx) * wy
    ww4 = (1.0 - wx) * (1.0 - wy)

    base = i1 * nsy + j1
    # Values at the four neighboring pixels
    v1 = sm[base]
    v2 = sm[base + 1]
    v3 = sm[base + nsy]
    v4 = sm[base + nsy + 1]

    # Element-wise interpolation with masking
    interp = ww1 * v1 + ww2 * v2 + ww3 * v3 + ww4 * v4
    lm_flat = jnp.where(valid, interp, 0.0)

    return lm_flat.reshape(posy1.shape)


@jit
def call_inverse_cic_JAX(img_in, yc1, yc2, yi1, yi2, dsi):
    ny1, ny2 = img_in.shape
    # Note: nsx = ny1, nsy = ny2
    return inverse_cic_JAX(img_in, yi1, yi2, yc1, yc2, dsi, ny1, ny2)

# ==============================================
# ai_to_ah — convert angular deflection to physical deflection
# ==============================================
@jit
def ai_to_ah_JAX(ai, zl, zs):
    return Da(zs) / Da2(zl, zs) * ai


# ==============================================
# Main function for multi-plane ray tracing
# ==============================================
def multiLensPlaneLensingSimsWithPICS_JAX(
    xf1, xf2, 
    alpha1_array_ref, alpha2_array_ref, 
    zl_lens_array, 
    zs_srcs, zs_srcs_ref=None
):
    """
    Inputs:
    xf1, xf2             -- (nx, ny) grid coordinates (JAX array or convertible)
    alpha1_array_ref     -- list or array, alpha1 map for each lens plane
    alpha2_array_ref     -- list or array, alpha2 map for each lens plane
    zl_lens_array        -- 1D array of lens redshifts
    zs_srcs_ref, zs_srcs -- reference and actual source redshifts
    Output:
    af1, af2             -- (nx, ny) total deflection maps
    """

    # Convert to JAX arrays
    xf1 = jnp.asarray(xf1)
    xf2 = jnp.asarray(xf2)
    alpha1_array_ref = jnp.asarray(alpha1_array_ref)
    alpha2_array_ref = jnp.asarray(alpha2_array_ref)
    zl_lens_array = jnp.asarray(zl_lens_array)

    # Keep only lens planes below the source redshift
    mask = zl_lens_array < zs_srcs
    zl_eff = zl_lens_array[mask]
    alpha1_ref = alpha1_array_ref[mask]
    alpha2_ref = alpha2_array_ref[mask]
    nzpl = zl_eff.shape[0]

    dsi = xf1[1,1] - xf1[0,0]  # Pixel angular size
    nx, ny = xf1.shape

    # Rescale each alpha map for redshift
    if zs_srcs_ref is not None:
        alpha1_array = []
        alpha2_array = []
        for i in range(nzpl):
            zl = zl_eff[i]
            factor = Da(zs_srcs_ref) / Da2(zl, zs_srcs_ref) * Da2(zl, zs_srcs) / Da(zs_srcs)
            alpha1_array.append(alpha1_ref[i] * factor)
            alpha2_array.append(alpha2_ref[i] * factor)
    else:
        alpha1_array = alpha1_ref
        alpha2_array = alpha2_ref

    # Store xj1, xj2 for each layer
    xj1_list = []
    xj2_list = []
    zj_list  = []

    for i in tqdm(range(nzpl), desc="Computing multi-plane lensing"):
        if i == 0:
            # First layer: use observed-plane coordinates
            xj1, xj2 = xf1, xf2
            zj = zl_eff[0]
        elif i == 1:
            # Second layer: depends on first layer
            z1, z2 = zl_eff[0], zl_eff[1]
            x01 = jnp.zeros_like(xf1)
            x02 = jnp.zeros_like(xf2)
            x11, x12 = xj1_list[0], xj2_list[0]
            ah11 = ai_to_ah_JAX(alpha1_array[0], z1, zs_srcs)
            ah12 = ai_to_ah_JAX(alpha2_array[0], z1, zs_srcs)
            bij = Da(z1) * Da2(0.0, z2) / (Da(z2) * Da2(0.0, z1))
            xj1 = x11 * bij - (bij - 1.0) * x01 - ah11 * Da2(z1, z2) / Da(z2)
            xj2 = x12 * bij - (bij - 1.0) * x02 - ah12 * Da2(z1, z2) / Da(z2)
            zj = z2
        else:
            # Third layer and beyond: depends on previous two layers
            zi   = zl_eff[i]
            zim1 = zl_eff[i-1]
            zim2 = zl_eff[i-2]
            xjm11 = xj1_list[i-1]; xjm12 = xj2_list[i-1]
            xjm21 = xj1_list[i-2]; xjm22 = xj2_list[i-2]
            ahm11 = ai_to_ah_JAX(call_inverse_cic_JAX(alpha1_array[i-1],0.,0.,xjm11,xjm12,dsi), zim1, zs_srcs)
            ahm12 = ai_to_ah_JAX(call_inverse_cic_JAX(alpha2_array[i-1],0.,0.,xjm11,xjm12,dsi), zim1, zs_srcs)
            bij = Da(zim1) * Da2(zim2, zi) / (Da(zi) * Da2(zim2, zim1))
            xj1 = xjm11 * bij - (bij - 1.0) * xjm21 - ahm11 * Da2(zim1, zi) / Da(zi)
            xj2 = xjm12 * bij - (bij - 1.0) * xjm22 - ahm12 * Da2(zim1, zi) / Da(zi)
            zj = zi

        xj1_list.append(xj1)
        xj2_list.append(xj2)
        zj_list.append(zj)

    # Final accumulation of all layers' deflections
    af1 = jnp.zeros_like(xf1)
    af2 = jnp.zeros_like(xf2)
    for i in tqdm(range(nzpl), desc="ray tracing"):
        a1 = call_inverse_cic_JAX(alpha1_array[i], 0.0, 0.0, xj1_list[i], xj2_list[i], dsi)
        a2 = call_inverse_cic_JAX(alpha2_array[i], 0.0, 0.0, xj1_list[i], xj2_list[i], dsi)
        af1 = af1 + a1
        af2 = af2 + a2

    return af1, af2


class SIDM_parametric_Multiplane(object):
    def __init__(self, tnfw_params=None, filename=None,
                 bsz_arc=200, dsx_arc=0.2, nnn=2000, xi1=None, xi2=None,
                 zsource=1.0, halonum=None):
        self.zsource = zsource

        # --- Grid setup ---
        if xi1 is None or xi2 is None:
            self.bsz_arc = bsz_arc
            self.dsx_arc = dsx_arc
            self.nnn = nnn
            self.xi2, self.xi1 = make_c_coor(self.bsz_arc, self.nnn)
        else:
            self.xi1     = xi1
            self.xi2     = xi2
            self.bsz_arc = xi1.max() - xi1.min()
            self.dsx_arc = xi1[1,0] - xi1[0,0]
            self.nnn     = xi1.shape[0]

        # --- Read TNFW parameters ---
        if filename is not None:
            self.tnfw_params = self.read_tnfw_params(filename, halonum)
        else:
            if tnfw_params is None:
                raise ValueError("Either filename or tnfw_params must be provided.")
            self.tnfw_params = tnfw_params[:halonum] if halonum else tnfw_params

        # --- Extract attribute lists ---
        self.rs_list, self.rhos_list = [], []
        self.redshift_list           = []
        self.pos_x_list, self.pos_y_list = [], []
        self.mass                    = []
        self.halo_age_list, self.tau = [], []
        for params in self.tnfw_params:
            self.mass.append(params['mass'])
            self.rs_list.append(params['tnfw_params']['rs'])
            self.rhos_list.append(params['tnfw_params']['rhos'])
            self.redshift_list.append(params['redshift'])
            self.pos_x_list.append(params['position_x'])
            self.pos_y_list.append(params['position_y'])
        
            if 'halo_age' in params:
                self.halo_age_list.append(params['halo_age'])
                self.state = 'Transfer'
            elif 'tau' in params:
                self.tau.append(params['tau'])
                self.state = 'Known_tau'
            else:
                raise ValueError("Either 'halo_age' or 'tau' must be provided.")

        # --- Convert to JAX arrays ---
        self.rs_array       = jnp.array(self.rs_list)
        self.rhos_array     = jnp.array(self.rhos_list)
        self.redshift_array = jnp.array(self.redshift_list)
        self.pos_x_array    = jnp.array(self.pos_x_list)
        self.pos_y_array    = jnp.array(self.pos_y_list)
        if self.state == 'Transfer':
            self.halo_age_array = jnp.array(self.halo_age_list)
        else:
            self.tau_array      = jnp.array(self.tau)

        # --- Cache flags ---
        self._tau_array_computed = False
        self._sigma_cr_array     = None
        self.a_values = self.b_values = self.c_values = None
        self.d_values = self.e_values = None

        # --- Force redshift sorting ---
        self.sort_by_redshift()

        # --- Unique redshifts and inverse index for grouped sums ---
        self.grouped_redshift_array, self._inv_z_idx = jnp.unique(
            self.redshift_array, return_inverse=True
        )
        print(f"number of unique redshifts: {self.grouped_redshift_array.shape[0]}")

    def sort_by_redshift(self):
        idx = jnp.argsort(self.redshift_array)
        order = list(idx)
        # Reorder lists
        self.tnfw_params     = [self.tnfw_params[i] for i in order]
        self.rs_list         = [self.rs_list[i] for i in order]
        self.rhos_list       = [self.rhos_list[i] for i in order]
        self.redshift_list   = [self.redshift_list[i] for i in order]
        self.pos_x_list      = [self.pos_x_list[i] for i in order]
        self.pos_y_list      = [self.pos_y_list[i] for i in order]
        if self.state == 'Transfer':
            self.halo_age_list = [self.halo_age_list[i] for i in order]
        else:
            self.tau           = [self.tau[i] for i in order]
        # Reorder arrays
        self.rs_array       = self.rs_array[idx]
        self.rhos_array     = self.rhos_array[idx]
        self.redshift_array = self.redshift_array[idx]
        self.pos_x_array    = self.pos_x_array[idx]
        self.pos_y_array    = self.pos_y_array[idx]
        if self.state == 'Transfer':
            self.halo_age_array = self.halo_age_array[idx]
        else:
            self.tau_array      = self.tau_array[idx]

    def read_tnfw_params(self, filename, num_items=None):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data[:num_items] if num_items else data

    def _compute_tau_if_needed(self):
        if self._tau_array_computed:
            return
        if self.state == 'Transfer':
            taus = [
                get_tau(self.rhos_array[i],
                        self.rs_array[i],
                        self.halo_age_array[i])
                for i in range(len(self.rs_array))
            ]
            self.tau_array = jnp.array(taus)
        self._tau_array_computed = True

    def _prepare_common_arrays(self, batch_size=30):
        if self._sigma_cr_array is not None:
            return
        self._compute_tau_if_needed()

        n  = len(self.rs_array)
        nb = (n + batch_size - 1) // batch_size

        sigma_cr_list = []
        a_list = []; b_list = []
        c_list = []; d_list = []; e_list = []
        sigma_crit_vec = vmap(lambda z: SigmaCrit(z, self.zsource))

        for bi in range(nb):
            s, e = bi*batch_size, min((bi+1)*batch_size, n)
            zbat   = self.redshift_array[s:e]
            taubat = self.tau_array[s:e]
            sigma_cr_list.append(sigma_crit_vec(zbat))
            a_b, b_b, c_b, d_b, e_b = get_para(taubat)
            a_list.append(a_b); b_list.append(b_b)
            c_list.append(c_b); d_list.append(d_b)
            e_list.append(e_b)

        self._sigma_cr_array = jnp.concatenate(sigma_cr_list)
        self.a_values = jnp.concatenate(a_list)
        self.b_values = jnp.concatenate(b_list)
        self.c_values = jnp.concatenate(c_list)
        self.d_values = jnp.concatenate(d_list)
        self.e_values = jnp.concatenate(e_list)
    
    def _compute_grouped_alpha_maps(self, batch_size=30):
        """
        Compute combined raw α1 and α2 maps for each redshift bin.
        Batch size controls memory; segment_sum handles grouped accumulation.
        """
        # 1) Ensure SigmaCrit and fit parameters are ready
        self._prepare_common_arrays(batch_size)
        G     = self.grouped_redshift_array.shape[0]
        nx, ny = self.xi1.shape

        # 2) Initialize accumulators
        grouped_alpha1 = jnp.zeros((G, nx, ny), dtype=jnp.float32)
        grouped_alpha2 = jnp.zeros((G, nx, ny), dtype=jnp.float32)


        # 3) Define vmap for single-halo angular deflection
        def _alpha_single(Rp, rhos, rs, sigma_cr, a, b, c, d, e, zl):
            a_hat = fit_alpha(Rp/rs, a, b, c, d, e)
            return func_alpha(rs, rhos, sigma_cr, a_hat, zl)
        alpha_vm = vmap(_alpha_single,
                        in_axes=(0,0,0,0,0,0,0,0,0,0),
                        out_axes=0)

        # 4) Iterate over halos in batches
        nlens = len(self.rs_array)
        for start in tqdm(range(0, nlens, batch_size), desc="Computing α maps"):
            end = min(start + batch_size, nlens)
            idx = slice(start, end)

            # 4.1) Grid offsets and angular distance for this batch
            arc_dx = self.xi1[None] - self.pos_x_array[idx, None, None]
            arc_dy = self.xi2[None] - self.pos_y_array[idx, None, None]
            R_ang  = jnp.sqrt(arc_dx**2 + arc_dy**2)

            # 4.2) Physical radius Rp
            Da_vec  = vmap(Da)(self.redshift_array[idx])[:,None,None]
            phys_dx = arc_dx / apr * Da_vec
            phys_dy = arc_dy / apr * Da_vec
            R_phys  = jnp.sqrt(phys_dx**2 + phys_dy**2)

            # 4.3) Batch core α evaluation
            batch_alpha = alpha_vm(
                R_phys,
                self.rhos_array[idx],
                self.rs_array[idx],
                self._sigma_cr_array[idx],
                self.a_values[idx],
                self.b_values[idx],
                self.c_values[idx],
                self.d_values[idx],
                self.e_values[idx],
                self.redshift_array[idx]
            )  # shape = (batch, nx, ny)

            # 4.4) Split into α1 and α2
            alpha1_batch = batch_alpha * (arc_dy / (R_ang + 1e-8))
            alpha2_batch = batch_alpha * (arc_dx / (R_ang + 1e-8))

            # 4.5) Flatten and accumulate by redshift group
            flat1 = alpha1_batch.reshape((end - start, -1)).astype(jnp.float32)
            flat2 = alpha2_batch.reshape((end - start, -1)).astype(jnp.float32)
            inv_idx = self._inv_z_idx[idx]  # length = end-start

            sum1 = segment_sum(flat1, inv_idx, G)  # shape (G, nx*ny)
            sum2 = segment_sum(flat2, inv_idx, G)

            del flat1, flat2, alpha1_batch, alpha2_batch
            del arc_dx, arc_dy, R_ang, phys_dx, phys_dy, R_phys, batch_alpha
            # 4.6) Reshape and add to totals
            grouped_alpha1 = grouped_alpha1 + sum1.reshape((G, nx, ny))
            grouped_alpha2 = grouped_alpha2 + sum2.reshape((G, nx, ny))



        return grouped_alpha2, grouped_alpha1
