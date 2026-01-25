from scipy.integrate import quad
import scipy
import numpy as np
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
# Cosmology, Units: Msun, kpc, Gyr
h = 0.674
Omegam0 = 0.3157
Omegab0 = 0.04936
sigma8 = 0.8116
Omegar0 = 5.3815e-5
Omegak0 = 0
OmegaL0 = 1 - Omegam0
G = 43007.3 / 10**10  # kpc/Msun(km/sec)^2

def Ez(z):
    return np.sqrt(OmegaL0 + Omegak0 * (1 + z)**2 + Omegam0 * (1 + z)**3 + Omegar0 * (1 + z)**4)

def Hz(z):
    H0 = h / 9.777752  # Gyr^-1
    return H0 * Ez(z)

def tL(z):
    integrand = lambda zx: 1 / (Hz(zx) * (1 + zx))
    return quad(integrand, 0, z)[0]

def rhocz(z):
    H0 = h / 9.777752  # Gyr^-1
    return (3 / (8 * np.pi * G)) * (0.978469)**2 * H0**2 * Ez(z)**2

def Zeta(z):
    OmegaM_z = (Omegam0 * (1 + z)**3) / Ez(z)**2
    return (18 * np.pi**2 + 82 * (OmegaM_z - 1) - 39 * (OmegaM_z - 1)**2) / OmegaM_z

# CDM halo profile based on mass concentration relation: Ref: Dutton & Maccio MNRAS 441, 3359–3374 (2014)
def CvirDuttonz0(Mh, nsigma):
    return 10**(1.025 + 0.11 * nsigma) * (Mh / (10**12 * h**-1))**-0.097

def MtotNFW(r, rhos, rs):
    return 4 * np.pi * rhos * rs**3 * (-1 + rs / (r + rs) - np.log(rs) + np.log(r + rs))

def VmaxNFW(rhos, rs):
    return 1.648 * np.sqrt(G * rhos * rs**2)

def RmaxNFW(rs):
    return 2.16258 * np.asarray(rs, dtype=float)

def Vmax_ratio(tau):
    """
    Compute V_max / V_max,0 as a function of tau.
    """
    return (1
            + 0.1777 * tau
            - 4.399  * tau**3
            + 16.66  * tau**4
            - 18.87  * tau**5
            + 9.077  * tau**7
            - 2.436  * tau**9)

def Rmax_ratio(tau):
    """
    Compute R_max / R_max,0 as a function of tau.
    """
    return (1
            + 0.007623 * tau
            - 0.7200  * tau**2
            + 0.3376  * tau**3
            - 0.1375  * tau**4)

def rhoNFW(r, Mvir, n):
    Rvir = (3 * Mvir / (4 * np.pi * rhocz(0) * Omegam0 * Zeta(0)))**(1/3)  # kpc
    cvir = CvirDuttonz0(Mvir, n)  # Median
    rs = Rvir / cvir  # kpc
    rhos = Mvir / (4 * np.pi * rs**3 * (np.log(1 + cvir) - cvir / (1 + cvir)))
    return rhos / ((r / rs) * (1 + r / rs)**2)

def rhoNFW_rs_rhos(r, rs, rhos):
    return rhos / ((r / rs) * (1 + r / rs)**2)

# Velocity dispersion function (from Jeans equation)
def Fx(x):
    return 0.5 * x * (1 + x)**2 * (np.pi**2 - np.log(x) - 1/x - 1/(1 + x)**2 - 
                                   6/(1 + x) + (1 + 1/x**2 - 4/x - 2/(1 + x)) * np.log(1 + x) + 
                                   3 * np.log(1 + x)**2 + 6 * special.polylog(2, -x))

def sigmaNFW(r, rhos, rs):
    return np.sqrt(4 * np.pi * G * rhos * rs**2 * Fx(r / rs))



# Parametric SIDM model: http://arxiv.org/abs/2305.16176
sigma0 = 1000
w0 = 40


def tc(rhoss, rss, sigmaOverm):
    return 150 / 0.75 * 1 / (rss * rhoss * sigmaOverm * 2.0899970609705147e-10) * 1 / np.sqrt(4 * np.pi * G * rhoss)

def rhost(tr, rhos):
    return rhos * (2.03305816 + 0.73806287 * tr + 7.26368767 * tr**5 - 12.72976657 * tr**7 + 9.91487857 * tr**9 - 0.1448 * (1 - 2.03305816) * np.log(tr + 0.001))

def rst(tr, rs):
    return rs * (0.71779858 - 0.10257242 * tr + 0.24743911 * tr**2 - 0.40794176 * tr**3 - 0.1448 * (1 - 0.71779858) * np.log(tr + 0.001))

def rct(tr, rs):
    return rs * (2.55497727 * np.sqrt(tr) - 3.63221179 * tr + 2.13141953 * tr**2 - 1.41516784 * tr**3 + 0.46832269 * tr**4)

def rhoSIDM(r, Mvir, n):
    Rvir = (3 * Mvir / (4 * np.pi * rhocz(0) * Omegam0 * Zeta(0)))**(1/3)  # kpc
    cvir = CvirDuttonz0(Mvir, n)  # Median
    rs = Rvir / cvir  # kpc
    rhos = Mvir / (4 * np.pi * rs**3 * (np.log(1 + cvir) - cvir / (1 + cvir)))
    #print(f"The paramatric: rhos={rhos}, rs = {rs}")
    veff = VmaxNFW(rhos, rs) * 0.64
    sigmaeff = 1 / (512 * veff**8) * quad(lambda v: v**7 * np.exp(-v**2 / (4 * veff**2)) * 2/3 * sigmaVis(v, sigma0, w0), 0.001, 1000)[0]
    zf = -0.0064 * np.log10(Mvir)**2 + 0.0237 * np.log10(Mvir) + 1.8837
    tform = 14 - tL(zf)
    #print(f"M_sigmaeff:{sigmaeff}")

    tc0 = tc(rhos, rs, sigmaeff)
    
    tr = tform / tc0
    if tr > 1:
        tr = 1.1

    return rhost(tr, rhos) / ((r**4 + rct(tr, rs)**4)**(1/4) / rst(tr, rs) * (1 + r / rst(tr, rs))**2), tr

def rhoSIDM_rhos_rs(r, rhos,rs,Mvir,sigma0=100, w0=60):
    veff = VmaxNFW(rhos, rs) * 0.64
    # print(veff)
    sigmaeff = 1 / (512 * veff**8) * quad(lambda v: v**7 * np.exp(-v**2 / (4 * veff**2)) * 2/3 * sigmaVis(v, sigma0, w0), 0.001, 1000)[0]
    #print(f"sigmaeff:{sigmaeff}")
    zf = -0.0064 * np.log10(Mvir)**2 + 0.0237 * np.log10(Mvir) + 1.8837
    tform = 14 - tL(zf)
    #print(f"TL = {tL(zf)}")

    tc0 = tc(rhos, rs, sigmaeff)
    
    tr = tform / tc0
    if tr > 1.08:
        tr = 108

    return rhost(tr, rhos) / ((r**4 + rct(tr, rs)**4)**(1/4) / rst(tr, rs) * (1 + r / rst(tr, rs))**2), tr,sigmaeff


def rhoSIDM_rhos_rs_tf(r, rhos,rs,tl):
    veff = VmaxNFW(rhos, rs) * 0.64
    sigmaeff = 1 / (512 * veff**8) * quad(lambda v: v**7 * np.exp(-v**2 / (4 * veff**2)) * 2/3 * sigmaVis(v, sigma0, w0), 0.001, 1000)[0]

    # print(f"sigmaeff:{sigmaeff}")

    tform = 14 - tl
    tc0 = tc(rhos, rs, sigmaeff)
    tr = tform / tc0
    if tr > 1.08:
        tr = 1.08

    return rhost(tr, rhos) / ((r**4 + rct(tr, rs)**4)**(1/4) / rst(tr, rs) * (1 + r / rst(tr, rs))**2), tr


# def tcb(sigmax,rhox,rx,rhoH,rH):
#     gamma=1.6
#     gamma2=20
#     reff=(rhox*pow(rx,3) + gamma*rhoH*pow(rH,3)*0.5)/(rhox*pow(rx,2) + gamma*rhoH*pow(rH,2)*0.5)
#     rhoeff=(rhox*rx/reff + gamma2*rhoH*pow(rH,3)/(reff*pow(reff+rH,2)) )
#     val= 150/0.9/(sigmax*(rhoeff)*2.09e-10)*pow(4*3.141593*G*(rhox*rx*rx+gamma*rhoH*rH*rH*0.5),-0.5)
#     return val

# # works for the cosmological parameters in YNY23
# def tlb(z): # in Gyr
#     return 13.647247606199668 - 11.020482589612016*np.log(1.5800327517186143/pow(1 + z,1.5) + np.sqrt(1. + 2.4965034965034962/pow(1 + z,3.)))
# def sigmaVis(v, sigma0, w):
#     return sigma0 * (6 * w**6) / v**6 * ((2 + v**2 / w**2) * jnp.log(1 + v**2 / w**2) - 2 * v**2 / w**2)
# def sigmaeff(veff,sigma0,w):
#     fint = lambda lnv:pow(np.exp(lnv),7)*np.exp(-pow(np.exp(lnv),2)/(4*pow(veff,2)))*(2/3)*sigmaVis(np.exp(lnv),sigma0,w)*np.exp(lnv)
#     val=scipy.integrate.quad(fint, np.log(0.01), np.log(15000))
#     return val[0]/(512*pow(veff,8))
# def get_sigmaxb(rhoss,rss,rhoH,rH,sigma0,w0):
#     veff = 1.648*np.sqrt(G*rhoss)*rss*0.64
#     veff = veff*np.sqrt((rhoss*rss*rss+rhoH*rH*rH/2)/(rhoss*rss*rss))
#     sidmx=sigmaeff(veff,sigma0,w0)
#     return sidmx

# def get_taub(Mvir,rhoss,rss,rhoH,rH,sigma0=100,w0=60):
#     sigmax=get_sigmaxb(rhoss,rss,rhoH,rH,sigma0,w0)
#     tcb0=tcb(sigmax,rhoss,rss,rhoH,rH)
#     zf = -0.0064 * np.log10(Mvir)**2 + 0.0237 * np.log10(Mvir) + 1.8837
#     tlb0=tlb(zf)
#     tform = 13.647-tlb0

#     tau = tform/tcb0
#     if tau > 1.08:
#         tau = 1.08
#     return tau


def tcb(sigmax, rhox, rx, rhoH, rH):
    gamma = 1.6
    gamma2 = 20
    reff = (rhox * jnp.power(rx, 3) + gamma * rhoH * jnp.power(rH, 3) * 0.5) / (rhox * jnp.power(rx, 2) + gamma * rhoH * jnp.power(rH, 2) * 0.5)
    rhoeff = (rhox * rx / reff + gamma2 * rhoH * jnp.power(rH, 3) / (reff * jnp.power(reff + rH, 2)))
    val = 150 / 0.9 / (sigmax * (rhoeff) * 2.09e-10) * jnp.power(4 * 3.141593 * G * (rhox * rx * rx + gamma * rhoH * rH * rH * 0.5), -0.5)
    return val

def tlb(z):  # in Gyr
    return 13.647247606199668 - 11.020482589612016 * jnp.log(1.5800327517186143 / jnp.power(1 + z, 1.5) + jnp.sqrt(1. + 2.4965034965034962 / jnp.power(1 + z, 3.)))

def sigmaVis(v, sigma0, w):

    return sigma0 * (6 * w**6) / v**6 * ((2 + v**2 / w**2) * jnp.log(1 + v**2 / w**2) - 2 * v**2 / w**2)

# Numerical integration using trapezoidal rule
def numerical_integration(f, a, b, n=1000):
    # Trapezoidal rule for numerical integration
    x = jnp.linspace(a, b, n)
    y = f(x)
    dx = (b - a) / (n - 1)
    return jnp.sum((y[:-1] + y[1:]) * dx / 2)

def sigmaeff(veff, sigma0, w):
    def fint(lnv):
        exp_lnv = jnp.exp(lnv)
        return jnp.power(exp_lnv, 7) * jnp.exp(-jnp.power(exp_lnv, 2) / (4 * jnp.power(veff, 2))) * (2 / 3) * sigmaVis(exp_lnv, sigma0, w) * exp_lnv

    # Using trapezoidal rule to replace scipy's quad
    val = numerical_integration(fint, jnp.log(0.01), jnp.log(15000))
    return val / (512 * jnp.power(veff, 8))

def get_sigmaxb(rhoss, rss, rhoH, rH, sigma0, w0):
    veff = 1.648 * jnp.sqrt(G * rhoss) * rss * 0.64
    veff = veff * jnp.sqrt((rhoss * rss * rss + rhoH * rH * rH / 2) / (rhoss * rss * rss))
    sidmx = sigmaeff(veff, sigma0, w0)
    return sidmx

def get_taub(Mvir, rhoss, rss, rhoH, rH, sigma0=100, w0=60):
    # kpc
    sigmax = get_sigmaxb(rhoss, rss, rhoH, rH, sigma0, w0)
    tcb0 = tcb(sigmax, rhoss, rss, rhoH, rH)
    zf = -0.0064 * jnp.log10(Mvir) ** 2 + 0.0237 * jnp.log10(Mvir) + 1.8837
    tlb0 = tlb(zf)
    tform = 13.647 - tlb0

    tau = tform / tcb0
    tau  = jnp.where(tau < 1e-4, 0, tau)  # Ensure tau is non-negative
    tau = jnp.where(tau > 1.08, 1.08, tau)  # Replace if condition with jnp.where
    return tau



from scipy.optimize import fsolve

def Mb_to_rH_rhoH(mb):
    resigma = np.random.normal(0, 0.1)
    rH = 0.0938884074712349 * pow(1 + 4.329004329004329e-11 * mb, 0.65) * pow(mb, 0.1) * pow(10, resigma)
    
    # Guard against zero or extremely small rH
    if rH < 1e-10:
        return 0, 0
    else:
        rhoH = mb / (2. * 3.1415923 * pow(rH, 3))
        return rH, rhoH

def Md_to_rH_rhoH(md):
    resigma = np.random.normal(0, 0.1)
    msigma=np.random.normal(0, 0.15)
    m1=pow(10,11.59)
    # Compute mb while avoiding md/m1 being zero or tiny
    if md / m1 < 1e-10:
        return 0, 0
    else:
        mb = md * 2 * 0.0351 / (pow(md / m1, -1.376) + pow(md / m1, 0.608)) * pow(10, msigma)
        # print("!!mb = {:.2e} Msun, md = {:.2e} Msun".format(mb, md))
    rH = 0.0938884074712349 * pow(1 + 4.329004329004329e-11 * mb, 0.65) * pow(mb, 0.1) * pow(10, resigma)
    
    # Guard against zero or extremely small rH
    if rH < 1e-10:
        return 0, 0
    else:
        rhoH = mb / (2. * 3.1415923 * pow(rH, 3))
        return rH, rhoH
    
def Vmax_Rmax_to_rhoss_rss(Vmax, Rmax):
    # Guard against zero or extremely small Rmax
    if Rmax < 1e-10:
        return 0, 0
    else:
        rss = Rmax / 2.1626
        GG = 4.30073 * 1e-6
        rhoss = pow(Vmax / 1.648 / rss, 2) / GG
        return rss, rhoss
    

def MtotNFW(r,rhoss,rss):
    if(r<=0): return 0
    return 4*np.pi*rhoss*pow(rss,3)*(-1.0 + rss/(r + rss) - np.log(rss) + np.log(r + rss))

def solve_Rvir(r,rhoss,rss):
    rho200c=25509.3 
    OM0 = 0.3089
    Or0 = 5.3815* 10**(-5)
    Ok0 = 0;
    OL0 = 1. - OM0
    Ez=np.sqrt(OL0 + Ok0+ OM0+ Or0)
    OM=(OM0)/Ez**2
    eta= (18*np.pi**2 + 82*(OM-1)- 39 *(OM - 1)**2) # the one used, Planck
    rhohbar=eta*rho200c/200
    return MtotNFW(r,rhoss,rss)-4/3*np.pi*r**3*rhohbar


def Rss_Rhoss_to_c200(rss,rhoss):
    Rvir = fsolve(solve_Rvir,10000*rss, args=(rhoss, rss))[0]
    Mvir= MtotNFW(Rvir,rhoss,rss)
    c200=Rvir/rss
    return Mvir,Rvir,c200

# Function to compute concentration parameter c
def compute_c(hrhalf, hrvir,cs_initial_guess):
    def equation(cs):
        if cs <= 0:
            return np.inf  # Return a large number to indicate a bad solution
        else:
            return hrhalf / hrvir - (0.6082 - 0.1843 * np.log10(cs) - 0.1011 * (np.log10(cs))**2 + 0.03918 * (np.log10(cs))**3)

    # cs_initial_guess = 5.0
    cs_solution, _, ier, _ = fsolve(equation, cs_initial_guess, full_output=True, maxfev=100000)
    
    if ier == 1:  # Successful solution
        return cs_solution[0]  # Return the first element to ensure a scalar value
    else:
        return np.nan  # Return NaN if no solution found
    



import numpy as np

def xi_approx(sigma_prime: float,
              sigma: float,
              beta: float,
              nu_loss: float,
              rho_s: float,
              r_s: float,
              G: float = 4.3009172700363e-6):
    """
    Approximate ξ (valid only when ν_loss < 0.2).
    Units must be consistent; G defaults to (kpc * (km/s)^2) / Msun.
    """
    # ratio = σ' / (β σ)
    ratio = sigma_prime / (beta * sigma)
    root = np.sqrt(ratio)
    denom = 0.035 * np.sqrt(4 * np.pi * G * rho_s * r_s**2)
    return np.exp(-(nu_loss * root) / denom)

def tcprime_from_tc(tc: float,
                    sigma_prime: float,
                    sigma: float,
                    beta: float,
                    nu_loss: float,
                    rho_s: float,
                    r_s: float,
                    G: float = 4.3009172700363e-6,
                    xi_override: float | None = None):
    """
    t_c' = ξ * t_c
    Use xi_override when provided; otherwise use the approximate ξ (for ν_loss < 0.2).
    """
    xi = xi_override if xi_override is not None else xi_approx(
        sigma_prime=sigma_prime,
        sigma=sigma,
        beta=beta,
        nu_loss=nu_loss,
        rho_s=rho_s,
        r_s=r_s,
        G=G
    )
    return xi * tc

def tc_from_tcprime(tc_prime: float,
                    sigma_prime: float,
                    sigma: float,
                    beta: float,
                    nu_loss: float,
                    rho_s: float,
                    r_s: float,
                    G: float = 4.3009172700363e-6,
                    xi_override: float | None = None):
    """
    t_c = t_c' / ξ
    Use xi_override when provided; otherwise use the approximate ξ (for ν_loss < 0.2).
    """
    xi = xi_override if xi_override is not None else xi_approx(
        sigma_prime=sigma_prime,
        sigma=sigma,
        beta=beta,
        nu_loss=nu_loss,
        rho_s=rho_s,
        r_s=r_s,
        G=G
    )
    return tc_prime / xi
