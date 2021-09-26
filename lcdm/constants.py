# Physical/math constants

# c = 2.99792458e5       				# km/s
T_CMB = 2.7255						    # CMB temperature (K) 
C = 2.99792458e8       					# speed of ligth (m/s) (formerly _c_)
G = 6.67428e-11        					# Newton constant (m^3/Kg/s^2)
PI = 3.1415926535897932384626433832795
MPC_PER_M = 3.085677581282e22        	# conversion factor from meters to megaparsecs
K_B = 1.3806504e-23                   # Boltzmann cosntant (m^2 kg / s^2 / K)
H_PLANCK = 6.62606896e-34                  # Planck's constant (m^2 kg / s)
# Stefan-Boltzmann constant (Kg / K^4 / s^3)
SIGMA_B = 2. * pow(PI,5) * pow(K_B,4) / 15. / pow(H_PLANCK,3) / pow(C,2)  

ROOT_DIR = "/Users/fabio/code/fchibana/tachyons"

def radiation_density(h):
    """Current radiation energy density Omega^0_g

    Args:
        h ([double]): dimensionless Hubble constant (H0/100 s Mpc / km)

    Returns:
        [double]: Current Omega_g
    """
    return (4.*SIGMA_B/C*pow(T_CMB,4.)) / (3.*C*C*1.e10*h*h/MPC_PER_M/MPC_PER_M/8./PI/G)