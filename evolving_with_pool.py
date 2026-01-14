import gala.dynamics as gd
import gala.potential as gp
import numpy as np
import astropy.units as u
from multiprocessing import Pool

# construct a simple time-evolving NFW potential
times = np.linspace(0, 10, 100) * u.Gyr
masses = np.linspace(1e11, 5e11, 100) * u.Msun
pot = gp.TimeInterpolatedPotential(
    gp.NFWPotential, times, m=masses, r_s=20 * u.kpc, units="galactic"
)

# set up initial conditions and integration parameters
w0 = gd.PhaseSpacePosition(pos=[10.0, 0.0, 0.0] * u.kpc, vel=[0, 200.0, 10] * u.km / u.s)

# just repeat the same thing 10 times to demonstrate parallelization
w0s = [w0 for _ in range(10)]
args = [(w0i, 0, 10 * u.Gyr, 1 * u.Myr) for w0i in w0s]

# define a helper function for integration when doing pool mapping
def int_func(args):
    return pot.integrate_orbit(args[0], t1=args[1], t2=args[2], dt=args[3])

# do everything serially first - it works!
orbits = [int_func(arg) for arg in args]

# now do the same thing in parallel - this seems to hang
with Pool(processes=2) as pool:
    orbits_pool = pool.map(int_func, args)
