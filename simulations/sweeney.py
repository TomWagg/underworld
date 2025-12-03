import cogsworth
import numpy as np
import pandas as pd
from scipy.stats import maxwell
import gala.dynamics as gd
import gala.integrate as gi
import astropy.units as u
from multiprocessing import Pool
from tqdm import tqdm

from time import time
from copy import copy


"""The Sweeney+2022 paper does a simpler population synthesis that we can emulate instead of using COSMIC
and cogsworth. Instead, we just take the template population, select the remnant based on their initial masses
(as defined in Section 2 of their paper), apply the kick distributions from Eq 1-2, and integrate them through
the galactic potential.

The main differences here are that we use a different initial distribution, and a *slightly* different
galactic potential. However, this should be close enough for our purposes, and it's more easily comparable to
the other simulations.
"""


very_start = time()

# load the template population
template = cogsworth.pop.load("/mnt/ceph/users/twagg/underworld/template", parts=[])

print(f"Loaded template population in {time() - very_start:.2f} seconds")

start = time()

# setup a simple population dataframe
pop = pd.DataFrame({
    "mass": np.concatenate([template.initC["mass_1"], template.initC["mass_2"]]),
    "x": np.concatenate([template.initial_galaxy.x, template.initial_galaxy.x]),
    "y": np.concatenate([template.initial_galaxy.y, template.initial_galaxy.y]),
    "z": np.concatenate([template.initial_galaxy.z, template.initial_galaxy.z]),
    "v_R": np.concatenate([template.initial_galaxy.v_R, template.initial_galaxy.v_R]),
    "v_T": np.concatenate([template.initial_galaxy.v_T, template.initial_galaxy.v_T]),
    "v_z": np.concatenate([template.initial_galaxy.v_z, template.initial_galaxy.v_z]),
    "tau": np.concatenate([template.initial_galaxy.tau, template.initial_galaxy.tau]),
})

# remove anything below 8 solar masses (Sweeney defines remnants as > 8 Msun)
pop = pop[pop["mass"] > 8]

print(f"Setup population dataframe in {time() - start:.2f} seconds")
start = time()

# convert velocities to cartesian
pop["R"] = np.sqrt(pop["x"]**2 + pop["y"]**2)
pop["v_x"] = -pop["v_T"] * (pop["y"] / pop["R"]) + pop["v_R"] * (pop["x"] / pop["R"])
pop["v_y"] = pop["v_T"] * (pop["x"] / pop["R"]) + pop["v_R"] * (pop["y"] / pop["R"])

# draw random kicks using their distribution
n_low_peak = len(pop) // 5
kicks = np.concatenate((
    maxwell(scale=56).rvs(size=n_low_peak),
    maxwell(scale=336).rvs(size=len(pop) - n_low_peak)
))

# not really necessary, but shuffle to avoid any ordering effects
np.random.shuffle(kicks)
pop["kick"] = kicks

# adjust kicks for black holes
pop.loc[pop["mass"] >= 25, "kick"] *= 1.35 / 7.8

# no kick for direct collapse
pop.loc[pop["mass"] > 40, "kick"] = 0.0

# randomly distribute kick
kick_theta = np.arccos(np.random.uniform(-1, 1, size=len(pop)))
kick_phi = np.random.uniform(0, 2 * np.pi, size=len(pop))
pop["kick_x"] = pop["kick"] * np.sin(kick_theta) * np.cos(kick_phi)
pop["kick_y"] = pop["kick"] * np.sin(kick_theta) * np.sin(kick_phi)
pop["kick_z"] = pop["kick"] * np.cos(kick_theta)

pop.reset_index(drop=True, inplace=True)

print(f"Prepared kicks and initial conditions in {time() - start:.2f} seconds")
start = time()

t1 = template.max_ev_time - pop["tau"].values * u.Gyr
t2 = template.max_ev_time
dt = 1 * u.Myr

args = [
    (gd.PhaseSpacePosition(
        pos=np.array([pop["x"].iloc[i], pop["y"].iloc[i], pop["z"].iloc[i]]) * u.kpc,
        vel=np.array([
            pop["v_x"].iloc[i] + pop["kick_x"].iloc[i],
            pop["v_y"].iloc[i] + pop["kick_y"].iloc[i],
            pop["v_z"].iloc[i] + pop["kick_z"].iloc[i],
        ]) * u.km / u.s,
    ),
     t1[i], t2, copy(dt), template.galactic_potential) for i in range(len(pop))
]


def int_func(w0, t1, t2, dt, pot):
    orbit = None
    for _ in range(2):
        try:
            success = False
            orbit = pot.integrate_orbit(w0, t1=t1, t2=t2, dt=dt,
                                        Integrator=gi.DOPRI853Integrator, save_all=False)
            success = True
        except Exception:
            dt /= 8

        if success:
            break

    if orbit is None:
        ret = np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ])
    else:
        ret = np.array([
            orbit.pos.xyz[:, -1].to(u.kpc).value,
            orbit.vel.d_xyz[:, -1].to(u.km / u.s).value,
        ])
    return ret


with Pool(processes=32) as pool:
    rets = pool.starmap(int_func, tqdm(args, total=len(args)))

rets = np.array(rets)

pop[["x_final", "y_final", "z_final"]] = rets[:, 0, :]
pop[["v_x_final", "v_y_final", "v_z_final"]] = rets[:, 1, :]

print(f"Integrated orbits in {time() - start:.2f} seconds")

pop.to_hdf("/mnt/ceph/users/twagg/underworld/sweeney_remnants.h5", key="data", mode="w")
print(f"Saved population to disk in {time() - start:.2f} seconds")

print(f"Total script time: {time() - very_start:.2f} seconds")
