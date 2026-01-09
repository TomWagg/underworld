import cogsworth
import numpy as np
import pandas as pd
from scipy.stats import maxwell
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
import astropy.units as u
from multiprocessing import Pool
from tqdm import tqdm

from time import time
from copy import copy
import ebf


"""The Sweeney+2022 paper does a simpler population synthesis that we can emulate instead of using COSMIC
and cogsworth. Instead, we just take the template population, select the remnant based on their initial masses
(as defined in Section 2 of their paper), apply the kick distributions from Eq 1-2, and integrate them through
the galactic potential.

The main differences here are that we use a different initial distribution, and a *slightly* different
galactic potential. However, this should be close enough for our purposes, and it's more easily comparable to
the other simulations.
"""

def load_data(filename, filter=True):
    '''Load Galaxia data into DataFrame'''
    data = ebf.read(filename, '/')
    centre = np.array(data['center'])
    keys = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'age', 'smass', 'feh', 'popid']
    useful_data = []
    for key in keys:
        useful_data += [data[key]]
    useful_data = np.array(useful_data).T
    df = pd.DataFrame(useful_data, columns=keys)

    # Convert age to gigayears
    df['age'] = 10**df['age'] / 10**9

    # Make data centred on galactic centre
    df.loc[:, ['px', 'py', 'pz', 'vx', 'vy', 'vz']] += centre

    # Add rtype column which specifies remnant type
    rtype = []
    for mass in df['smass']:
        if mass > 25:
            rtype.append('Black Hole')
        elif mass > 8:
            rtype.append('Neutron Star')
        else:
            rtype.append('White Dwarf')
    df['rtype'] = rtype

    if filter:
        df = df[df['smass'] > 8]
    return df


very_start = time()

# load the template population
template = load_data("/mnt/home/twagg/ceph/underworld/sweeney_data/galaxia_f1e-3_bhm2.35.ebf")

print(f"Loaded template population in {time() - very_start:.2f} seconds")

start = time()

print(f"Setup population dataframe in {time() - start:.2f} seconds")
start = time()

# draw random kicks using their distribution
n_low_peak = len(template) // 5
kicks = np.concatenate((
    maxwell(scale=56).rvs(size=n_low_peak),
    maxwell(scale=336).rvs(size=len(template) - n_low_peak)
))

# not really necessary, but shuffle to avoid any ordering effects
np.random.shuffle(kicks)
template["kick"] = kicks

# adjust kicks for black holes
template.loc[template["smass"] >= 25, "kick"] *= 1.35 / 7.8

# no kick for direct collapse
template.loc[template["smass"] > 40, "kick"] = 0.0

# randomly distribute kick
kick_theta = np.arccos(np.random.uniform(-1, 1, size=len(template)))
kick_phi = np.random.uniform(0, 2 * np.pi, size=len(template))
template["kick_x"] = template["kick"] * np.sin(kick_theta) * np.cos(kick_phi)
template["kick_y"] = template["kick"] * np.sin(kick_theta) * np.sin(kick_phi)
template["kick_z"] = template["kick"] * np.cos(kick_theta)

template.reset_index(drop=True, inplace=True)
print(f"Prepared kicks and initial conditions in {time() - start:.2f} seconds")

start = time()

t1 = (12 - template["age"].values) * u.Gyr
t2 = 12 * u.Gyr
dt = 1 * u.Myr

pot = gp.BovyMWPotential2014()

args = [
    (gd.PhaseSpacePosition(
        pos=np.array([template["px"].iloc[i], template["py"].iloc[i], template["pz"].iloc[i]]) * u.kpc,
        vel=np.array([
            template["vx"].iloc[i] + template["kick_x"].iloc[i],
            template["vy"].iloc[i] + template["kick_y"].iloc[i],
            template["vz"].iloc[i] + template["kick_z"].iloc[i],
        ]) * u.km / u.s,
    ),
     t1[i], t2, copy(dt), pot) for i in range(len(template))
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

print(f"Finished preparing arguments in {time() - start:.2f} seconds")
start = time()

with Pool(processes=32) as pool:
    rets = pool.starmap(int_func, tqdm(args, total=len(args)))

rets = np.array(rets)

template[["x_final", "y_final", "z_final"]] = rets[:, 0, :]
template[["v_x_final", "v_y_final", "v_z_final"]] = rets[:, 1, :]

print(f"Integrated orbits in {time() - start:.2f} seconds")

template.to_hdf("/mnt/ceph/users/twagg/underworld/sweeney_remnants_direct.h5", key="data", mode="w")
print(f"Saved population to disk in {time() - start:.2f} seconds")

print(f"Total script time: {time() - very_start:.2f} seconds")
