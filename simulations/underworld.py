import cogsworth
import gala.potential as gp

import time

very_start = time.time()

print("Initiating cogsworth underworld simulation")

bpp_columns = [
    'tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 'sep', 'porb', 'ecc',
    'evol_type', 'RRLO_1', 'RRLO_2', 'massc_1', 'massc_2',
    'lum_1', 'lum_2', 'teff_1', 'teff_2', 'SN_1', 'SN_2'
]

# create a new cogsworth population that assumes 100% binarity
pot = gp.MilkyWayPotential2022()
initial_pop = cogsworth.pop.Population(n_binaries=10_000_000, processes=32,
                                       m1_cutoff=4,
                                       ini_file="/mnt/home/twagg/projects/underworld/simulations/params.ini",
                                       galactic_potential=pot,
                                       sfh_model=cogsworth.sfh.SandersBinney2015,
                                       sfh_params={
                                           "potential": pot,
                                           "time_bins": 5,
                                           "verbose": True
                                       },
                                       bpp_columns=bpp_columns,
                                       store_entire_orbits=False)
initial_pop.BSE_settings["binfrac"] = 1.0

# sample initial binaries
start = time.time()
initial_pop.sample_initial_binaries()
print(f"Sampled initial binaries in {time.time() - start:1.2f} seconds")

print("Binaries first!")

# perform steller evolution for binaries
start = time.time()
initial_pop.perform_stellar_evolution()
print(f"   Performed stellar evolution for binaries in {time.time() - start:1.2f} seconds")

print("Save the template population")
start = time.time()
initial_pop.save("/mnt/ceph/users/twagg/underworld/template", overwrite=True)
print(f"   Saved template population in {time.time() - start:1.2f} seconds")

# do galactic evolution only for the binaries that end up as underworld objects
start = time.time()
underworld_mask = ((initial_pop.final_bpp['kstar_1'] == 13) | (initial_pop.final_bpp['kstar_1'] == 14) |
                   (initial_pop.final_bpp['kstar_2'] == 13) | (initial_pop.final_bpp['kstar_2'] == 14))
binary_underworld = initial_pop[underworld_mask]
binary_underworld.perform_galactic_evolution()
print(f"   Performed galactic evolution for binaries in {time.time() - start:1.2f} seconds")

start = time.time()
binary_underworld.save("/mnt/ceph/users/twagg/underworld/binaries", overwrite=True)
print(f"   Saved binary underworld population in {time.time() - start:1.2f} seconds")

print(f"   Number of underworld binaries: {len(binary_underworld)}")

print("Now singles!")

# copy for singles
singles = initial_pop.copy()
singles.initC["porb"] = 1e20
singles.initC["ecc"] = 0.0

cols = ["natal_kick_1", "phi_1", "theta_1", "natal_kick_2", "phi_2", "theta_2"]
for col in cols:
    singles.initC[col] = -100.0

# perform steller evolution for singles
start = time.time()
singles.perform_stellar_evolution()
print(f"   Performed stellar evolution for singles in {time.time() - start:1.2f} seconds")

# do galactic evolution only for the singles that end up as underworld objects
start = time.time()
underworld_mask = ((singles.final_bpp['kstar_1'] == 13) | (singles.final_bpp['kstar_1'] == 14) |
                   (singles.final_bpp['kstar_2'] == 13) | (singles.final_bpp['kstar_2'] == 14))
single_underworld = singles[underworld_mask]
single_underworld.perform_galactic_evolution()
print(f"   Performed galactic evolution for singles in {time.time() - start:1.2f} seconds")

start = time.time()
single_underworld.save("/mnt/ceph/users/twagg/underworld/singles", overwrite=True)
print(f"   Saved single underworld population in {time.time() - start:1.2f} seconds")

print("Underworld simulation complete!")
print(f"Total time: {time.time() - very_start:1.2f} seconds")
