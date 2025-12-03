import cogsworth
import gala.potential as gp

import time

very_start = time.time()

kickflags = [5, 1, 1]
ecsn = [0, 2.25, 0]

for kickflag, ecsn in zip(kickflags, ecsn):
    print(f"Starting underworld simulation with kickflag={kickflag}, ecsn={ecsn}")

    start = time.time()
    initial_pop = cogsworth.pop.load("/mnt/ceph/users/twagg/underworld/template")
    print(f"   Loaded template population in {time.time() - start:1.2f} seconds")

    # convert columns back to strings
    initial_pop.bpp_columns = [col.decode('utf-8') for col in initial_pop.bpp_columns]

    cols = ["natal_kick_1", "phi_1", "theta_1", "natal_kick_2", "phi_2", "theta_2"]
    for col in cols:
        initial_pop.initC[col] = -100.0

    initial_pop.initC["kickflag"] = kickflag
    initial_pop.initC["ecsn"] = ecsn

    start = time.time()
    initial_pop.perform_stellar_evolution()
    print(f"   Performed stellar evolution in {time.time() - start:1.2f} seconds")

    # do galactic evolution only for the singles that end up as underworld objects
    start = time.time()
    underworld_mask = ((initial_pop.final_bpp['kstar_1'] == 13) | (initial_pop.final_bpp['kstar_1'] == 14) |
                       (initial_pop.final_bpp['kstar_2'] == 13) | (initial_pop.final_bpp['kstar_2'] == 14))
    underworld = initial_pop[underworld_mask]

    del initial_pop

    underworld.perform_galactic_evolution()
    print(f"   Performed galactic evolution for singles in {time.time() - start:1.2f} seconds")

    start = time.time()
    underworld.save(f"/mnt/ceph/users/twagg/underworld/binaries-kickflag-{kickflag}-ecsn-{ecsn}",
                    overwrite=True)
    print(f"   Saved underworld population in {time.time() - start:1.2f} seconds")

    del underworld

print("Underworld simulations complete!")
print(f"Total time: {time.time() - very_start:1.2f} seconds")
