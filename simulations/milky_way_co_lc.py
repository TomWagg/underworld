import cogsworth
import gala.potential as gp
import argparse
from os.path import join, exists
import pathlib
import astropy.units as u

import time


def find_the_bh_lc_pop(n_binaries, output_dir, processes, simulation_name):
    """
    Run CO LC simulation with cogsworth. This creates the fiducial simulation.

    Parameters
    ----------
    n_binaries : int
        Number of binaries to simulate.
    output_dir : str
        Output directory for saving results.
    processes : int
        Number of processes to use for parallel computation.
    simulation_name : str
        Name of the simulation for output files.
    """
    # quickly check if output directory exists
    if not exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist!")

    very_start = time.time()

    print("Initiating cogsworth underworld simulation")

    bpp_columns = [
        'tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 'sep', 'porb', 'ecc',
        'evol_type', 'RRLO_1', 'RRLO_2', 'massc_he_layer_1', 'massc_co_layer_1',
        'massc_he_layer_2', 'massc_co_layer_2',
        'lum_1', 'lum_2', 'teff_1', 'teff_2', 'SN_1', 'SN_2', 'omega_spin_1', 'omega_spin_2',
        'rad_1', 'rad_2',
    ]
    pot = gp.MilkyWayPotential(version='v2')

    # loop until we've sampled a Milky Way's worth of mass
    mass_sampled = 0
    n_loops = 0

    bh_lc_pops = []

    while mass_sampled < 1e7:
        n_loops += 1
        print(f"Loop {n_loops}: Currently sampled {mass_sampled:.2e} Msun")

        # create a new cogsworth population that assumes 100% binarity
        initial_pop = cogsworth.pop.Population(
            n_binaries=n_binaries,
            processes=processes,
            ini_file="/mnt/home/twagg/projects/underworld/simulations/params.ini",
            galactic_potential=pot,
            sfh_model=cogsworth.sfh.SandersBinney2015,
            sfh_params={
                "potential": pot,
                "time_bins": 5,
                "verbose": False
            },
            bpp_columns=bpp_columns,
            store_entire_orbits=False,
            error_file_path=None,
            v_dispersion=0 * u.km / u.s,
            sampling_params={
                'qmin': 0.0,
                'm2_min': 0.08,
                "binfrac_model": 1.0,
            },
            final_kstar1=[14],
            sampling_mask="mass_1 > 16 and mass_2 < 8"
        )

        print(f"   Created initial population with {n_binaries} binaries")

        # sample initial binaries
        start = time.time()
        initial_pop.sample_initial_binaries()
        print(f"   Sampled initial binaries in {time.time() - start:1.2f} seconds, left with {len(initial_pop)} binaries after sampling")

        # perform stellar evolution for binaries
        start = time.time()
        initial_pop.perform_stellar_evolution()
        print(f"   Performed stellar evolution in {time.time() - start:1.2f} seconds")

        # mask out the co-lc binaries
        bound = initial_pop.final_bpp["sep"] > 0
        bh_or_lc = (
            ((initial_pop.final_bpp['kstar_1'] == 14) & (initial_pop.final_bpp['kstar_2'] < 10)) |
            ((initial_pop.final_bpp['kstar_2'] == 14) & (initial_pop.final_bpp['kstar_1'] < 10))
        )
        bh_lc_mask = bound & bh_or_lc

        if bh_lc_mask.sum() > 0:
            bh_lc_pop = initial_pop[bh_lc_mask]
            bh_lc_pops.append(bh_lc_pop)
        print(f"   Found {bh_lc_mask.sum()} BH LC binaries in the population")

        mass_sampled += initial_pop.mass_binaries

    start = time.time()

    full_bh_lc_pop = cogsworth.pop.concat(*bh_lc_pops)

    print(f"Concatenated CO LC populations in {time.time() - start:1.2f} seconds")

    start = time.time()
    full_bh_lc_pop.perform_galactic_evolution()
    print(f"Performed galactic evolution for binaries in {time.time() - start:1.2f} seconds")

    simulation_folder = join(output_dir, simulation_name)
    if not exists(simulation_folder):
        pathlib.Path(simulation_folder).mkdir(parents=True, exist_ok=True) 

    start = time.time()
    full_bh_lc_pop.save(join(simulation_folder, f"{simulation_name}"), overwrite=True)
    print(f"Saved BH-LC population in {time.time() - start:1.2f} seconds")

    print(f"Number of BH-LC binaries: {len(full_bh_lc_pop)}, Total mass sampled: {mass_sampled:.2e} {full_bh_lc_pop.mass_binaries:.2e} Msun")

    print("BH-LC simulation complete!")
    print(f"Total time: {time.time() - very_start:1.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Run Underworld simulation with cogsworth")
    parser.add_argument(
        '-n',
        '--n_binaries',
        type=int,
        default=20_000_000,
        help='Number of binaries to simulate (default: 20,000,000)'
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default='/mnt/ceph/users/twagg/underworld/',
        help='Output directory for saving results (default: /mnt/ceph/users/twagg/underworld/)'
    )
    parser.add_argument(
        '-p',
        '--processes',
        type=int,
        default=32,
        help='Number of processes to use for parallel computation (default: 32)'
    )
    parser.add_argument(
        '-s',
        '--simulation-name',
        type=str,
        default='bh-lc',
        help='Name of the simulation for output files (default: bh-lc)'
    )

    args = parser.parse_args()

    find_the_bh_lc_pop(
        n_binaries=args.n_binaries,
        output_dir=args.output_dir,
        processes=args.processes,
        simulation_name=args.simulation_name,
    )


if __name__ == "__main__":
    main()
