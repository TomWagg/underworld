import cogsworth
import gala.potential as gp
import argparse
from os.path import join, exists
from os import mkdir

import time


def enter_the_underworld(n_binaries, output_dir, processes, simulation_name, file_suffix):
    """
    Run Underworld simulation with cogsworth. This creates the template and fiducial simulation.

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
    file_suffix : str
        Suffix to append to output filenames.
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
    ]

    # create a new cogsworth population that assumes 100% binarity
    pot = gp.MilkyWayPotential(version='v2')
    initial_pop = cogsworth.pop.Population(
        n_binaries=n_binaries,
        processes=processes,
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
        store_entire_orbits=False
    )
    initial_pop.BSE_settings["binfrac"] = 1.0

    # sample initial binaries
    start = time.time()
    initial_pop.sample_initial_binaries()
    print(f"Sampled initial binaries in {time.time() - start:1.2f} seconds")

    # perform stellar evolution for binaries
    start = time.time()
    initial_pop.perform_stellar_evolution()
    print(f"   Performed stellar evolution in {time.time() - start:1.2f} seconds")

    print("Save the template population")
    start = time.time()

    template_folder = join(output_dir, "template")
    if not exists(template_folder):
        mkdir(template_folder)

    initial_pop.save(join(template_folder, f"template{file_suffix}"), overwrite=True)
    print(f"   Saved template population in {time.time() - start:1.2f} seconds")

    # do galactic evolution only for the binaries that end up as underworld objects
    start = time.time()
    underworld_mask = ((initial_pop.final_bpp['kstar_1'] == 13) | (initial_pop.final_bpp['kstar_1'] == 14) |
                       (initial_pop.final_bpp['kstar_2'] == 13) | (initial_pop.final_bpp['kstar_2'] == 14))
    binary_underworld = initial_pop[underworld_mask]
    binary_underworld.perform_galactic_evolution()
    print(f"   Performed galactic evolution for binaries in {time.time() - start:1.2f} seconds")

    simulation_folder = join(output_dir, simulation_name)
    if not exists(simulation_folder):
        mkdir(simulation_folder)

    start = time.time()
    binary_underworld.save(join(simulation_folder, f"{simulation_name}{file_suffix}"), overwrite=True)
    print(f"   Saved underworld population in {time.time() - start:1.2f} seconds")

    print(f"   Number of underworld binaries: {len(binary_underworld)}")

    print("Underworld simulation complete!")
    print(f"Total time: {time.time() - very_start:1.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Run Underworld simulation with cogsworth")
    parser.add_argument(
        '-n',
        '--n_binaries',
        type=int,
        default=10_000_000,
        help='Number of binaries to simulate (default: 10,000,000)'
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
        default=64,
        help='Number of processes to use for parallel computation (default: 64)'
    )
    parser.add_argument(
        '-s',
        '--simulation-name',
        type=str,
        default='underworld_simulation',
        help='Name of the simulation for output files (default: underworld_simulation)'
    )
    parser.add_argument(
        '-f',
        '--file-suffix',
        type=str,
        default='',
        help='Suffix to append to output filenames (default: "")'
    )

    args = parser.parse_args()

    enter_the_underworld(
        n_binaries=args.n_binaries,
        output_dir=args.output_dir,
        processes=args.processes,
        simulation_name=args.simulation_name,
        file_suffix=args.file_suffix
    )


if __name__ == "__main__":
    main()
