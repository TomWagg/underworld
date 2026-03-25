import cogsworth
import argparse
from os.path import join, exists
import evolving_potential
import pathlib

import time


def evolve_the_galaxy(output_dir, processes, simulation_name, file_suffix):
    """
    Run Underworld simulation with cogsworth.

    Parameters
    ----------
    output_dir : str
        Output directory for saving results.
    processes : int
        Number of processes to use for parallel computation.
    file_suffix : str
        Suffix to append to output filenames.
    """
    # quickly check if output directory exists
    if not exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist!")

    simulation_folder = join(output_dir, simulation_name)
    if not exists(simulation_folder):
        pathlib.Path(simulation_folder).mkdir(parents=True, exist_ok=True)

    print("Starting Underworld simulation with these parameters:")
    print(f"   Output directory: {output_dir}")
    print(f"   Number of processes: {processes}")
    print(f"   Simulation name: {simulation_name}")
    print(f"   File suffix: {file_suffix}")
    print("       EVOLVING the potential")

    evolving_mw_pot = evolving_potential.get_milky_way_potential(evolve_dm=True, evolve_disk=True)

    very_start = time.time()

    # read the template population
    print("Reading the template population")
    start = time.time()
    underworld = cogsworth.pop.load(join(output_dir, f"fiducial/fiducial{file_suffix}"))
    print(f"   Loaded template population in {time.time() - start:1.2f} seconds")

    underworld.processes = processes
    underworld.galactic_potential = evolving_mw_pot

    # do galactic evolution only for the binaries that end up as underworld objects
    start = time.time()
    underworld.perform_galactic_evolution()
    print(f"   Performed galactic evolution in {time.time() - start:1.2f} seconds")

    start = time.time()
    underworld.save(join(simulation_folder, f"{simulation_name}{file_suffix}"), overwrite=True)
    print(f"   Saved underworld population in {time.time() - start:1.2f} seconds")

    print(f"   Number of underworld binaries: {len(underworld)}")

    print("Underworld simulation complete!")
    print(f"Total time: {time.time() - very_start:1.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Run Underworld simulation with cogsworth")
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

    evolve_the_galaxy(
        output_dir=args.output_dir,
        processes=args.processes,
        simulation_name=args.simulation_name,
        file_suffix=args.file_suffix,
    )


if __name__ == "__main__":
    main()
