import cogsworth
import gala.potential as gp
import argparse
from os.path import join, exists
import numpy as np
import astropy.units as u

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

    print("Starting Underworld simulation with these parameters:")
    print(f"   Output directory: {output_dir}")
    print(f"   Number of processes: {processes}")
    print(f"   Simulation name: {simulation_name}")
    print(f"   File suffix: {file_suffix}")
    print("       EVOLVING the potential")

    time_knots = np.linspace(0, 12, 20) * u.Gyr

    # use y = mx + c
    # t < 3 Gyr: linear from (0, 0.45) to (3, 0.88)
    # t >= 3 Gyr: linear from (3, 0.88) to (12, 1.0)
    mass_at_knot = np.zeros_like(time_knots.value) * u.Msun
    for i, t in enumerate(time_knots.value):
        if t < 3:
            mass_at_knot[i] = (0.45 + (0.88 - 0.45) / 3 * t) * u.Msun
        else:
            mass_at_knot[i] = (0.88 + (1.0 - 0.88) / (12 - 3) * (t - 3)) * u.Msun
    mass_at_knot *= 5.54e11

    evolving_mw_pot = gp.CCompositePotential(
        disk=gp.MN3ExponentialDiskPotential(
            m=4.77e10 * u.Msun,
            h_R=2.6 * u.kpc,
            h_z=0.3 * u.kpc,
            units="galactic"
        ),
        bulge=gp.HernquistPotential(
            m=5e9 * u.Msun,
            c=1.0 * u.kpc,
            units="galactic"
        ),
        nucleus=gp.HernquistPotential(
            m=1.81e9 * u.Msun,
            c=0.07 * u.kpc,
            units="galactic"
        ),
        halo=gp.TimeInterpolatedPotential(
            gp.NFWPotential,
            time_knots,
            m=mass_at_knot,
            r_s=15.63 * u.kpc,
            units="galactic"
        )
    )

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
    underworld.save(join(output_dir, f"{simulation_name}{file_suffix}"), overwrite=True)
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
