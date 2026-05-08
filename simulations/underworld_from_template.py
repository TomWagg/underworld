import cogsworth
import gala.potential as gp
import argparse
from os.path import join, exists
import pathlib
import h5py as h5

import sys
sys.path.append("../src")

import helpers
import time

POSTPROCESS_FOLDER = "/mnt/ceph/users/twagg/underworld/postprocessed/subfiles"


def vary_the_underworld(output_dir, processes, simulation_name, file_suffix,
                        params_to_vary={}, reset_kicks=True,
                        template_path="/mnt/ceph/users/twagg/underworld/sims/template/",
                        run_as_singles=False):
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
    params_to_vary : dict
        Dictionary of parameters to vary in the initial conditions.
    reset_kicks : bool
        Whether to reset supernova kicks to default values.
    template_path : str
        Path to the template population files folder.
    run_as_singles : bool
        Whether to treat all binaries as singles by giving them a very large initial separation.
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
    print(f"   Parameters to vary: {params_to_vary}")
    print(f"   Reset kicks: {reset_kicks}")
    print(f"   Template path: {template_path}")

    very_start = time.time()

    # read the template population
    print("Reading the template population")
    start = time.time()
    initial_pop = cogsworth.pop.load(join(template_path, f"template{file_suffix}"))
    print(f"   Loaded template population in {time.time() - start:1.2f} seconds")

    # delete the bpp, update initC columns as needed
    initial_pop._bpp = None
    initial_pop._kick_info = None
    initial_pop.error_file_path = None
    initial_pop.SSE_settings = {}
    initial_pop.BSE_settings = {}

    for param, value in params_to_vary.items():
        if param in initial_pop.initC.columns:
            print(f"Updating initial condition parameter: {param} to {value}")
            initial_pop.initC[param] = value
        else:
            raise ValueError(f"Parameter {param} not found in initial conditions columns!")

    if reset_kicks:
        for col in ["natal_kick_1", "phi_1", "theta_1", "natal_kick_2", "phi_2", "theta_2",
                    "mean_anomaly_1", "mean_anomaly_2"]:
            initial_pop.initC[col] = -100.0

        # drop randomseed column if it exists
        if "randomseed" in initial_pop.initC.columns:
            initial_pop.initC.drop(columns=["randomseed"], inplace=True)

    initial_pop.processes = processes
    initial_pop.galactic_potential = gp.MilkyWayPotential(version='v2')

    if run_as_singles:
        # give each binary an extremely large initial separation to ensure they are treated as singles
        initial_pop.initC['porb'] = 1e20

    # perform stellar evolution for binaries
    start = time.time()
    initial_pop.perform_stellar_evolution()
    print(f"   Performed stellar evolution in {time.time() - start:1.2f} seconds")

    # if simulation_name == "beta_0.5":
    #     print("Debugging beta now")
    #     initial_pop.initC.to_hdf("/mnt/ceph/users/twagg/underworld/problems.h5", key="initC")
    #     initial_pop.bpp.to_hdf("/mnt/ceph/users/twagg/underworld/problems.h5", key="bpp")
    #     initial_pop.kick_info.to_hdf("/mnt/ceph/users/twagg/underworld/problems.h5", key="kick_info")

    # do galactic evolution only for the binaries that end up as underworld objects
    start = time.time()
    underworld_mask = ((initial_pop.final_bpp['kstar_1'] == 13) | (initial_pop.final_bpp['kstar_1'] == 14) |
                       (initial_pop.final_bpp['kstar_2'] == 13) | (initial_pop.final_bpp['kstar_2'] == 14))
    underworld = initial_pop[underworld_mask]
    underworld.perform_galactic_evolution()
    print(f"   Performed galactic evolution in {time.time() - start:1.2f} seconds")

    start = time.time()
    underworld.save(join(simulation_folder, f"{simulation_name}{file_suffix}"), overwrite=True)
    print(f"   Saved underworld population in {time.time() - start:1.2f} seconds")

    print(f"   Number of underworld binaries: {len(underworld)}")

    # mask down to just the bound systems with 1 NS or BH and a star
    co_plus_star_mask = (
        (underworld.final_bpp['sep'] > 0) &
        ((underworld.final_bpp['kstar_1'].isin([13, 14])) & (underworld.final_bpp['kstar_2'] < 10)) |
        ((underworld.final_bpp['kstar_2'].isin([13, 14])) & (underworld.final_bpp['kstar_1'] < 10))
    )

    if co_plus_star_mask.sum() == 0:
        with h5.File(join(simulation_folder, f"{simulation_name}_co_plus_star{file_suffix}.h5"), 'w') as f:
            f.attrs["status"] = 404     # no binaries found hehe
        print(f"   No co+star binaries found. Saved empty file in {time.time() - start:1.2f} seconds")
    else:
        co_plus_star = underworld[co_plus_star_mask]
        co_plus_star.save(join(simulation_folder, f"{simulation_name}_co_plus_star{file_suffix}"), overwrite=True)
        print(f"   Saved co+star population in {time.time() - start:1.2f} seconds")
        print(f"   Number of co+star binaries: {len(co_plus_star)}")

    underworld.label = simulation_name

    # postprocess the file and save it
    kinematics, masses, bin_nums, sep, primary, companion = helpers.postprocess_populations(underworld)

    # save the postprocessed part
    helpers.save_postprocessed_data(
        [underworld], [join(POSTPROCESS_FOLDER, f"{simulation_name}{file_suffix}.h5")],
        kinematics, masses, bin_nums, sep, primary, companion
    )

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
    parser.add_argument(
        '--reset-kicks',
        action='store_true',
        help='Reset supernova kicks to default values (default: False)'
    )

    # add optional for a variable number of arguments to vary initial conditions (pairs of key=value)
    parser.add_argument(
        '--vary-params',
        nargs='*',
        help='Parameters to vary in the initial conditions as key:value pairs (e.g., --vary-params param1:val1 param2:val2)'
    )

    parser.add_argument(
        '--run-as-singles',
        action='store_true',
        help='Treat all binaries as singles by giving them a very large initial separation (default: False)',
        default=False
    )

    args = parser.parse_args()

    params_to_vary = {}
    if args.vary_params:
        for param_pair in args.vary_params:
            key, value = param_pair.split(':')
            try:
                # try to convert to float or int
                if '.' not in value:
                    value = int(value)
                else:
                    value = float(value)
            except ValueError:
                pass  # keep as string if conversion fails
            params_to_vary[key] = value

    vary_the_underworld(
        output_dir=args.output_dir,
        processes=args.processes,
        simulation_name=args.simulation_name,
        file_suffix=args.file_suffix,
        params_to_vary=params_to_vary,
        reset_kicks=args.reset_kicks,
        run_as_singles=args.run_as_singles,
    )


if __name__ == "__main__":
    main()
