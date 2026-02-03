import cogsworth
import gala.potential as gp
import argparse
from os.path import join, exists

import time


def vary_the_underworld(output_dir, processes, simulation_name, file_suffix,
                        params_to_vary={}, reset_kicks=False):
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
    """
    # quickly check if output directory exists
    if not exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist!")

    print("Starting Underworld simulation with these parameters:")
    print(f"   Output directory: {output_dir}")
    print(f"   Number of processes: {processes}")
    print(f"   Simulation name: {simulation_name}")
    print(f"   File suffix: {file_suffix}")
    print(f"   Parameters to vary: {params_to_vary}")
    print(f"   Reset kicks: {reset_kicks}")

    very_start = time.time()

    # read the template population
    print("Reading the template population")
    start = time.time()
    initial_pop = cogsworth.pop.load(join(output_dir, f"template/template{file_suffix}"))
    print(f"   Loaded template population in {time.time() - start:1.2f} seconds")

    # delete the bpp, update initC columns as needed
    initial_pop._bpp = None
    initial_pop._kick_info = None
    initial_pop.error_file_path = "./"

    defaults = {'mm_mu_ns': 400.0, 'mm_mu_bh': 200.0,
                'maltsev_mode': 0, 'maltsev_fallback': 0.5, 'maltsev_pf_prob': 0.1}
    for param, value in defaults.items():
        if param not in initial_pop.initC.columns:
            print(f"Setting default initial condition parameter: {param} to {value}")
            initial_pop.initC[param] = value

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

    if "massc_1" in initial_pop.bpp_columns:
        # swap to mass_co_layer_1, mass_co_layer_2, mass_he_layer_1, mass_he_layer_2
        initial_pop.bpp_columns.remove("massc_1")
        initial_pop.bpp_columns.remove("massc_2")
        initial_pop.bpp_columns += ["massc_co_layer_1", "massc_co_layer_2",
                                    "massc_he_layer_1", "massc_he_layer_2"]

    initial_pop.processes = processes
    initial_pop.galactic_potential = gp.MilkyWayPotential(version='v2')

    # perform stellar evolution for binaries
    start = time.time()
    initial_pop.perform_stellar_evolution()
    print(f"   Performed stellar evolution in {time.time() - start:1.2f} seconds")

    # do galactic evolution only for the binaries that end up as underworld objects
    start = time.time()
    underworld_mask = ((initial_pop.final_bpp['kstar_1'] == 13) | (initial_pop.final_bpp['kstar_1'] == 14) |
                       (initial_pop.final_bpp['kstar_2'] == 13) | (initial_pop.final_bpp['kstar_2'] == 14))
    underworld = initial_pop[underworld_mask]
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
        reset_kicks=args.reset_kicks
    )


if __name__ == "__main__":
    main()
