import cogsworth
import gala.potential as gp
import argparse
from os.path import join, exists
import pathlib
import astropy.units as u

import sys
sys.path.append("../src")

import helpers
import time

POSTPROCESS_FOLDER = "/mnt/ceph/users/twagg/underworld/postprocessed/subfiles"


def enter_the_underworld(n_binaries, output_dir, processes, simulation_name, file_suffix,
                         high_mass_slope=None, porb_model="sana12", q_power_law=0,
                         qmin=0.0, m2_min=0.08, save_template=True):
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
    high_mass_slope : float, optional
        High mass slope for the initial mass function (default: None, which uses cogsworth's default).
    porb_model : str or dict, optional
        Binary orbital period model (default: "sana12"). Can also be a dict with keys "min", "max", and "slope" for a custom power-law distribution.
    q_power_law : float, optional
        Power-law slope for the binary mass ratio distribution (default: 0, which is flat in q).
    qmin : float, optional
        Minimum mass ratio for the binary mass ratio distribution (default: 0.0).
    m2_min : float, optional
        Minimum mass for the secondary star (default: 0.08 Msun).
    save_template : bool, optional
        Whether to save the template population (default: True).
    """
    # quickly check if output directory exists
    if not exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist!")

    very_start = time.time()

    print("Initiating cogsworth underworld simulation")

    sampling_params = {"porb_model": porb_model, "q_power_law": q_power_law, "qmin": qmin, "m2_min": m2_min}

    if high_mass_slope is not None:
        sampling_params["primary_model"] = "custom"
        sampling_params["alphas"] = [-1.3, -2.3, high_mass_slope]
        sampling_params["mcuts"] = [0.08, 0.5, 1.0, 150.]

    bpp_columns = [
        'tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 'sep', 'porb', 'ecc',
        'evol_type', 'RRLO_1', 'RRLO_2', 'massc_he_layer_1', 'massc_co_layer_1',
        'massc_he_layer_2', 'massc_co_layer_2',
        'lum_1', 'lum_2', 'teff_1', 'teff_2', 'SN_1', 'SN_2', 'omega_spin_1', 'omega_spin_2',
        'rad_1', 'rad_2'
    ]

    # create a new cogsworth population that assumes 100% binarity
    pot = gp.MilkyWayPotential(version='v2')
    initial_pop = cogsworth.pop.Population(
        n_binaries=n_binaries,
        processes=processes,
        ini_file="/mnt/home/twagg/projects/underworld/simulations/params.ini",
        galactic_potential=pot,
        sfh_model=cogsworth.sfh.SandersBinney2015,
        sfh_params={
            "potential": pot,
            "time_bins": 5,
            "verbose": True
        },
        bpp_columns=bpp_columns,
        store_entire_orbits=False,
        error_file_path=None,
        v_dispersion=0 * u.km / u.s,
        sampling_mask="mass_1 > 4.0",
        sampling_params=sampling_params
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

    if save_template:
        print("Save the template population")
        start = time.time()

        template_folder = join(output_dir, "template")
        if not exists(template_folder):
            pathlib.Path(template_folder).mkdir(parents=True, exist_ok=True)

        initial_pop.save(join(template_folder, f"template{file_suffix}"), overwrite=True)
        print(f"   Saved template population in {time.time() - start:1.2f} seconds")

    # do galactic evolution only for the binaries that end up as underworld objects
    start = time.time()
    underworld_mask = ((initial_pop.final_bpp['kstar_1'] == 13) | (initial_pop.final_bpp['kstar_1'] == 14) |
                       (initial_pop.final_bpp['kstar_2'] == 13) | (initial_pop.final_bpp['kstar_2'] == 14))
    underworld = initial_pop[underworld_mask]
    underworld.perform_galactic_evolution()
    print(f"   Performed galactic evolution for binaries in {time.time() - start:1.2f} seconds")

    simulation_folder = join(output_dir, simulation_name)
    if not exists(simulation_folder):
        pathlib.Path(simulation_folder).mkdir(parents=True, exist_ok=True) 

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
    parser.add_argument('-P', '--porb-model', default="sana12", type=str,
                        help='Binary orbital period model')
    parser.add_argument('-M', '--porb-max', default=None, type=float,
                        help='Maximum log10 porb')
    parser.add_argument('-q', '--q-power-law', default=0, type=float,
                        help='Binary mass ratio power law')
    parser.add_argument('-m', '--high-mass-slope', default=None, type=float,
                        help='High mass slope')
    parser.add_argument('--qmin', default=0.0, type=float,
                        help='Minimum mass ratio for the binary mass ratio distribution (default: 0.0)')
    parser.add_argument('--m2-min', default=0.08, type=float,
                        help='Minimum mass for the secondary star (default: 0.08 Msun)')
    parser.add_argument('--no-template', dest='save_template', action='store_false',
                        help='Whether to save the template population (default: True)')
    args = parser.parse_args()

    if args.m2_min < 0:
        args.m2_min = None

    # check if args.porb_model is a number and convert to dict if so
    try:
        args.porb_model = {
            "min": 0.15,
            "max": 5.5 if args.porb_max is None else args.porb_max,
            "slope": float(args.porb_model)
        }
    except ValueError:
        pass

    enter_the_underworld(
        n_binaries=args.n_binaries,
        output_dir=args.output_dir,
        processes=args.processes,
        simulation_name=args.simulation_name,
        file_suffix=args.file_suffix,
        high_mass_slope=args.high_mass_slope,
        porb_model=args.porb_model,
        q_power_law=args.q_power_law,
        qmin=args.qmin,
        m2_min=args.m2_min,
        save_template=args.save_template
    )


if __name__ == "__main__":
    main()
