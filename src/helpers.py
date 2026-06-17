import numpy as np
import astropy.units as u
import cogsworth
import warnings
import logging
import h5py as h5
from os.path import join
import pandas as pd


def load_distributed_pop(base_path, sim_name, parts, label=None, colour=None):
    if isinstance(parts, int):
        parts = list(range(parts))
    n_parts = len(parts)
    print(f"Loading distributed population{' ' + label if label is not None else ''} from {n_parts} parts")
    pops = [None for _ in parts]
    for i, part in enumerate(parts):
        logging.getLogger("cogsworth").setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pops[i] = cogsworth.pop.load(join(base_path, sim_name, f"{sim_name}_part{part:d}"))
        logging.getLogger("cogsworth").setLevel(logging.WARNING)

        pops[i].initial_binaries
        pops[i].initial_galaxy
        pops[i].initC
        pops[i].final_bpp
        pops[i].bin_nums
        pops[i].final_pos
        pops[i].final_vel
        pops[i]._file = None

        extra = f" of {label} " if label is not None else " "
        print(f"   loaded part {i+1}/{n_parts}{extra}with {len(pops[i])} binaries")

    pop = cogsworth.pop.concat(*pops)

    pop.label = label
    pop.colour = colour
    pop.bpp["row_num"] = np.arange(len(pop.bpp))

    return pop


def get_shaped_bin_nums(pops):
    bin_nums = {}
    for pop in pops:
        ns_bin_nums = np.concatenate((
            pop.final_bpp[pop.final_bpp["kstar_1"] == 13]["bin_num"].values,
            pop.final_bpp[pop.final_bpp["kstar_2"] == 13]["bin_num"].values
        ))
        bh_bin_nums = np.concatenate((
            pop.final_bpp[pop.final_bpp["kstar_1"] == 14]["bin_num"].values,
            pop.final_bpp[pop.final_bpp["kstar_2"] == 14]["bin_num"].values
        ))
        co_bin_nums = np.concatenate((ns_bin_nums, bh_bin_nums))
        bin_nums[pop.label] = {
            "NS": ns_bin_nums,
            "BH": bh_bin_nums,
            "CO": co_bin_nums,
        }
    return bin_nums


def get_kinematics(pops):
    kinematics = {}

    for pop in pops:
        kinematics[pop.label] = {}

        ns_pos = np.concatenate((pop.final_primary_pos[pop.final_bpp["kstar_1"] == 13],
                                pop.final_secondary_pos[pop.final_bpp["kstar_2"] == 13]))
        bh_pos = np.concatenate((pop.final_primary_pos[pop.final_bpp["kstar_1"] == 14],
                                 pop.final_secondary_pos[pop.final_bpp["kstar_2"] == 14]))
        co_pos = np.concatenate((ns_pos, bh_pos))

        kinematics[pop.label]["pos"] = {
            "NS": ns_pos,
            "BH": bh_pos,
            "CO": co_pos,
        }

        ns_vel = np.concatenate((pop.final_primary_vel[pop.final_bpp["kstar_1"] == 13],
                                pop.final_secondary_vel[pop.final_bpp["kstar_2"] == 13]))
        bh_vel = np.concatenate((pop.final_primary_vel[pop.final_bpp["kstar_1"] == 14],
                                pop.final_secondary_vel[pop.final_bpp["kstar_2"] == 14]))
        co_vel = np.concatenate((ns_vel, bh_vel))

        kinematics[pop.label]["vel"] = {
            "NS": ns_vel,
            "BH": bh_vel,
            "CO": co_vel,
        }

        kinematics[pop.label]["escaped"] = {}
        for co_type, pos, vel in zip(
            ["NS", "BH", "CO"],
            [ns_pos, bh_pos, co_pos],
            [ns_vel, bh_vel, co_vel],
        ):
            V = np.linalg.norm(vel.to(u.km/u.s).value, axis=1)
            escaped = V >= np.sqrt(-2 * pop.galactic_potential(pos.T)).to(u.km / u.s).value

            kinematics[pop.label]["escaped"][co_type] = escaped

    return kinematics


def get_average_mass_at_z(abs_zs, bh_masses, z_range=np.geomspace(0.1, 10, 1000), window_width=0.1):
    """Calculate the average black hole mass as a function of absolute distance from the Galactic plane.

    Parameters
    ----------
    abs_zs : list of np.ndarray
        List of arrays containing the absolute z distances of black holes for each population.
    bh_masses : list of np.ndarray
        List of arrays containing the black hole masses corresponding to abs_zs.
    z_range : np.ndarray, optional
        Array of z values at which to calculate the average mass, by default np.geomspace(0.1, 10, 1000)
    window_width : float, optional
        Width of the window around each z value to consider for averaging, by default 0.1

    Returns
    -------
    mean_masses : np.ndarray
        2D array of mean masses with shape (len(abs_zs), len(z_range)).
    """
    mean_masses = np.zeros((len(abs_zs), len(z_range)))
    half_window = window_width / 2
    for i, z_centre in enumerate(z_range):
        for j in range(len(abs_zs)):
            in_window = (abs_zs[j] >= (z_centre - half_window)) & (abs_zs[j] < (z_centre + half_window))
            mean_masses[j, i] = np.mean(bh_masses[j][in_window]) if np.sum(in_window) > 0 else np.nan

    return mean_masses


def get_underworld_binaries(pops, verbose=False):
    co_binary_labels = ["BH-BH", "BH-NS", "BH-WD", "BH-Star", "NS-NS", "NS-WD", "NS-Star"]
    co_binary_kstar_groups = [
        ([14], [14]),
        ([14], [13]),
        ([14], [10, 11, 12]),
        ([14], list(range(0, 10))),
        ([13], [13]),
        ([13], [10, 11, 12]),
        ([13], list(range(0, 10))),
    ]

    underworld_binaries = {}
    for pop in pops:
        underworld_binaries[pop.label] = {}
        co_binaries = pop.final_bpp[
            ((pop.final_bpp["kstar_1"].isin([13, 14])) | (pop.final_bpp["kstar_2"].isin([13, 14]))) &
            (pop.final_bpp["sep"] > 0)
        ]

        underworld_binaries[pop.label] = {
            label: co_binaries[
                ((co_binaries["kstar_1"].isin(kstar_group[0]))
                 & (co_binaries["kstar_2"].isin(kstar_group[1]))) |
                ((co_binaries["kstar_1"].isin(kstar_group[1]))
                 & (co_binaries["kstar_2"].isin(kstar_group[0])))
            ] for (label, kstar_group) in zip(co_binary_labels, co_binary_kstar_groups)
        }

        scale_up = 6e10 / pop.mass_binaries

        if verbose:
            print(f"{pop.label} Underworld Binaries (scale up by {scale_up:.0f}x):")
            for label, table in underworld_binaries[pop.label].items():
                avg_bh_mass = np.mean(np.concatenate((
                    table["mass_1"][table["kstar_1"] == 14],
                    table["mass_2"][table["kstar_2"] == 14],
                ))) if (table["kstar_1"] == 14).any() or (table["kstar_2"] == 14).any() else 0
                avg_ns_mass = np.mean(np.concatenate((
                    table["mass_1"][table["kstar_1"] == 13],
                    table["mass_2"][table["kstar_2"] == 13],
                ))) if (table["kstar_1"] == 13).any() or (table["kstar_2"] == 13).any() else 0

                avg_sep = np.mean(table["sep"])
                print(f"  {label}:{' ' * (9 - len(label))} {len(table):.0f}  \t{len(table) * scale_up:.1e} (scaled)\t{avg_bh_mass:.1f} Msun \t{avg_ns_mass:.1f} Msun \t {avg_sep:.1e} Rsun")
                if label == "BH-Star":
                    print()
            print()
    return underworld_binaries


def postprocess_populations(*pops):
    kinematics = get_kinematics(pops)
    bin_nums = get_shaped_bin_nums(pops)

    masses = {}
    sep = {}
    primary = {}
    companion = {}
    for pop in pops:
        primary_bh = pop.final_bpp["kstar_1"] == 14
        secondary_bh = pop.final_bpp["kstar_2"] == 14
        primary_ns = pop.final_bpp["kstar_1"] == 13
        secondary_ns = pop.final_bpp["kstar_2"] == 13

        final_bpp_primary_bh = pop.final_bpp[primary_bh]
        final_bpp_secondary_bh = pop.final_bpp[secondary_bh]
        final_bpp_primary_ns = pop.final_bpp[primary_ns]
        final_bpp_secondary_ns = pop.final_bpp[secondary_ns]

        masses[pop.label] = {
            "BH": np.concatenate((final_bpp_primary_bh["mass_1"], final_bpp_secondary_bh["mass_2"])),
            "NS": np.concatenate((final_bpp_primary_ns["mass_1"], final_bpp_secondary_ns["mass_2"]))
        }
        masses[pop.label]["CO"] = np.concatenate((masses[pop.label]["NS"], masses[pop.label]["BH"]))

        sep[pop.label] = {
            "BH": np.concatenate((final_bpp_primary_bh["sep"], final_bpp_secondary_bh["sep"])),
            "NS": np.concatenate((final_bpp_primary_ns["sep"], final_bpp_secondary_ns["sep"])),
        }
        sep[pop.label]["CO"] = np.concatenate((sep[pop.label]["NS"], sep[pop.label]["BH"]))

        primary[pop.label] = {
            "BH": np.concatenate((
                np.repeat(True, len(final_bpp_primary_bh)),
                np.repeat(False, len(final_bpp_secondary_bh)),
            )),
            "NS": np.concatenate((
                np.repeat(True, len(final_bpp_primary_ns)),
                np.repeat(False, len(final_bpp_secondary_ns)),
            )),
        }
        primary[pop.label]["CO"] = np.concatenate((primary[pop.label]["NS"], primary[pop.label]["BH"]))

        companion[pop.label] = {
            "BH": np.concatenate((
                pop.final_bpp["kstar_2"][primary_bh],
                pop.final_bpp["kstar_1"][secondary_bh],
            )),
            "NS": np.concatenate((
                pop.final_bpp["kstar_2"][primary_ns],
                pop.final_bpp["kstar_1"][secondary_ns],
            )),
        }
        companion[pop.label]["CO"] = np.concatenate((companion[pop.label]["NS"], companion[pop.label]["BH"]))

    return kinematics, masses, bin_nums, sep, primary, companion


def save_postprocessed_data(pops, files, kinematics, masses, bin_nums, sep, primary, companion):
    """Save the post-processed data to an HDF5 file.

    Parameters
    ----------
    pops : cogsworth.Pop
        The population objects containing the initial galaxy information.
    files : str
        The base filenames to save the processed data to (without extension).
    kinematics : dict
        Dictionary containing the kinematic data for each population and component.
    masses : dict
        Dictionary containing the mass data for each population and component.
    bin_nums : dict
        Dictionary containing the binary numbers for each population and component.
    sep : dict
        Dictionary containing the separations for each population and component.
    primary : dict
        Dictionary containing a mask of whether the CO is the primary in its binary
    companion : dict
        Dictionary containing the companion type for each CO
    """
    for pop, file in zip(pops, files):
        with h5.File(file, "w") as f:
            f.attrs["mass_binaries"] = pop.mass_binaries
            f.attrs["mass_singles"] = pop.mass_singles
            for comp in ["NS", "BH"]:
                f.create_dataset(f"{comp}/pos", data=kinematics[pop.label]["pos"][comp])
                f.create_dataset(f"{comp}/vel", data=kinematics[pop.label]["vel"][comp])
                f.create_dataset(f"{comp}/escaped", data=kinematics[pop.label]["escaped"][comp])
                f.create_dataset(f"{comp}/bin_nums", data=bin_nums[pop.label][comp])
                f.create_dataset(f"{comp}/mass", data=masses[pop.label][comp])
                f.create_dataset(
                    f"{comp}/tau",
                    data=pop.initial_galaxy.tau[np.searchsorted(pop.bin_nums, bin_nums[pop.label][comp])]
                )
                f.create_dataset(
                    f"{comp}/init_z",
                    data=pop.initial_galaxy.z[np.searchsorted(pop.bin_nums, bin_nums[pop.label][comp])]
                )
                f.create_dataset(f"{comp}/sep", data=sep[pop.label][comp])
                f.create_dataset(f"{comp}/primary", data=primary[pop.label][comp])
                f.create_dataset(f"{comp}/companion", data=companion[pop.label][comp])


def save_postprocessed_data_one_dict(data, file):
    """Save the post-processed data to an HDF5 file."""
    with h5.File(file, "w") as f:
        f.attrs["mass_binaries"] = data["mass_binaries"]
        f.attrs["mass_singles"] = data["mass_singles"]
        for comp in ["NS", "BH"]:
            f.create_dataset(f"{comp}/pos", data=data["pos"][comp])
            f.create_dataset(f"{comp}/vel", data=data["vel"][comp])
            f.create_dataset(f"{comp}/escaped", data=data["escaped"][comp])
            f.create_dataset(f"{comp}/bin_nums", data=data["bin_nums"][comp])
            f.create_dataset(f"{comp}/mass", data=data["mass"][comp])
            f.create_dataset(f"{comp}/tau", data=data["tau"][comp])
            f.create_dataset(f"{comp}/init_z", data=data["init_z"][comp])
            f.create_dataset(f"{comp}/sep", data=data["sep"][comp])
            f.create_dataset(f"{comp}/primary", data=data["primary"][comp])
            f.create_dataset(f"{comp}/companion", data=data["companion"][comp])


class DummyPop:
    def __init__(self, label, colour):
        self.label = label
        self.colour = colour


def load_postprocessed_data(files, labels, folder="/mnt/ceph/users/twagg/underworld/postprocessed"):
    """Load the post-processed data from an HDF5 file.

    Parameters
    ----------
    file : str
        The base filename to load the processed data from (without extension).

    Returns
    -------
    dict
        Dictionary containing the loaded data for each component.
    """
    data_dict = {}
    for file, label in zip(files, labels):
        print(f"Loading post-processed data for {label} from {file}")
        data = {}
        path = join(folder, file)
        if not file.endswith(".h5"):
            path += ".h5"
        with h5.File(path, "r") as f:
            data["mass_binaries"] = f.attrs["mass_binaries"]
            data["mass_singles"] = f.attrs["mass_singles"]
            for key in f["BH"].keys():
                data[key] = {}
                for comp in ["NS", "BH"]:
                    data[key][comp] = f[f"{comp}/{key}"][:]
                data[key]["CO"] = np.concatenate((data[key]["NS"], data[key]["BH"]))
        data["n_BH"] = len(data["mass"]["BH"])
        data["n_NS"] = len(data["mass"]["NS"])
        data_dict[label] = data
    return data_dict


def load_postprocessed_pops(files, labels, colours, folder="/mnt/ceph/users/twagg/underworld/postprocessed"):
    pops = [DummyPop(label, colour) for label, colour in zip(labels, colours)]
    data = load_postprocessed_data(files, labels, folder=folder)
    return pops, data


def get_kick_stats(pop):
    """Extract the natal kick statistics for black holes in the population,
    including the natal kick magnitude, the change in systemic velocity, and whether the binary was disrupted.
    
    Parameters
    ----------
    pop : cogsworth.Population
        The population object containing the binary population synthesis data.  
    """
    # find primary BH formation rows
    primary_bh_rows = pop.bpp[pop.bpp["row_num"].isin(pop.bpp[pop.bpp["evol_type"] == 15]["row_num"] + 1) & (pop.bpp["kstar_1"] == 14)]
    primary_bh_explosion_bin_nums = primary_bh_rows["bin_num"].values

    # get the final BH mass and companion mass for the primary BHs
    primary_bh_mass = primary_bh_rows["mass_1"].values
    primary_companion_mass = primary_bh_rows["mass_2"].values

    # now look at pre-SN properties for the primary BHs, get pre-SN separation and progenitor mass
    primary_pre_sn = pop.bpp.loc[primary_bh_explosion_bin_nums][pop.bpp.loc[primary_bh_explosion_bin_nums, "kstar_1"] < 13].drop_duplicates(subset="bin_num", keep="last")
    primary_sep = primary_pre_sn["sep"].values
    primary_progenitor_mass = primary_pre_sn["mass_1"].values

    # repeat for secondary BHs
    secondary_bh_rows = pop.bpp[pop.bpp["row_num"].isin(pop.bpp[pop.bpp["evol_type"] == 16]["row_num"] + 1) & (pop.bpp["kstar_2"] == 14)]
    secondary_bh_explosion_bin_nums = secondary_bh_rows["bin_num"].values
    secondary_bh_mass = secondary_bh_rows["mass_2"].values
    secondary_companion_mass = secondary_bh_rows["mass_1"].values
    secondary_pre_sn = pop.bpp.loc[secondary_bh_explosion_bin_nums][pop.bpp.loc[secondary_bh_explosion_bin_nums, "kstar_2"] < 13].drop_duplicates(subset="bin_num", keep="last")
    secondary_sep = secondary_pre_sn["sep"].values
    secondary_progenitor_mass = secondary_pre_sn["mass_2"].values

    # now use the bin_nums to get the specific kick information for primaries
    primary_kick_rows = pop.kick_info.loc[primary_bh_explosion_bin_nums][pop.kick_info.loc[primary_bh_explosion_bin_nums]["star"] == 1]
    primary_kicks = primary_kick_rows["natal_kick"].values
    primary_deltav = np.linalg.norm(primary_kick_rows[["delta_vsysx_1", "delta_vsysy_1", "delta_vsysz_1"]], axis=1)
    primary_disrupted = primary_kick_rows["disrupted"].values

    # same for secondaries if they exist
    secondary_kicks = []
    secondary_deltav = []
    secondary_disrupted = []
    if len(secondary_bh_explosion_bin_nums) > 0:
        secondary_kick_rows = pop.kick_info.loc[secondary_bh_explosion_bin_nums][pop.kick_info.loc[secondary_bh_explosion_bin_nums]["star"] == 2]
        secondary_deltav = np.linalg.norm(secondary_kick_rows[["delta_vsysx_2", "delta_vsysy_2", "delta_vsysz_2"]], axis=1)
        secondary_kicks = secondary_kick_rows["natal_kick"].values
        secondary_disrupted = secondary_kick_rows["disrupted"].values

    is_primary = np.concatenate([np.repeat(True, len(primary_kicks)), np.repeat(False, len(secondary_kicks))])

    # combine primary and secondary stats into a single DataFrame
    kick_stat_dict = {}
    kick_stat_dict["natal_kick"] = np.concatenate([primary_kicks, secondary_kicks])
    kick_stat_dict["delta_v"] = np.concatenate([primary_deltav, secondary_deltav])
    kick_stat_dict["disrupted"] = np.concatenate([primary_disrupted, secondary_disrupted])
    kick_stat_dict["bh_mass"] = np.concatenate([primary_bh_mass, secondary_bh_mass])
    kick_stat_dict["companion_mass"] = np.concatenate([primary_companion_mass, secondary_companion_mass])
    kick_stat_dict["separation_at_sn"] = np.concatenate([primary_sep, secondary_sep])
    kick_stat_dict["progenitor_mass"] = np.concatenate([primary_progenitor_mass, secondary_progenitor_mass])
    kick_stat_dict["is_primary"] = is_primary
    kick_stat_dict["bin_num"] = np.concatenate([primary_bh_explosion_bin_nums, secondary_bh_explosion_bin_nums])

    kick_stat_df = pd.DataFrame(kick_stat_dict)
    return kick_stat_df
