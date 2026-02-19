import numpy as np
import astropy.units as u
import cogsworth
import warnings
import logging
import h5py as h5
from os.path import join


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
        primary_pos = pop.final_pos[:len(pop)]
        secondary_pos = pop.final_pos[:len(pop)].copy()
        secondary_pos[pop.disrupted] = pop.final_pos[len(pop):]

        ns_pos = np.concatenate((primary_pos[pop.final_bpp["kstar_1"] == 13],
                                secondary_pos[pop.final_bpp["kstar_2"] == 13]))
        bh_pos = np.concatenate((primary_pos[pop.final_bpp["kstar_1"] == 14],
                                secondary_pos[pop.final_bpp["kstar_2"] == 14]))
        co_pos = np.concatenate((ns_pos, bh_pos))

        kinematics[pop.label]["pos"] = {
            "NS": ns_pos,
            "BH": bh_pos,
            "CO": co_pos,
        }

        primary_vel = pop.final_vel[:len(pop)]
        secondary_vel = pop.final_vel[:len(pop)].copy()
        secondary_vel[pop.disrupted] = pop.final_vel[len(pop):]

        ns_vel = np.concatenate((primary_vel[pop.final_bpp["kstar_1"] == 13],
                                secondary_vel[pop.final_bpp["kstar_2"] == 13]))
        bh_vel = np.concatenate((primary_vel[pop.final_bpp["kstar_1"] == 14],
                                secondary_vel[pop.final_bpp["kstar_2"] == 14]))
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


def save_postprocessed_data(pops, files, kinematics, masses, bin_nums):
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
    """
    for pop, file in zip(pops, files):
        with h5.File(f"{file}_processed.h5", "w") as f:
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


class DummyPop:
    def __init__(self, label, colour):
        self.label = label
        self.colour = colour


def load_postprocessed_data(files, labels):
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
        data = {}
        with h5.File(f"{file}_processed.h5", "r") as f:
            data["mass_binaries"] = f.attrs["mass_binaries"]
            data["mass_singles"] = f.attrs["mass_singles"]
            for key in f["BH"].keys():
                data[key] = {}
                for comp in ["NS", "BH"]:
                    data[key][comp] = f[f"{comp}/{key}"][:]
                data[key]["CO"] = np.concatenate((data[key]["NS"], data[key]["BH"]))
        data_dict[label] = data
    return data_dict


def load_postprocessed_pops(files, labels, colours):
    pops = [DummyPop(label, colour) for label, colour in zip(labels, colours)]
    data = load_postprocessed_data(files, labels)
    return pops, data
