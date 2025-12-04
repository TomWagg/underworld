import numpy as np
import astropy.units as u


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
                print(f"  {label}:{' ' * (9 - len(label))} {len(table):.0f}  \t{len(table) * scale_up:.1e} (scaled)")
                if label == "BH-Star":
                    print()
            print()
    return underworld_binaries
