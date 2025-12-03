import numpy as np


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
    return kinematics
