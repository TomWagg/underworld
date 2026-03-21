import helpers
import os
import numpy as np
import time

do_now = [
    'fiducial_m2min'
]
do_later = [
    'time-evolving-pot'
]
done = [
    'fiducial', 'singles', 'bhflag_0', 'bhflag_3', 'fryer_rapid', 'kickflag_1', 'mandel_muller',
    'maltsev_fallback_0.0', 'maltsev_fallback_0.25', 'maltsev_fallback_0.5', 'maltsev_fallback_0.75',
    'maltsev_fallback_1.0', 'maltsev_pf_prob_0.0', 'maltsev_pf_prob_1.0',
    'beta_0.0', 'beta_0.5', 'beta_1.0',
    'alpha_0.1', 'alpha_0.5', 'alpha_2.0', 'alpha_10.0',
    'qcrit_caseB_0.001', 'qcrit_caseB_1000',
]

files = do_now

BASE_PATH = "/mnt/ceph/users/twagg/underworld/"
postprocess_folder = os.path.join(BASE_PATH, "postprocessed")
N_PARTS = 34

for file in files:
    print("-------------------------\n\n")
    postprocess_file = os.path.join(postprocess_folder, f"{file}_processed.h5")
    # if os.path.exists(postprocess_file):
    #     print(f"File {file}_processed.h5 already exists, skipping...")
    #     continue

    start = time()
    print("Processing file:", file)

    pops = [helpers.load_distributed_pop("/mnt/ceph/users/twagg/underworld",
                                         file, N_PARTS, label=file, colour="")]

    print("Calculating kinematics and bin numbers...")

    kinematics = helpers.get_kinematics(pops)
    bin_nums = helpers.get_shaped_bin_nums(pops)

    print("Calculating masses...")

    masses = {}
    for pop in pops:
        masses[pop.label] = {
            "BH": np.concatenate((
                pop.final_bpp["mass_1"][pop.final_bpp["kstar_1"] == 14],
                pop.final_bpp["mass_2"][pop.final_bpp["kstar_2"] == 14],
            )),
            "NS": np.concatenate((
                pop.final_bpp["mass_1"][pop.final_bpp["kstar_1"] == 13],
                pop.final_bpp["mass_2"][pop.final_bpp["kstar_2"] == 13],
            )),
        }
        masses[pop.label]["CO"] = np.concatenate((masses[pop.label]["NS"], masses[pop.label]["BH"]))

    print("Recording binarity")

    sep = {}
    primary = {}
    companion = {}
    for pop in pops:
        sep[pop.label] = {
            "BH": np.concatenate((
                pop.final_bpp["sep"][pop.final_bpp["kstar_1"] == 14],
                pop.final_bpp["sep"][pop.final_bpp["kstar_2"] == 14],
            )),
            "NS": np.concatenate((
                pop.final_bpp["sep"][pop.final_bpp["kstar_1"] == 13],
                pop.final_bpp["sep"][pop.final_bpp["kstar_2"] == 13],
            )),
        }
        sep[pop.label]["CO"] = np.concatenate((sep[pop.label]["NS"], sep[pop.label]["BH"]))

        primary[pop.label] = {
            "BH": np.concatenate((
                np.repeat(True, len(pop.final_bpp["kstar_1"][pop.final_bpp["kstar_1"] == 14])),
                np.repeat(False, len(pop.final_bpp["kstar_2"][pop.final_bpp["kstar_2"] == 14])),
            )),
            "NS": np.concatenate((
                np.repeat(True, len(pop.final_bpp["kstar_1"][pop.final_bpp["kstar_1"] == 13])),
                np.repeat(False, len(pop.final_bpp["kstar_2"][pop.final_bpp["kstar_2"] == 13])),
            )),
        }
        primary[pop.label]["CO"] = np.concatenate((primary[pop.label]["NS"], primary[pop.label]["BH"]))

        companion[pop.label] = {
            "BH": np.concatenate((
                pop.final_bpp["kstar_2"][pop.final_bpp["kstar_1"] == 14],
                pop.final_bpp["kstar_1"][pop.final_bpp["kstar_2"] == 14],
            )),
            "NS": np.concatenate((
                pop.final_bpp["kstar_2"][pop.final_bpp["kstar_1"] == 13],
                pop.final_bpp["kstar_1"][pop.final_bpp["kstar_2"] == 13],
            )),
        }
        companion[pop.label]["CO"] = np.concatenate((companion[pop.label]["NS"], companion[pop.label]["BH"]))

    print("Saving processed data...")

    helpers.save_postprocessed_data(
        pops, [postprocess_file], kinematics, masses, bin_nums, sep, primary, companion
    )
    print(f"Finished processing {file} in {time() - start:.2f} seconds.")

