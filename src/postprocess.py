import helpers
import os
import numpy as np

files = ['fiducial', 'beta_0.0', 'beta_1.0',
         'alpha_0.5', 'alpha_2.0', 'alpha_10.0',
         'qcrit_caseB_1000',
         'fryer_rapid', 'mandel_muller', 'maltsev_fallback_0.0', 'maltsev_fallback_0.25',
         'maltsev_fallback_0.5', 'maltsev_fallback_0.75', 'maltsev_fallback_1.0',
         'maltsev_pf_prob_0.0', 'maltsev_pf_prob_1.0',
         'bhflag_3', 'kickflag_1']
deleted = ['alpha_0.1', 'beta_0.5', 'qcrit_caseB_0.001']

BASE_PATH = "/mnt/ceph/users/twagg/underworld/"
postprocess_folder = os.path.join(BASE_PATH, "postprocessed")
N_PARTS = 6

for file in files:
    print("-------------------------\n\n")
    postprocess_file = os.path.join(postprocess_folder, f"{file}_processed.h5")
    if os.path.exists(postprocess_file):
        print(f"File {file}_processed.h5 already exists, skipping...")
        continue

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

    print("Saving processed data...")

    helpers.save_postprocessed_data(pops, [postprocess_file], kinematics, masses, bin_nums)
