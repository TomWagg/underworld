import helpers
import os
import numpy as np
from time import time
from os.path import join
import cogsworth

do_now = [
    "bhflag_0", 'bhflag_3', 'fryer_rapid', 'kickflag_1', 'mandel_muller',
    'maltsev_fallback_0.0', 'maltsev_fallback_0.25', 'maltsev_fallback_0.5', 'maltsev_fallback_0.75',
    'maltsev_fallback_1.0', 'maltsev_pf_prob_0.0', 'maltsev_pf_prob_1.0',
    'imf_2.7', 'imf_1.9', 'porb_slope_0', 'porb_slope_m1', 'porb_max_1000',
    'q_power_law_m1', 'q_power_law_p1', 'qmin_pre_ms',
    'singles',
    'alpha_0.1', 'alpha_0.5', 'alpha_2.0', 'alpha_10.0',
    'beta_0.0', 'beta_0.5', 'beta_1.0',
    'qcrit_caseB_0.001', 'qcrit_caseB_1000',
    'time-evolving-pot',
]
do_later = [
]
done = [
    'fiducial', 
]

files = do_now

BASE_PATH = "/mnt/ceph/users/twagg/underworld/"
postprocess_folder = os.path.join(BASE_PATH, "postprocessed")
N_PARTS = 34

for file in files:
    print("-------------------------\n\n")
    postprocess_file = os.path.join(postprocess_folder, f"{file}.h5")
    if os.path.exists(postprocess_file):
        print(f"File {file}.h5 already exists, skipping...")
        continue

    start = time()
    print("Processing file:", file)

    for part in range(N_PARTS):
        # check if the postprocessed part already exists
        if os.path.exists(os.path.join(postprocess_folder, "subfiles", f"{file}_part{part:d}.h5")):
            print(f"    Part {part} already processed, skipping...")
            continue

        print(f"    Processing part {part+1}/{N_PARTS}...")

        p = cogsworth.pop.load(join(BASE_PATH, "sims", file, f"{file}_part{part:d}"))

        p.initial_binaries
        p.initial_galaxy
        p.initC
        p.final_bpp
        p.bin_nums
        p.final_pos
        p.final_vel
        p._file = None

        p.label = f"{file}_part{part:d}"

        kinematics, masses, bin_nums, sep, primary, companion = helpers.postprocess_populations(p)

        # save the postprocessed part
        helpers.save_postprocessed_data(
            [p], [os.path.join(postprocess_folder, "subfiles", f"{file}_part{part:d}.h5")],
            kinematics, masses, bin_nums, sep, primary, companion
        )

    print(f"Finished processing all parts for {file} in {time() - start:.2f} seconds.")

    start = time()
    print("Combining parts")
    data = helpers.load_postprocessed_data(
        [f"{file}_part{part:d}.h5" for part in range(N_PARTS)],
        labels=[f"{file}_part{part:d}" for part in range(N_PARTS)],
        folder=os.path.join(postprocess_folder, "subfiles")
    )

    combined_data = {}

    # concatenate the data across parts
    keys_to_concat = ["pos", "vel", "escaped", "bin_nums", "mass", "tau",
                      "init_z", "sep", "primary", "companion"]
    # with subkeys of BH and NS
    for key in keys_to_concat:
        combined_data[key] = {}
        for subkey in ["BH", "NS"]:
            combined_data[key][subkey] = np.concatenate([data[f"{file}_part{part:d}"][key][subkey] for part in range(N_PARTS)])

    combined_data["mass_binaries"] = sum(data[f"{file}_part{part:d}"]["mass_binaries"] for part in range(N_PARTS))
    combined_data["mass_singles"] = sum(data[f"{file}_part{part:d}"]["mass_singles"] for part in range(N_PARTS))

    print(f"Finished combining parts for {file} in {time() - start:.2f} seconds.")

    print("Saving processed data...")
    helpers.save_postprocessed_data_one_dict(combined_data, postprocess_file)
    print(f"Finished processing {file} in {time() - start:.2f} seconds.")

