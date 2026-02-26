import matplotlib.pyplot as plt
import numpy as np

maltsev_fb_colours = [plt.get_cmap("Reds", 6)(i) for i in range(1, 6)]
maltsev_pf_colours = [plt.get_cmap("RdPu", 5)(i) for i in range(2, 4)]
alpha_colours = plt.cm.Reds(np.linspace(0.4, 1, 4))
beta_colours = plt.cm.Blues(np.linspace(0.4, 1, 3))

SIMULATION_FILES = [
    'fiducial',
    # singles
    'singles',
    # time-evolving potential
    # ---
    # kicks
    'bhflag_0', 'bhflag_3', 'kickflag_1',
    # remnant mass prescriptions
    'fryer_rapid', 'mandel_muller', 'maltsev_fallback_0.5',
    'maltsev_fallback_0.0', 'maltsev_fallback_0.25', 'maltsev_fallback_0.75', 'maltsev_fallback_1.0',
    'maltsev_pf_prob_0.0', 'maltsev_pf_prob_1.0',
    # time evolving potential
    'time-evolving-pot',
    # mt physics
    'beta_0.0', 'beta_0.5', 'beta_1.0',
    'alpha_0.1', 'alpha_0.5', 'alpha_2.0', 'alpha_10.0',
    'qcrit_caseB_0.001', 'qcrit_caseB_1000',
]

SIMULATION_COLOURS = [
    "#333",
    "#05b3a4",
    "#65B860", "darkgreen",  "#C0D061",
    "slategrey", "tab:purple", maltsev_fb_colours[2],
    maltsev_fb_colours[0], maltsev_fb_colours[1], maltsev_fb_colours[3], maltsev_fb_colours[4],
    maltsev_pf_colours[0], maltsev_pf_colours[1],
    "#FF9D1E",
    *beta_colours,
    *alpha_colours,
    "tab:pink", "tab:purple"
]

FILE_TO_COLOUR = {f: c for f, c in zip(SIMULATION_FILES, SIMULATION_COLOURS)}