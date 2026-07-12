import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

maltsev_fb_colours = [plt.get_cmap("Reds", 6)(i) for i in range(1, 6)]
maltsev_pf_colours = [plt.get_cmap("RdPu", 5)(i) for i in range(2, 4)]
alpha_colours = plt.cm.Reds(np.linspace(0.4, 1, 4))
beta_colours = plt.cm.Blues(np.linspace(0.4, 1, 3))
initC_colours = plt.colormaps["tab20"]([0, 1, 2, 3, 4, 6, 7, 9])

maltsev_fb_colours = [mpl.colors.to_hex(c) for c in maltsev_fb_colours]
maltsev_pf_colours = [mpl.colors.to_hex(c) for c in maltsev_pf_colours]
alpha_colours = [mpl.colors.to_hex(c) for c in alpha_colours]
beta_colours = [mpl.colors.to_hex(c) for c in beta_colours]
initC_colours = [mpl.colors.to_hex(c) for c in initC_colours]

SIMULATION_FILES = [
    'fiducial',
    # ---
    # kicks
    'bhflag_0', 'kickflag_1', 'bhflag_3', 
    # remnant mass prescriptions
    'fryer_rapid', 'mandel_muller', 'maltsev_fallback_0.5',
    'maltsev_fallback_0.0', 'maltsev_fallback_0.25', 'maltsev_fallback_0.75', 'maltsev_fallback_1.0',
    'maltsev_pf_prob_0.0', 'maltsev_pf_prob_1.0',
    # time evolving potential
    'time-evolving-pot',
    # singles
    'singles',
    # initial conditions
    'imf_1.9', 'imf_2.7',
    'q_power_law_m1', 'q_power_law_p1',
    'qmin_pre_ms',
    'porb_slope_0', 'porb_slope_m1',
    'porb_max_1000',
    # mt physics
    'beta_0.0', 'beta_0.5', 'beta_1.0',
    'alpha_0.1', 'alpha_0.5', 'alpha_2.0', 'alpha_10.0',
    'qcrit_caseB_0.001', 'qcrit_caseB_1000',
]

SIMULATION_COLOURS = [
    "#333",
    "#65B860", "#C0D061", "darkgreen", 
    "slategrey", "tab:purple", maltsev_fb_colours[2],
    maltsev_fb_colours[0], maltsev_fb_colours[1], maltsev_fb_colours[3], maltsev_fb_colours[4],
    maltsev_pf_colours[0], maltsev_pf_colours[1],
    "#FF9D1E",
    # initial conditions
    "#05b3a4",
    *initC_colours,
    # mt physics
    *beta_colours,
    *alpha_colours,
    "tab:pink", "tab:purple",
]

FILE_TO_COLOUR = {f: c for f, c in zip(SIMULATION_FILES, SIMULATION_COLOURS)}