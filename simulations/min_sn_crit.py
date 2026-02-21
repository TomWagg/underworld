"""We're going to run a grid of stars from 5 to 9 solar masses for a range of metallicities to assess what masses lead to core-collapse in COSMIC."""

from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve
import numpy as np
import matplotlib.pyplot as plt


plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': 0.7*fs,
          'legend.title_fontsize': 0.8*fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)

mass_step = 0.025
mass_range = np.arange(5.8, 9 + mass_step, mass_step)

metallicity_grid = np.geomspace(1e-4, 0.03, 200)

M, Z = np.meshgrid(mass_range, metallicity_grid, indexing='ij')
m_flat = M.ravel()
z_flat = Z.ravel()


initial_binaries = InitialBinaryTable.InitialBinaries(
    m1=M.ravel(),
    m2=np.zeros(len(m_flat)),
    porb=-1 * np.ones(len(m_flat)),
    ecc=-1 * np.ones(len(m_flat)),
    tphysf=13700 * np.ones(len(m_flat)),
    kstar1=np.ones(len(m_flat)),
    kstar2=np.zeros(len(m_flat)),
    metallicity=z_flat,
)

BSEDict = {
    "pts1": 0.001, "pts2": 0.01, "pts3": 0.02, "zsun": 0.014, "windflag": 3,
    "eddlimflag": 0, "neta": 0.5, "bwind": 0.0, "hewind": 0.5, "beta": 0.125,
    "xi": 0.5, "acc2": 1.5, "LBV_flag": 1, "alpha1": 1.0, "lambdaf": 0.0,
    "ceflag": 1, "cekickflag": 2, "cemergeflag": 1, "cehestarflag": 0,
    "qcflag": 5,
    "qcrit_array": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    "kickflag": 5, "sigma": 265.0, "bhflag": 1, "bhsigmafrac": 1.0,
    "sigmadiv": -20.0, "ecsn": 2.25, "ecsn_mlow": 1.6, "aic": 1, "ussn": 1,
    "polar_kick_angle": 90.0,
    "natal_kick_array": [[-100.0, -100.0, -100.0, -100.0, 0.0], [-100.0, -100.0, -100.0, -100.0, 0.0]],
    "mm_mu_ns": 400.0, "mm_mu_bh": 200.0, "remnantflag": 4,
    "fryer_mass_limit": 0, "mxns": 3.0, "rembar_massloss": 0.5,
    "wd_mass_lim": 1, "maltsev_mode": 0, "maltsev_fallback": 0.5,
    "maltsev_pf_prob": 0.1, "pisn": -2, "ppi_co_shift": 0.0,
    "ppi_extra_ml": 0.0, "bhspinflag": 0, "bhspinmag": 0.0, "grflag": 1,
    "eddfac": 10, "gamma": -2, "don_lim": -1, "acc_lim": -1, "tflag": 1,
    "ST_tide": 1,
    "fprimc_array": [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0],
    "ifflag": 1, "wdflag": 1, "epsnov": 0.001, "bdecayfac": 1,
    "bconst": 3000, "ck": 1000, "rejuv_fac": 1.0, "rejuvflag": 0,
    "bhms_coll_flag": 0, "htpmb": 1, "ST_cr": 1, "rtmsflag": 0
}


bpp, bcm, initC, kick_info = Evolve.evolve(
    initial_binaries, BSEDict=BSEDict,
    bpp_columns=["tphys", "mass_1", "massc_co_layer_1", "massc_he_layer_1", "evol_type", "kstar_1", "SN_1"],
    nproc=32
)
final_bpp = bpp.drop_duplicates(subset="bin_num", keep="last")

# find the minimum initial mass in initC where final_bpp has kstar_1 == 13
exploding_stars = final_bpp[(final_bpp["kstar_1"] == 13) & (final_bpp["SN_1"] == 1)]
exploding_initC = initC[initC["bin_num"].isin(exploding_stars["bin_num"])]
min_exploding_mass = exploding_initC.groupby("metallicity")["mass_1"].min().reset_index()

fig, ax = plt.subplots(figsize=(10, 8))

colours = final_bpp["SN_1"].map({0: plt.get_cmap("viridis")(0), 1: plt.get_cmap("viridis")(0.5), 2: plt.get_cmap("magma")(0.6)}).values

ax.scatter(initC["metallicity"], initC["mass_1"],
           c=colours,
           s=12,
           rasterized=True)

ax.plot(min_exploding_mass["metallicity"], min_exploding_mass["mass_1"],
        c='black', lw=5)

for sn_type, color in zip([0, 1, 2], [plt.get_cmap("viridis")(0), plt.get_cmap("viridis")(0.5), plt.get_cmap("magma")(0.6)]):
    ax.scatter(
        np.nan, np.nan,
        c=color,
        label={0: "No SN", 1: "Core-Collapse SN", 2: "ECSN"}[sn_type],
    )

# some very crude analytic model
Z_range = np.geomspace(1e-4, 0.03, 1000)
line = np.log10(Z_range) * 1.1 + 10.1
line[Z_range < 1.05e-3] = 6.825
plt.plot(Z_range, line, c='white', ls='--', label="CCSN 'model'")

line = np.log10(Z_range) * 1.1 + 10.1
line[(Z_range < 1.05e-3)] = np.log10(Z_range)[(Z_range < 1.05e-3)] * 0.8 + 9.2
line[Z_range < 3e-4] = 6.38
plt.plot(Z_range, line, c='white', ls=':', label="Any SN 'model'")

ax.legend(loc='upper left', handletextpad=0.5)

ax.set(
    xlabel="Metallicity",
    ylabel="Initial Mass (M$_\odot$)",
    xscale="log",
    xlim=(1e-4, 0.03),
    ylim=(6, 9),
)

plt.show()
