
import gizmo_analysis as gizmo
import numpy as np
import argparse
from os.path import join, isfile
import astropy.units as u
import astropy.constants as const


def get_circularity(r, v, get_v_circ):
    """Compute the circularity parameter for particles given their positions and velocities."""
    R = np.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2)
    Lz = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]

    vc = get_v_circ(R)
    circularity = Lz / (R * vc)

    return circularity


def get_pos_vel_m(part, part_type, get_v_circ, circularity_thresh=0.5, dist_thresh=300.0):
    """Get positions, velocities, and masses of particles of a given type that are likely part of the disk."""
    pos = part[part_type].prop("host.distance.principal")
    vel = part[part_type].prop("host.velocity.principal")
    mass = np.array(part[part_type]["mass"])

    d = np.linalg.norm(pos, axis=1)

    circularity = get_circularity(pos, vel, get_v_circ)
    mask = (circularity > circularity_thresh) & (d < dist_thresh)

    return pos[mask], vel[mask], mass[mask]


def get_mass_growth(sim_dir, redshift, radii, out_dir, overwrite=False, circularity_thresh=0.5):
    """Compute enclosed mass profiles at given redshift for a simulation.

    Parameters
    ----------
    sim_dir : str
        Path to the simulation directory.
    redshift : float
        Redshift of the snapshot to analyze.
    radii : array-like
        Radii at which to compute enclosed mass (in kpc).
    out_dir : str
        Directory to save output plots.
    overwrite : bool, optional
        Whether to overwrite existing output files. Default is False.
    circularity_thresh : float, optional
        Threshold for circularity to identify disk particles. Default is 0.5.
    """
    out_file = join(out_dir, f"z_{redshift:.2f}.npz")

    # check if output file already exists
    if isfile(out_file) and not overwrite:
        print(f"Output file {out_file} already exists. Skipping computation.")
        return None, None

    print(f"Processing simulation at {sim_dir} for z={redshift}")
    print(f"Input radii (kpc): {radii}")

    part = gizmo.io.Read.read_snapshots(
        ["dark", "star", "gas"], "redshift", redshift, sim_dir, snapshot_directory="output"
    )
    print(f"Read {len(part['dark'])} dark matter particles at z={redshift}")

    all_pos = np.concatenate([part["star"].prop("host.distance.principal"),
                              part["dark"].prop("host.distance.principal"),
                              part["gas"].prop("host.distance.principal")])
    all_mass = np.concatenate([part["star"]["mass"], part["dark"]["mass"], part["gas"]["mass"]])
    all_R = (all_pos[:, 0]**2 + all_pos[:, 1]**2)**0.5

    sort_idx = np.argsort(all_R)
    r_sorted = all_R[sort_idx]
    m_sorted = all_mass[sort_idx]

    m_enc = np.cumsum(m_sorted)

    R_grid = np.linspace(1e-5, 100.0, 5000)
    indices = np.searchsorted(r_sorted, R_grid, side="right") - 1
    indices = np.clip(indices, 0, len(m_enc) - 1)

    M_grid = m_enc[indices]

    v_circ = np.sqrt(const.G * M_grid * u.Msun / (R_grid * u.kpc)).to(u.km / u.s).value

    get_v_circ_interp = lambda r: np.interp(r, R_grid, v_circ)

    print(f"Computed circular velocity profile at z={redshift}")

    dm_pos, dm_vel, dm_mass = get_pos_vel_m(part, "dark", get_v_circ_interp,
                                            circularity_thresh=circularity_thresh)

    # calculate the enclosed mass profiles for dark matter, then stars+gas
    dm_r = np.linalg.norm(dm_pos, axis=1)
    dm_M_enc = np.array([dm_mass[dm_r < rr].sum() for rr in radii])

    star_pos, star_vel, star_mass = get_pos_vel_m(part, "star", get_v_circ_interp,
                                                  circularity_thresh=circularity_thresh)
    gas_pos, gas_vel, gas_mass = get_pos_vel_m(part, "gas", get_v_circ_interp,
                                               circularity_thresh=circularity_thresh)

    baryon_pos = np.vstack((star_pos, gas_pos))  # (N_star + N_gas, 3) in kpc
    baryon_mass = np.hstack((np.array(star_mass), np.array(gas_mass)))  # (N_star + N_gas,) in Msun
    baryon_r = np.linalg.norm(baryon_pos, axis=1)  # physical radius in kpc
    baryon_M_enc = np.array([baryon_mass[baryon_r < rr].sum() for rr in radii])
    print(f"Computed enclosed mass profile at z={redshift}")

    R = np.sqrt(star_pos[:, 0]**2 + star_pos[:, 1]**2)
    z = star_pos[:, 2]

    Rbins = np.linspace(1, 30, 100)
    zbins = np.linspace(-5, 5, 101)
    H, xe, ye = np.histogram2d(R, z, bins=(Rbins, zbins), weights=star_mass)

    # volume factor
    V = 2 * np.pi * (Rbins[1:]**2 - Rbins[:-1]**2)[:, None] * (zbins[1:] - zbins[:-1])[None, :]

    # adjust density by volume
    dens = H / V

    np.savez(out_file, radii=radii, dm_M_enc=dm_M_enc, baryon_M_enc=baryon_M_enc,
             dens=dens, xe=xe, ye=ye)
    print(f"Saved enclosed mass profile to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute enclosed mass profiles at specified redshift."
    )
    parser.add_argument(
        "--sim-dir", type=str, help="Path to the simulation directory."
    )
    parser.add_argument(
        "--redshift", type=float, help="Redshift of the snapshot to analyze."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Directory to save output plots.",
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[1, 10, 20, 30, 50, 100, 200, 300],
        help="Radii (in kpc) at which to compute enclosed mass.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing output files.",
    )
    parser.add_argument(
        "--circularity-thresh",
        type=float,
        help="Threshold for circularity to identify disk particles.",
    )

    args = parser.parse_args()

    get_mass_growth(args.sim_dir, args.redshift, args.radii, args.out_dir,
                    args.overwrite, args.circularity_thresh)
