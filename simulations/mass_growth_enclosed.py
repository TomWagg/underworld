
import gizmo_analysis as gizmo
import numpy as np
import argparse
from os.path import join, isfile


def compute_half_mass_scales(pos, mass):
    """Compute half-mass radius and half-mass height for a set of particles.

    Parameters
    ----------
    pos : np.ndarray
        Positions of particles (N, 3) in kpc.
    mass : np.ndarray
        Masses of particles (N,) in Msun.

    Returns
    -------
    R_half : float
        Half-mass radius in kpc.
    z_half : float
        Half-mass height in kpc.
    """
    R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    z = np.abs(pos[:, 2])

    # only look at the central 50 kpc
    dist = np.linalg.norm(pos, axis=1)
    mask = dist < 50
    R = R[mask]
    z = z[mask]
    mass = mass[mask]

    # half mass radius
    sorted_indices_R = np.argsort(R)
    cum_mass_R = np.cumsum(mass[sorted_indices_R])
    R_half = R[sorted_indices_R][np.searchsorted(cum_mass_R, cum_mass_R[-1] / 2)]

    # half mass height
    sorted_indices_z = np.argsort(z)
    cum_mass_z = np.cumsum(mass[sorted_indices_z])
    z_half = z[sorted_indices_z][np.searchsorted(cum_mass_z, cum_mass_z[-1] / 2)]

    return R_half, z_half


def get_mass_growth(sim_dir, redshift, radii, out_dir, overwrite=False):
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
    """
    out_file = join(out_dir, f"z_{redshift:.2f}.npz")

    # check if output file already exists
    if isfile(out_file) and not overwrite:
        print(f"Output file {out_file} already exists. Skipping computation.")
        return None, None

    print(f"Processing simulation at {sim_dir} for z={redshift}")
    print(f"Input radii (kpc): {radii}")
    ≈
    print(f"Read {len(part['dark'])} dark matter particles at z={redshift}")

    # calculate the enclosed mass profiles for dark matter, then stars+gas
    dm_pos = part["dark"].prop("host.distance.principal")  # (N, 3) in kpc
    dm_mass = np.array(part["dark"]["mass"])  # (N_dm,) in Msun
    dm_r = np.linalg.norm(dm_pos, axis=1)  # physical radius in kpc
    dm_M_enc = np.array([dm_mass[dm_r < rr].sum() for rr in radii])

    baryon_pos = np.vstack(
        (
            part["star"].prop("host.distance.principal"),
            part["gas"].prop("host.distance.principal"),
        )
    )  # (N_star + N_gas, 3) in kpc
    baryon_mass = np.hstack(
        (np.array(part["star"]["mass"]), np.array(part["gas"]["mass"]))
    )  # (N_star + N_gas,) in Msun
    baryon_r = np.linalg.norm(baryon_pos, axis=1)  # physical radius in kpc
    baryon_M_enc = np.array([baryon_mass[baryon_r < rr].sum() for rr in radii])
    print(f"Computed enclosed mass profile at z={redshift}")

    dm_R_half, dm_z_half = compute_half_mass_scales(dm_pos, dm_mass)
    baryon_R_half, baryon_z_half = compute_half_mass_scales(baryon_pos, baryon_mass)
    print(f"Computed half-mass scales at z={redshift}")

    np.savez(out_file, radii=radii, dm_M_enc=dm_M_enc, baryon_M_enc=baryon_M_enc,
             dm_R_half=dm_R_half, dm_z_half=dm_z_half,
             baryon_R_half=baryon_R_half, baryon_z_half=baryon_z_half)
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

    args = parser.parse_args()

    get_mass_growth(args.sim_dir, args.redshift, args.radii, args.out_dir, args.overwrite)
