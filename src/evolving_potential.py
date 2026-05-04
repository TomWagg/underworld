import numpy as np
import astropy.units as u
import gala.potential as gp


@np.vectorize
def dm_mass_growth_at_tau(tau):
    if tau > 9:
        return (2.32 - (0.88 - 0.4) / 3 * tau)
    else:
        return (1 - 12 / 900 * tau)


def baryon_mass_growth_at_tau(tau):
    return 0.8 / 12 * (12 - tau) + 0.2


def get_milky_way_potential(evolve_dm=True, evolve_disk=True, just_halo=False):
    static_mw = gp.MilkyWayPotential(version='v2')
    if not evolve_dm and not evolve_disk:
        return static_mw

    time_knots = np.linspace(0, 12, 20) * u.Gyr
    lookback_times = 12 * u.Gyr - time_knots

    m_dm = dm_mass_growth_at_tau(lookback_times.to(u.Gyr).value) * static_mw["halo"].parameters["m"].value
    m_bary = baryon_mass_growth_at_tau(lookback_times.to(u.Gyr).value) * (static_mw["disk"].parameters["m"].value)

    if evolve_disk:
        disc = gp.TimeInterpolatedPotential(
            gp.MN3ExponentialDiskPotential,
            time_knots,
            m=m_bary,
            h_R=2.6 * u.kpc,
            h_z=0.3 * u.kpc,
            units="galactic"
        )
    else:
        disc = gp.MN3ExponentialDiskPotential(
            m=static_mw["disk"].parameters["m"],
            h_R=2.6 * u.kpc,
            h_z=0.3 * u.kpc,
            units="galactic"
        )

    if evolve_dm:
        halo = gp.TimeInterpolatedPotential(
            gp.NFWPotential,
            time_knots,
            m=m_dm,
            r_s=15.63 * u.kpc,
            units="galactic"
        )
    else:
        halo = gp.NFWPotential(
            m=static_mw["halo"].parameters["m"],
            r_s=15.63 * u.kpc,
            units="galactic"
        )

    evolving_mw = gp.CCompositePotential(
        disk=disc,
        bulge=gp.HernquistPotential(
            m=5e9 * u.Msun,
            c=1.0 * u.kpc,
            units="galactic"
        ),
        nucleus=gp.HernquistPotential(
            m=1.81e9 * u.Msun,
            c=0.07 * u.kpc,
            units="galactic"
        ),
        halo=halo
    )

    if just_halo:
        return halo

    return evolving_mw
