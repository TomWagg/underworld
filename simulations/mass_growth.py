#!/usr/bin/env python3

import glob
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


def parse_header_for_cosmology(fname):
    """parse the header of one rockstar file to get a, Om, Ol, h."""
    a = None
    Om = None
    Ol = None
    h = None

    with open(fname, "r") as f:
        for line in f:
            if not line.startswith("#"):
                break

            if line.startswith("#a"):
                # example: "#a = 0.062500"
                a = float(line.split("=")[1])
            elif line.startswith("#Om"):
                # example: "#Om = 0.272000; Ol = 0.728000; h = 0.702000"
                line_str = line[1:].strip()  # drop '#'
                parts = [x.strip() for x in line_str.split(";") if "=" in x]
                for p in parts:
                    name, val = [x.strip() for x in p.split("=")]
                    val = float(val)
                    if name == "Om":
                        Om = val
                    elif name == "Ol":
                        Ol = val
                    elif name == "h":
                        h = val

    if a is None:
        raise RuntimeError(f"could not find scale factor in header of {fname}")

    # if Ol is missing, assume flat universe
    if Om is not None and Ol is None:
        Ol = 1.0 - Om

    return dict(a=a, Om=Om, Ol=Ol, h=h)


def read_rockstar_catalog(fname):
    """read one rockstar-galaxies catalog into a dataframe with an 'a' column."""
    a = None
    names = None
    with open(fname, "r") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("#a"):
                a = float(line.split("=")[1])
            if names is None and line.startswith("#ID"):
                # line like "#id desc_id m_200b ..."
                names = line[1:].strip().split()
    df = pd.read_csv(
        fname,
        sep='\s+',
        comment="#",
        names=names,
    )

    df["a"] = a
    df["z"] = 1.0 / a - 1.0
    return df


def load_all_catalogues(pattern="out_*.list", n_procs=None):
    """load all rockstar catalogues matching a glob pattern using a process pool."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"no files matched pattern: {pattern}")

    if n_procs is None:
        n_procs = min(cpu_count(), 32)  # cap at 32 since you mentioned 32 cpus

    print(f"found {len(files)} files; reading with {n_procs} processes")

    with Pool(processes=n_procs) as pool:
        catalogues = pool.map(read_rockstar_catalog, files)

    return catalogues, files


def build_cosmology_from_header(example_file):
    """build an astropy cosmology from one file's header."""

    header = parse_header_for_cosmology(example_file)
    Om0 = header["Om"]
    h = header["h"]

    if Om0 is None or h is None:
        raise RuntimeError("could not parse Om or h from header")

    H0 = 100.0 * h  # km/s/Mpc

    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    return cosmo, header


def add_time_columns(growth_df, cosmo):
    """add cosmic age and lookback time columns to the growth dataframe."""
    z = growth_df["z"].values
    age = cosmo.age(z).to(u.Gyr).value          # Gyr
    lookback = cosmo.lookback_time(z).to(u.Gyr).value  # Gyr

    growth_df["age_Gyr"] = age
    growth_df["lookback_Gyr"] = lookback


def find_main_halo_last_snapshot(catalogues):
    """find the main halo in the last snapshot (most massive M200b, type==0 if present)."""
    last_cat = catalogues[-1]

    if "Type" in last_cat.columns:
        candidates = last_cat[last_cat["Type"] == 0]
        if len(candidates) == 0:
            candidates = last_cat
    else:
        candidates = last_cat

    main = candidates.sort_values("M200b", ascending=False).iloc[0]
    main_id = int(main["ID"])
    print("main halo at final snapshot:")
    print(main[["ID", "M200b", "Mvir", "a", "z"]])

    return main_id


def track_main_progenitor(catalogues, main_id):
    """
    follow the main progenitor of the final main halo back in time.

    catalogues should be a list sorted by increasing scale factor, each entry:
        {"fname": ..., "a": float, "df": dataframe}
    """
    growth_records = []
    current_id = main_id
    n_snaps = len(catalogues)

    for snap_idx in range(n_snaps - 1, -1, -1):
        a_snap = catalogues[snap_idx]["a"].iloc[0]
        df = catalogues[snap_idx]

        if snap_idx == n_snaps - 1:
            # final snapshot: select by id
            row = df.loc[df["ID"] == current_id]
            if row.empty:
                print(f"warning: main halo id {current_id} not found in final snapshot")
                break
            row = row.iloc[0]
        else:
            # earlier snapshot: find progenitors whose DescID == current main id
            progenitors = df.loc[df["DescID"] == current_id]
            if progenitors.empty:
                print(f"stopped at snapshot index {snap_idx}: no progenitor with DescID={current_id}")
                break

            # choose the most massive progenitor as "main"
            row = progenitors.loc[progenitors["M200b"].idxmax()]
            current_id = int(row["ID"])

        growth_records.append(
            {
                "a": float(a_snap),
                "z": float(row["z"]),
                "ID": int(row["ID"]),
                "M200b": float(row["M200b"]),
                "Mvir": float(row["Mvir"]),
                "M200c": float(row["M200c"]) if "M200c" in row.index else np.nan,
            }
        )

    # currently in order from latest -> earliest; reverse to get earliest -> latest
    growth_records.reverse()
    growth_df = pd.DataFrame(growth_records)
    return growth_df


def plot_mass_growth(growth_df, h, outfile="mass_growth.png"):
    """plot physical M200b vs cosmic age."""
    # convert from Msun/h to Msun
    growth_df["M200b_Msun"] = growth_df["M200b"] / h

    plt.figure()
    plt.plot(growth_df["age_Gyr"], growth_df["M200b_Msun"])
    plt.xlabel("Physical time [Gyr]")
    plt.ylabel(r"$M_{200b}$ [M$_\odot$]")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"saved plot to {outfile}")
    plt.close()


def main():
    # pattern for your files; adjust if needed
    pattern = "/mnt/home/mgrudic/public_www/fire2_public_release/core/m12i_res7100/halo/rockstar_dm/catalog/out_*.list"

    # load all catalogues
    catalogues, files = load_all_catalogues(pattern=pattern, n_procs=32)

    # build cosmology from the first file (all share the same header)
    cosmo, header = build_cosmology_from_header(files[0])
    h = header["h"]

    print("using cosmology:")
    print(f"  Om0 = {header['Om']}")
    print(f"  Ol0 = {header['Ol']}")
    print(f"  h   = {h}")

    # find main halo at last snapshot
    main_id = find_main_halo_last_snapshot(catalogues)

    # track main progenitor back in time
    growth = track_main_progenitor(catalogues, main_id)

    # add cosmic time columns
    add_time_columns(growth, cosmo)

    # plot mass growth vs cosmic age
    plot_mass_growth(growth, h, outfile="mass_growth_M200b_vs_time.png")

    # optionally save the growth history to a csv
    growth.to_csv("main_halo_growth.csv", index=False)
    print("wrote main halo growth history to main_halo_growth.csv")


if __name__ == "__main__":
    main()
