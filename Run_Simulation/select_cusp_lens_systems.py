#!/usr/bin/env python3
# select_cusp_from_fits.py
# Select cusp-configuration systems from a FITS catalog with merged lensing results.
from Rcusp_tool import select_cusp_systems_with_progress
import argparse



def save_all_cusp_data_observable(cusp_data, output_file: str | None = None):
    """
    Convert cusp_data list into an astropy Table, apply observability cuts,
    assign cusp_id, and optionally write to FITS.
    """
    if len(cusp_data) == 0:
        raise ValueError("cusp_data is empty. Nothing to save.")

    colnames = cusp_data[0]["data"].dtype.names
    cols = {col: [entry["data"][col] for entry in cusp_data] for col in colnames}
    table = Table(cols)

    table["Rcusp"] = [entry["R_cusp"] for entry in cusp_data]
    table["phi"]   = [entry["phi"] for entry in cusp_data]
    table["phi1"]  = [entry["phi1"] for entry in cusp_data]
    table["phi2"]  = [entry["phi2"] for entry in cusp_data]

    images_A = [entry["images"]["A"] for entry in cusp_data]
    images_B = [entry["images"]["B"] for entry in cusp_data]
    images_C = [entry["images"]["C"] for entry in cusp_data]
    images_D = [entry["images"]["D"] for entry in cusp_data]

    table["imageA_x"], table["imageA_y"] = zip(*images_A)
    table["imageB_x"], table["imageB_y"] = zip(*images_B)
    table["imageC_x"], table["imageC_y"] = zip(*images_C)
    table["imageD_x"], table["imageD_y"] = zip(*images_D)

    # observability cuts
    mask_obs = (
        (table["source_redshift"] < 4)
        & (table["apparent_mag_first_arrival_i_band"] < 26)
        & (table["v_disp"] > 50.0)
        & ((table["image_sep"] / 2.0) > 0.07)
    )
    table = table[mask_obs]
    table["cusp_id"] = np.arange(1, len(table) + 1)

    if output_file is not None:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        table.write(output_file, overwrite=True)
        print(f"Saved {len(table)} observable cusp systems -> {output_file}")

    # keep a compact list output (optional downstream use)
    filtered_list = []
    for row in table:
        filtered_list.append({
            "data": row,
            "R_cusp": row["Rcusp"],
            "phi": row["phi"],
            "images": {
                "A": np.array([row["imageA_x"], row["imageA_y"]]),
                "B": np.array([row["imageB_x"], row["imageB_y"]]),
                "C": np.array([row["imageC_x"], row["imageC_y"]]),
                "D": np.array([row["imageD_x"], row["imageD_y"]]),
            }
        })

    return filtered_list


def main():
    parser = argparse.ArgumentParser(description="Select cusp systems from merged FITS catalog.")
    parser.add_argument("--fits", required=True, help="Input FITS file (with merged columns).")
    parser.add_argument("--out-fits", default="Theory_Mock/cusp_all_observable_multipole.fits", help="Output FITS path.")
    args = parser.parse_args()

    cusp_data = select_cusp_systems_with_progress(args.fits)

    _ = save_all_cusp_data_observable(cusp_data, output_file=args.out_fits)


if __name__ == "__main__":
    main()