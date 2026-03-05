#!/usr/bin/env python3
# merge_calc_results_to_fits.py
# Merge lensing calculation results stored in JSON files into a FITS catalog

import os
import glob
import json
import argparse
import numpy as np
from astropy.io import fits


def merge_calc_to_fits(fits_path: str, json_dir: str, output_fits: str) -> None:
    """
    Merge lensing calculation results from JSON files into a FITS catalog.

    Parameters
    ----------
    fits_path : str
        Path to the original FITS catalog.

    json_dir : str
        Directory containing JSON result files (calc_results_*.json).

    output_fits : str
        Path where the merged FITS file will be written.

    Notes
    -----
    The function reads all JSON files in json_dir and merges the results
    into the second extension table of the FITS file using the index `idx`.

    New columns added to the FITS table:
        - idx
        - images
        - image_sep
        - apparent_mag_first_arrival_i_band
        - image_number
    """

    # ---- Read original FITS file ----
    with fits.open(fits_path) as hdul:

        src_primary = hdul[0]        # Primary HDU
        src_hdu = hdul[1]            # Binary table HDU
        src_data = src_hdu.data

        nrows = len(src_data)

        # ---- First pass through JSON files: collect results ----
        idx_to_res = {}
        max_str_len = 1

        json_files = sorted(glob.glob(os.path.join(json_dir, "calc_results_*.json")))
        print(f"Found {len(json_files)} JSON files")

        for fp in json_files:

            print(f"Reading {fp}")

            with open(fp, "r") as f:
                obj = json.load(f)

            for rec in obj.get("data", []):

                idx = int(rec["idx"])

                # Skip invalid indices
                if not (0 <= idx < nrows):
                    continue

                res = rec.get("result", {})

                idx_to_res[idx] = {

                    "images": str(res.get("calc_images", "")),

                    "image_sep":
                        float(res.get("calc_image_sep", np.nan))
                        if res.get("calc_image_sep") is not None else np.nan,

                    "apparent_mag_first_arrival_i_band":
                        float(res.get("calc_apparent_mag_first_arrival_i_band", np.nan))
                        if res.get("calc_apparent_mag_first_arrival_i_band") is not None else np.nan,

                    "image_number":
                        float(res.get("calc_image_number", np.nan))
                        if res.get("calc_image_number") is not None else np.nan,
                }

                # Track maximum string length for FITS column format
                max_str_len = max(max_str_len, len(idx_to_res[idx]["images"]))

        print(f"Collected results for {len(idx_to_res)} rows")

        # ---- Create arrays for new columns ----

        idx_arr = np.arange(nrows, dtype=np.int32)

        img_arr = np.full(nrows, '', dtype=f'S{max_str_len}')
        sep_arr = np.full(nrows, np.nan, dtype=np.float64)
        app_arr = np.full(nrows, np.nan, dtype=np.float64)
        num_arr = np.full(nrows, np.nan, dtype=np.float64)

        # Fill arrays with collected results
        for idx, res in idx_to_res.items():

            img_arr[idx] = res["images"].encode("ascii", "ignore")[:max_str_len]
            sep_arr[idx] = res["image_sep"]
            app_arr[idx] = res["apparent_mag_first_arrival_i_band"]
            num_arr[idx] = res["image_number"]

        # ---- Construct new FITS columns ----

        col_idx = fits.Column(name="idx", format="J", array=idx_arr)

        new_cols = [

            fits.Column(
                name="images",
                format=f"A{max_str_len}",
                array=img_arr
            ),

            fits.Column(
                name="image_sep",
                format="D",
                array=sep_arr
            ),

            fits.Column(
                name="apparent_mag_first_arrival_i_band",
                format="D",
                array=app_arr
            ),

            fits.Column(
                name="image_number",
                format="D",
                array=num_arr
            ),
        ]

        # Combine original columns with new columns
        merged_hdu = fits.BinTableHDU.from_columns(
            fits.ColDefs([col_idx] + list(src_hdu.columns) + new_cols)
        )

        # ---- Write output FITS ----

        hdul_out = fits.HDUList([
            fits.PrimaryHDU(header=src_primary.header),
            merged_hdu
        ])

        hdul_out.writeto(output_fits, overwrite=True)

        print("================================")
        print("FITS merge completed successfully")
        print(f"Output file: {output_fits}")
        print("================================")


def main():

    parser = argparse.ArgumentParser(
        description="Merge lensing JSON results into a FITS catalog"
    )

    parser.add_argument(
        "--fits",
        required=True,
        help="Input FITS catalog"
    )

    parser.add_argument(
        "--json-dir",
        required=True,
        help="Directory containing JSON files"
    )

    parser.add_argument(
        "--out-fits",
        required=True,
        help="Output merged FITS file"
    )

    args = parser.parse_args()

    merge_calc_to_fits(
        fits_path=args.fits,
        json_dir=args.json_dir,
        output_fits=args.out_fits
    )


if __name__ == "__main__":
    main()