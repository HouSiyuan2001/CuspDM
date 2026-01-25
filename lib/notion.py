from notion_client import Client
import numpy as np
from Lensing_tool import Da0


def get_all_lens_names(response):
    """
    Extract all system names from a Notion database response.
    Assumes response["results"] matches the structure returned by a Notion database query,
    where each entry has properties["Name"]["title"][0]["plain_text"].
    """
    names = []
    for item in response.get("results", []):
        try:
            title_data = item["properties"]["Name"]["title"]
            if title_data:
                names.append(title_data[0]["plain_text"])
        except Exception:
            continue
    return names

def get_lens_data_by_name(response, target_name):
    for result in response["results"]:
        props = result["properties"]

        # Fetch Name
        name_block = props["Name"]["title"]
        if not name_block:
            continue
        name = name_block[0]["plain_text"]

        if name != target_name:
            continue

        # Keep only entries where type == "cusp"
        type_values = props["type"]["multi_select"]
        type_names = [t["name"].strip().lower() for t in type_values]
        if not any(t == "cusp" for t in type_names):
            continue

        zlens = props["zlens"].get("number")
        theta = props["theta(arcsec)"].get("number")
        # print(name_block, theta)
        phi = props["phi"].get("number")

        axis_type = [t for t in type_names if t != "cusp"]
        axis_type = axis_type[0] if axis_type else None

        if zlens is None or theta is None:
            continue

        Da_zlens = Da0(zlens)  # Units: Mpc
        # Conversion constant: arcsec per rad
        apr = 1.0 / np.pi * 180.0 * 3600.0  # ≈ 206264.8
        arcsec_1 = 1 / apr * Da_zlens  # Units: Mpc
        theta_mpc = theta * arcsec_1 * 1e3  # θ (arcsec) → kpc

        rich_text = props.get("center-pixel", {}).get("rich_text", [])
        center_pixel_text = rich_text[0].get("plain_text", "") if rich_text else None

        return {
            "name": name,
            "zlens": zlens,
            "zsource": props.get("zsource", {}).get("number"), 
            "phi": phi,
            "phi_sigma": props.get("phi_sigma", {}).get("number"),
            "phi1": props.get("phi1", {}).get("number"),
            "phi1_sigma": props.get("phi1_sigma", {}).get("number"),
            "phi2": props.get("phi2", {}).get("number"),
            "phi2_sigma": props.get("phi2_sigma", {}).get("number"),
            "theta_arcsec": theta,
            "theta_kpc": theta_mpc,
            "Rcusp_sigma": props.get("Sigma_Rcusp_use", {}).get("number"),
            "Rcusp": props.get("Rcusp_use", {}).get("number"),
            "axis_type": axis_type,
            "Gamma": props.get("Gamma", {}).get("number"),
            "type_rcusp" : props.get("type_Rcusp_use", {}).get("select", {}).get("name"),
            "Gamma_sigma": props.get("Gamma_sigma", {}).get("number"),
            "gamma_external": props.get("gamma_external", {}).get("number"),
            "gamma_external_sigma": props.get("gamma_external_sigma", {}).get("number"),
            "phi_external": props.get("phi_external", {}).get("number"),
            "phi_external_sigma": props.get("phi_external_sigma", {}).get("number"),
            "phi_lens": props.get("phi_lens", {}).get("number"),
            "phi_lens_sigma": props.get("phi_lens_sigma", {}).get("number"),
            "q": props.get("q", {}).get("number"),
            "q_sigma": props.get("q_sigma", {}).get("number"),
            "thetaE": props.get("thetaE", {}).get("number"),
            "thetaE_sigma": props.get("thetaE_sigma", {}).get("number"),
            "x_imageA": props.get("x_imageA", {}).get("number"),
            "y_imageA": props.get("y_imageA", {}).get("number"),
            "x_imageA_sigma": props.get("x_imageA_sigma", {}).get("number"),
            "y_imageA_sigma": props.get("y_imageA_sigma", {}).get("number"),
            "x_imageB": props.get("x_imageB", {}).get("number"),
            "y_imageB": props.get("y_imageB", {}).get("number"),
            "x_imageB_sigma": props.get("x_imageB_sigma", {}).get("number"),
            "y_imageB_sigma": props.get("y_imageB_sigma", {}).get("number"),
            "x_imageC": props.get("x_imageC", {}).get("number"),
            "y_imageC": props.get("y_imageC", {}).get("number"),
            "x_imageC_sigma": props.get("x_imageC_sigma", {}).get("number"),
            "y_imageC_sigma": props.get("y_imageC_sigma", {}).get("number"),
            "x_imageD": props.get("x_imageD", {}).get("number"),
            "y_imageD": props.get("y_imageD", {}).get("number"),
            "x_imageD_sigma": props.get("x_imageD_sigma", {}).get("number"),
            "y_imageD_sigma": props.get("y_imageD_sigma", {}).get("number"),
            "center-pixel": center_pixel_text,
            "size": props.get("size", {}).get("number"),
            "nnn": props.get("nnn", {}).get("number"),
            "Sample_reff": props.get("Sample_reff", {}).get("number"),
            "delta_theta_FDM": props.get("delta_theta_FDM", {}).get("number"),
            
        }

    return None

def get_simple_params_from_lens(response, target_name):
    for result in response["results"]:
        props = result["properties"]
        zlens = props["zlens"].get("number")
        # Check Name
        name_block = props.get("Name", {}).get("title", [])
        if not name_block:
            continue
        name = name_block[0].get("plain_text", "")
        if name != target_name:
            continue

        # Extract parameters
        bsz_arc = props.get("bsz_arc", {}).get("number")
        nnn = props.get("nnn", {}).get("number")
        ys1 = props.get("ys1", {}).get("number")
        ys2 = props.get("ys2", {}).get("number")
        phi = props["phi"].get("number")
        return {
            "name": name,
            "zlens": zlens,
            "zsource": props.get("zsource", {}).get("number"), 
            "thetaE": props.get("thetaE", {}).get("number"),
            "bsz_arc": bsz_arc,
            "nnn": nnn,
            "ys1": ys1,
            "ys2": ys2,
            "x_imageA": props.get("x_imageA", {}).get("number"),
            "y_imageA": props.get("y_imageA", {}).get("number"),
            "x_imageA_sigma": props.get("x_imageA_sigma", {}).get("number"),
            "y_imageA_sigma": props.get("y_imageA_sigma", {}).get("number"),
            "x_imageB": props.get("x_imageB", {}).get("number"),
            "y_imageB": props.get("y_imageB", {}).get("number"),
            "x_imageB_sigma": props.get("x_imageB_sigma", {}).get("number"),
            "y_imageB_sigma": props.get("y_imageB_sigma", {}).get("number"),
            "x_imageC": props.get("x_imageC", {}).get("number"),
            "y_imageC": props.get("y_imageC", {}).get("number"),
            "x_imageC_sigma": props.get("x_imageC_sigma", {}).get("number"),
            "y_imageC_sigma": props.get("y_imageC_sigma", {}).get("number"),
            "x_imageD": props.get("x_imageD", {}).get("number"),
            "y_imageD": props.get("y_imageD", {}).get("number"),
            "x_imageD_sigma": props.get("x_imageD_sigma", {}).get("number"),
            "y_imageD_sigma": props.get("y_imageD_sigma", {}).get("number"),
            "Rcusp_sigma": props.get("Rcusp_sigma_use", {}).get("number"),
            "Rcusp": props.get("Rcusp_use", {}).get("number"),
        }

    return None

def get_Mock_lens_system_params(data, idx, base_name):
    system = data[idx]
    name = f"{base_name}_{system['cusp_id']}"

    params = {
        "name": name,
        "zlens": system["lens_redshift"],
        "zsource": system["source_redshift"],
        "phi": system["phi"],
        "phi_sigma": None,
        "phi1": system["phi1"],
        "phi1_sigma": None,
        "phi2": system["phi2"],
        "phi2_sigma": None,
        "Rcusp": system["Rcusp"],
        "gamma_external": system["amp_shear"],
        "phi_external": system["pa_shear"],
        "phi_lens": 0,
        "lambda_q": system["lambda_q"],
        "q": system["q_SIE"],
        "thetaE": None,
        "v_disp": system["v_disp"],
        "Sample_reff": None,
        "delta_theta_FDM": None,
        "nnn": 1500,
        "x_imageA": system["imageA_x"],
        "y_imageA": system["imageA_y"],
        "x_imageB": system["imageB_x"],
        "y_imageB": system["imageB_y"],
        "x_imageC": system["imageC_x"],
        "y_imageC": system["imageC_y"],
        "x_imageD": system["imageD_x"],
        "y_imageD": system["imageD_y"],
        "source_xlocation": system["source_xlocation"],
        "source_ylocation": system["source_ylocation"],

    }

    return params


def get_Mock_lens_system_params_mul(data, idx, base_name):
    system = data[idx]
    name = f"{base_name}_{system['cusp_id']}"

    def _get(system_record, key, default=0):
        try:
            return system_record[key]
        except Exception:
            return default

    params = {
        "name": name,
        "zlens": system["lens_redshift"],
        "zsource": system["source_redshift"],
        "phi": system["phi"],
        "phi_sigma": None,
        "phi1": system["phi1"],
        "phi1_sigma": None,
        "phi2": system["phi2"],
        "phi2_sigma": None,
        "Rcusp": system["Rcusp"],
        "gamma_external": system["amp_shear"],
        "phi_external": system["pa_shear"],
        "phi_lens": 0,
        "lambda_q": system["lambda_q"],
        "q": system["q_SIE"],
        "thetaE": None,
        "v_disp": system["v_disp"],
        "Sample_reff": None,
        "delta_theta_FDM": None,
        "nnn": 1500,
        "x_imageA": system["imageA_x"],
        "y_imageA": system["imageA_y"],
        "x_imageB": system["imageB_x"],
        "y_imageB": system["imageB_y"],
        "x_imageC": system["imageC_x"],
        "y_imageC": system["imageC_y"],
        "x_imageD": system["imageD_x"],
        "y_imageD": system["imageD_y"],
        "source_xlocation": system["source_xlocation"],
        "source_ylocation": system["source_ylocation"],

        # Fields for multipole terms (default to 0 if missing)
        "a3_over_a_signed": _get(system, "a3_over_a_signed", 0),
        "delta_phi_m3": _get(system, "delta_phi_m3", 0),
        "a4_over_a_signed": _get(system, "a4_over_a_signed", 0),
        "delta_phi_m4": _get(system, "delta_phi_m4", 0),

        "center_x": 0,
        "center_y": 0,
        "gamma_slope": 2,
    }

    return params

# Example usage
# file_path = "Theory_Mock/cusp_high_observable.fits"
# lens_params = get_Mock_lens_system_params(file_path)
# print(lens_params)
