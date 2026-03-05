import numpy as np
from astropy.io import fits
import jax.numpy as jnp
from tqdm import tqdm

def compute_theoretical_angles(edge_points, negative_point, center_point):
    """
    Inputs:
        edge_points: np.ndarray, shape (3, 2) - coordinates of the three clustered points
        negative_point: np.ndarray, shape (2,) - the point treated as the cusp negative point
        center_point: np.ndarray, shape (2,) - vertex point (typically (0, 0))
    
    Returns:
        dict {
            "phi":   total angle (between the two points excluding negative_point),
            "phi1":  angle between cusp_negative_point and one edge point,
            "phi2":  angle between cusp_negative_point and the other edge point
        }
    """
    # Find the two points other than negative_point.
    mask = ~np.all(edge_points == negative_point, axis=1)
    other_two_pts = edge_points[mask]

    # phi: angle formed by other_two_pts.
    v1 = other_two_pts[0] - center_point
    v2 = other_two_pts[1] - center_point
    cos_phi = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    phi = np.degrees(np.arccos(np.clip(cos_phi, -1.0, 1.0)))

    # phi1 and phi2: angles between negative_point and each of the other points.
    v0 = negative_point - center_point

    cos_phi1 = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
    phi1 = np.degrees(np.arccos(np.clip(cos_phi1, -1.0, 1.0)))

    cos_phi2 = np.dot(v0, v2) / (np.linalg.norm(v0) * np.linalg.norm(v2))
    phi2 = np.degrees(np.arccos(np.clip(cos_phi2, -1.0, 1.0)))

    return {
        "phi": phi,
        "phi1": phi1,
        "phi2": phi2
    }
def Get_Rcusp_and_phi(mu_info):
    '''
    This version skips point classification and uses the three highest magnifications.
    '''
    abs_mu_with_info = [(abs(info["mu"]), info) for info in mu_info]

    # Sort by absolute value (ascending).
    abs_mu_with_info_sorted = sorted(abs_mu_with_info, key=lambda x: x[0])
    
    # Take the largest three and largest four.
    top3_infos = [entry[1] for entry in abs_mu_with_info_sorted[-3:]]
    top4_infos = [entry[1] for entry in abs_mu_with_info_sorted[-4:]]

    # Collect positions for cluster-style selection.
    positions4 = np.array([info["position"] for info in top4_infos])
    positions3 = np.array([info["position"] for info in top3_infos])
    center_idx = np.argmin([
    np.sum(np.linalg.norm(positions3[i] - np.delete(positions3, i, axis=0), axis=1))
    for i in range(3)
    ])
    cusp_negative_point = positions3[center_idx]
   
    mu_list = [abs(info["mu"]) for info in top3_infos]
    mu_signed = mu_list.copy()

    mu_signed[center_idx] *= -1

    # Rcusp calculation.
    mu1, mu2, mu3 = mu_signed
    R_cusp = jnp.abs(mu1 + mu2 + mu3) / (jnp.abs(mu1) + jnp.abs(mu2) + jnp.abs(mu3))

    # Resolve edge and center information.
    if center_idx is not None:
        edge_points = [info for i, info in enumerate(top3_infos) if i != center_idx]
    else:
        edge_points = top3_infos  # fallback case

    center_point = abs_mu_with_info_sorted[0][1]

    return {
        "mu_info": top3_infos,
        "R_cusp": R_cusp,
        "edge_points": edge_points,
        "negative_point": cusp_negative_point,  
        "center_point": center_point
    }


def select_cusp_systems_with_progress(file_path, 
                                      R_cusp_threshold=1, 
                                      phi_threshold=140, 
                                      symmetry_tolerance=5):
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        four_images = data[data['image_number'] == 5]
    
    cusp_candidates = []

    for row in tqdm(four_images, desc="Filtering cusp configurations"):
        mus = row['magnification']      
        xs = row['image_xlocation']
        ys = row['image_ylocation']

        mu_info = [
            {"mu": mus[i], "position": np.array([xs[i], ys[i]])}
            for i in range(4)
        ]
        mu_info.append({"mu": 0.0, "position": np.array([0.0, 0.0])})

        result = Get_Rcusp_and_phi(mu_info)

        edge_pts = np.array([info["position"] for info in result["edge_points"]])
        negative_point = result["negative_point"]
        negative_pos = (
            negative_point["position"] if isinstance(negative_point, dict) 
            else np.array(negative_point)
        )
        center_point = np.array(result["center_point"]["position"])

        angle_info = compute_theoretical_angles(edge_pts, negative_pos, center_point)

        # Symmetry condition.
        half_phi = angle_info["phi"] / 2
        cond_phi1 = abs(angle_info["phi1"] - half_phi) <= symmetry_tolerance
        cond_phi2 = abs(angle_info["phi2"] - half_phi) <= symmetry_tolerance

        if (
            (result["R_cusp"] < R_cusp_threshold) and 
            (angle_info["phi"] < phi_threshold) and
            cond_phi1 and cond_phi2
        ):
            all_positions = [np.array([xs[i], ys[i]]) for i in range(4)]
            edge_positions = [tuple(pt) for pt in edge_pts]
            negative_position_tuple = tuple(negative_pos)
            remaining = [
                tuple(pos) for pos in all_positions 
                if tuple(pos) not in edge_positions and tuple(pos) != negative_position_tuple
            ]
            imageD = np.array(remaining[0]) if remaining else None

            cusp_candidates.append({
                "data": row,
                "R_cusp": float(result["R_cusp"]),
                "phi": angle_info["phi"],
                "phi1": angle_info["phi1"],
                "phi2": angle_info["phi2"],
                "images": {
                    "A": negative_pos,
                    "B": edge_pts[0],
                    "C": edge_pts[1],
                    "D": imageD
                }
            })

    print(f"Selected {len(cusp_candidates)} cusp configuration systems")
    return cusp_candidates
