import os
import numpy as np
from scipy.stats import gaussian_kde, norm
import sys
sys.path.append("../lib")

from notion import get_lens_data_by_name

def compare_models_by_bayes_factor(
    base_dir,
    simu_obj_list,
    response,
    sim_type_list,
    total_runs,
    baseline_model="CDM",  # Baseline model
    show_scale=True
):
    jeffreys_scale = [
        (1/float('inf'), 1, "Evidence favors M0"),
        (1, 3, "Anecdotal evidence for M1"),
        (3, 10, "Moderate evidence for M1"),
        (10, 30, "Strong evidence for M1"),
        (30, 100, "Very strong evidence for M1"),
        (100, float('inf'), "Decisive evidence for M1")
    ]

    def interpret_bayes_factor(bf):
        for low, high, desc in jeffreys_scale:
            if low < bf <= high:
                return desc
        return "?"

    results = {}

    for simu_obj in simu_obj_list:
        object_lensing = get_lens_data_by_name(response, simu_obj)
        ob_Rcusp = object_lensing["Rcusp_use"]
        ob_sigma = object_lensing["Rcusp_sigma"]

        evidence_dict = {}

        for sim_type in sim_type_list:
            all_Rcusp = []
            all_weights = []

            for i in range(1, total_runs + 1):
                rcusp_file = os.path.join(base_dir, f"{simu_obj}/{i}/{simu_obj}_{sim_type}_{i}_Rcusp.npz")
                if not os.path.exists(rcusp_file):
                    continue
                data = np.load(rcusp_file)
                Rcusp_arr = data["Rcusp_arr"]
                weights = data["weights"]
                all_Rcusp.append(Rcusp_arr)
                all_weights.append(weights * len(weights))

            if not all_Rcusp:
                continue

            Rcusp_all = np.concatenate(all_Rcusp)
            weights_all = np.concatenate(all_weights)

            kde = gaussian_kde(Rcusp_all, weights=weights_all)
            x_eval = np.linspace(min(Rcusp_all), max(Rcusp_all), 500)
            pdf = kde(x_eval)
            pdf /= np.trapz(pdf, x_eval)

            obs_likelihood = norm.pdf(ob_Rcusp, loc=x_eval, scale=ob_sigma)
            evidence = np.trapz(pdf * obs_likelihood, x_eval)
            evidence_dict[sim_type] = evidence

        base_Z = evidence_dict.get(baseline_model, None)
        if base_Z is None:
            print(f"❌ System {simu_obj} missing baseline model {baseline_model} data, skipping")
            continue

        print(f"\n📊 [System: {simu_obj}] Observed R_cusp = {ob_Rcusp:.3f} ± {ob_sigma:.3f}")
        print(f"Relative to baseline model: {baseline_model}")
        print(f"{'Model':<12} | {'Evidence':>10} | {'log₁₀(BF)':>10} | {'~σ':>6} | {'Explanation':<35}")
        print("-" * 85)

        results[simu_obj] = {}

        for model, Z in evidence_dict.items():
            BF = Z / base_Z
            logBF = np.log10(BF)
            sigma_equiv = 2.145 * np.sqrt(logBF) if logBF > 0 else 0
            interp = interpret_bayes_factor(BF) if show_scale else ""
            print(f"{model:<12} | {Z:10.4e} | {logBF:10.3f} | {sigma_equiv:6.2f} | {interp:<35}")
            results[simu_obj][model] = {
                "evidence": Z,
                "BF_vs_" + baseline_model: BF,
                "log10BF": logBF,
                "sigma_equiv": sigma_equiv,
                "interpretation": interp
            }

    return results

def compare_models_by_bayes_factor_pvalue(
    base_dir,
    simu_obj_list,
    response,
    sim_type_list,
    total_runs,
    baseline_model="CDM",
    show_scale=True
):
    jeffreys_scale = [
        (1/float('inf'), 1, "Evidence favors M0"),
        (1, 3, "Anecdotal evidence for M1"),
        (3, 10, "Moderate evidence for M1"),
        (10, 30, "Strong evidence for M1"),
        (30, 100, "Very strong evidence for M1"),
        (100, float('inf'), "Decisive evidence for M1")
    ]

    def interpret_bayes_factor(bf):
        for low, high, desc in jeffreys_scale:
            if low < bf <= high:
                return desc
        return "?"

    results = {}

    for simu_obj in simu_obj_list:
        object_lensing = get_lens_data_by_name(response, simu_obj)
        ob_Rcusp = object_lensing["Rcusp_use"]
        ob_sigma = object_lensing["Rcusp_sigma"]

        evidence_dict = {}
        pcum_dict = {}

        for sim_type in sim_type_list:
            all_Rcusp = []
            all_weights = []

            for i in range(1, total_runs + 1):
                rcusp_file = os.path.join(base_dir, f"{simu_obj}/{i}/{simu_obj}_{sim_type}_{i}_Rcusp.npz")
                if not os.path.exists(rcusp_file):
                    continue
                data = np.load(rcusp_file)
                Rcusp_arr = data["Rcusp_arr"]
                weights = data["weights"]
                all_Rcusp.append(Rcusp_arr)
                all_weights.append(weights * len(weights))

            if not all_Rcusp:
                continue

            Rcusp_all = np.concatenate(all_Rcusp)
            weights_all = np.concatenate(all_weights)

            kde = gaussian_kde(Rcusp_all, weights=weights_all)
            x_eval = np.linspace(min(Rcusp_all), max(Rcusp_all), 1000)
            pdf_vals = kde(x_eval)
            pdf_vals /= np.trapz(pdf_vals, x_eval)

            # Evidence
            obs_likelihood = norm.pdf(ob_Rcusp, loc=x_eval, scale=ob_sigma)
            evidence = np.trapz(pdf_vals * obs_likelihood, x_eval)
            evidence_dict[sim_type] = evidence

            # Cumulative probability within ±σ around R_obs

            in_window = x_eval >= ob_Rcusp
            p_cum = np.trapz(pdf_vals[in_window], x_eval[in_window])
            pcum_dict[sim_type] = p_cum

        base_Z = evidence_dict.get(baseline_model, None)
        if base_Z is None:
            print(f"❌ System {simu_obj} missing baseline model {baseline_model} data, skipping")
            continue

        print(f"\n📊 [System: {simu_obj}] Observed R_cusp = {ob_Rcusp:.3f} ± {ob_sigma:.3f}")
        print(f"Relative to baseline model: {baseline_model}")
        print(f"{'Model':<12} | {'Evidence':>10} | {'log₁₀(BF)':>10} | {'~σ':>6} | {'p_cum':>8} | {'Explanation':<35}")
        print("-" * 100)

        results[simu_obj] = {}

        for model, Z in evidence_dict.items():
            BF = Z / base_Z
            logBF = np.log10(BF)
            sigma_equiv = 2.145 * np.sqrt(logBF) if logBF > 0 else 0

            p_cum = pcum_dict.get(model, np.nan)
            interp = interpret_bayes_factor(BF) if show_scale else ""

            print(f"{model:<12} | {Z:10.4e} | {logBF:10.3f} | {sigma_equiv:6.2f} | {p_cum:8.2e} | {interp:<35}")

            results[simu_obj][model] = {
                "evidence": Z,
                "BF_vs_" + baseline_model: BF,
                "log10BF": logBF,
                "sigma_equiv": sigma_equiv,
                "p_cum": p_cum,
                "interpretation": interp
            }

    return results

def compare_models_joint_bayes_factor(
    base_dir,
    simu_obj_list,
    response,
    sim_type_list,
    total_runs,
    baseline_model="CDM",
    show_scale=True
):

    jeffreys_scale = [
        (1/float('inf'), 1, "Evidence favors M0"),
        (1, 3, "Anecdotal evidence for M1"),
        (3, 10, "Moderate evidence for M1"),
        (10, 30, "Strong evidence for M1"),
        (30, 100, "Very strong evidence for M1"),
        (100, float('inf'), "Decisive evidence for M1")
    ]

    def interpret_bayes_factor(bf):
        for low, high, desc in jeffreys_scale:
            if low < bf <= high:
                return desc
        return "?"

    logZ_total = {}

    for simu_obj in simu_obj_list:
        object_lensing = get_lens_data_by_name(response, simu_obj)
        ob_Rcusp = object_lensing["Rcusp_use"]
        ob_sigma = object_lensing["Rcusp_sigma"]

        for sim_type in sim_type_list:
            all_Rcusp = []
            all_weights = []

            for i in range(1, total_runs + 1):
                rcusp_file = os.path.join(base_dir, f"{simu_obj}/{i}/{simu_obj}_{sim_type}_{i}_Rcusp.npz")
                if not os.path.exists(rcusp_file):
                    continue
                data = np.load(rcusp_file)
                Rcusp_arr = data["Rcusp_arr"]
                weights = data["weights"]
                all_Rcusp.append(Rcusp_arr)
                all_weights.append(weights * len(weights))

            if not all_Rcusp:
                continue

            Rcusp_all = np.concatenate(all_Rcusp)
            weights_all = np.concatenate(all_weights)

            kde = gaussian_kde(Rcusp_all, weights=weights_all)
            x_eval = np.linspace(min(Rcusp_all), max(Rcusp_all), 500)
            pdf = kde(x_eval)
            pdf /= np.trapz(pdf, x_eval)

            obs_likelihood = norm.pdf(ob_Rcusp, loc=x_eval, scale=ob_sigma)
            evidence = np.trapz(pdf * obs_likelihood, x_eval)

            if sim_type not in logZ_total:
                logZ_total[sim_type] = 0

            if evidence > 0:
                logZ_total[sim_type] += np.log(evidence)

    base_logZ = logZ_total.get(baseline_model, None)
    if base_logZ is None:
        print(f"❌ Joint evidence for baseline model {baseline_model} is missing; cannot compare.")
        return

    print(f"\n🔗 Joint model comparison (baseline model: {baseline_model})")
    print(f"{'Model':<12} | {'log₁₀(BF)':>10} | {'~σ':>6} | {'Explanation':<35}")
    print("-" * 70)

    results = {}
    for model, logZ in logZ_total.items():
        delta_log = logZ - base_logZ
        log10BF = delta_log / np.log(10)
        BF = 10 ** log10BF
        sigma_equiv = 2.145 * np.sqrt(log10BF) if log10BF > 0 else 0
        interp = interpret_bayes_factor(BF) if show_scale else ""
        print(f"{model:<12} | {log10BF:10.3f} | {sigma_equiv:6.2f} | {interp:<35}")
        results[model] = {
            "log10BF_vs_" + baseline_model: log10BF,
            "BF": BF,
            "sigma_equiv": sigma_equiv,
            "interpretation": interp
        }

    return results


def kde_marginal_likelihood_2d(
    sim_samples,             # dict: {model: {"R": array, "phi": array}}
    observations,            # list of dicts: {"R":..,"phi":..,"sigma_R":..,"sigma_phi":..}
    ref_model=None,
    bw_method="scott",       # "scott" | "silverman" | float
    jitter=1e-9,
    chunk=50000,
    return_grid=False
):
    """
    Build 2D Gaussian KDE \hat p_M(R,phi) with measurement errors folded into kernel width (H + Sigma_i):
        p_i = (1/n) Σ_j N([R_i,phi_i] | [R_j,phi_j], H + Sigma_i)
        Z_M = Π_i p_i
    """
    import numpy as np
    import math

    # ===================== FIX: keep 'scott'/'silverman' strings from being treated as floats =====================
    def _bandwidth_matrix(X, method="scott"):
        # X: (n,2)
        n, d = X.shape
        cov = np.cov(X, rowvar=False)

        # Only treat numeric types as scale factors
        if isinstance(method, (int, float, np.floating)):
            h = float(method)
        else:
            if method == "scott":
                h = n ** (-1.0 / (d + 4))
            elif method == "silverman":
                h = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
            else:
                raise ValueError("bw_method must be a number, or 'scott'/'silverman'")

        H = (h ** 2) * cov
        # Numerical stability: enforce symmetry and add diagonal jitter
        H = 0.5 * (H + H.T) + np.eye(2) * jitter
        return H
    # ==========================================================================================

    def _gauss_pdf_sum(x, samples, S, chunk=50000):
        """
        Compute (1/n) * sum_j N(x | samples_j, S)
        x: (2,), samples: (n,2), S: (2,2)
        """
        n = samples.shape[0]
        # Numerical stability
        S = 0.5 * (S + S.T)
        # Avoid singular matrices
        detS = np.linalg.det(S)
        if not np.isfinite(detS) or detS <= 0:
            S = S + np.eye(2) * 1e-9
            detS = np.linalg.det(S)

        invS = np.linalg.inv(S)
        norm = 1.0 / (2.0 * np.pi * np.sqrt(detS) + 1e-300)

        acc = 0.0
        for start in range(0, n, chunk):
            sl = slice(start, min(start + chunk, n))
            D = samples[sl] - x  # (m,2)
            q = np.einsum("ni,ij,nj->n", D, invS, D)
            acc += np.exp(-0.5 * q).sum()
        return norm * acc / n

    def _logZ_for_one_model(R, PHI, obs, bw_method):
        X = np.column_stack([R, PHI]).astype(float)
        H = _bandwidth_matrix(X, method=bw_method)

        logp_list = []
        for o in obs:
            x = np.array([o["R"], o["phi"]], dtype=float)
            Sigma = np.diag([o["sigma_R"] ** 2, o["sigma_phi"] ** 2])
            S_eff = H + Sigma  # Convolution: kernel covariance + measurement covariance
            p = _gauss_pdf_sum(x, X, S_eff, chunk=chunk)
            logp_list.append(np.log(p + 1e-300))
        return float(np.sum(logp_list))

    # Compute log marginal likelihood for each model
    logZ = {}
    for model, d in sim_samples.items():
        Rm = np.asarray(d["R"]).ravel()
        Ph = np.asarray(d["phi"]).ravel()
        if len(Rm) == 0:
            logZ[model] = -np.inf
            continue
        logZ[model] = _logZ_for_one_model(Rm, Ph, observations, bw_method)

    Z = {m: math.exp(v) for m, v in logZ.items()}
    if ref_model is None:
        ref_model = max(Z, key=Z.get)

    BF = {m: (Z[m] / (Z[ref_model] + 1e-300)) for m in Z}
    log10_BF = {m: (0.0 if m == ref_model else math.log10(BF[m] + 1e-300)) for m in Z}
    sigma_equiv = {m: (0.0 if m == ref_model else math.sqrt(2.0 * max(0.0, math.log(BF[m] + 1e-300)))) for m in Z}

    return {
        "Z": Z,
        "logZ": logZ,
        "BF_vs_ref": BF,
        "log10_BF": log10_BF,
        "sigma_equiv": sigma_equiv,
        "ref_model": ref_model,
    }


def build_sim_samples_for_axis(sim_dict, sim_type_list):
    import numpy as np
    sim_samples = {}
    for sim_type in sim_type_list:
        if sim_dict[sim_type]["Rcusp"]:
            R = np.concatenate(sim_dict[sim_type]["Rcusp"])
            P = np.concatenate(sim_dict[sim_type]["phi"])
            sim_samples[sim_type] = {"R": R, "phi": P}
    return sim_samples

def make_observations_from_notion(
    notion_response,
    axis_filter=None,   # 'short' | 'long' | None
    rcusp_max=1,
    require_sigma=True,
    exclude_names=None
):
    """
    Extract observations from a Notion response, keeping fields phi, Rcusp, axis_type, sigma_R, sigma_phi.
    Returns a list of dicts like:
      {"R": float, "phi": float, "sigma_R": float, "sigma_phi": float, "name": str}
    """
    import math
    from notion import get_lens_data_by_name, get_all_lens_names
    exclude_names = set(exclude_names or [])     # &&&!!!(2025-11-21 15:58)

    def _to_float(x):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except Exception:
            return None

    names = get_all_lens_names(notion_response) or []
    obs = []

    for name in names:
        # Skip excluded names when provided
        if name in exclude_names:              # &&&!!!(2025-11-21 15:45)
            continue

        d = get_lens_data_by_name(notion_response, name)
        if not isinstance(d, dict):
            continue

        phi = _to_float(d.get("phi"))
        R = _to_float(d.get("Rcusp"))
        if phi is None or R is None:
            continue
        if R > rcusp_max:
            continue

        at = (d.get("axis_type") or "").strip().lower()
        if axis_filter is not None and at != axis_filter:
            continue
        if at not in ("long", "short"):
            continue

        sigma_phi = _to_float(d.get("phi_sigma"))
        sigma_R = _to_float(d.get("Rcusp_sigma"))


        if require_sigma and (sigma_phi is None or sigma_R is None):
            continue

        obs.append({
            "R": R,
            "phi": phi,
            "sigma_R": float(sigma_R if sigma_R is not None else 0.05),
            "sigma_phi": float(sigma_phi if sigma_phi is not None else 2),
            "name": name
        })

    return obs


# Weight-aware collection helpers

def collect_phi_rcusp_all_indices_Mock_with_weights(
    simu_obj_list,
    sim_type_list,
    data_dir="Theory_Mock",
    max_index=100,
    percentile_threshold=None
):
    """
    Collect (Rcusp, phi, weight) and merge into arrays per obj.
    Returns all_results[obj][axis_type][sim_type] with Rcusp, phi, w arrays (raw weights).
    """
    import os, glob, json
    import numpy as np
    from tqdm import tqdm

    all_system_results = {}

    for obj in simu_obj_list:
        axis_type_dict = {}
        print(f"📦 Processing system/object: {obj}")
        for index in tqdm(range(1, max_index + 1), desc=f"Index 1 to {max_index}"):
            dir_path = f"{data_dir}/{obj}/{index}"
            if not os.path.isdir(dir_path):
                continue

            # Read params.json to obtain axis_type
            axis_type_from_params = None
            json_candidate = os.path.join(dir_path, f"{obj}_{index}_params.json")
            json_path = json_candidate if os.path.exists(json_candidate) else None
            if json_path is None:
                matches = glob.glob(os.path.join(dir_path, "*_params.json"))
                if matches:
                    json_path = matches[0]
            if json_path:
                try:
                    with open(json_path, "r") as f:
                        params = json.load(f)
                    axis_type_from_params = params.get("axis_type", None)
                except Exception:
                    axis_type_from_params = None
            if axis_type_from_params is None:
                continue

            # Initialize structure for this axis_type
            if axis_type_from_params not in axis_type_dict:
                axis_type_dict[axis_type_from_params] = {
                    st: {"Rcusp": [], "phi": [], "w": []} for st in sim_type_list
                }

            for sim_type in sim_type_list:
                Rcusp_chunks, phi_chunks, w_chunks = [], [], []

                for phi_val in np.arange(20, 151, 10):
                    matched = glob.glob(os.path.join(dir_path, f"*_{sim_type}_{phi_val}_Rcusp_phi.npz"))
                    if not matched:
                        legacy = os.path.join(dir_path, f"{obj}_{sim_type}_{phi_val}_Rcusp_phi.npz")
                        if os.path.exists(legacy):
                            matched = [legacy]

                    for file_path in matched:
                        try:
                            data = np.load(file_path)
                            Rcusp_arr  = np.asarray(data["Rcusp_arr"])
                            angles_arr = np.asarray(data["angles"])
                            weights     = np.asarray(data["weights"])
                            axis_types  = data["axis_types"]

                            # Axis mask
                            if np.isscalar(axis_types) or isinstance(axis_types, (str, bytes)):
                                is_same = (str(axis_types) == str(axis_type_from_params))
                                axis_mask = np.ones_like(weights, dtype=bool) if bool(is_same) else np.zeros_like(weights, dtype=bool)
                            else:
                                axis_types = np.asarray(axis_types)
                                if axis_types.shape != weights.shape:
                                    continue
                                axis_mask = (axis_types.astype(str) == str(axis_type_from_params))

                            # Optional weight threshold
                            if percentile_threshold is not None:
                                thresh = np.percentile(weights, percentile_threshold)
                                weight_mask = (weights >= thresh)
                            else:
                                weight_mask = np.ones_like(weights, dtype=bool)

                            mask = axis_mask & weight_mask
                            if np.any(mask):
                                Rcusp_chunks.append(Rcusp_arr[mask])
                                phi_chunks.append(angles_arr[mask])
                                w_chunks.append(weights[mask])
                        except Exception:
                            continue

                if Rcusp_chunks:
                    axis_type_dict[axis_type_from_params][sim_type]["Rcusp"].append(np.concatenate(Rcusp_chunks))
                    axis_type_dict[axis_type_from_params][sim_type]["phi"].append(np.concatenate(phi_chunks))
                    axis_type_dict[axis_type_from_params][sim_type]["w"].append(np.concatenate(w_chunks))

        # Merge all indices for this obj into single arrays
        for axis_key in axis_type_dict:
            for sim_type in sim_type_list:
                R_list = axis_type_dict[axis_key][sim_type]["Rcusp"]
                P_list = axis_type_dict[axis_key][sim_type]["phi"]
                W_list = axis_type_dict[axis_key][sim_type]["w"]
                if R_list:
                    axis_type_dict[axis_key][sim_type] = {
                        "Rcusp": np.concatenate(R_list),
                        "phi":   np.concatenate(P_list),
                        "w":     np.concatenate(W_list),
                    }
                else:
                    axis_type_dict[axis_key][sim_type] = {
                        "Rcusp": np.array([]),
                        "phi":   np.array([]),
                        "w":     np.array([]),
                    }

        all_system_results[obj] = axis_type_dict

    return all_system_results
# Merge different systems grouped by axis_type while retaining weights
def merge_by_axis_type_with_weights(all_results, sim_type_list):
    import numpy as np
    merged = {}
    for obj_name, axis_type_dict in all_results.items():
        for axis_type_name, sim_data_dict in axis_type_dict.items():
            if axis_type_name not in merged:
                merged[axis_type_name] = {st: {"Rcusp": [], "phi": [], "w": []} for st in sim_type_list}
            for sim_type, data_pair in sim_data_dict.items():
                if data_pair["Rcusp"].size > 0:
                    merged[axis_type_name][sim_type]["Rcusp"].append(data_pair["Rcusp"])
                    merged[axis_type_name][sim_type]["phi"].append(data_pair["phi"])
                    merged[axis_type_name][sim_type]["w"].append(data_pair["w"])
    # Flatten lists into arrays
    for axis_type_name, sim_dict in merged.items():
        for sim_type, d in sim_dict.items():
            if d["Rcusp"]:
                sim_dict[sim_type] = {
                    "Rcusp": np.concatenate(d["Rcusp"]),
                    "phi":   np.concatenate(d["phi"]),
                    "w":     np.concatenate(d["w"]),
                }
            else:
                sim_dict[sim_type] = {"Rcusp": np.array([]), "phi": np.array([]), "w": np.array([])}
    return merged

# Two-level directory variant
def collect_phi_rcusp_all_indices_Mock_with_weights_nested(
    simu_obj_list,
    sim_type_list,
    data_dir="Theory_Mock",
    max_index=100,
    percentile_threshold=None
):
    """
    Collect (Rcusp, phi, weight) under two-level structure {data_dir}/{obj}/{index}/{inner}/.
    Returns nested dict all_results[obj][axis_type][sim_type] with Rcusp/phi/w arrays.
    """
    import os, glob, json
    import numpy as np
    from tqdm import tqdm

    all_system_results = {}

    for obj in simu_obj_list:
        axis_type_dict = {}
        print(f"📦 Processing system/object: {obj}")
        for index in tqdm(range(1, max_index + 1), desc=f"Index 1 to {max_index}"):
            outer_dir = f"{data_dir}/{obj}/{index}"
            if not os.path.isdir(outer_dir):
                continue
            # print(outer_dir)
            # Traverse inner layer
            for inner in os.listdir(outer_dir):
                dir_path = os.path.join(outer_dir, inner)
                if not os.path.isdir(dir_path):
                    continue
                # print(dir_path)
                # Read params.json to obtain axis_type
                axis_type_from_params = None
                json_candidate = os.path.join(dir_path, f"{obj}_{index}_params.json")
                json_path = json_candidate if os.path.exists(json_candidate) else None
                if json_path is None:
                    matches = glob.glob(os.path.join(dir_path, "*_params.json"))
                    if matches:
                        json_path = matches[0]
                if json_path:
                    try:
                        with open(json_path, "r") as f:
                            params = json.load(f)
                        axis_type_from_params = params.get("axis_type", None)
                    except Exception:
                        axis_type_from_params = None
                if axis_type_from_params is None:
                    continue

                # Initialize structure for this axis_type
                if axis_type_from_params not in axis_type_dict:
                    axis_type_dict[axis_type_from_params] = {
                        st: {"Rcusp": [], "phi": [], "w": []} for st in sim_type_list
                    }

                for sim_type in sim_type_list:
                    Rcusp_chunks, phi_chunks, w_chunks = [], [], []

                    for phi_val in np.arange(20, 151, 10):
                        matched = glob.glob(os.path.join(dir_path, f"*_{sim_type}_{phi_val}_Rcusp_phi.npz"))
                        if not matched:
                            legacy = os.path.join(dir_path, f"{obj}_{sim_type}_{phi_val}_Rcusp_phi.npz")
                            if os.path.exists(legacy):
                                matched = [legacy]

                        for file_path in matched:
                            try:
                                data = np.load(file_path)
                                Rcusp_arr  = np.asarray(data["Rcusp_arr"])
                                angles_arr = np.asarray(data["angles"])
                                weights     = np.asarray(data["weights"])
                                axis_types  = data["axis_types"]

                                # Axis mask
                                if np.isscalar(axis_types) or isinstance(axis_types, (str, bytes)):
                                    is_same = (str(axis_types) == str(axis_type_from_params))
                                    axis_mask = np.ones_like(weights, dtype=bool) if bool(is_same) else np.zeros_like(weights, dtype=bool)
                                else:
                                    axis_types = np.asarray(axis_types)
                                    if axis_types.shape != weights.shape:
                                        continue
                                    axis_mask = (axis_types.astype(str) == str(axis_type_from_params))

                                # Optional weight threshold
                                if percentile_threshold is not None:
                                    thresh = np.percentile(weights, percentile_threshold)
                                    weight_mask = (weights >= thresh)
                                else:
                                    weight_mask = np.ones_like(weights, dtype=bool)

                                mask = axis_mask & weight_mask
                                if np.any(mask):
                                    Rcusp_chunks.append(Rcusp_arr[mask])
                                    phi_chunks.append(angles_arr[mask])
                                    w_chunks.append(weights[mask])
                            except Exception:
                                continue

                    if Rcusp_chunks:
                        axis_type_dict[axis_type_from_params][sim_type]["Rcusp"].append(np.concatenate(Rcusp_chunks))
                        axis_type_dict[axis_type_from_params][sim_type]["phi"].append(np.concatenate(phi_chunks))
                        axis_type_dict[axis_type_from_params][sim_type]["w"].append(np.concatenate(w_chunks))

        # Merge all index+inner data for this obj into single arrays
        for axis_key in axis_type_dict:
            for sim_type in sim_type_list:
                R_list = axis_type_dict[axis_key][sim_type]["Rcusp"]
                P_list = axis_type_dict[axis_key][sim_type]["phi"]
                W_list = axis_type_dict[axis_key][sim_type]["w"]
                if R_list:
                    axis_type_dict[axis_key][sim_type] = {
                        "Rcusp": np.concatenate(R_list),
                        "phi":   np.concatenate(P_list),
                        "w":     np.concatenate(W_list),
                    }
                else:
                    axis_type_dict[axis_key][sim_type] = {
                        "Rcusp": np.array([]),
                        "phi":   np.array([]),
                        "w":     np.array([]),
                    }

        all_system_results[obj] = axis_type_dict

    return all_system_results

# Build sim_samples with weights for KDE likelihood
def build_sim_samples_for_axis_with_weights(sim_dict, sim_type_list):
    sim_samples = {}
    for sim_type in sim_type_list:
        R = sim_dict[sim_type]["Rcusp"]
        P = sim_dict[sim_type]["phi"]
        W = sim_dict[sim_type]["w"]
        if R.size > 0:
            sim_samples[sim_type] = {"R": R, "phi": P, "w": W}
    return sim_samples


# Weighted covariance & effective sample size helpers
# &&&!!!(2025-09-03 19:00)
def _cov_w(X, w, eps=1e-12):
    import numpy as np
    w = np.asarray(w, dtype=float).ravel()
    w = np.clip(w, 0.0, None)
    sw = w.sum()
    if not np.isfinite(sw) or sw <= 0:
        return np.cov(X, rowvar=False)
    mu = (w[:, None] * X).sum(axis=0) / sw
    Y  = X - mu
    cov = (w[:, None] * Y).T @ Y / sw  # Population-style; stable for KDE
    cov = 0.5 * (cov + cov.T) + np.eye(X.shape[1]) * eps
    return cov

# Effective sample size
# &&&!!!(2025-09-03 19:00)
def _n_eff(w):
    import numpy as np
    w = np.asarray(w, dtype=float).ravel()
    s1 = w.sum()
    s2 = (w * w).sum()
    return float((s1 * s1) / max(s2, 1e-300))


def kde_marginal_likelihood_2d_weighted(
    sim_samples,             # {model: {"R": array, "phi": array, "w": array or None}}
    observations,            # [{"R","phi","sigma_R","sigma_phi", ...}]
    ref_model=None,
    bw_method="scott",
    jitter=1e-9,
    chunk=50000
):
    """
    Weighted version of KDE marginal likelihood:
        p_i = (1/sum_j w_j) * Σ_j w_j * N(x_i | x_j, H + Σ_i)
    Falls back to equal weights when w is None.
    """
    import numpy as np, math

    # Weighted bandwidth matrix supporting n_eff and weighted covariance
    # &&&!!!(2025-09-03 19:00)
    def _bandwidth_matrix(X, method="scott", w=None, use_neff=True, weighted_cov=True):
        import numpy as np
        n, d = X.shape

        # Weighted covariance by default; fallback to unweighted when weights absent/disabled
        cov = _cov_w(X, w) if (weighted_cov and w is not None) else np.cov(X, rowvar=False)

        # Choose n or n_eff to compute h
        n_used = _n_eff(w) if (w is not None and use_neff) else float(n)

        if isinstance(method, (int, float, np.floating)):
            h = float(method)
        else:
            if method == "scott":
                h = n_used ** (-1.0 / (d + 4))
            elif method == "silverman":
                h = (n_used * (d + 2) / 4.0) ** (-1.0 / (d + 4))
            else:
                raise ValueError("bw_method must be a number, or 'scott'/'silverman'")

        H = (h ** 2) * cov
        H = 0.5 * (H + H.T) + np.eye(2) * jitter
        return H

    def _gauss_pdf_sum_weighted(x, samples, S, w=None, chunk=50000):
        n = samples.shape[0]
        S = 0.5 * (S + S.T)
        detS = np.linalg.det(S)
        if not np.isfinite(detS) or detS <= 0:
            S = S + np.eye(2) * 1e-9
            detS = np.linalg.det(S)

        invS = np.linalg.inv(S)
        norm = 1.0 / (2.0 * np.pi * np.sqrt(detS) + 1e-300)

        if w is None:
            sum_w = float(n)
        else:
            w = np.asarray(w, dtype=float).ravel()
            sum_w = float(np.sum(w) + 1e-300)

        acc = 0.0
        for start in range(0, n, chunk):
            sl = slice(start, min(start + chunk, n))
            D = samples[sl] - x
            q = np.einsum("ni,ij,nj->n", D, invS, D)
            kern = np.exp(-0.5 * q)
            if w is None:
                acc += kern.sum()
            else:
                acc += (w[sl] * kern).sum()
        return norm * acc / sum_w

    # Pass W to bandwidth matrix and enable weighting
    # &&&!!!(2025-09-03 19:00)
    def _logZ_for_one_model(R, PHI, W, obs, bw_method):
        X = np.column_stack([R, PHI]).astype(float)
        H = _bandwidth_matrix(X, method=bw_method, w=W, use_neff=True, weighted_cov=True)  # Use weighted bandwidth
        logp_list = []
        for o in obs:
            x = np.array([o["R"], o["phi"]], dtype=float)
            Sigma = np.diag([o["sigma_R"] ** 2, o["sigma_phi"] ** 2])
            S_eff = H + Sigma
            p = _gauss_pdf_sum_weighted(x, X, S_eff, w=W, chunk=chunk)
            logp_list.append(np.log(p + 1e-300))
        return float(np.sum(logp_list))

    logZ = {}
    for model, d in sim_samples.items():
        Rm = np.asarray(d["R"]).ravel()
        Ph = np.asarray(d["phi"]).ravel()
        W  = d.get("w", None)
        if W is not None:
            W = np.asarray(W).ravel()
            if W.size != Rm.size:
                # Ignore weights when length mismatches
                W = None
        if len(Rm) == 0:
            logZ[model] = -np.inf
            continue
        logZ[model] = _logZ_for_one_model(Rm, Ph, W, observations, bw_method)

    Z = {m: math.exp(v) for m, v in logZ.items()}
    if ref_model is None and Z:
        ref_model = max(Z, key=Z.get)
    BF = {m: (Z[m] / (Z.get(ref_model, 1.0) + 1e-300)) for m in Z}
    log10_BF = {m: (0.0 if m == ref_model else math.log10(BF[m] + 1e-300)) for m in Z}
    sigma_equiv = {m: (0.0 if m == ref_model else math.sqrt(2.0 * max(0.0, math.log(BF[m] + 1e-300)))) for m in Z}

    return {
        "Z": Z,
        "logZ": logZ,
        "BF_vs_ref": BF,
        "log10_BF": log10_BF,
        "sigma_equiv": sigma_equiv,
        "ref_model": ref_model,
    }

import math
from typing import Dict, Any

def combine_independent_bayes_results(res_a: Dict[str, Any], res_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine two independent Bayes-evidence results (e.g., from different axis tests).
    Assumes independence: Z_combined = Z_a * Z_b. Recomputes BF/logBF from combined Z.
    If both inputs have 'sigma_equiv', combine them in quadrature as an approximate
    Gaussian-equivalent significance combiner.

    Expected input structure (per result):
      {
        'Z': {model: float, ...},
        'logZ': {model: float, ...},
        'BF_vs_ref': {model: float, ...},
        'log10_BF': {model: float, ...},
        'sigma_equiv': {model: float, ...},  # optional
        'ref_model': 'CDM'
      }
    """
    # Reference model consistency (fallback to res_a if mismatch)
    ref_model = res_a.get('ref_model', None)
    if res_b.get('ref_model', ref_model) != ref_model:
        # Prefer res_a's ref if inconsistent
        ref_model = res_a.get('ref_model')

    # Models present in both results
    models = sorted(set(res_a.get('Z', {}).keys()) & set(res_b.get('Z', {}).keys()))
    if not models:
        raise ValueError("No overlapping models between the two results.")

    # Combine evidences
    Z_combined = {}
    logZ_combined = {}
    for m in models:
        Za = float(res_a['Z'][m])
        Zb = float(res_b['Z'][m])
        Zc = Za * Zb
        Z_combined[m] = Zc
        # Use logs for stability when possible
        if Za > 0.0 and Zb > 0.0:
            logZa = math.log(Za)
            logZb = math.log(Zb)
            logZ_combined[m] = logZa + logZb
        else:
            logZ_combined[m] = float('-inf')

    # Bayes factors vs reference from combined Z
    Z_ref = Z_combined[ref_model]
    BF_vs_ref = {m: (Z_combined[m] / Z_ref) if Z_ref > 0.0 else float('inf') for m in models}
    # log10 Bayes factors
    log10_BF = {m: (math.log10(BF_vs_ref[m]) if BF_vs_ref[m] > 0.0 else float('-inf')) for m in models}

    # Combine sigma_equiv if available in both (quadrature, treating them as independent z-scores)
    sigma_equiv = {}
    has_sigma_a = 'sigma_equiv' in res_a and isinstance(res_a['sigma_equiv'], dict)
    has_sigma_b = 'sigma_equiv' in res_b and isinstance(res_b['sigma_equiv'], dict)
    if has_sigma_a and has_sigma_b:
        for m in models:
            sa = float(res_a['sigma_equiv'].get(m, 0.0))
            sb = float(res_b['sigma_equiv'].get(m, 0.0))
            sigma_equiv[m] = math.hypot(sa, sb)  # sqrt(sa^2 + sb^2)
    else:
        # If unavailable, default to zeros (user can recompute with their own mapping if needed)
        sigma_equiv = {m: 0.0 for m in models}

    combined = {
        'Z': Z_combined,
        'logZ': logZ_combined,
        'BF_vs_ref': BF_vs_ref,
        'log10_BF': log10_BF,
        'sigma_equiv': sigma_equiv,
        'ref_model': ref_model,
        'components': {
            'A': res_a,
            'B': res_b
        }
    }
    return combined
