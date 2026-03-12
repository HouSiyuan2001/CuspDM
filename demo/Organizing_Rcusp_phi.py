import os
import pickle
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB = os.path.join(ROOT, "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

from Bayesian import collect_phi_rcusp_all_indices_Mock_with_weights_nested, merge_by_axis_type_with_weights

def save_pickle(obj, filename):
    """
    Save an object to a pickle file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"[Saved] {filename}")

def load_pickle(filename):
    """
    Load an object from a pickle file.
    """
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    print(f"[Loaded] {filename}")
    return obj


# === Example usage ===
# 1) Collect (with weights)
all_results = collect_phi_rcusp_all_indices_Mock_with_weights_nested(
    simu_obj_list=["all_sim"],
    sim_type_list=["None","CDM", "SIDM", "FDM"],
    data_dir="Theory_Mock",
    max_index=300,
    percentile_threshold=None
)

# print(all_results)
# 2) Merge and group by axis_type (with weights)
merged_by_axis_type = merge_by_axis_type_with_weights(all_results, ["None","CDM", "SIDM", "FDM"])
save_pickle(merged_by_axis_type, "SavedResults/merged_by_axis_type.pkl")

# usage: merged_by_axis_type = load_pickle("SavedResults/merged_by_axis_type.pkl")
