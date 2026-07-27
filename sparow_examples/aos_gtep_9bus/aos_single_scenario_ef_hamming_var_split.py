import json
import or_topas.aos
from sparow_examples.aos_gtep_9bus.aos_single import create_sp
from sparow.sp.util import relax_second_stage

OUTPUT_FILE = "aos_single_scenario_ef_hamming_full_vars.json"
FS_OUTPUT_FILE = "aos_single_scenario_ef_hamming_first_stage_vars.json"


print("\n--- Creating model ---")
sp = create_sp()
sp.add_transformation(relax_second_stage)
model = sp.create_EF(compact_repn=True)

# ------------------------------------------------------------------
# Collect the authoritative first-stage name set *from the EF*.
# Prefer the SPAROW maps; they already know the exact names that
# appear on this concrete EF (including any scenario-block prefixes
# introduced by compact_repn).
# ------------------------------------------------------------------
b0 = next(iter(sp.bundles))
fs_name_set = set()
if hasattr(sp, "int_to_FirstStageVar") and b0 in sp.int_to_FirstStageVar:
    fs_name_set = {v.name for v in sp.int_to_FirstStageVar[b0].values()}
elif hasattr(sp, "int_to_FirstStageVarName") and sp.int_to_FirstStageVarName:
    fs_name_set = set(sp.int_to_FirstStageVarName.values())

if not fs_name_set:
    # Defensive pattern fallback (should not be needed once maps are live)
    print("  WARNING: SPAROW first-stage maps empty; falling back to name patterns")
    def is_first_stage(name: str) -> bool:
        n = (name or "").lower()
        return ("investmentstage" in n) and (
            "binary_indicator" in n or "renewable" in n
        )
else:
    def is_first_stage(name: str) -> bool:
        return name in fs_name_set

print(f"  First-stage name set size: {len(fs_name_set)}")
if fs_name_set:
    for n in sorted(fs_name_set)[:6]:
        print(f"    {n}")
    if len(fs_name_set) > 6:
        print(f"    ... +{len(fs_name_set)-6} more")


print("\n--- Running AOS ---")
rel_opt_gap = 0.00001
sol_max = 3
search_mode = "hamming"
variables = None   # keep current behaviour; see remark below
aos_results = or_topas.aos.enumerate_binary_solutions(
    model,
    variables=variables,
    num_solutions=sol_max,
    rel_opt_gap=rel_opt_gap,
    solver="gurobi",
    search_mode=search_mode,
)

print("\n--- Results Summary ---")
print(f"Metadata: Max Sols {sol_max}, Rel_Gap {rel_opt_gap}")
print(f"Number of Solutions {len(aos_results)}")
for i, s in enumerate(aos_results):
    print(f"Sol {i}: objective: {s.objective().value}")


# ------------------------------------------------------------------
# Full archive (unchanged)
# ------------------------------------------------------------------
print("\n--- Saving full results ---")
aos_dict = aos_results.to_dict()
with open(OUTPUT_FILE, "w") as OUTPUT:
    json.dump(aos_dict, OUTPUT, indent=0)


# ------------------------------------------------------------------
# First-stage-only archive
# ------------------------------------------------------------------
def extract_first_stage(sol):
    """Project a Solution onto the first-stage variables.

    Returns a dict that is immediately compatible with
    sparow.sp.util.constrain_EF_model(..., first_stage_variables=...).
    """
    fs = {}
    for vinfo in sol.variables():
        if vinfo.name and is_first_stage(vinfo.name):
            # keep the raw float; downstream code can round if needed
            fs[vinfo.name] = float(vinfo.value) if vinfo.value is not None else None
    objs = list(sol.objectives())
    obj_val = float(objs[0].value) if objs else None
    return {
        "id": getattr(sol, "id", None),
        "objective": obj_val,
        "first_stage_variables": fs,          # name → value
        # optional richer form if you ever need the full VariableInfo metadata:
        # "variables": [v.to_dict() for v in sol.variables() if is_first_stage(v.name)],
    }

fs_archive = {
    "metadata": {
        "description": "First-stage projection of AOS solutions (investment indicators only)",
        "source_script": "aos_single_scenario_ef_hamming.py",
        "num_solutions": len(aos_results),
        "rel_opt_gap": rel_opt_gap,
        "search_mode": search_mode,
        "first_stage_name_set_size": len(fs_name_set),
    },
    "solutions": [extract_first_stage(s) for s in aos_results],
}

# diagnostic
if fs_archive["solutions"]:
    n_fs = len(fs_archive["solutions"][0]["first_stage_variables"])
    fs_archive["metadata"]["first_stage_vars_retained_per_sol"] = n_fs
    print(f"\n--- First-stage archive ---")
    print(f"  Retained {n_fs} first-stage variables per solution")
    if n_fs == 0:
        print("  WARNING: zero first-stage variables matched. "
              "Inspect fs_name_set vs. VariableInfo.name values.")
else:
    print("  WARNING: no solutions to project")

with open(FS_OUTPUT_FILE, "w") as f:
    json.dump(fs_archive, f, indent=2)
print(f"  Written to {FS_OUTPUT_FILE}")