#!/usr/bin/env python3
"""
EF-based AOS probe on Rung 2b sizes (multi-stage, modest ops).

Uses or_topas.aos.gurobi_generate_solutions on the residual-relaxed extensive
form to check whether alternative near-optimal solutions exist before
investing in aos-benders.

Sizes (Rung 2b)
---------------
  stages=2, num_reps=1, len_reps=12, num_commit=8, num_dispatch=1
"""
from __future__ import annotations

import json
import time

import or_topas.aos
from sparow.sp.util import relax_second_stage

try:
    from sparow_examples.aos_gtep_benders_minimal.scale_tests import create_sp
except ImportError:
    from scale_tests import create_sp

OUTPUT_FILE = "aos_2b_ef.json"

rel_opt_gap = 0.01
sol_max = 100

print("\n--- Creating model (Rung 2b sizes) ---", flush=True)
t0 = time.perf_counter()
sp = create_sp(
    stages=2,
    num_reps=1,
    len_reps=12,
    num_commit=8,
    num_dispatch=1,
)
sp.add_transformation(relax_second_stage)
model = sp.create_EF(compact_repn=True)
print(f"  EF built in {time.perf_counter() - t0:.1f}s", flush=True)

print("\n--- Running AOS ---", flush=True)
print(f"  rel_opt_gap={rel_opt_gap}, num_solutions={sol_max}", flush=True)
t0 = time.perf_counter()
aos_results = or_topas.aos.gurobi_generate_solutions(
    model,
    num_solutions=sol_max,
    rel_opt_gap=rel_opt_gap,
)
print(f"  AOS finished in {time.perf_counter() - t0:.1f}s", flush=True)

if True:
    print("\n--- Results Summary ---", flush=True)
    print(f"Metadata: Max Sols {sol_max}, Rel_Opt_Gap {rel_opt_gap}")
    print(f"Number of Solutions {len(aos_results)}")
    for i, s in enumerate(aos_results):
        print(f"Sol {i}: objective: {s.objective().value}")

if False:
    print("\n--- Printing results ---", flush=True)
    for s in aos_results:
        print(s)
        print(s.objective().value)

# print("\n--- Saving results to an output file ---", flush=True)
# aos_dict = aos_results.to_dict()
# with open(OUTPUT_FILE, "w") as OUTPUT:
#     json.dump(aos_dict, OUTPUT, indent=0)
# print(f"Wrote {OUTPUT_FILE}", flush=True)
# print("Done.", flush=True)
