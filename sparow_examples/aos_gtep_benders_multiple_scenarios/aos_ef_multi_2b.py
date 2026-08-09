#!/usr/bin/env python3
"""
EF-based AOS probe on multi-scenario Rung 2b sizes.

Mirrors the working single-scenario aos_ef_2b.py pattern exactly:

  sp = create_sp(...)
  sp.add_transformation(relax_second_stage)
  model = sp.create_EF(compact_repn=True)
  aos_results = or_topas.aos.gurobi_generate_solutions(model, ...)

Two scenarios (from scale_tests model_data):
  load_0p95  → alpha=0.95, Probability=0.5
  load_1p05  → alpha=1.05, Probability=0.5

Sizes (Rung 2b)
---------------
  stages=2, num_reps=1, len_reps=12, num_commit=8, num_dispatch=1
"""
from __future__ import annotations

import logging
import time

import or_topas.aos
from sparow.sp.util import relax_second_stage

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

try:
    from sparow_examples.aos_gtep_benders_multiple_scenarios.scale_tests import create_sp
except ImportError:
    from scale_tests import create_sp

rel_opt_gap = 0.01
sol_max = 100

print("\n--- Creating multi-scenario SP (Rung 2b sizes) ---", flush=True)
t0 = time.perf_counter()
sp = create_sp(
    stages=2,
    num_reps=1,
    len_reps=12,
    num_commit=8,
    num_dispatch=1,
)
print(f"  bundles: {list(sp.bundles)}", flush=True)
sp.add_transformation(relax_second_stage)
model = sp.create_EF(compact_repn=True)
print(f"  EF built in {time.perf_counter() - t0:.1f}s", flush=True)

print("\n--- Running EF AOS ---", flush=True)
print(f"  rel_opt_gap={rel_opt_gap}, num_solutions={sol_max}", flush=True)
t0 = time.perf_counter()
aos_results = or_topas.aos.gurobi_generate_solutions(
    model,
    num_solutions=sol_max,
    rel_opt_gap=rel_opt_gap,
)
print(f"  AOS finished in {time.perf_counter() - t0:.1f}s", flush=True)

print("\n--- Results Summary ---", flush=True)
print(f"Metadata: Max Sols {sol_max}, Rel_Opt_Gap {rel_opt_gap}")
print(f"Number of Solutions {len(aos_results)}")
for i, s in enumerate(aos_results):
    try:
        print(f"Sol {i}: objective: {s.objective().value}")
    except Exception as e:
        print(f"Sol {i}: (could not read objective: {e})")

print("Done.", flush=True)
