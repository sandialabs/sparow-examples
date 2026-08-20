#!/usr/bin/env python3
"""phase0_inspect.py – run once on the local conda branch and paste the full output back.

Usage (no PYTHONPATH changes):
  1. Copy this file into the same directory that already contains
     egret_sparow_staged.py (and the other original staged helpers).
  2. cd into that directory.
  3. python phase0_inspect.py 2>&1 | tee phase0_inspect.log
  4. Paste the full terminal output (and optionally the log) back.

Do not modify the original attachments.
"""
from __future__ import annotations
import copy
import inspect
import sys
import traceback


def main():
    print("=== Phase-0 environment inspection ===")
    print(f"Python: {sys.version}")

    # 1. Core imports + file locations
    import sparow
    import egret
    import or_topas
    import pyomo
    import gurobipy

    print(f"sparow     : {sparow.__file__}")
    print(f"egret      : {egret.__file__}")
    print(f"or_topas   : {or_topas.__file__}")
    print(f"pyomo      : {pyomo.__file__}")
    print(f"gurobipy   : {gurobipy.__file__}")

    from sparow.benders import BendersSolver

    print("\n--- BendersSolver signatures ---")
    print("_setup_topas_subproblem:", inspect.signature(BendersSolver._setup_topas_subproblem))
    print(
        "_transform_to_subproblem_model:",
        inspect.signature(BendersSolver._transform_to_subproblem_model),
    )
    print("solve_and_return_model present:", hasattr(BendersSolver, "solve_and_return_model"))
    if hasattr(BendersSolver, "solve_and_return_model"):
        print(
            "solve_and_return_model signature:",
            inspect.signature(BendersSolver.solve_and_return_model),
        )

    # 2. relax_second_stage
    print("\n--- relax_second_stage ---")
    try:
        from sparow.sp.util import relax_second_stage

        print("FOUND")
        src = inspect.getsource(relax_second_stage)
        print(src[:1500] + (" ..." if len(src) > 1500 else ""))
    except Exception as e:
        print(f"NOT FOUND / error: {type(e).__name__}: {e}")

    # 3. or_topas Benders data class
    print("\n--- or_topas Benders_Serial ---")
    try:
        from or_topas.benders.benders_serial import Benders_Serial

        print(
            "Benders_Serial imported OK, has add_subproblem:",
            hasattr(Benders_Serial, "add_subproblem"),
        )
    except Exception as e:
        print(f"Import failed: {type(e).__name__}: {e}")
        try:
            from or_topas.benders import Benders_Serial

            print("Fallback import OK")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")

    # 4. Contingencies / deepcopy smoke test on synthetic SP
    print("\n--- Contingencies deepcopy smoke test (synthetic tiny UC) ---")
    try:
        # Use the staged helpers that live next to this script / on PYTHONPATH
        from egret_sparow_staged import (
            stage0_check_environment,
            stage1_build_egret_model,
            stage2_inspect_variables,
            stage3_verify_first_stage_names,
            stage5_egret_sparow_sp,
        )

        stage0_check_environment()
        md, model = stage1_build_egret_model(force_synthetic=True, n_periods=4)
        binary_comps, _, suggested = stage2_inspect_variables(model)
        fs = [b["name"] for b in binary_comps if "uniton" in b["name"].lower()] or suggested
        stage3_verify_first_stage_names(model, fs)
        sp = stage5_egret_sparow_sp(
            md, first_stage_names=fs, scenario_scales={"base": 1.0}
        )
        print(f"SP built: bundles={list(sp.bundles)}")

        # Report Contingencies presence
        n_cont = 0
        for b in sp.bundles:
            m = None
            try:
                m = sp.get_model(b) if hasattr(sp, "get_model") else None
            except Exception:
                pass
            if m is None:
                try:
                    m = sp.s[None, b]
                except Exception:
                    pass
            if m is not None and (
                hasattr(m, "Contingencies") or hasattr(m, "contingencies")
            ):
                n_cont += 1
                print(f"  bundle {b}: Contingencies-like component present")
        print(f"Bundles with Contingencies-like components: {n_cont}")

        sp2 = copy.deepcopy(sp)
        print("deepcopy(sp) SUCCEEDED")
        print(f"type(sp2) = {type(sp2)}")
    except Exception:
        print("deepcopy / build FAILED:")
        traceback.print_exc()

    print("\n=== Phase-0 inspection complete – please paste the entire output back ===")


if __name__ == "__main__":
    main()
