Running UC Steps:

0. TODO - Add logic to track max memory usage
1. Be in sparow_examples/egret
2. Run the egret single scenario tests: python egret_sparow_staged.py
3. Run the egret multiple scenario tests: python egret_sparow_staged_multi.py \
    --force-synthetic \
    --n-periods 4 \
    --first-stage-names UnitOn \
    --run-benders \
    --max-benders-iterations 80 \
    2>&1 | tee egret_multi_rung1_benders.log
4. Run the egret aos tests: python run_aos_analysis_tiny_uc.py
5. Install https://github.com/power-grid-lib/pglib-uc
6. Running AOS analysis:
    a) Single Scenario Control File: python driver_egret_aos.py --intensity 3 --out_dir results/ --egret_file path/to/instance.json \\
        --first-stage-names UnitOn,UnitStart,UnitStop
    b) Multiple Scenario Control File: python driver_egret_aos_multi.py --intensity 3 --out_dir results/ --scales 0.9,1.0,1.1 --probs 0.3,0.4,0.3
    c) Bash File: still to come
