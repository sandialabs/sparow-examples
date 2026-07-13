import json
from or_topas.aos import lp_enum
from sparow.ef import ExtensiveFormSolver
from sparow_examples.aos_gtep_9bus.aos_single import create_sp

#
# Create newsvendor, solve and return model
#
sp = create_sp()
solver = ExtensiveFormSolver()
solver.set_options(solver="gurobi")
results = solver.solve_and_return_EF(sp)
# TODO - remove solve step here

import sys
sys.exit(0)
# Run AOS
aos_results = lp_enum.enumerate_linear_solutions(results.model, solver="gurobi")

if True:
    for s in aos_results:
        print(s)
        print(s.objective().value)

# Dump results to a JSON file
aos_dict = aos_results.to_dict()
with open('newsvendor.json','w') as OUTPUT:
    json.dump(aos_dict, OUTPUT, indent=4)

