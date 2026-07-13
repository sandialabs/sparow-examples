import json
from or_topas.aos import lp_enum
from sparow.sp.examples import (
    simple_newsvendor,
)
from sparow.ef import ExtensiveFormSolver

#
# Create newsvendor, solve and return model
#
app = simple_newsvendor()
solver = ExtensiveFormSolver()
solver.set_options(solver="gurobi")
results = solver.solve_and_return_EF(app.sp)
# TODO - remove solve step here

#results.model.pprint()

# Fixing newsvendor bounds
for index in results.model.s:
    results.model.s[index].x.ub = 1000
for index in results.model.s:
    results.model.s[index].y.lb = -1000
    results.model.s[index].y.ub = 1000

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

