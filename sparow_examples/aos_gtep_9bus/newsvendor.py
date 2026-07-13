from sparow.sp.examples import (
    simple_newsvendor,
)
from sparow.ef import ExtensiveFormSolver

app = simple_newsvendor()
solver = ExtensiveFormSolver()
solver.set_options(solver="gurobi")
results = solver.solve_and_return_EF(app.sp)
results_dict = results.solutions.to_dict()
soln = next(iter(results_dict["solutions"].values()))

