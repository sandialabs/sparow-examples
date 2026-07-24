# import facility location exemplar from sparow_examples
#from sparow_examples.facilityloc.grid_facilityloc import *
 
# import parallel PH solver (mpi-sppy wrapper)
from sparow.ph.ph_mpisppy import ProgressiveHedgingSolver_MPISPPY
from sparow.sp.examples import simple_newsvendor
from IPython import embed

import pprint
 
# SP model object
sp = simple_newsvendor()


# solving in parallel with the sparow wrapper around mpisppy
solver = ProgressiveHedgingSolver_MPISPPY()
 
solver.set_options(
    solver="ipopt",  # solving with gurobi
    max_iterations=100,  # this will default to 100
    loglevel="DEBUG",  # can replace with DEBUG, VERBOSE, etc.
    default_rho=1.5,  # rho by default will already be 1.5
    mpisppy_options=[
        "--tee-rank0-solves",
        "--lagrangian",
        "--xhatshuffle",
        "--rel-gap=0.01",
    ],  # can customize these options as well
)
 
results_mpi = solver.solve(sp['sp'])  # returns the solution object
 
if getattr(solver, "mpi_rank", 0) == 0:  # all the information gets sent to rank 0
    pprint.pprint(results_mpi.to_dict())  # pretty-print results
