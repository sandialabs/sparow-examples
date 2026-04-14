import pprint   # for pretty-printing the results dict
from sparow_examples.facilityloc.grid_facilityloc import *  # change this import to whatever exemplar you're running
from sparow.ph.ph_mpisppy import ProgressiveHedgingSolver_MPISPPY   # import sparow wrapper around mpisppy

sp = random_HF_LF1_grid_facilityloc()   # replace sp obj with whatever exemplar you're running
solver = ProgressiveHedgingSolver_MPISPPY() # solving with the sparow wrapper around mpisppy
solver.set_options(
    solver="gurobi",    # not sure we support other solvers right now
    max_iterations=10,  # i think this is 100 by default?
    loglevel="INFO",    # can replace with DEBUG, VERBOSE, etc.
    default_rho = 1.5,  # rho by default will already be 1.5
    mpisppy_options=["--lagrangian", "--xhatshuffle", "--rel-gap=0.01", "--default-rho=1.5"] # can customize these also
)

results_mpi = solver.solve(sp, solver="gurobi") # solution obj
if getattr(solver, "mpi_rank", 0) == 0:         # all the information gets sent to rank 0
    pprint.pprint(results_mpi.to_dict())        # pretty-print results
