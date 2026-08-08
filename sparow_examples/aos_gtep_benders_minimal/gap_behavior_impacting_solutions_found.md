The interesting thing about these two files is that by shrinking the rel_gap we get
different solutions even as gurobi attempts to find points in optimality order.

The higher candidates relative gap gave solutions that were all infeasible.
When we shrank the gap, we got different solutions, some of which were feasible.

This difference is anamolous.
Likely indicates that the tree limb fathoming that happens in one tree, which is changed by the gap tolerances, impacts the solutions returned for optimality ordering
when there are more solutions within machine precision than you ask for in the pool.